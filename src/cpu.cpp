#include "cpu.hpp"
#include "common.hpp"
#include <cmath>
#include <functional>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>

// Note: float vector _mm256_add_ps(vsum, squared) overflows
// Also this might double the used stack size if not handled correctly

// Note: not aligned! (must use *_loadu_*) (header size already makes the data
// addr not divisible by 32), plus it depends on the original allocator (even
// malloc does not guarantee alignement past 16)

inline void _cpu_extend_8xf32_to_2x4xf64_then_sum(const __m256 &value_8xf32,
                                                  __m256d &lo_d_sum,
                                                  __m256d &hi_d_sum) {
  __m256d hi_d = _mm256_cvtps_pd(_mm256_extractf128_ps(value_8xf32, 1));
  __m256d lo_d = _mm256_cvtps_pd(_mm256_castps256_ps128(value_8xf32));
  lo_d_sum = _mm256_add_pd(lo_d_sum, lo_d);
  hi_d_sum = _mm256_add_pd(hi_d_sum, hi_d);
}

inline __m256 _cpu_goback_2x4xf64_to_8xf32(const __m256d &lo_d_sum_4xf64,
                                           const __m256d &hi_d_sum_4xf64) {
  __m128 lo_sum_f = _mm256_cvtpd_ps(lo_d_sum_4xf64);
  __m128 hi_sum_f = _mm256_cvtpd_ps(hi_d_sum_4xf64);
  return _mm256_insertf128_ps(_mm256_castps128_ps256(lo_sum_f), hi_sum_f, 1);
}

// Reduction

double f32_norm(const vecx *v) {
  double sum = 0.0f;
  switch (v->header.dtype) {
  case FLOAT_32: {
    const float *data = static_cast<const float *>(v->data);
    uint64_t i = 0;
    const size_t block = 256 / 32;

    __m256d lo_d_sum = _mm256_setzero_pd();
    __m256d hi_d_sum = _mm256_setzero_pd();
    for (; i + block <= v->header.size; i += block) {
      __m256 value_8xf32 = _mm256_loadu_ps(data + i);
      __m256 squared = _mm256_mul_ps(value_8xf32, value_8xf32);
      _cpu_extend_8xf32_to_2x4xf64_then_sum(squared, lo_d_sum, hi_d_sum);
    }
    __m256 vsum = _cpu_goback_2x4xf64_to_8xf32(lo_d_sum, hi_d_sum);

    float tmp[8];
    _mm256_storeu_ps(tmp, vsum);
    for (size_t j = 0; j < 8; ++j)
      sum += tmp[j];

    // rest
    for (; i < v->header.size; ++i)
      sum += data[i] * data[i];

    break;
  }
  case QINT_8: {
    const int8_t *data = static_cast<const int8_t *>(v->data);
    size_t i = 0;
    const size_t block = 256 / 8;

    __m256d lo_d_sum = _mm256_setzero_pd();
    __m256d hi_d_sum = _mm256_setzero_pd();

    for (; i + block <= v->header.size; i += block) {
      __m256i bytes_32xi8 =
          _mm256_loadu_si256(reinterpret_cast<__m256i const *>(data + i));

      for (const __m256 &ext_8xf32 :
           _cpu_dequantize_fast(bytes_32xi8, v->header.qparams)) {
        __m256 squared = _mm256_mul_ps(ext_8xf32, ext_8xf32);
        _cpu_extend_8xf32_to_2x4xf64_then_sum(squared, lo_d_sum, hi_d_sum);
      }
    }
    __m256 vsum = _cpu_goback_2x4xf64_to_8xf32(lo_d_sum, hi_d_sum);
    float tmp[8];
    _mm256_storeu_ps(tmp, vsum);
    for (size_t j = 0; j < 8; ++j)
      sum += tmp[j];

    // rest
    for (; i < v->header.size; ++i) {
      float value = _cpu_dequantize_i8(data[i], v->header.qparams);
      sum += value * value;
    }

    break;
  }
  default:
    UNREACHABLE;
  }

  return sqrtf(sum);
}

// Binary Ops
inline vecx_status
_cpu_op_apply(std::function<__m256(const __m256 &, const __m256 &)> op_simd,
              std::function<float(const float &, const float &)> op_trivial,
              const vecx *a, const vecx *b, void *dest) {
  if (a->header.dtype != b->header.dtype) {
    return a->header.dtype == FLOAT_32 ? VECX_ERR_BAD_OP_F32_X_QI8
                                       : VECX_ERR_BAD_OP_QI8_X_F32;
  }
  if (a->header.size != b->header.size) {
    return VECX_ERR_BAD_OP_BAD_SIZE;
  }

  const vecx_dtype dtype = a->header.dtype;
  const size_t size = a->header.size;

  vecx_header header = {size, FLOAT_32, {}};
  float *result = (float *)vecx_pack_header_into(header, dest);
  std::printf("dsads");

  switch (dtype) {
  case FLOAT_32: {
    const float *a_data = a->data_as<float>();
    const float *b_data = b->data_as<float>();
    uint64_t i = 0;
    const size_t block = 256 / 32;

    for (; i + block <= size; i += block) {
      __m256 a_value_8xf32 = _mm256_loadu_ps(a_data + i);
      __m256 b_value_8xf32 = _mm256_loadu_ps(b_data + i);

      __m256 op_res = op_simd(a_value_8xf32, b_value_8xf32);
      _mm256_storeu_ps(result + i, op_res);
    }

    for (; i < size; ++i)
      result[i] = op_trivial(a_data[i], b_data[i]);
    break;
  }
  case QINT_8: {
    const uint8_t *a_data = static_cast<const uint8_t *>(a->data);
    const uint8_t *b_data = static_cast<const uint8_t *>(b->data);
    size_t i = 0;
    const size_t block = 256 / 8;

    size_t cursor = 0;
    for (; i + block <= size; i += block) {
      __m256i a_bytes_32xi8 =
          _mm256_loadu_si256(reinterpret_cast<__m256i const *>(a_data + i));
      __m256i b_bytes_32xi8 =
          _mm256_loadu_si256(reinterpret_cast<__m256i const *>(b_data + i));

      std::array<__m256, 4> a_chunks =
          _cpu_dequantize_fast(a_bytes_32xi8, a->header.qparams);
      std::array<__m256, 4> b_chunks =
          _cpu_dequantize_fast(b_bytes_32xi8, b->header.qparams);

      for (int j = 0; j < 4; ++j) {
        __m256 op_res = op_simd(a_chunks[j], b_chunks[j]);
        _mm256_storeu_ps(result + cursor, op_res);
        cursor += 8 /*floats*/;
      }
    }

    size_t k = cursor;
    for (; k < size; ++k)
      result[k] = op_trivial(a_data[k], b_data[k]);
    break;
  }
  default:
    UNREACHABLE;
  }

  return VECX_OK;
}

vecx_status vecx_add(const vecx *a, const vecx *b, void *dest) {
  const auto binop = [](const __m256 &a, const __m256 &b) {
    return _mm256_add_ps(a, b);
  };
  const auto op_trivial = [](const float &a, const float &b) { return a + b; };
  return _cpu_op_apply(binop, op_trivial, a, b, dest);
}

vecx_status vecx_sub(const vecx *a, const vecx *b, void *dest) {
  const auto op_simd = [](const __m256 &a, const __m256 &b) {
    return _mm256_sub_ps(a, b);
  };
  const auto op_trivial = [](const float &a, const float &b) { return a - b; };
  return _cpu_op_apply(op_simd, op_trivial, a, b, dest);
}

vecx_status vecx_mult(const vecx *a, const vecx *b, void *dest) {
  const auto op_simd = [](const __m256 &a, const __m256 &b) {
    return _mm256_mul_ps(a, b);
  };
  const auto op_trivial = [](const float &a, const float &b) { return a * b; };
  return _cpu_op_apply(op_simd, op_trivial, a, b, dest);
}

vecx_status vecx_div(const vecx *a, const vecx *b, void *dest) {
  const auto op_simd = [](const __m256 &a, const __m256 &b) {
    return _mm256_div_ps(a, b);
  };
  const auto op_trivial = [](const float &a, const float &b) { return a / b; };
  return _cpu_op_apply(op_simd, op_trivial, a, b, dest);
}

void init_device() {}
