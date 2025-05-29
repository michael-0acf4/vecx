#include "backend.hpp"

#include <cmath>
#include <cstdint>
#include <immintrin.h>

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

// Dequantize a 256 block that packs 32 int8

std::array<__m256, 4> _cpu_dequantize_fast(const __m256i &bytes_32xi8,
                                           const quant_params &qparams) {
  // Note: cast are 'logical' (just like static_cast)
  //    [32 i8 -> 16 i8 + 16 i8]
  // -> [32 i8 -> [ [16 i8 ~> [8 i32] + [8 i32]] ] + [...]]
  // -> [32 i8 -> [ [16 i8 ~> [8 i32] + [8 i32]] ] + [...]]

  __m128i low = _mm256_castsi256_si128(bytes_32xi8);
  __m256i ext0 = _mm256_cvtepi8_epi32(low);
  __m256i ext1 = _mm256_cvtepi8_epi32(_mm_bsrli_si128(low, 8));

  __m128i high = _mm256_extracti128_si256(bytes_32xi8, 1);
  __m256i ext2 = _mm256_cvtepi8_epi32(high);
  __m256i ext3 = _mm256_cvtepi8_epi32(_mm_bsrli_si128(high, 8));

  __m256 scale = _mm256_set1_ps(qparams.scale);
  __m256i zp_vec = _mm256_set1_epi32(qparams.zero);

  std::array<__m256, 4> out;
  size_t i = 0;
  for (const __m256i &packed_i32 : {ext0, ext1, ext2, ext3}) {
    __m256i plus_zero = _mm256_sub_epi32(packed_i32, zp_vec);
    __m256 plus_zero_f = _mm256_cvtepi32_ps(plus_zero);
    __m256 scaled = _mm256_mul_ps(plus_zero_f, scale);

    out[i++] = scaled;
  }

  return out;
}

vecx_result vecx_dequantize_to_f32(const vecx *v, void *dest) {
  if (v->header.dtype == FLOAT_32) {
    vecx_pack_into(*v, dest);
    return vecx_result::ok();
  }

  const int8_t *data = static_cast<const int8_t *>(v->data);
  size_t i = 0;
  const size_t block = 256 / 8;

  vecx_header header = {v->header.size, FLOAT_32, {}};
  float_t *result = (float_t *)vecx_pack_header_into(header, dest);

  size_t cursor = 0;
  for (; i + block <= v->header.size; i += block) {
    __m256i bytes_32xi8 =
        _mm256_loadu_si256(reinterpret_cast<__m256i const *>(data + i));

    for (const __m256 &ext_8xf32 :
         _cpu_dequantize_fast(bytes_32xi8, v->header.qparams)) {
      // Note: i increments 'block' amount
      // extX => 8 floats
      _mm256_storeu_ps(result + cursor * 8, ext_8xf32);
      cursor++;
    }
  }

  for (; i < v->header.size; i++)
    result[i] = _cpu_dequantize_i8(data[i], v->header.qparams);

  return vecx_result::ok();
}

// Reduction

double vecx_norm(const vecx *v) {
  double sum = 0.0f;
  switch (v->header.dtype) {
  case FLOAT_32: {
    const float *data = v->data_as<float>();
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

    for (; i < v->header.size; ++i)
      sum += data[i] * data[i];

    break;
  }
  case QINT_8: {
    const int8_t *data = v->data_as<int8_t>();
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
template <__m256 (*op_simd)(__m256, __m256), float (*op_trivial)(float, float)>
inline vecx_result _cpu_op_apply(const vecx *a, const vecx *b, void *dest) {
  vecx_result check = validate_layout_similarities(a, b);
  if (check.is_err()) {
    return check;
  }

  const vecx_dtype dtype = a->header.dtype;
  const size_t size = a->header.size;

  vecx_header header = {size, FLOAT_32, {}};
  float *result = (float *)vecx_pack_header_into(header, dest);

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
    const int8_t *a_data = a->data_as<int8_t>();
    const int8_t *b_data = b->data_as<int8_t>();
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

      for (int j = 0; j < 4; ++j, cursor += 8 /*floats*/) {
        __m256 op_res = op_simd(a_chunks[j], b_chunks[j]);
        _mm256_storeu_ps(result + cursor, op_res);
      }
    }

    size_t k = cursor;
    for (; k < size; ++k) {
      result[k] = op_trivial(_cpu_dequantize_i8(a_data[k], a->header.qparams),
                             _cpu_dequantize_i8(b_data[k], b->header.qparams));
    }
    break;
  }
  default:
    UNREACHABLE;
  }

  return vecx_result::ok();
}

// Note:
// SIMD _mm256_mul_ps wrapped within a Lambda for example crashes when passed as
// argument, ideally we want to inline any SIMD

inline __m256 add_simd(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }
inline float add_trivial(float a, float b) { return a + b; }

inline __m256 sub_simd(__m256 a, __m256 b) { return _mm256_sub_ps(a, b); }
inline float sub_trivial(float a, float b) { return a - b; }

inline __m256 mul_simd(__m256 a, __m256 b) { return _mm256_mul_ps(a, b); }
inline float mul_trivial(float a, float b) { return a * b; }

inline __m256 div_simd(__m256 a, __m256 b) { return _mm256_div_ps(a, b); }
inline float div_trivial(float a, float b) { return a / b; }

vecx_result vecx_add(const vecx *a, const vecx *b, void *dest) {
  return _cpu_op_apply<add_simd, add_trivial>(a, b, dest);
}

vecx_result vecx_sub(const vecx *a, const vecx *b, void *dest) {
  return _cpu_op_apply<sub_simd, sub_trivial>(a, b, dest);
}

vecx_result vecx_mult(const vecx *a, const vecx *b, void *dest) {
  return _cpu_op_apply<mul_simd, mul_trivial>(a, b, dest);
}

vecx_result vecx_div(const vecx *a, const vecx *b, void *dest) {
  return _cpu_op_apply<div_simd, div_trivial>(a, b, dest);
}

template <__m256 (*op_simd)(__m256, __m256), float (*op_trivial)(float, float)>
inline vecx_result _cpu_op_apply_broadcast_scalar(const vecx *v, float scalar,
                                                  void *dest) {
  const vecx_dtype dtype = v->header.dtype;
  const size_t size = v->header.size;

  vecx_header header = {size, FLOAT_32, {}};
  float *result = (float *)vecx_pack_header_into(header, dest);

  switch (dtype) {
  case FLOAT_32: {
    const float *v_data = v->data_as<float>();
    uint64_t i = 0;
    const size_t block = 256 / 32;

    __m256 brdcst_8xf32 = _mm256_set1_ps(scalar);
    for (; i + block <= size; i += block) {
      __m256 value = _mm256_loadu_ps(v_data + i);
      __m256 op_res = op_simd(value, brdcst_8xf32);
      _mm256_storeu_ps(result + i, op_res);
    }

    for (; i < size; ++i)
      result[i] = op_trivial(v_data[i], scalar);
    break;
  }
  case QINT_8: {
    const int8_t *v_data = v->data_as<int8_t>();
    size_t i = 0;
    const size_t block = 256 / 8;

    size_t cursor = 0;
    __m256 brdcst_8xf32 = _mm256_set1_ps(scalar);
    for (; i + block <= size; i += block) {
      __m256i bytes_32xi8 =
          _mm256_loadu_si256(reinterpret_cast<__m256i const *>(v_data + i));

      std::array<__m256, 4> chunks =
          _cpu_dequantize_fast(bytes_32xi8, v->header.qparams);

      for (int j = 0; j < 4; ++j, cursor += 8 /*floats*/) {
        __m256 op_res = op_simd(chunks[j], brdcst_8xf32);
        _mm256_storeu_ps(result + cursor, op_res);
      }
    }

    size_t k = cursor;
    for (; k < size; ++k) {
      result[k] =
          op_trivial(_cpu_dequantize_i8(v_data[k], v->header.qparams), scalar);
    }
    break;
  }
  default:
    UNREACHABLE;
  }

  return vecx_result::ok();
}

vecx_result vecx_scalar(const vecx *v, float scalar, void *dest) {
  return _cpu_op_apply_broadcast_scalar<mul_simd, mul_trivial>(v, scalar, dest);
}

void init_device() {}
