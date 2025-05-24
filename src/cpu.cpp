#include "cpu.hpp"
#include "common.hpp"
#include <cmath>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>

// Note: float vector _mm256_add_ps(vsum, squared) overflows
// Also this might double the used stack size if not handled correctly
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
    const auto &handler = [&](const size_t cursor, const __m256 &scaled_8xf32) {
      __m256 squared = _mm256_mul_ps(scaled_8xf32, scaled_8xf32);
      _cpu_extend_8xf32_to_2x4xf64_then_sum(squared, lo_d_sum, hi_d_sum);
    };
    _cpu_dequantize_i8_single_vec_routine(i, block, data, v->header.size,
                                          v->header.qparams, handler);
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

double f32_dot_float_x_quant(const vecx *a_float, const vecx *b_quant) {
  double sum = 0.0f;

  size_t max_size = std::max(a_float->header.size, b_quant->header.size);

  const float *a_data = static_cast<const float *>(a_float->data);
  const float *b_data = static_cast<const float *>(b_quant->data);

  uint64_t i = 0, j = 0;
  const size_t a_block_8xf32 = 256 / 32;
  const size_t b_block_32xi8 = 256 / 8;

  __m256d lo_d_sum = _mm256_setzero_pd();
  __m256d hi_d_sum = _mm256_setzero_pd();
  size_t b_cursor = 0;

  while ((i + (4 * a_block_8xf32) <= a_float->header.size ||
          j + b_block_32xi8 <= b_quant->header.size)) {
    __m256 mult = _mm256_set1_ps(0.0);
    __m256i value_enc_32xi32 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_data + i));
    const auto handler = [&](size_t cursor, __m256 value_8xf32) {
      __m256 a_value_8xf32 = _mm256_loadu_ps(a_data + i);
      mult = _mm256_mul_ps(a_value_8xf32, a_value_8xf32);
      i += a_block_8xf32;
    };
    _cpu_dequantize_fast(value_enc_32xi32, b_cursor, b_quant->header.qparams,
                         handler);
    j += b_block_32xi8;

    // std::cout << "HIT: " << i << "," << j << "\n";

    _cpu_extend_8xf32_to_2x4xf64_then_sum(mult, lo_d_sum, hi_d_sum);
  }
  __m256 vsum = _cpu_goback_2x4xf64_to_8xf32(lo_d_sum, hi_d_sum);

  float tmp[8];
  _mm256_storeu_ps(tmp, vsum);
  for (size_t j = 0; j < 8; ++j)
    sum += tmp[j];

  // rest
  for (;; ++i, ++j) {
    if (i >= a_float->header.size || j >= b_quant->header.size) {
      // 0 * r = l * 0 = 0
      break;
    }
    sum += a_data[i] * _cpu_dequantize_i8(b_data[j], b_quant->header.qparams);
  }

  return sum;
}

void init_device() {}
