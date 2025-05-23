#include "cpu.hpp"
#include "common.hpp"
#include <cmath>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>

double f32_norm(const vecx *v) {
  float sum = 0.0f;

  switch (v->dtype) {
  case FLOAT_32: {
    const float *data = static_cast<const float *>(v->data);
    uint64_t i = 0;
    const size_t block = 256 / 32;

    __m256 vsum = _mm256_setzero_ps();
    for (; i + block <= v->size; i += block) {
      __m256 vdata = _mm256_loadu_ps(data + i);
      __m256 vmul = _mm256_mul_ps(vdata, vdata);
      vsum = _mm256_add_ps(vsum, vmul);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, vsum);
    for (int j = 0; j < 8; ++j)
      sum += tmp[j];

    // rest
    for (; i < v->size; ++i)
      sum += data[i] * data[i];
    break;
  }
  case QINT_8: {
    const int8_t *data = static_cast<const int8_t *>(v->data);
    size_t i = 0;
    const size_t block = 256 / 8;

    __m256 vsum = _mm256_setzero_ps();
    const auto &handler = [&](const __m256 &scaled) {
      __m256 squared = _mm256_mul_ps(scaled, scaled);
      vsum = _mm256_add_ps(vsum, squared);
    };
    _cpu_dequantize_i8_single_vec_routine(i, block, data, v->size, v->qparams,
                                          handler);

    float tmp[8];
    _mm256_storeu_ps(tmp, vsum);
    for (int j = 0; j < 8; ++j)
      sum += tmp[j];

    // rest
    for (; i < v->size; ++i) {
      float value = _cpu_dequantize_i8(data[i], v->qparams);
      sum += value * value;
    }
    break;
  }
  default:
    UNREACHABLE;
  }

  return sqrtf(sum);
}

void init_device() {}
