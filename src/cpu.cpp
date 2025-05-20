#include "cpu.hpp"
#include "common.hpp"
#include <cmath>
#include <immintrin.h> // simd
#include <math.h>
#include <stdint.h>

double f32_norm(const vecx *v) {
  float sum = 0.0f;

  switch (v->dtype) {
  case FLOAT_32: { // AVX2
    const float *data = (const float *)v->data;
    uint64_t i = 0;
    __m256 vsum = _mm256_setzero_ps();

    // 8 floats chunks
    for (; i + 8 <= v->size; i += 8) {
      __m256 vdata = _mm256_loadu_ps(data + i);
      __m256 vmul = _mm256_mul_ps(vdata, vdata);
      vsum = _mm256_add_ps(vsum, vmul);
    }

    float tmp[8];
    _mm256_storeu_ps(tmp, vsum);
    for (int j = 0; j < 8; ++j)
      sum += tmp[j];

    // rem
    for (; i < v->size; ++i)
      sum += data[i] * data[i];

    break;
  }
  default:
    UNREACHABLE();
  }

  return sqrtf(sum);
}

void init_device() {}
