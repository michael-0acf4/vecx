#include "cpu.hpp"
#include "common.hpp"
#include <cmath>
#include <immintrin.h> // simd
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
    for (; i + block <= v->size; i += block) {
      // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=2,4079&text=_mm256_loadu_epi8
      // __m256i bytes = _mm256_loadu_epi8(data + i); // AVX512?
      __m256i bytes = _mm256_loadu_si256(
          reinterpret_cast<__m256i const *>(data + i)); // ! packed 8 * 32 int32

      // 32 int32 -> 16 int32 + 16 int32
      __m128i low = _mm256_castsi256_si128(bytes);
      __m128i high = _mm256_extracti128_si256(bytes, 1);

      // Note: cast are 'logical' (just like static_cast)
      // [16 int32 -> [8 int32] + [8 int32]]
      // + [16 int32 -> [8 int32] + [8 int32]]
      __m256i ext0 = _mm256_cvtepi8_epi32(low);
      __m256i ext1 = _mm256_cvtepi8_epi32(_mm_bsrli_si128(low, 8));
      __m256i ext2 = _mm256_cvtepi8_epi32(high);
      __m256i ext3 = _mm256_cvtepi8_epi32(_mm_bsrli_si128(high, 8));

      __m256 scale = _mm256_set1_ps(v->qparams.scale);
      __m256i zp_vec = _mm256_set1_epi32(v->qparams.zero);
      for (const __m256i &packed_i32 : {ext0, ext1, ext2, ext3}) {
        // TODO: refactor, this is the body of the computation
        // the rest is just a blob interpretation game and can be copied over
        // all future impl
        __m256i plus_zero = _mm256_sub_epi32(packed_i32, zp_vec);
        __m256 plus_zero_f = _mm256_cvtepi32_ps(plus_zero);
        __m256 scaled = _mm256_mul_ps(plus_zero_f, scale);
        __m256 squared = _mm256_mul_ps(scaled, scaled);
        vsum = _mm256_add_ps(vsum, squared);
      }
    }
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
    UNREACHABLE();
  }

  return sqrtf(sum);
}

void init_device() {}
