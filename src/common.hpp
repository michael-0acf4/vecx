#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <iostream>

#define UNREACHABLE                                                            \
  do {                                                                         \
    assert(0);                                                                 \
    __builtin_unreachable();                                                   \
  } while (0)

typedef enum vecx_dtype { FLOAT_32 = 1, QINT_8 = 2 } vecx_dtype;

typedef enum vecx_status {
  VECX_OK = 0,
  VECX_ERR_BAD_VECX_HEADER = -1,
  VECX_ERR_INVALID_LAYOUT = -2,
  VECX_ERR_INVALID_SIZE = -3,
  VECX_ERR_UNKNOWN_DTYPE = -4,
  VECX_ERR_RECOVER_NULL = -5,
  VECX_ERR_GENERIC = -1000,
} vecx_status;

typedef struct quant_params {
  float scale;
  int32_t zero;
} quant_params;

typedef struct vecx {
  uint64_t size;
  vecx_dtype dtype;
  quant_params qparams;
  const void *data;
} vecx;

vecx_status vecx_parse_blob(const void *blob, int blob_size, vecx *out_vecx);
uint64_t vecx_type_size(const vecx_dtype &dtype);
vecx vecx_dequantize_to_f32(const vecx &v);

template <typename Fn>
inline void _cpu_dequantize_i8_single_vec_routine(size_t &offset, size_t block,
                                                  const int8_t *data,
                                                  size_t size,
                                                  const quant_params &qparams,
                                                  Fn &&handler) {
  for (; offset + block <= size; offset += block) {
    // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=2,4079&text=_mm256_loadu_epi8
    // __m256i bytes = _mm256_loadu_epi8(data + i); // AVX512?
    __m256i bytes = _mm256_loadu_si256(reinterpret_cast<__m256i const *>(
        data + offset)); // ! packed 8 * 32 int32

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

    __m256 scale = _mm256_set1_ps(qparams.scale);
    __m256i zp_vec = _mm256_set1_epi32(qparams.zero);
    for (const __m256i &packed_i32 : {ext0, ext1, ext2, ext3}) {
      __m256i plus_zero = _mm256_sub_epi32(packed_i32, zp_vec);
      __m256 plus_zero_f = _mm256_cvtepi32_ps(plus_zero);
      __m256 scaled = _mm256_mul_ps(plus_zero_f, scale);

      handler(scaled);
    }
  }
}

inline float _cpu_dequantize_i8(int8_t value, const quant_params &qparams);
inline float _cpu_dequantize_i8(int8_t value, const quant_params &qparams) {
  return qparams.scale *
         static_cast<float>(static_cast<int32_t>(value) - qparams.zero);
}

inline int8_t _cpu_quantize_i8(float value, const quant_params &qparams);
inline int8_t _cpu_quantize_i8(float value, const quant_params &qparams) {
  return static_cast<int8_t>(std::fmax(
      std::fmin(INT8_MAX, std::round(value / qparams.scale + qparams.zero)),
      INT8_MIN));
}
