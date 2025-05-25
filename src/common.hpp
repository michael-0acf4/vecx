#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <immintrin.h>
#include <iostream>

#define UNREACHABLE                                                            \
  do {                                                                         \
    assert(0);                                                                 \
    __builtin_unreachable();                                                   \
  } while (0)

typedef enum vecx_dtype { FLOAT_32 = 1, QINT_8 = 2 } vecx_dtype;
uint64_t vecx_type_size(const vecx_dtype &dtype);

typedef enum vecx_status {
  VECX_OK = 0,
  VECX_ERR_BAD_VECX_HEADER = -1,
  VECX_ERR_INVALID_LAYOUT = -2,
  VECX_ERR_INVALID_SIZE = -3,
  VECX_ERR_UNKNOWN_DTYPE = -4,
  VECX_ERR_RECOVER_NULL = -5,
  VECX_ERR_BAD_OP_F32_X_QI8 = -6,
  VECX_ERR_BAD_OP_QI8_X_F32 = -7,
  VECX_ERR_BAD_OP_BAD_SIZE = -8,
  VECX_ERR_GENERIC = -1000,
} vecx_status;

typedef struct quant_params {
  float scale;
  int32_t zero;
} quant_params;

typedef struct vecx_header {
  uint64_t size;
  vecx_dtype dtype;
  quant_params qparams;

  inline size_t bytes_count_data_region() const {
    return vecx_type_size(dtype) * size;
  }

  inline size_t bytes_count_total() const {
    return vecx_header::canon_size() + bytes_count_data_region();
  }

  static inline size_t canon_size() { return 4 /*vecx*/ + 8 + 1 + (4 + 4); }
} vecx_header;

typedef struct vecx {
  vecx_header header;
  const void *data;

  template <typename T> inline const T *data_as() const {
    return reinterpret_cast<const T *>(data);
  }

  template <typename T> inline T item_as(size_t pos) const {
    return reinterpret_cast<const T *>(data)[pos];
  }
} vecx;

vecx_status vecx_parse_blob(const void *blob, size_t blob_size, vecx *out_vecx);
void *vecx_allocate_blob(const vecx_header &header);

// Note: ownership of heap data is up to the caller
// The reason of this design is that sqlite has its own allocator
// An alternative cleaner implementation would have been a std::function or
// function pointer but even then function signature may diverge (e.g.
// sqlite3_malloc only accepts int as memory size)

void vecx_dequantize_to_f32(const vecx &v, void *dest);

// Pack vecx header and return the memory addresss coming after it.
// The caller must ensure that dest is of the correct size.
void *vecx_pack_header_into(const vecx_header &header, void *dest);

// Pack vecx vector into dest
// The caller must ensure that dest is of the correct size.
void vecx_pack_into(const vecx &v_src, void *dest);

// Dequantize a 256 block that packs 32 int8
std::array<__m256, 4> _cpu_dequantize_fast(const __m256i &bytes_32xi8,
                                           const quant_params &qparams);

// inline float _cpu_dequantize_i8(int8_t value, const quant_params &qparams);
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

inline vecx_status validate_layout_similarities(const vecx *a, const vecx *b) {
  if (a->header.dtype != b->header.dtype) {
    return a->header.dtype == FLOAT_32 ? VECX_ERR_BAD_OP_F32_X_QI8
                                       : VECX_ERR_BAD_OP_QI8_X_F32;
  }
  if (a->header.size != b->header.size) {
    return VECX_ERR_BAD_OP_BAD_SIZE;
  }

  return VECX_OK;
}