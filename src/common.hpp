#pragma once

#include <cassert>
#include <cstdint>
#include <iostream>

#define UNREACHABLE()                                                          \
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

inline float _cpu_dequantize_i8(int8_t value, const quant_params &qparams);
inline float _cpu_dequantize_i8(int8_t value, const quant_params &qparams) {
  return qparams.scale *
         static_cast<float>(static_cast<int32_t>(value) - qparams.zero);
}
