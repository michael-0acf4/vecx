
#pragma once

#include <stdint.h>

typedef enum vecx_dtype {
  INT_32 = 1,
  INT_64 = 2,
  FLOAT_32 = 3,
} vecx_dtype;

typedef enum vecx_status {
  VECX_OK = 0,
  VECX_ERR_BAD_VECX_HEADER = -1,
  VECX_ERR_INVALID_LAYOUT = -2,
  VECX_ERR_INVALID_SIZE = -3,
  VECX_ERR_UNKNOWN_DTYPE = -4,
  VECX_ERR_RECOVER_NULL = -5,
  VECX_ERR_GENERIC = -1000,
} vecx_status;

typedef struct vecx {
  uint64_t size;
  vecx_dtype dtype;
  const void *data;
} vecx;

vecx_status vecx_parse_blob(const void *blob, int blob_size, vecx *out_vecx);
uint64_t vecx_type_size(vecx_dtype dtype);
