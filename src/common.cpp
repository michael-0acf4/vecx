#include "common.hpp"
#include <cstdlib>
#include <cstring>

uint64_t vecx_type_size(vecx_dtype dtype) {
  switch (dtype) {
  case FLOAT_32:
  case INT_32:
    return 4;
  case INT_64:
    return 8;
  default:
    return 0;
  }
}

/**
 * vecx Layout:
 *
 * * Magic "vecx" (4 bytes)
 * * vecx_dtype (1 byte)
 * * size (8 bytes)
 * * data pointer (size * canon size of vecx_dtype)
 */
vecx_status vecx_parse_blob(const void *blob, int blob_size, vecx *out_vecx) {
  if (!blob)
    return VECX_ERR_BAD_VECX_HEADER;

  if (!out_vecx)
    return VECX_ERR_GENERIC;

  const int header_size = 4 + 1 + sizeof(uint64_t);
  if (blob_size < header_size)
    return VECX_ERR_INVALID_LAYOUT;

  const uint8_t *data = (const uint8_t *)blob;
  uint64_t offset = 0;

  // magic
  if (data[0] != 'v' || data[1] != 'e' || data[2] != 'c' || data[3] != 'x')
    return VECX_ERR_BAD_VECX_HEADER;
  offset += 4;

  // dtype
  vecx_dtype dtype = (vecx_dtype)data[offset];
  if (dtype != INT_32 && dtype != INT_64 && dtype != FLOAT_32)
    return VECX_ERR_UNKNOWN_DTYPE;
  out_vecx->dtype = dtype;
  offset += 1;

  // size
  uint64_t size;
  memcpy(&size, data + offset, sizeof(size));
  out_vecx->size = size;
  offset += sizeof(uint64_t);

  // expected size
  int type_size = vecx_type_size(out_vecx->dtype);
  uint64_t expected_total = header_size + size * type_size;
  if ((uint64_t)blob_size != expected_total) {
    // printf("%d =? %d + %d * %d", blob_size, header_size, size, type_size);
    return VECX_ERR_INVALID_SIZE;
  }

  out_vecx->data = data + offset;

  return VECX_OK;
}
