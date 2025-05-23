#include "common.hpp"
#include <cstdlib>
#include <cstring>
#include <immintrin.h>

uint64_t vecx_type_size(const vecx_dtype &dtype) {
  switch (dtype) {
  case FLOAT_32:
    return 4;
  case QINT_8:
    return 1;
  default:
    return 0;
  }
}

/**
 * vecx Layout:
 *
 * * Magic "vecx" (4 bytes)
 * * vecx_dtype (1 byte)
 * * quant_params (4 + 4 = 8 bytes)
 * * size (8 bytes)
 * * data pointer (size * canon size of vecx_dtype)
 */
vecx_status vecx_parse_blob(const void *blob, int blob_size, vecx *out_vecx) {
  if (!blob)
    return VECX_ERR_BAD_VECX_HEADER;

  if (!out_vecx)
    return VECX_ERR_GENERIC;

  const int header_size = 4 + 1 + 8 + sizeof(uint64_t);
  if (blob_size < header_size)
    return VECX_ERR_INVALID_LAYOUT;

  const uint8_t *data = (const uint8_t *)blob;
  uint64_t offset = 0;

  // magic
  if (data[0] != 'v' || data[1] != 'e' || data[2] != 'c' || data[3] != 'x')
    return VECX_ERR_BAD_VECX_HEADER;
  offset += 4;

  // dtype
  vecx_dtype dtype = static_cast<vecx_dtype>(data[offset]);
  if (dtype != FLOAT_32 && dtype != QINT_8)
    return VECX_ERR_UNKNOWN_DTYPE;
  out_vecx->dtype = dtype;
  offset += 1;

  // quantization parameters
  quant_params qparams;
  memcpy(&qparams, data + offset, sizeof(qparams));
  out_vecx->qparams = qparams;
  offset += sizeof(qparams);

  // size
  uint64_t size;
  memcpy(&size, data + offset, sizeof(size));
  out_vecx->size = size;
  offset += sizeof(uint64_t);

  // expected size
  int type_size = vecx_type_size(out_vecx->dtype);
  uint64_t expected_total = header_size + size * type_size;
  if ((uint64_t)blob_size != expected_total) {
    // printf("%d =? %d + %d * %d = %d (%d)\n", blob_size, header_size, size,
    //        type_size, expected_total, out_vecx->dtype);
    // printf("scale %f zero %d", out_vecx->qparams.scale,
    // out_vecx->qparams.zero);
    return VECX_ERR_INVALID_SIZE;
  }

  out_vecx->data = data + offset;

  return VECX_OK;
}

vecx vecx_dequantize_to_f32(const vecx &v) {
  if (v.dtype == FLOAT_32) {
    return v;
  }
  const int8_t *data = static_cast<const int8_t *>(v.data);
  size_t i = 0;
  const size_t block = 256 / 8;

  float_t *result = (float_t *)malloc(v.size * sizeof(float));

  // size_t pos = 0;
  // const auto handler = [&](const __m256 &scaled) {
  //   // Note: i increments 'block' amount
  //   _mm256_storeu_ps(result + pos, scaled);
  //   pos += 4 * sizeof(float);
  // };
  // _cpu_dequantize_i8_single_vec_routine(i, block, data, v.size, v.qparams,
  //                                       handler);

  for (; i < v.size; i++)
    result[i] = _cpu_dequantize_i8(data[i], v.qparams);

  return {v.size, FLOAT_32, {}, result};
}