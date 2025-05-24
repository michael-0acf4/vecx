#include "common.hpp"
#include <cstdlib>
#include <cstring>
#include <functional>
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
vecx_status vecx_parse_blob(const void *blob, size_t blob_size,
                            vecx *out_vecx) {
  if (!blob)
    return VECX_ERR_BAD_VECX_HEADER;

  if (!out_vecx)
    return VECX_ERR_GENERIC;

  const int header_size = vecx_header::canon_size();
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
  out_vecx->header.dtype = dtype;
  offset += 1;

  // quantization parameters
  quant_params qparams;
  memcpy(&qparams, data + offset, sizeof(qparams));
  out_vecx->header.qparams = qparams;
  offset += sizeof(qparams);

  // size
  uint64_t size;
  memcpy(&size, data + offset, sizeof(size));
  out_vecx->header.size = size;
  offset += sizeof(uint64_t);

  // expected size
  int type_size =
      vecx_type_size(static_cast<vecx_dtype>(out_vecx->header.dtype));
  uint64_t expected_total = header_size + size * type_size;
  if (blob_size != expected_total) {
    // printf("%d =? %d + %d * %d = %d (%d)\n", blob_size, header_size, size,
    //        type_size, expected_total, out_vecx->header.dtype);
    // printf("scale %f zero %d", out_vecx->header.qparams.scale,
    //        out_vecx->header.qparams.zero);
    return VECX_ERR_INVALID_SIZE;
  }

  out_vecx->data = data + offset;

  return VECX_OK;
}

void vecx_dequantize_to_f32(const vecx &v, void *dest) {
  if (v.header.dtype == FLOAT_32) {
    vecx_pack_into(v, dest);
    return;
  }

  const int8_t *data = static_cast<const int8_t *>(v.data);
  size_t i = 0;
  const size_t block = 256 / 8;

  vecx_header header = {v.header.size, FLOAT_32, {}};
  float_t *result = (float_t *)vecx_pack_header_into(header, dest);

  const auto handler = [&](const size_t cursor, const __m256 &scaled) {
    // Note: i increments 'block' amount
    // extX => 8 floats
    _mm256_storeu_ps(result + cursor * 8, scaled);
  };
  _cpu_dequantize_i8_single_vec_routine(i, block, data, v.header.size,
                                        v.header.qparams, handler);

  for (; i < v.header.size; i++)
    result[i] = _cpu_dequantize_i8(data[i], v.header.qparams);
}

void *vecx_pack_header_into(const vecx_header &header, void *dest) {
  uint8_t *data = (uint8_t *)dest;
  size_t offset = 0;

  memcpy(data + offset, "vecx", 4);
  offset += 4;

  uint8_t dtype = static_cast<uint8_t>(header.dtype);
  memcpy(data + offset, &dtype, 1);
  offset += 1;

  memcpy(data + offset, &header.qparams, sizeof(header.qparams));
  offset += sizeof(header.qparams);

  memcpy(data + offset, &header.size, sizeof(header.size));
  offset += sizeof(header.size);

  return data + offset;
}

void vecx_pack_into(const vecx &v_src, void *dest) {
  void *next = vecx_pack_header_into(v_src.header, dest);
  memcpy(next, v_src.data, v_src.header.bytes_count_data_region());
}
