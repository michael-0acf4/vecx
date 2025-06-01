#include "common.hpp"
#include <cstdlib>
#include <cstring>
#include <sstream>

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

std::string vecx_type_name(const vecx_dtype &dtype) {
  switch (dtype) {
  case FLOAT_32:
    return "F32";
  case QINT_8:
    return "QI8";
  default:
    return "UND";
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
vecx_result vecx_parse_blob(const void *blob, size_t blob_size,
                            vecx *out_vecx) {
  if (!blob)
    return vecx_result::bad_vecx_header("source blob is NULL");

  if (!out_vecx)
    return vecx_result::error(VECX_ERR_GENERIC,
                              "Fatal: source vecx not allocated");

  const int header_size = vecx_header::canon_size();
  if (blob_size < header_size)
    return vecx_result::invalid_layout();

  const uint8_t *data = (const uint8_t *)blob;
  uint64_t offset = 0;

  // magic
  if (data[0] != 'v' || data[1] != 'e' || data[2] != 'c' || data[3] != 'x')
    return vecx_result::bad_vecx_header("vecx section malformed");
  offset += 4;

  // dtype
  vecx_dtype dtype = static_cast<vecx_dtype>(data[offset]);
  if (dtype != FLOAT_32 && dtype != QINT_8)
    return vecx_result::unknown_dtype(dtype);
  out_vecx->header.dtype = dtype;
  offset += 1;

  // quantization parameters
  quant_params qparams;
  std::memcpy(&qparams, data + offset, sizeof(qparams));
  out_vecx->header.qparams = qparams;
  offset += sizeof(qparams);

  // size
  uint64_t size;
  std::memcpy(&size, data + offset, sizeof(size));
  out_vecx->header.size = size;
  offset += sizeof(uint64_t);

  // expected size
  int type_size =
      vecx_type_size(static_cast<vecx_dtype>(out_vecx->header.dtype));
  uint64_t expected_total = header_size + size * type_size;
  if (blob_size != expected_total) {
    return vecx_result::invalid_size(blob_size);
  }

  out_vecx->data = data + offset;

  return vecx_result::ok();
}

void *vecx_pack_header_into(const vecx_header &header, void *dest) {
  uint8_t *data = (uint8_t *)dest;
  size_t offset = 0;

  std::memcpy(data + offset, "vecx", 4);
  offset += 4;

  uint8_t dtype = static_cast<uint8_t>(header.dtype);
  std::memcpy(data + offset, &dtype, 1);
  offset += 1;

  std::memcpy(data + offset, &header.qparams, sizeof(header.qparams));
  offset += sizeof(header.qparams);

  std::memcpy(data + offset, &header.size, sizeof(header.size));
  offset += sizeof(header.size);

  return data + offset;
}

void vecx_pack_into(const vecx &v_src, void *dest) {
  void *next = vecx_pack_header_into(v_src.header, dest);
  std::memcpy(next, v_src.data, v_src.header.bytes_count_data_region());
}

std::string vecx_show(const vecx &v) {
  size_t window = 4;
  size_t size = v.header.size;
  std::ostringstream ss;
  ss << (v.header.dtype == FLOAT_32 ? "F32" : "QI8");
  ss << " [ ";

  const auto display_item = [&](size_t pos) {
    return v.header.dtype == FLOAT_32
               ? v.item_as<float>(pos)
               : _cpu_dequantize_i8(v.item_as<int8_t>(pos), v.header.qparams);
  };

  if (size <= 2 * window) {
    for (size_t i = 0; i < size; ++i) {
      ss << display_item(i) << " ";
    }
  } else {
    for (size_t i = 0; i < window; ++i) {
      ss << display_item(i) << " ";
    }
    ss << "... ";
    for (size_t i = size - window; i < size; ++i) {
      ss << display_item(i) << " ";
    }
  }

  ss << "]";
  return ss.str();
}

vecx_result parse_inline_vec(const char *text, size_t size,
                             std::vector<float> *dest) {
  std::string expr(text, size);
  std::stringstream ss(expr);
  std::string token;

  while (std::getline(ss, token, ',')) {
    size_t start = token.find_first_not_of(" \t\n");
    size_t end = token.find_last_not_of(" \t\n");
    if (start != std::string::npos) {
      std::string trimmed = token.substr(start, end - start + 1);
      dest->push_back(std::stof(trimmed));
    }
  }

  return vecx_result::ok();
}

vecx_result pack_inline_vec(const std::vector<float> &values, size_t left_pad,
                            size_t right_pad, float fill, void *dest) {
  float *data = static_cast<float *>(dest);

  size_t total = left_pad + values.size() + right_pad;
  size_t i = 0;

  for (; i < left_pad; ++i)
    data[i] = fill;

  for (; i < (left_pad + values.size()); ++i)
    data[i] = values[i - left_pad];

  for (; i < total; ++i)
    data[i] = fill;

  return vecx_result::ok();
}
