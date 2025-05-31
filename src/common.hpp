#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

#define UNREACHABLE                                                            \
  do {                                                                         \
    assert(0);                                                                 \
    __builtin_unreachable();                                                   \
  } while (0)

typedef enum vecx_dtype { FLOAT_32 = 1, QINT_8 = 2 } vecx_dtype;
uint64_t vecx_type_size(const vecx_dtype &dtype);
std::string vecx_type_name(const vecx_dtype &dtype);

typedef enum vecx_status {
  VECX_OK = 0,
  VECX_ERR_BAD_VECX_HEADER = -1,
  VECX_ERR_INVALID_LAYOUT = -2,
  VECX_ERR_INVALID_SIZE = -3,
  VECX_ERR_UNKNOWN_DTYPE = -4,
  VECX_ERR_RECOVER_NULL = -5,
  VECX_ERR_BAD_OPERAND = -6,
  VECX_ERR_BAD_OP_BAD_SIZE = -7,
  VECX_ERR_BAD_OP_BAD_DTYPE = -8,
  VECX_ERR_DEVICE_PROBLEM = -9,
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

  inline size_t bytes_count_data_region() const { return type_size() * size; }

  inline size_t type_size() const { return vecx_type_size(dtype); }

  inline size_t bytes_count_total() const {
    return vecx_header::canon_size() + bytes_count_data_region();
  }

  static inline size_t canon_size() { return 4 /*vecx*/ + 8 + 1 + (4 + 4); }
} vecx_header;

typedef struct vecx_result {
  vecx_status status;
  std::string error_payload;
  // This is a very redundant operation
  // Should be fine assuming no unnecessary heap allocation with SSO
  inline bool is_ok() const { return status == VECX_OK; }
  inline bool is_err() const { return status != VECX_OK; }

  inline operator vecx_status() const { return status; }

  static inline vecx_result ok() { return {VECX_OK, ""}; }

  static inline vecx_result error(vecx_status status,
                                  const std::string &&message) {
    // TODO: add debug flag
    // std::cerr << "[ERROR] " << message << "\n";
    return {status, message};
  }

  static inline vecx_result bad_vecx_header(const std::string &&ctx) {
    return error(VECX_ERR_BAD_VECX_HEADER, "Bad vecx header: " + ctx);
  }

  static inline vecx_result invalid_layout() {
    return error(VECX_ERR_INVALID_LAYOUT, "Invalid vecx layout");
  }

  static inline vecx_result invalid_size(size_t size) {
    return error(VECX_ERR_INVALID_SIZE,
                 "Invalid size: " + std::to_string(size));
  }

  static inline vecx_result unknown_dtype(int dtype) {
    return error(VECX_ERR_UNKNOWN_DTYPE,
                 "Unknown dtype: kind " + std::to_string(dtype));
  }

  static inline vecx_result recover_null() {
    // e.g. When l or r-operand is NULL-ish, we can just return the non null
    // operand
    return error(VECX_ERR_RECOVER_NULL, "");
  }

  static inline vecx_result bad_operand() {
    return error(VECX_ERR_BAD_OPERAND, "Bad operand.");
  }

  static inline vecx_result bad_op_bad_size(size_t lsize, size_t rsize) {
    return error(VECX_ERR_BAD_OP_BAD_SIZE,
                 "Incompatible operand size: " + std::to_string(lsize) +
                     " vs " + std::to_string(rsize));
  }

  static inline vecx_result bad_op_bad_dtype(vecx_dtype ltype,
                                             vecx_dtype rtype) {
    return error(VECX_ERR_BAD_OP_BAD_SIZE,
                 "Incompatible operand type: " + vecx_type_name(ltype) +
                     " vs " + vecx_type_name(rtype));
  }

  static inline vecx_result device_error(const std::string &message) {
    return error(VECX_ERR_DEVICE_PROBLEM, "Device error " + message);
  }

  static inline vecx_result device_error(const char *message) {
    return error(VECX_ERR_DEVICE_PROBLEM,
                 "Device error " + std::string(message));
  }
} result;

typedef struct vecx {
  vecx_header header;
  const void *data;

  inline size_t mem_size_required(vecx_dtype dest) const {
    return vecx_header::canon_size() +
           header.size * (dest == FLOAT_32 ? sizeof(float) : sizeof(int8_t));
  }

  template <typename T> inline const T *data_as() const {
    return reinterpret_cast<const T *>(data);
  }

  template <typename T> inline T item_as(size_t pos) const {
    return reinterpret_cast<const T *>(data)[pos];
  }
} vecx;

vecx_result vecx_parse_blob(const void *blob, size_t blob_size, vecx *out_vecx);

// Note: ownership of heap data is up to the caller
// The reason of this design is that sqlite has its own allocator
// An alternative cleaner implementation would have been a std::function or
// function pointer but even then function signature may diverge (e.g.
// sqlite3_malloc only accepts int as memory size)

// Pack vecx header and return the memory addresss coming after it.
// The caller must ensure that dest is of the correct size.
void *vecx_pack_header_into(const vecx_header &header, void *dest);

// Pack vecx vector into dest
// The caller must ensure that dest is of the correct size.
void vecx_pack_into(const vecx &v_src, void *dest);

std::string vecx_show(const vecx &v);

// inline float _cpu_dequantize_i8(int8_t value, const quant_params &qparams);
inline float _cpu_dequantize_i8(int8_t value, const quant_params &qparams) {
  return qparams.scale *
         static_cast<float>(static_cast<int32_t>(value) - qparams.zero);
}

inline int8_t _cpu_quantize_i8(float value, const quant_params &qparams) {
  return static_cast<int8_t>(std::fmax(
      std::fmin(INT8_MAX, std::round(value / qparams.scale + qparams.zero)),
      INT8_MIN));
}

inline vecx_result validate_layout_similarities(const vecx *a, const vecx *b) {
  if (a->header.dtype != b->header.dtype) {
    return vecx_result::bad_op_bad_dtype(a->header.dtype, b->header.dtype);
  }

  if (a->header.size != b->header.size) {
    return vecx_result::bad_op_bad_size(a->header.size, b->header.size);
  }

  return vecx_result::ok();
}
