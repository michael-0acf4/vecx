
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1

#include "backend.hpp"
#include <memory>

void x_emit_error(sqlite3_context *ctx, vecx_status status) {
  switch (status) {
  case VECX_ERR_BAD_VECX_HEADER:
    sqlite3_result_error(ctx, "Underlying blob is not a vecx object", -1);
    break;
  case VECX_ERR_UNKNOWN_DTYPE:
    sqlite3_result_error(ctx, "Underlying vecx object is of an unknown dtype",
                         -1);
    break;
  case VECX_ERR_INVALID_LAYOUT:
    sqlite3_result_error(ctx, "Underlying vecx object layout is not supported",
                         -1);
    break;
  case VECX_ERR_INVALID_SIZE:
    sqlite3_result_error(
        ctx, "Underlying vecx object has an invalid reported size", -1);
    break;
  case VECX_ERR_BAD_OP_F32_X_QI8:
    sqlite3_result_error(ctx, "Cannot directly operate F32 and QI8", -1);
    break;
  case VECX_ERR_BAD_OP_QI8_X_F32:
    sqlite3_result_error(ctx, "Cannot directly operate QI8 and F32", -1);
    break;
  case VECX_ERR_BAD_OP_BAD_SIZE:
    sqlite3_result_error(
        ctx, "Cannot directly operate between two vectors of different size",
        -1);
    break;
  case VECX_ERR_GENERIC:
  default:
    sqlite3_result_error(ctx, "Unknown error while processing vecx object", -1);
    break;
  }
}

void x_size(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  if (argc != 1) {
    sqlite3_result_null(ctx);
    return;
  }

  if (sqlite3_value_type(argv[0]) != SQLITE_BLOB) {
    sqlite3_result_null(ctx);
    return;
  }

  const void *blob = sqlite3_value_blob(argv[0]);
  uint64_t blob_size = sqlite3_value_bytes(argv[0]);
  vecx vec;
  vecx_status status = vecx_parse_blob(blob, blob_size, &vec);

  if (status < 0)
    x_emit_error(ctx, status);
  else
    sqlite3_result_int64(ctx, (int64_t)vec.header.size);
}

void x_type(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  if (argc != 1) {
    sqlite3_result_null(ctx);
    return;
  }

  if (sqlite3_value_type(argv[0]) != SQLITE_BLOB) {
    sqlite3_result_null(ctx);
    return;
  }

  const void *blob = sqlite3_value_blob(argv[0]);
  uint64_t blob_size = sqlite3_value_bytes(argv[0]);
  vecx vec;
  vecx_status status = vecx_parse_blob(blob, blob_size, &vec);

  if (status < 0)
    x_emit_error(ctx, status);
  else {
    switch (vec.header.dtype) {
    case FLOAT_32:
      sqlite3_result_text(ctx, "F32", -1, SQLITE_STATIC);
      break;
    case QINT_8:
      sqlite3_result_text(ctx, "QI8", -1, SQLITE_STATIC);
      break;
    default:
      x_emit_error(ctx, VECX_ERR_UNKNOWN_DTYPE);
      break;
    }
  }
}

void x_dequantize(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  if (argc != 1) {
    sqlite3_result_null(ctx);
    return;
  }

  if (sqlite3_value_type(argv[0]) != SQLITE_BLOB) {
    sqlite3_result_null(ctx);
    return;
  }

  const void *blob = sqlite3_value_blob(argv[0]);
  uint64_t blob_size = sqlite3_value_bytes(argv[0]);
  vecx qvec;
  vecx_status status = vecx_parse_blob(blob, blob_size, &qvec);

  if (status < 0)
    x_emit_error(ctx, status);
  else {
    size_t size = vecx_header::canon_size() + qvec.header.size * sizeof(float);
    void *blob = sqlite3_malloc(size);
    vecx_dequantize_to_f32(&qvec, blob);

    sqlite3_result_blob(ctx, blob, size, sqlite3_free);
  }
}

void x_norm(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  if (argc != 1) {
    sqlite3_result_null(ctx);
    return;
  }

  if (sqlite3_value_type(argv[0]) != SQLITE_BLOB) {
    sqlite3_result_null(ctx);
    return;
  }

  const void *blob = sqlite3_value_blob(argv[0]);
  uint64_t blob_size = sqlite3_value_bytes(argv[0]);
  vecx vec;
  vecx_status status = vecx_parse_blob(blob, blob_size, &vec);

  if (status < 0)
    x_emit_error(ctx, status);
  else
    sqlite3_result_double(ctx, vecx_norm(&vec));
}

enum OP { ADD, SUB, MUL, DIV };

inline void apply_op(OP op, sqlite3_context *ctx, int argc,
                     sqlite3_value **argv) {
  if (argc != 2) {
    sqlite3_result_null(ctx);
    return;
  }

  if (sqlite3_value_type(argv[0]) != SQLITE_BLOB ||
      sqlite3_value_type(argv[1]) != SQLITE_BLOB) {
    sqlite3_result_null(ctx);
    return;
  }

  const void *a_blob = sqlite3_value_blob(argv[0]);
  const void *b_blob = sqlite3_value_blob(argv[1]);

  uint64_t a_blob_size = sqlite3_value_bytes(argv[0]);
  uint64_t b_blob_size = sqlite3_value_bytes(argv[1]);

  vecx a, b;
  vecx_status a_status = vecx_parse_blob(a_blob, a_blob_size, &a);
  if (a_status < 0) {
    x_emit_error(ctx, a_status);
    return;
  }

  vecx_status b_status = vecx_parse_blob(a_blob, b_blob_size, &b);
  if (b_status < 0) {
    x_emit_error(ctx, b_status);
    return;
  }

  size_t size = vecx_header::canon_size() + a.header.size * sizeof(float);
  void *dest_blob = sqlite3_malloc(size);
  switch (op) {
  case ADD:
    vecx_add(&a, &b, dest_blob);
    break;
  case SUB:
    vecx_sub(&a, &b, dest_blob);
    break;
  case MUL:
    vecx_mult(&a, &b, dest_blob);
    break;
  case DIV:
    vecx_div(&a, &b, dest_blob);
    break;
  default:
    x_emit_error(ctx, VECX_ERR_GENERIC);
    return;
  }

  sqlite3_result_blob(ctx, dest_blob, size, sqlite3_free);
}

void x_add(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  apply_op(ADD, ctx, argc, argv);
}

void x_sub(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  apply_op(SUB, ctx, argc, argv);
}

void x_mul(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  apply_op(MUL, ctx, argc, argv);
}

void x_div(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  apply_op(DIV, ctx, argc, argv);
}

void x_info(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  if (argc != 0) {
    sqlite3_result_null(ctx);
    return;
  }

#ifdef ENABLE_CUDA_MODE
  sqlite3_result_text(ctx, "Backend: GPU (CUDA)", -1, SQLITE_STATIC);
#else
  sqlite3_result_text(ctx, "Backend: CPU", -1, SQLITE_STATIC);
#endif
}

void x_show(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  if (argc != 1) {
    sqlite3_result_null(ctx);
    return;
  }

  if (sqlite3_value_type(argv[0]) != SQLITE_BLOB) {
    sqlite3_result_null(ctx);
    return;
  }

  const void *blob = sqlite3_value_blob(argv[0]);
  uint64_t blob_size = sqlite3_value_bytes(argv[0]);
  vecx vx;
  vecx_status status = vecx_parse_blob(blob, blob_size, &vx);

  if (status < 0)
    sqlite3_result_text(ctx, "UNKNOWN VECX FORMAT", -1, SQLITE_STATIC);
  else {
    std::string info = vecx_show(vx);
    sqlite3_result_text(ctx, info.c_str(), info.size(), SQLITE_TRANSIENT);
  }
}

// nvcc + cl does not automatically export the symbols
#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

EXPORT int sqlite3_vecx_init(sqlite3 *db, char **pzErrMsg,
                             const sqlite3_api_routines *pApi) {

  init_device();

  SQLITE_EXTENSION_INIT2(pApi);

  sqlite3_create_function(db, "x_size", 1, SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                          0, x_size, 0, 0);
  sqlite3_create_function(db, "x_type", 1, SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                          0, x_type, 0, 0);
  sqlite3_create_function(db, "x_show", 1, SQLITE_DETERMINISTIC, 0, x_show, 0,
                          0);
  sqlite3_create_function(db, "x_norm", 1, SQLITE_DETERMINISTIC, 0, x_norm, 0,
                          0);
  sqlite3_create_function(db, "x_dequantize", 1, SQLITE_DETERMINISTIC, 0,
                          x_dequantize, 0, 0);

  sqlite3_create_function(db, "x_add", 2, SQLITE_DETERMINISTIC, 0, x_add, 0, 0);
  sqlite3_create_function(db, "x_sub", 2, SQLITE_DETERMINISTIC, 0, x_sub, 0, 0);
  sqlite3_create_function(db, "x_mul", 2, SQLITE_DETERMINISTIC, 0, x_mul, 0, 0);
  sqlite3_create_function(db, "x_div", 2, SQLITE_DETERMINISTIC, 0, x_div, 0, 0);

  sqlite3_create_function(db, "x_info", 0, SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                          0, x_info, 0, 0);

  return SQLITE_OK;
}
