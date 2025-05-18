
#include <memory>

#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1

#include "common.hpp"

#ifdef ENABLE_CUDA_MODE
#include "gpu.cuh"
#else
#include "cpu.hpp"
#endif

void vecx_emit_error(sqlite3_context *ctx, vecx_status status) {
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
  case VECX_ERR_GENERIC:
  default:
    sqlite3_result_error(ctx, "Unknown error while processing vecx object", -1);
    break;
  }
}

void vecx_size(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  if (argc != 1) {
    sqlite3_result_null(ctx);
    return;
  }

  const void *blob = sqlite3_value_blob(argv[0]);
  uint64_t blob_size = sqlite3_value_bytes(argv[0]);
  vecx vec;
  vecx_status status = vecx_parse_blob(blob, blob_size, &vec);

  if (status < 0)
    vecx_emit_error(ctx, status);
  else
    sqlite3_result_int64(ctx, (int64_t)vec.size);
}

void vecx_type(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  if (argc != 1) {
    sqlite3_result_null(ctx);
    return;
  }

  const void *blob = sqlite3_value_blob(argv[0]);
  uint64_t blob_size = sqlite3_value_bytes(argv[0]);
  vecx vec;
  vecx_status status = vecx_parse_blob(blob, blob_size, &vec);

  if (status < 0)
    vecx_emit_error(ctx, status);
  else {
    switch (vec.dtype) {
    case INT_32:
      sqlite3_result_text(ctx, "I32", -1, SQLITE_STATIC);
      break;
    case INT_64:
      sqlite3_result_text(ctx, "I64", -1, SQLITE_STATIC);
      break;
    case FLOAT_32:
      sqlite3_result_text(ctx, "F32", -1, SQLITE_STATIC);
      break;
    default:
      vecx_emit_error(ctx, VECX_ERR_UNKNOWN_DTYPE);
      break;
    }
  }
}

void vecx_norm(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  if (argc != 1) {
    sqlite3_result_null(ctx);
    return;
  }

  const void *blob = sqlite3_value_blob(argv[0]);
  uint64_t blob_size = sqlite3_value_bytes(argv[0]);
  vecx vec;
  vecx_status status = vecx_parse_blob(blob, blob_size, &vec);

  if (status < 0)
    vecx_emit_error(ctx, status);
  else
    sqlite3_result_double(ctx, f32_norm(&vec));
}

extern "C" int sqlite3_vecx_init(sqlite3 *db, char **pzErrMsg,
                                 const sqlite3_api_routines *pApi) {
  SQLITE_EXTENSION_INIT2(pApi);

  sqlite3_create_function(db, "vecx_size", 1,
                          SQLITE_UTF8 | SQLITE_DETERMINISTIC, 0, vecx_size, 0,
                          0);
  sqlite3_create_function(db, "vecx_type", 1,
                          SQLITE_UTF8 | SQLITE_DETERMINISTIC, 0, vecx_type, 0,
                          0);
  sqlite3_create_function(db, "vecx_norm", 1,
                          SQLITE_UTF8 | SQLITE_DETERMINISTIC, 0, vecx_norm, 0,
                          0);

  return SQLITE_OK;
}
