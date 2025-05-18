#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1

#include <stdint.h>
#include <memory.h>

// #include <stdio.h>

typedef enum vecx_dtype
{
    INT_32 = 1,
    INT_64 = 2,
    FLOAT_32 = 3,
} vecx_dtype;

uint64_t vecx_type_size(vecx_dtype dtype)
{
    switch (dtype)
    {
    case FLOAT_32:
    case INT_32:
        return 4;
    case INT_64:
        return 8;
    default:
        return 0;
    }
}

typedef enum vecx_status
{
    VECX_OK = 0,
    VECX_ERR_BAD_VECX_HEADER = -1,
    VECX_ERR_INVALID_LAYOUT = -2,
    VECX_ERR_INVALID_SIZE = -3,
    VECX_ERR_UNKNOWN_DTYPE = -4,
    VECX_ERR_RECOVER_NULL = -5,
    VECX_ERR_GENERIC = -1000,
} vecx_status;

typedef struct vecx
{
    uint64_t size;
    vecx_dtype dtype;
    const void *data;
} vecx;

/**
 * vecx Layout:
 *
 * * Magic "vecx" (32 bits)
 * * vecx_dtype (8 bits)
 * * size (64 bits)
 * * data pointer (size * vecx_dtype)
 */
vecx_status vecx_parse_blob(const void *blob, int blob_size, vecx *out_vecx)
{
    if (!blob)
        return VECX_ERR_BAD_VECX_HEADER;

    if (!out_vecx)
        return VECX_ERR_GENERIC;

    const int header_size = 4 + 1 + 8;
    if (blob_size < header_size)
        return VECX_ERR_INVALID_LAYOUT;

    const uint8_t *rest = (const uint8_t *)blob;

    // header
    if (rest[0] != 'v' || rest[1] != 'e' || rest[2] != 'c' || rest[3] != 'x')
        return VECX_ERR_BAD_VECX_HEADER;
    rest += 4;

    // dtype
    vecx_dtype dtype = (vecx_dtype)rest[0];
    if (dtype != INT_32 && dtype != INT_64 && dtype != FLOAT_32)
        return VECX_ERR_UNKNOWN_DTYPE;
    out_vecx->dtype = dtype;
    rest += 1;

    // size
    uint64_t size;
    memcpy(&size, rest, sizeof(size));
    out_vecx->size = size;
    rest += sizeof(uint64_t);

    // expected size
    int type_size = vecx_type_size(out_vecx->dtype);

    uint64_t expected_total = header_size + size * type_size;
    if ((uint64_t)blob_size != expected_total)
        return VECX_ERR_INVALID_SIZE;

    out_vecx->data = rest;

    return VECX_OK;
}

void vecx_emit_error(sqlite3_context *ctx, vecx_status status)
{
    switch (status)
    {
    case VECX_ERR_BAD_VECX_HEADER:
        sqlite3_result_error(ctx, "Underlying blob is not a vecx object", -1);
        break;
    case VECX_ERR_UNKNOWN_DTYPE:
        sqlite3_result_error(ctx, "Underlying vecx object is of an unknown dtype", -1);
        break;
    case VECX_ERR_INVALID_LAYOUT:
        sqlite3_result_error(ctx, "Underlying vecx object layout is not supported", -1);
        break;
    case VECX_ERR_INVALID_SIZE:
        sqlite3_result_error(ctx, "Underlying vecx object has an invalid reported size", -1);
        break;
    case VECX_ERR_GENERIC:
    default:
        sqlite3_result_error(ctx, "Unknown error while processing vecx object", -1);
        break;
    }
}

static void vecx_size(sqlite3_context *ctx, int argc, sqlite3_value **argv)
{
    if (argc != 1)
    {
        sqlite3_result_null(ctx);
        return;
    }

    const void *blob = sqlite3_value_blob(argv[0]);
    uint64_t blob_size = sqlite3_value_bytes(argv[0]);
    vecx vec;
    int status = vecx_parse_blob(blob, blob_size, &vec);

    if (status < 0)
        vecx_emit_error(ctx, status);
    else
        sqlite3_result_int64(ctx, (int64_t)vec.size);
}

static void vecx_type(sqlite3_context *ctx, int argc, sqlite3_value **argv)
{
    if (argc != 1)
    {
        sqlite3_result_null(ctx);
        return;
    }

    const void *blob = sqlite3_value_blob(argv[0]);
    uint64_t blob_size = sqlite3_value_bytes(argv[0]);
    vecx vec;
    int status = vecx_parse_blob(blob, blob_size, &vec);

    if (status < 0)
        vecx_emit_error(ctx, status);
    else
    {
        switch (vec.dtype)
        {
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

int sqlite3_vecx_init(sqlite3 *db, char **pzErrMsg, const sqlite3_api_routines *pApi)
{
    SQLITE_EXTENSION_INIT2(pApi);

    sqlite3_create_function(db, "vecx_size", 1, SQLITE_UTF8 | SQLITE_DETERMINISTIC, 0, vecx_size, 0, 0);
    sqlite3_create_function(db, "vecx_type", 1, SQLITE_UTF8 | SQLITE_DETERMINISTIC, 0, vecx_type, 0, 0);

    return SQLITE_OK;
}
