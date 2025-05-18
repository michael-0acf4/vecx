#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1

#include <stdio.h>

static void add_two(sqlite3_context *ctx, int argc, sqlite3_value **argv)
{
    if (argc != 2)
    {
        sqlite3_result_null(ctx);
        return;
    }

    int a = sqlite3_value_int(argv[0]);
    int b = sqlite3_value_int(argv[1]);

    sqlite3_result_int(ctx, a + b);
}

int sqlite3_add_init(sqlite3 *db, char **pzErrMsg, const sqlite3_api_routines *pApi)
{
    SQLITE_EXTENSION_INIT2(pApi);
    return sqlite3_create_function(db, "add", 2, SQLITE_UTF8, 0, add_two, 0, 0);
}