import sqlite3
from vecx_spec import pack_vecx_f32_blob, unpack_vecx_f32_blob

a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
b = [4, 0.0, 0.0, 0.0, 3.0]

conn = sqlite3.connect(":memory:")
conn.enable_load_extension(True)  # !
conn.load_extension("../vecx.dll")  # !

conn.execute("CREATE TABLE Test (a BLOB, b BLOB);")

conn.execute(
    "INSERT INTO Test (a, b) VALUES (?, ?)",
    (pack_vecx_f32_blob(a), pack_vecx_f32_blob(b)),
)
conn.commit()

cur = conn.cursor()
for row in cur.execute("SELECT a, b FROM Test"):
    for col, blob in zip(["a", "b"], row):
        print(f" {col}: {unpack_vecx_f32_blob(blob)[:20]} ... {len(blob)} bytes")

print("Simple data check")
for row in conn.execute(
    "SELECT vecx_size(a), vecx_type(a), vecx_norm(a), vecx_size(b), vecx_type(b), vecx_norm(b)  FROM Test"
):
    print(f"Row: {row}")
