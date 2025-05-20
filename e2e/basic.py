import os
import sqlite3
from vecx_spec import Vecx, VECX_DTYPE

a = Vecx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=VECX_DTYPE.FLOAT32)
b = Vecx([1.0] * 177013, dtype=VECX_DTYPE.FLOAT32)

conn = sqlite3.connect(":memory:")
conn.enable_load_extension(True)  # !

ext = ".dll" if os.name == "nt" else ""
conn.load_extension(f"./bin/vecx{ext}")

conn.execute("CREATE TABLE Test (a BLOB, b BLOB);")

conn.execute(
    "INSERT INTO Test (a, b) VALUES (?, ?)",
    (a.pack(), b.pack()),
)
conn.commit()

cur = conn.cursor()
for row in cur.execute("SELECT a, b FROM Test"):
    for col, blob in zip(["a", "b"], row):
        x = Vecx.unpack(blob)
        print(f" {col}: {x[:20]} ... {len(blob)} bytes, {len(x)} elements")


for row in conn.execute("SELECT vecx_info()"):
    print(f"INFO: {row}")

print("Simple data check")
for row in conn.execute(
    "SELECT vecx_size(a), vecx_type(a), vecx_norm(a), vecx_size(b), vecx_type(b), vecx_norm(b)  FROM Test"
):
    print(f"Row: {row}")
