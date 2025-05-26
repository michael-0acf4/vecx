import os
import sqlite3
from vecx_spec import Vecx

a = Vecx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], auto_quantize=True)
b = Vecx([1.0] * (625 * 625), auto_quantize=True)

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


for row in conn.execute("SELECT x_info()"):
    print(f"INFO: {row}")

print("Simple data check")
for row in conn.execute(
    """
    SELECT 
        x_size(x_dequantize(a)),
        x_type(x_dequantize(a)),
        x_show(a),
        x_size(x_dequantize(b)),
        x_show(b),
        x_norm(b),
        x_norm(x_dequantize(b)),
        x_show(
            x_div(
                x_add(x_dequantize(a), x_mul(a, a)),
                x_dequantize(x_dequantize(x_dequantize(a))) --TODO: better error when non matching
            )
        ),
        x_show(NULL)
    FROM Test
    """
):
    print(f"Row: {row}")
