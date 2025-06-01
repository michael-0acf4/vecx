import os
import sqlite3
import time
import random
from vecx_spec import Vecx

a = Vecx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], auto_quantize=True)
b = Vecx([1.0] * (625 * 625))

conn = sqlite3.connect(":memory:")
conn.enable_load_extension(True)  # !

ext = ".dll" if os.name == "nt" else ""
conn.load_extension(f"./bin/vecx{ext}")

conn.execute("CREATE TABLE Test (a BLOB, b BLOB, c FLOAT);")
for _ in range(1000):
    # b.np_data[0] = b.np_data[0] + random.random() / 100  # no cache
    conn.execute(
        "INSERT INTO Test (a, b, c) VALUES (?, ?, ?)",
        (a.pack(), b.pack(), 1234.0),
    )
conn.commit()
print("inserted..")

for row in conn.execute("SELECT x_info()"):
    print(f"INFO: {row}")

rows = []
start = time.perf_counter()
res = conn.execute(
    """
    SELECT 
        x_size(x_dequantize(a)),
        x_type(x_dequantize(a)),
        x_show(a),
        x_size(x_dequantize(b)),
        x_show(b),
        x_norm(b),
        x_norm(x_dequantize(b)), -- 625
        (sqrt(x_dot(x_dequantize(b), b)) + 375.0), -- 1000
        x_cosim(a, a),
        x_show(
            x_div(
                x_add(x_dequantize(a), x_mul(a, a)),
                x_mulk(
                    -- x_dequantize(x_dequantize(x_dequantize(a))),
                    a,
                    2.0
                )
            )
        ),
        x_norm(x_vec('4.0, 3.0')), -- 25
        x_size(x_vec('4.0, 3.0', 2, 3, 1.5)), -- 2 + 2 + 3
        x_show(x_vec('4.0, 3.0', 2, 3, 1.5)), -- F32 [ 1.5 1.5 4 3 1.5 1.5 1.5 ]
        x_show(NULL)
    FROM Test
    """
)
for row in res:
    # actual execution, sqlite_step applies one row at a time
    rows.append(row)
end = time.perf_counter()
print("Compute + Pulling done", (end - start) * 1000, "ms")

print("rows", len(rows))
print("tail", rows[-1])
