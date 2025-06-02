import sqlite3
import os

conn = sqlite3.connect(":memory:")
cur = conn.cursor()
conn.enable_load_extension(True)  # !
ext = ".dll" if os.name == "nt" else ""
conn.load_extension(f"./bin/vecx{ext}")

with open("./e2e/example.sql") as f:
    cur.executescript(f.read())

closest_to_id = 2  # banana
cur.execute(
    """
    SELECT
        w.id,
        w.word,
        floor(100 * (1 + x_cosim(w.emb, t.emb)) / 2) similarity
    FROM
        Words AS w
    JOIN
        (SELECT * FROM Words WHERE id = ?) AS t
    ORDER BY similarity DESC;
    """,
    (closest_to_id,),
)
for row in cur.fetchall():
    print(row)


conn.close()
