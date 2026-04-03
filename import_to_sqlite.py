
import sqlite3, pandas as pd, os

CSV = os.path.join("output", "clustered_complaints_200k.csv")
DB = os.path.join("output", "complaints.db")

if not os.path.exists(CSV):
    raise SystemExit("CSV not found. Run pipeline first.")

conn = sqlite3.connect(DB)
chunksize = 50000
reader = pd.read_csv(CSV, chunksize=chunksize, dtype=str)
first = True
for chunk in reader:
    
    if 'id' in chunk.columns:
        chunk['id'] = pd.to_numeric(chunk['id'], errors='coerce').fillna(0).astype(int)
    if 'kmeans_label' in chunk.columns:
        chunk['kmeans_label'] = pd.to_numeric(chunk['kmeans_label'], errors='coerce').fillna(-1).astype(int)
    chunk.to_sql('complaints', conn, if_exists='append', index=False)
    print("Appended chunk, current DB size:", os.path.getsize(DB))
conn.execute("CREATE INDEX IF NOT EXISTS idx_label ON complaints(kmeans_label)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_state ON complaints(state)")
conn.commit()
conn.close()
print("SQLite DB created at:", DB)
