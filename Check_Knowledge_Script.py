import sqlite3
import numpy as np

db_path = "knowledge_db.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    cursor.execute("SELECT * FROM knowledge_items")
    rows = cursor.fetchall()
    print(f"Found {len(rows)} rows.")
    for row in rows:
        id, topic, emb_len, emb_blob = row
        print(f"ID: {id}, Topic: {topic}, Blob Len: {emb_len}")
        if emb_blob:
            try:
                emb_arr = np.frombuffer(emb_blob, dtype=np.float32)
                print(f"  Shape: {emb_arr.shape}, First 5: {emb_arr[:5]}")
            except Exception as e:
                print(f"  Error parsing blob: {e}")
        else:
            print("  Embedding is None/Empty")

except Exception as e:
    print(f"Error: {e}")
finally:
    conn.close()