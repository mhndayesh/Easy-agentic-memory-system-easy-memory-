import json, numpy as np, os, sys
from sentence_transformers import SentenceTransformer

BASE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(BASE, "index", "librarian_index.json")

print("[*] Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

print("[*] Loading index...")
with open(INDEX, "r") as f:
    idx = json.load(f)

keys = list(idx.keys())
vecs = np.array([idx[k]["vector"] for k in keys]).astype("float32")

query = "Crimson Pineapple favorite fruit"
print(f"\n[QUERY] '{query}'")
qv = model.encode([query], convert_to_numpy=True)
sims = np.dot(vecs, qv[0])
top5 = np.argsort(sims)[-5:][::-1]

print("\n--- TOP 5 RESULTS ---")
for rank, i in enumerate(top5, 1):
    print(f"\n#{rank} [score={sims[i]:.4f}] {keys[i]}")
    print(f"   TEXT: {idx[keys[i]]['text'][:200]}")
