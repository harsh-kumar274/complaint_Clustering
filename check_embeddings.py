import numpy as np
import os

path = "output/embeddings_memmap.npy"

if not os.path.exists(path):
    print("Memmap missing:", path)
    exit()


mm = np.memmap(path, dtype="float32", mode="r")

try:
    emb = mm.reshape(-1, 384)
    print("Embeddings shape:", emb.shape)
    print("First row mean/std:", emb[0].mean(), emb[0].std())
    print("Global mean/std:", emb.mean(), emb.std())
except Exception as e:
    print("Reshape error:", e)
