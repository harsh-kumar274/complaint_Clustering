import numpy as np
import os

p = "output/labels_kmeans_k12.npy"

if not os.path.exists(p):
    print("Labels file missing:", p)
else:
    a = np.load(p)
    print("labels shape:", a.shape)
    print("unique labels:", np.unique(a))
    print("first 20 labels:", a[:20])
