
import os, re, json, math, random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import umap
import nltk 

CSV_PATH = "synthetic_complaints.csv"  
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
N = 200_000                             
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
BATCH_SIZE_EMBED = 256
MEMMAP_PATH = os.path.join(OUTPUT_DIR, "embeddings_memmap.npy")
K = 12
MB_BATCH = 4096
MB_EPOCHS = 3
UMAP_SAMPLE = 15000
TOP_K_KEYWORDS = 12
random.seed(42); np.random.seed(42)


import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
STOP = set(stopwords.words("english"))

def clean_text_simple(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+"," ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = nltk.word_tokenize(text)
    toks = [t for t in toks if t not in STOP and len(t) > 1]
    return " ".join(toks)


def create_memmap(path, n_rows, dim):
    if os.path.exists(path):
        mm = np.memmap(path, dtype='float32', mode='r+', shape=(n_rows, dim))
        print("Opened existing memmap:", path)
    else:
        mm = np.memmap(path, dtype='float32', mode='w+', shape=(n_rows, dim))
        print("Created memmap:", path)
    return mm

def generate_embeddings_memmap():
    print("Loading model:", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)
    mm = create_memmap(MEMMAP_PATH, N, EMBED_DIM)
    reader = pd.read_csv(CSV_PATH, usecols=['complaint'], chunksize=BATCH_SIZE_EMBED, iterator=True, dtype=str)
    offset = 0
    for chunk in tqdm(reader, desc="Embedding batches"):
        texts = chunk['complaint'].astype(str).apply(clean_text_simple).tolist()
        emb_batch = model.encode(texts, batch_size=len(texts), show_progress_bar=False)
        mm[offset: offset + len(texts)] = np.array(emb_batch, dtype='float32')
        offset += len(texts)
    del mm
    print("Embeddings saved to memmap:", MEMMAP_PATH)


def train_minibatch_kmeans():
    emb = np.memmap(MEMMAP_PATH, dtype='float32', mode='r', shape=(N, EMBED_DIM))
    mbk = MiniBatchKMeans(n_clusters=K, batch_size=MB_BATCH, random_state=42)
    for epoch in range(MB_EPOCHS):
        print(f"MBK epoch {epoch+1}/{MB_EPOCHS}")
        for start in range(0, N, MB_BATCH):
            end = min(N, start + MB_BATCH)
            mbk.partial_fit(emb[start:end])
    joblib.dump(mbk, os.path.join(OUTPUT_DIR, f"minibatch_kmeans_k{K}.joblib"))
    del emb
    return mbk


def predict_labels(mbk):
    emb = np.memmap(MEMMAP_PATH, dtype='float32', mode='r', shape=(N, EMBED_DIM))
    labels = np.empty(N, dtype=np.int32)
    for start in tqdm(range(0, N, MB_BATCH), desc="Predict labels"):
        end = min(N, start + MB_BATCH)
        labels[start:end] = mbk.predict(emb[start:end])
    np.save(os.path.join(OUTPUT_DIR, f"labels_kmeans_k{K}.npy"), labels)
    del emb
    return labels


def extract_keywords(labels):
    df = pd.read_csv(CSV_PATH, usecols=['complaint'], dtype=str)
    results = {}
    for c in range(K):
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            results[c] = {"count":0, "keywords":[], "examples":[]}
            continue
        docs = df.iloc[idxs]['complaint'].astype(str).apply(clean_text_simple).tolist()
        v = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,2))
        X = v.fit_transform(docs)
        sums = X.sum(axis=0).A1
        terms = v.get_feature_names_out()
        top_idx = sums.argsort()[::-1][:TOP_K_KEYWORDS]
        results[c] = {"count": int(len(docs)), "keywords":[terms[i] for i in top_idx], "examples": docs[:3]}
    with open(os.path.join(OUTPUT_DIR, f"kmeans_topics_k{K}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def compute_umap_sample():
    emb = np.memmap(MEMMAP_PATH, dtype='float32', mode='r', shape=(N, EMBED_DIM))
    idx = np.random.choice(N, size=min(UMAP_SAMPLE, N), replace=False)
    sample_emb = emb[idx]
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_2d = reducer.fit_transform(sample_emb)
    np.save(os.path.join(OUTPUT_DIR, "umap_sample.npy"), np.column_stack((idx, umap_2d)))
    del emb
    return idx, umap_2d


def write_final_csv(labels):
    out_csv = os.path.join(OUTPUT_DIR, "clustered_complaints_200k.csv")
    reader = pd.read_csv(CSV_PATH, chunksize=50000, dtype=str)
    start = 0; first = True
    for chunk in reader:
        sz = len(chunk)
        chunk = chunk.reset_index(drop=True)
        chunk["kmeans_label"] = labels[start:start+sz]
        if first:
            chunk.to_csv(out_csv, mode="w", index=False)
            first = False
        else:
            chunk.to_csv(out_csv, mode="a", index=False, header=False)
        start += sz
    print("Final labeled CSV saved to:", out_csv)


if __name__ == "__main__":
    if not os.path.exists(MEMMAP_PATH):
        generate_embeddings_memmap()
    else:
        print("Memmap exists. Skipping embedding generation.")
    mbk = train_minibatch_kmeans()
    labels = predict_labels(mbk)
    _ = extract_keywords(labels)
    compute_umap_sample()
    write_final_csv(labels)
    print("ALL DONE.")
