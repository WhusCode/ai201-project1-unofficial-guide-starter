"""
build_embeddings.py - Embedding + Vector Store + Retrieval
Domain: Guides and Walkthroughs for Hollow Knight: Silksong

Reads chunks.json (from build_chunks.py), embeds every chunk with all-MiniLM-L6-v2,
builds a FAISS index using cosine similarity, saves the index + metadata to disk, then
runs the evaluation questions through retrieve() so you can inspect retrieval quality.

Usage (with your .venv activated):
    pip install sentence-transformers faiss-cpu numpy
    python build_embeddings.py
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- Config (from planning.md) ---
CHUNKS_FILE = "chunks.json"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
INDEX_FILE = "silksong.index"     # FAISS index saved here
META_FILE = "chunks_meta.json"    # chunk text + source, aligned to index order
TOP_K = 5

# The 5 evaluation questions from planning.md, used to sanity-check retrieval.
EVAL_QUESTIONS = [
    "How do I increase the total amount of silk I have in Silksong?",
    "How many bosses are there in Silksong?",
    "What is the Father of the Flame and how do I beat it in Silksong?",
    "What is the first skill I acquire in Silksong?",
    "What tool buffs my Silk Skills in Silksong?",
]


def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if not chunks:
        raise ValueError(f"{path} is empty. Run build_chunks.py first.")
    return chunks


def build_store(chunks, model):
    """Embed all chunk texts and build a cosine-similarity FAISS index."""
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks with {EMBED_MODEL}...")
    # normalize_embeddings=True lets us use inner product as cosine similarity
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product on normalized vectors = cosine
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved index ({index.ntotal} vectors, dim {dim}) to {INDEX_FILE}")
    print(f"Saved metadata to {META_FILE}\n")
    return index


def retrieve(query, model, index, meta, k=TOP_K):
    """Return the top-k most similar chunks to `query` as (score, chunk) tuples."""
    q_emb = model.encode(
        [query], normalize_embeddings=True
    ).astype("float32")
    scores, idxs = index.search(q_emb, k)
    results = []
    for score, i in zip(scores[0], idxs[0]):
        if i == -1:
            continue
        results.append((float(score), meta[i]))
    return results


def main():
    chunks = load_chunks(CHUNKS_FILE)
    model = SentenceTransformer(EMBED_MODEL)
    index = build_store(chunks, model)
    meta = chunks  # same order as the embeddings we added

    # Sanity-check retrieval against the evaluation questions
    print("=== RETRIEVAL CHECK (top-{} per question) ===".format(TOP_K))
    for q in EVAL_QUESTIONS:
        print(f"\nQ: {q}")
        for rank, (score, chunk) in enumerate(retrieve(q, model, index, meta), 1):
            preview = chunk["text"].replace("\n", " ").strip()
            if len(preview) > 140:
                preview = preview[:140] + "..."
            print(f"  {rank}. [{score:.3f}] {chunk['source']} #{chunk['chunk_index']}"
                  f"  {preview}")


if __name__ == "__main__":
    main()