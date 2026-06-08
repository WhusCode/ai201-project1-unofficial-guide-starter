"""
build_embeddings_chroma.py - Embedding + ChromaDB Vector Store + Retrieval (Milestone 4)
Domain: Guides and Walkthroughs for Hollow Knight: Silksong

Reads chunks.json (from build_chunks.py), embeds every chunk with a sentence-transformers
model, stores them in a persistent ChromaDB collection with source + chunk_index metadata,
then runs the evaluation questions through retrieve() and prints chunks with DISTANCE scores.

ChromaDB returns cosine DISTANCE (0 = identical, higher = less similar), so lower is better
here. This matches the milestone checkpoint wording ("distance scores below 0.5").

Usage (with your .venv activated):
    pip install sentence-transformers chromadb
    python build_embeddings_chroma.py
"""

import json
import chromadb
from chromadb.utils import embedding_functions

# --- Config (from planning.md) ---
CHUNKS_FILE = "chunks.json"
# all-mpnet-base-v2 chosen over the all-MiniLM-L6-v2 default: it resolved a semantic-match
# failure (the "what tool buffs Silk Skills" -> Volt Filament query). See planning.md.
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHROMA_DIR = "chroma_store"        # persistent on-disk store
COLLECTION_NAME = "silksong"
TOP_K = 5

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


def build_collection(chunks):
    """Embed chunks and (re)build a persistent ChromaDB collection with cosine distance."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Start fresh each run so re-runs don't duplicate entries
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # Chroma handles embedding internally via this function (uses sentence-transformers)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},  # use cosine distance, not default L2
    )

    documents = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]
    ids = [f"{c['source']}-{c['chunk_index']}-{i}" for i, c in enumerate(chunks)]

    print(f"Embedding {len(documents)} chunks with {EMBED_MODEL} and storing in ChromaDB...")
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Stored {collection.count()} chunks in collection '{COLLECTION_NAME}'.\n")
    return collection


def retrieve(query, collection, k=TOP_K):
    """Return the top-k chunks for `query` as a list of (distance, document, metadata)."""
    res = collection.query(query_texts=[query], n_results=k)
    out = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        out.append((dist, doc, meta))
    return out


def main():
    chunks = load_chunks(CHUNKS_FILE)
    collection = build_collection(chunks)

    print(f"=== RETRIEVAL CHECK (top-{TOP_K} per question, lower distance = better) ===")
    for q in EVAL_QUESTIONS:
        print(f"\nQ: {q}")
        for rank, (dist, doc, meta) in enumerate(retrieve(q, collection), 1):
            preview = doc.replace("\n", " ").strip()
            if len(preview) > 140:
                preview = preview[:140] + "..."
            print(f"  {rank}. (dist {dist:.3f}) {meta['source']} "
                  f"#{meta['chunk_index']}  {preview}")


if __name__ == "__main__":
    main()