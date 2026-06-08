"""
rag_query.py - Generation stage (Milestone 5), local / no API key
Domain: Guides and Walkthroughs for Hollow Knight: Silksong

Same pipeline as the Claude version, but generation runs on a LOCAL model via Ollama, so
there is no API key and no cost. Connects to the ChromaDB collection built by
build_embeddings_chroma.py, retrieves the top-k chunks, and asks the local model to answer
using ONLY those chunks. If the context lacks the answer, the model is told to say so.

Setup (with your .venv activated):
    1. Install Ollama from https://ollama.com/download and let it run.
    2. ollama pull llama3.2          (or llama3.2:1b for low RAM, qwen2.5:7b for better quality)
    3. pip install ollama chromadb sentence-transformers

Run:
    python rag_query.py                 # runs the 5 evaluation questions
    python rag_query.py "your question" # answers a single custom question
"""

import sys
import chromadb
from chromadb.utils import embedding_functions
import ollama

# --- Config ---
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "silksong"
TOP_K = 5
OLLAMA_MODEL = "llama3.2"   # must match what you pulled with `ollama pull`

SYSTEM_PROMPT = (
    "You are a Hollow Knight: Silksong guide assistant. Answer the user's question using "
    "ONLY the numbered sources provided in the context. Do not use outside knowledge. "
    "If the sources do not contain enough information to answer, say clearly that the "
    "provided sources do not cover it - do not guess or invent details. When you use a "
    "fact, cite the source it came from like [01_dualshockers.txt]. Keep answers concise "
    "and specific."
)


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def retrieve(query, collection, k=TOP_K):
    res = collection.query(query_texts=[query], n_results=k)
    out = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        out.append((dist, doc, meta))
    return out


def build_context(results):
    blocks = []
    for n, (dist, doc, meta) in enumerate(results, 1):
        blocks.append(
            f"[Source {n}: {meta['source']} #{meta['chunk_index']}, distance {dist:.3f}]\n"
            f"{doc.strip()}"
        )
    return "\n\n".join(blocks)


def generate_answer(query, collection):
    results = retrieve(query, collection)
    context = build_context(results)
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer using only the context above."
    )
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    answer = resp["message"]["content"]
    sources = [f"{m['source']} #{m['chunk_index']}" for _, _, m in results]
    return answer, sources


EVAL_QUESTIONS = [
    "How do I increase the total amount of silk I have in Silksong?",
    "How many bosses are there in Silksong?",
    "What is the Father of the Flame and how do I beat it in Silksong?",
    "What is the first skill I acquire in Silksong?",
    "What tool buffs my Silk Skills in Silksong?",
]


def main():
    collection = get_collection()
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else EVAL_QUESTIONS
    for q in questions:
        print("=" * 70)
        print(f"Q: {q}\n")
        answer, sources = generate_answer(q, collection)
        print(answer)
        print(f"\n[retrieved from: {', '.join(sources)}]\n")


if __name__ == "__main__":
    main()