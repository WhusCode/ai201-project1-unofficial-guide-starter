"""
build_chunks.py - Milestone 3: Document ingestion + chunking
Domain: Guides and Walkthroughs for Hollow Knight: Silksong

Loads plain-text source files from ./documents, cleans them, splits them into
512-token chunks with 100-token overlap using the all-MiniLM-L6-v2 tokenizer,
then prints a cleaned preview + 5 representative chunks and counts the result.

Usage (with your .venv activated):
    pip install langchain-text-splitters transformers
    python build_chunks.py
"""

import os
import re
import glob
import html
import json

from transformers import AutoTokenizer
# If the import below fails on an older LangChain, use:
#   from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Config (from planning.md) ---
DOCUMENTS_DIR = "documents"
CHUNK_SIZE = 128        # tokens. Smaller chunks isolate individual facts for precise retrieval
CHUNK_OVERLAP = 30      # tokens
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_FILE = "chunks.json"


def load_documents(folder):
    """Load every .txt file in `folder`. Returns list of {source, text} dicts."""
    paths = sorted(glob.glob(os.path.join(folder, "*.txt")))
    if not paths:
        raise FileNotFoundError(
            f"No .txt files found in '{folder}/'. "
            "Save each source's text as a .txt file there first."
        )
    docs = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            docs.append({"source": os.path.basename(path), "text": f.read()})
    return docs


def clean_text(text):
    """Strip HTML, entities, and common boilerplate; collapse whitespace."""
    text = html.unescape(text)                 # &amp; &nbsp; &#39; -> normal chars
    text = re.sub(r"<[^>]+>", " ", text)        # remove leftover HTML/XML tags
    boilerplate = [
        r"read more", r"share this", r"sign in", r"log in", r"subscribe",
        r"cookie", r"advertisement", r"related articles?", r"you might also like",
    ]
    for pat in boilerplate:
        text = re.sub(rf"(?im)^.*{pat}.*$", "", text)
    text = re.sub(r"[ \t]+", " ", text)         # collapse runs of spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)      # collapse blank-line runs
    return text.strip()


def main():
    print("Loading documents...")
    docs = load_documents(DOCUMENTS_DIR)
    print(f"  Loaded {len(docs)} documents.\n")

    print("Cleaning documents...")
    for d in docs:
        d["clean"] = clean_text(d["text"])

    # Sanity check: read one cleaned document. If you still see nav text or
    # HTML entities here, trim the source .txt file and re-run.
    sample = docs[0]
    print(f"\n--- Cleaned preview of '{sample['source']}' (first 800 chars) ---")
    print(sample["clean"][:800])
    print("--- end preview ---\n")

    print("Building token-based splitter...")
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    print("Chunking...")
    all_chunks = []
    skipped = 0
    for d in docs:
        for i, piece in enumerate(splitter.split_text(d["clean"])):
            # Skip empty or whitespace-only chunks before they reach the embedder
            if not piece or not piece.strip():
                skipped += 1
                continue
            all_chunks.append({
                "source": d["source"],
                "chunk_index": i,
                "text": piece,
            })
    if skipped:
        print(f"  Skipped {skipped} empty/whitespace-only chunk(s).")

    # Save chunks for the next milestone (embedding + vector store)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # Inspect 5 representative chunks, spread across the corpus
    n = len(all_chunks)
    print("\n=== 5 REPRESENTATIVE CHUNKS ===")
    idxs = [0, n // 4, n // 2, (3 * n) // 4, n - 1] if n >= 5 else range(n)
    for idx in idxs:
        c = all_chunks[idx]
        n_tokens = len(tokenizer.encode(c["text"], add_special_tokens=False))
        print(f"\n[chunk {idx} | {c['source']} | ~{n_tokens} tokens]")
        print(c["text"])
        print("-" * 60)

    print("\n=== SUMMARY ===")
    print(f"Documents:    {len(docs)}")
    print(f"Total chunks: {n}")
    print(f"Saved to:     {OUTPUT_FILE}")
    if n < 50:
        print("WARNING: <50 chunks - chunks may be too large.")
    elif n > 2000:
        print("WARNING: >2000 chunks - chunks may be too small.")
    else:
        print("Chunk count is in the healthy 50-2000 range.")


if __name__ == "__main__":
    main()