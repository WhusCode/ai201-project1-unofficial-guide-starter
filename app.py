"""
app.py - Gradio interface for the Silksong RAG pipeline (Milestone 5)
Domain: Guides and Walkthroughs for Hollow Knight: Silksong

Wraps the local-Ollama generation pipeline (rag_query.py) in a Gradio web UI.
You type a question; the system retrieves the top-k chunks from ChromaDB, the local model
answers using ONLY those chunks, and the source list is filled in PROGRAMMATICALLY from the
retrieved metadata (not trusted to the LLM) so attribution is always guaranteed.

Setup (with your .venv activated, and Ollama running):
    pip install gradio
    python app.py
Then open http://localhost:7860
"""

import gradio as gr

# Reuse the working pipeline pieces from the Ollama generation script.
from rag_query import get_collection, retrieve, build_context, OLLAMA_MODEL
import ollama

SYSTEM_PROMPT = (
    "You are a Hollow Knight: Silksong guide assistant. Answer the user's question using "
    "ONLY the numbered sources provided in the context. Do not use outside knowledge. "
    "If the sources do not contain enough information to answer, say clearly that the "
    "provided sources do not cover it - do not guess or invent details. Keep answers "
    "concise and specific."
)

# Load the vector store once at startup, not on every query.
collection = get_collection()


def ask(question):
    """End-to-end: retrieve -> generate -> return answer + programmatic source list."""
    results = retrieve(question, collection)
    context = build_context(results)
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
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
    # Source attribution is built in CODE from retrieval metadata, so it is guaranteed
    # regardless of whether the model chose to cite anything in its text.
    sources = [f"{m['source']} (chunk #{m['chunk_index']}, distance {d:.3f})"
               for d, _, m in results]
    return {"answer": answer, "sources": sources}


def handle_query(question):
    if not question or not question.strip():
        return "Please enter a question.", ""
    result = ask(question)
    sources = "\n".join(f"\u2022 {s}" for s in result["sources"])
    return result["answer"], sources


with gr.Blocks(title="Silksong Unofficial Guide") as demo:
    gr.Markdown("# Hollow Knight: Silksong - Unofficial Guide\n"
                "Ask about bosses, abilities, tools, progression, and more. "
                "Answers come only from the collected guide sources.")
    inp = gr.Textbox(label="Your question",
                     placeholder="e.g. What is the first skill I acquire in Silksong?")
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=5)
    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()