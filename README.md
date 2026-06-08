# The Unofficial Guide — Project 1

A retrieval-augmented generation (RAG) system that answers questions about
*Hollow Knight: Silksong* using a corpus of community guides, wikis, and walkthroughs. You
ask a question in a web UI; the system retrieves the most relevant chunks from its document
store and a language model answers using only those chunks, citing the sources it drew from.

---

## Domain

This system covers **guides and walkthroughs for *Hollow Knight: Silksong*** — boss
strategies, ability/skill unlocks, tools, enemy (Hunter's Journal) entries, progression
routes, and beginner mechanics.

This knowledge is valuable because *Silksong* is a deliberately opaque game: it withholds
explanations by design, expecting players to learn through experimentation, and the official
channels (the game itself, Team Cherry's marketing) intentionally provide almost no guidance
on how systems work or where things are. As a result, the practical knowledge that helps
players actually progress — which boss drops which tool, where an ability is unlocked, what
order to tackle areas in — lives entirely in player-made resources scattered across review
sites, wikis, forums, and video transcripts. No single official source collects it, and
because the game is recent, even the community resources are incomplete and still being
filled in. This project consolidates ten of those community sources into one searchable,
grounded assistant.

---

## Document Sources

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | DualShockers — Complete Guide & Walkthrough (guide directory) | Web article | https://www.dualshockers.com/hollow-knight-silksong-complete-guide-walkthrough/ → `documents/01_dualshockers.txt` |
| 2 | Polygon — Secret permadeath (Steel Soul) mode | Web article | https://www.polygon.com/silksong-permadeath-hard-mode-steel-soul-how-to-unlock/ → `documents/02_polygon.txt` |
| 3 | MapGenie — Silksong Maps | Interactive map (text descriptions) | https://mapgenie.io/hollow-knight-silksong → `documents/03_mapgenie.txt` |
| 4 | Fextralife — All Bosses Guide | Wiki | https://hollowknightsilksong.wiki.fextralife.com/Bosses → `documents/04_fextralife.txt` |
| 5 | Steam Community — All Hunter's Journal Entries (True Hunter guide) | Community guide | https://steamcommunity.com/sharedfiles/filedetails/?id=3577313146 → `documents/05_steam.txt` |
| 6 | Game8 — Beginner's Guide: Tips and Tricks | Web article | https://game8.co/games/Hollow-Knight-Silksong/archives/546274 → `documents/06_game8.txt` |
| 7 | Nintendo Life — Progression Guide: Recommended Route | Web article | https://www.nintendolife.com/guides/hollow-knight-silksong-progression-guide-recommended-route → `documents/07_nintendolife.txt` |
| 8 | YouTube — How to Unlock All Act 1 Traversal Abilities in Order | Video transcript | https://youtu.be/YUAI7Q8y23U → `documents/08_youtube.txt` |
| 9 | Reddit (r/HollowKnight) — 100% Completion Requirements thread | Forum thread | https://www.reddit.com/r/HollowKnight/comments/1ndrak6/ → `documents/09_reddit.txt` |
| 10 | Rock Paper Shotgun — The best Tools in Silksong | Web article | https://www.rockpapershotgun.com/the-best-tools-in-hollow-knight-silksong → `documents/10_rockpapershotgun.txt` |

The sources were chosen for variety of subtopic and shape: list-style references (the boss
guide, the 237-entry Hunter's Journal, the tools rankings), flowing prose walkthroughs (the
progression guide, the ability-unlock transcript, the beginner tips), and a community forum
thread, so the corpus covers both "what exists" lookups and "how do I" explanations.

---

## Chunking Strategy

**Chunk size:** 128 tokens, counted with the embedding model's own tokenizer via LangChain's
`RecursiveCharacterTextSplitter.from_huggingface_tokenizer`, so chunk sizes line up with what
the embedder actually processes.

**Overlap:** 30 tokens, used so that a fact sitting near a boundary (a boss's location and its
drop, or a multi-step strategy) is not cleanly severed between two chunks.

**Why these choices fit your documents:** The size was tuned against the corpus, not guessed.
Most of these documents are short, self-contained facts — one boss per line, one tool per
entry, one journal bug per line — rather than long prose. Large chunks bury an individual
fact among dozens of unrelated ones, which dilutes the embedding and makes precise retrieval
impossible, so smaller chunks isolate each fact for matching. A hard constraint also pushed
the size down: the embedding model truncates input past 256 tokens, so oversized chunks would
only be half-embedded; and the initial 512-token run produced just 44 chunks (below the
50-chunk minimum), confirming chunks were too large. 128/30 keeps every chunk fully embedded
and isolates individual facts. Preprocessing before chunking: HTML entities are unescaped,
residual HTML/XML tags are stripped, common boilerplate lines (nav, "read more", cookie/ad
text) are removed, whitespace and blank-line runs are collapsed, and empty/whitespace-only
chunks are filtered out before embedding.

**Final chunk count:** 162 chunks across the 10 documents.

---

## Embedding Model

**Model used:** `sentence-transformers/all-mpnet-base-v2`, run locally with no API key and no
rate limit. The project started on the recommended `all-MiniLM-L6-v2` but switched to
`all-mpnet-base-v2` after testing showed MiniLM could not connect paraphrased queries to the
right chunks (see Failure Case Analysis). mpnet's larger 768-dimension embeddings give
noticeably better semantic matching on this corpus.

**Production tradeoff reflection:** With cost off the table, the biggest axis is **accuracy on
domain-specific text** — game guides are dense with proper nouns (boss names like
"Skarrsinger Karmelita", abilities like "Silk Soar") that general models tokenize awkwardly,
so a larger or corpus-fine-tuned model would resolve those better. **Context length** matters
because a few sources (the YouTube transcript, long walkthrough sections) carry multi-sentence
strategies that a 256/384-token model truncates; a long-context embedder (e.g. OpenAI's
`text-embedding-3`, 8k tokens) would allow larger chunks without losing the tail.
**Latency** is the cost of going bigger: mpnet is already several times slower than MiniLM,
and an API-hosted model adds network round-trips that hurt an interactive UI.
**Multilingual support** is irrelevant here (the corpus is all English) but would matter for a
localized guide. **Local vs. API-hosted** is the deployment lever: local models are free and
private but capped by user hardware, while API models offer higher accuracy and no local
compute at the cost of per-query fees and an external dependency. For this project, local
mpnet was the right balance of accuracy and zero cost.

---

## Grounded Generation

**System prompt grounding instruction:** The model receives this system prompt, which
*requires* grounding rather than merely suggesting it:

> "You are a Hollow Knight: Silksong guide assistant. Answer the user's question using ONLY
> the numbered sources provided in the context. Do not use outside knowledge. If the sources
> do not contain enough information to answer, say clearly that the provided sources do not
> cover it — do not guess or invent details. Keep answers concise and specific."

Two structural choices reinforce this. The retrieved chunks are formatted into a **numbered,
source-labeled context block** (each prefixed with its source filename, chunk index, and
distance), and the user turn ends with "Answer using only the context above," anchoring the
model to a finite, visible set of evidence rather than its training memory. The prompt also
explicitly authorizes refusal, which is what makes the system decline on uncovered questions
instead of fabricating a plausible answer.

**How source attribution is surfaced in the response:** Attribution is **guaranteed
programmatically, not left to the model.** After generation, the code builds the source list
directly from the retrieval metadata —
`f"{m['source']} (chunk #{m['chunk_index']}, distance {d:.3f})"` — and displays it in a
separate "Retrieved from" field in the Gradio UI. The cited sources therefore always reflect
exactly what was retrieved, even if the language model omits them from its prose.

---

## Evaluation Report

Run on the final system (ChromaDB retrieval with `all-mpnet-base-v2`, k=5, local generation).

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | How do I increase the total amount of silk I have? | Collect Spool Fragments; two combine to add one silk. | Declined — stated the sources don't explain how to increase total silk, noting extra silk is mentioned only for healing/abilities. | Partially relevant | Inaccurate (honest decline — corpus lacks the specific fact) |
| 2 | How many bosses are there? | 44 main bosses (and 4 mini-bosses). | "44 main bosses and 4 mini-bosses have been discovered," cited to `04_fextralife.txt`. | Relevant (top distance 0.330) | Accurate |
| 3 | What is the Father of the Flame and how do I beat it? | Detailed fight strategy (destroy lanterns/arms, dodge fireballs, hit the core). | Declined — stated the sources don't contain a Father of the Flame strategy. | Off-target (best distance 0.567) | Inaccurate (honest decline — coverage gap) |
| 4 | What is the first skill I acquire? | The Silkspear. | "The first ability you acquire is the Silkspear," cited to `08_youtube.txt`. | Relevant | Accurate |
| 5 | What tool buffs my Silk Skills? | The Volt Filament. | "Blue Tools buff Silk Skills," citing the Blue Tools category — did not name the Volt Filament specifically. | Partially relevant (distance 0.349 on the category chunk, not the specific tool) | Partially accurate |

**Summary:** 2 accurate (Q2, Q4), 1 partially accurate (Q5), 2 honest declines on
under-covered questions (Q1, Q3). The system never hallucinated — every question it could not
answer from the corpus, it declined rather than invented.

---

## Failure Case Analysis

**Question that failed:** "What tool buffs my Silk Skills in Silksong?" (Q5)

**What the system returned:** Instead of naming the **Volt Filament** (the correct answer,
present in `10_rockpapershotgun.txt` as "Volt Filament — adds a pink lightning effect that
drastically improves the damage of all Silk Skills"), the system returned the more general
"Blue Tools buff Silk Skills," citing the Blue Tools *category* description. The specific
answer-bearing chunk never entered the top-5 retrieval set.

**Root cause (tied to a specific pipeline stage):** This is a **retrieval failure**, not a
coverage gap — the answer is in the corpus, but two stages combined to keep it out of the
results. (1) **Embedding stage / lexical–semantic mismatch:** the query asks what "buffs"
Silk Skills, while the Volt Filament chunk says it "improves the damage" of Silk Skills. The
model must bridge "buffs" ↔ "improves damage"; with `all-MiniLM-L6-v2` it couldn't, and the
chunk didn't appear in the top 5 at all. Switching to `all-mpnet-base-v2` improved this enough
to surface the Blue Tools *category* chunk (whose text literally repeats "buffs" and "Silk
Skills") but still not the Volt Filament line itself. (2) **Chunking stage / keyword
dilution:** the category-intro chunk packs the query's keywords densely, so it out-ranks the
Volt Filament chunk, which leads with "pink lightning effect" before reaching "Silk Skills" —
the specific fact loses to the generic one on surface vocabulary. (A useful contrast is Q3,
Father of the Flame, which looks similar but is a true **coverage gap**: no chunk contains the
strategy at all. Q5 is present-but-not-surfaced; Q3 is simply absent — different failures
needing different fixes.)

**What you would change to fix it:** For Q5, **document enrichment** — prepend a short keyword
header to each tool chunk at build time (e.g. "Tool: Volt Filament. Buffs Silk Skills.") so
the answer-bearing chunk carries the query's vocabulary and stops losing to the category
intro. More broadly, **hybrid retrieval** (combining vector search with a BM25 keyword search
and merging rankings) directly fixes the "exact words are in the chunk but ranked low" class
of failure. For Q3's coverage gap, the only real fix is adding a source that actually contains
the Father of the Flame strategy — no retrieval tuning can surface information that isn't in
the corpus.

---

## Spec Reflection

**One way the spec helped you during implementation:** The `planning.md` Chunking Strategy and
Retrieval Approach sections forced me to commit to concrete, reproducible parameters (chunk
size, overlap, embedding model, top-k) before writing code, so implementation became a matter
of wiring up decisions I had already reasoned through rather than improvising. It also gave me
a baseline to measure against: when retrieval underperformed, I could see exactly which
parameter I was changing and why, instead of tweaking blindly. The spec's "if you change the
numbers, say why" requirement turned each tuning step into a documented, defensible decision.

**One way your implementation diverged from the spec, and why:** I diverged on three
parameters, each driven by test evidence. Chunk size dropped from the planned 512/100 to
128/30, because the embedding model truncates past 256 tokens (so 512-token chunks would be
half-embedded) and the first run produced only 44 chunks, below the 50-chunk floor. The
embedding model changed from `all-MiniLM-L6-v2` to `all-mpnet-base-v2` after MiniLM failed to
connect the "what tool buffs Silk Skills" query to the right chunk. And generation runs on a
local Ollama model (`llama3.2`) instead of the recommended Groq API, to keep the system
entirely free and local with no API key. The vector store is ChromaDB with cosine distance,
one of the two stores the planning allowed.

---

## AI Usage

**Instance 1**

- *What I gave the AI:* My `planning.md` Chunking Strategy section (512-token chunks, 100
  overlap, `RecursiveCharacterTextSplitter`) and my list of source file types, and asked it to
  implement the ingestion + chunking script.
- *What it produced:* A `build_chunks.py` that loads the `.txt` files, cleans them (HTML
  entities, tag stripping, boilerplate removal), splits them token-based using the embedding
  model's tokenizer at my specified 512/100, and prints sample chunks plus a count.
- *What I changed or overrode:* After the first run produced only 44 chunks (below the 50-chunk
  floor) and the tokenizer warned about sequences exceeding the model's limit, I overrode the
  chunk size to 256/50 and then 128/30, because my sources are short fact-style entries rather
  than long prose and the embedder truncates past 256 tokens. I also had it add an empty-chunk
  filter. The final 128/30 setting produced 162 well-isolated chunks.

**Instance 2**

- *What I gave the AI:* The retrieval output for my five evaluation questions, pointing out
  that "what tool buffs my Silk Skills" was not surfacing the Volt Filament chunk even though
  it's in the corpus.
- *What it produced:* An explanation that the failure was a lexical–semantic mismatch ("buffs"
  vs. the chunk's "improves damage") compounded by keyword dilution, with MiniLM leaning too
  heavily on surface vocabulary. It proposed, in order of effort, raising k, enriching chunk
  text with keywords, swapping to `all-mpnet-base-v2`, and hybrid retrieval.
- *What I changed or overrode:* I chose the model swap to `all-mpnet-base-v2` and re-ran the
  comparison rather than accepting the failure or jumping to hybrid retrieval. I confirmed it
  improved the result (the Blue Tools chunk surfaced) but verified it still didn't fully
  resolve the specific-tool retrieval, and I documented that remaining limitation honestly in
  the Failure Case Analysis rather than hiding it.