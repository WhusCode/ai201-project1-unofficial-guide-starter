# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->

I choose the domain "Guides and Walkthroughs for Hollow Knight: Silksong". This knowledge is valuable for beginners to the game and people struggling while playing the game as it is fairly new, and it is hard to find through official channels as they want the players to learn from playing the game instead of giving out extra help.

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | dualSHOCKERS        | Hollow Knight: Silksong: Complete Guide & Walkthrough | https://www.dualshockers.com/hollow-knight-silksong-complete-guide-walkthrough/ |
| 2 | Polygon             | Silksong has a secret permadeath mode, as if it needed to get harder | https://www.polygon.com/silksong-permadeath-hard-mode-steel-soul-how-to-unlock/ |
| 3 | MapGenie            | Hollow Knight: Silksong Maps | https://mapgenie.io/hollow-knight-silksong |
| 4 | Fextralife          | Hollow Knight Silksong All Bosses Guide | https://hollowknightsilksong.wiki.fextralife.com/Bosses |
| 5 | Steam               | All Hunter’s Journal Entries - True Hunter Achievement Guide | https://steamcommunity.com/sharedfiles/filedetails/?id=3577313146 |
| 6 | Game8               | Hollow Knight: Silksong - Beginner's Guide: Tips and Tricks | https://game8.co/games/Hollow-Knight-Silksong/archives/546274 |
| 7 | Nintendo Life       | Hollow Knight: Silksong: Progression Guide - Recommended Route | https://www.nintendolife.com/guides/hollow-knight-silksong-progression-guide-recommended-route |
| 8 | YouTube             | Silksong: How to Unlock All Act 1 Traversal Abilities in Order (Complete Guide) | https://youtu.be/YUAI7Q8y23U?si=QzsNqmNYxNN8JYL8 |
| 9 | Reddit              | [Guide] 100% Completion Requirements - What Counts and What Doesn't [Heavy Spoilers] | https://www.reddit.com/r/HollowKnight/comments/1ndrak6/guide_100_completion_requirements_what_counts_and/ |
| 10| Rock Paper Shotgun  | The best Tools in Hollow Knight Silksong | https://www.rockpapershotgun.com/the-best-tools-in-hollow-knight-silksong |

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:**
256 tokens, because it's a good balance for guide paragraphs, and fits most embedding model limits

**Overlap:**
50 tokens, because I want to prevent cutting off important information midway through

**Reasoning:**
Silksong guides have structured list of items with their own descriptions as well as narrative walkthroughs. Which is why I want the model to have a balanced number of tokens for chunk and overlap size so that enough context is preserved without being too large and most of the words aren't cut off mid-explanation.

Edit:
Reduced from 512/100 to 128/30 because the all-MiniLM-L6-v2 embedding model truncates input at 256 tokens; 512-token chunks would only be half-embedded, and the initial run produced just 44 chunks (below the 50 minimum), indicating chunks were too large for precise retrieval
---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:**
all-mpnet-base-v2: The details needs to be accurate, a lot of games share similar key words but the model will be slower

**Top-k:**
k = 8: queries might be too vague if players don't know exactly what to ask for to beat the game

**Production tradeoff reflection:**
In production, I would consider using all-mpnet-base-v2 to be more accurate on terms related to Silksong compared to other games. The tradeoff is cost and latency. Cost for the API key and latency due to network traffic. Context length matters here as players may or may not know what to ask for which is why 8k tokens is ideal.

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | How do I increase the total amount of silk I have in Silksong? | To increase the total amount of silk in Silksong, you will need to collect Spool Fragments. Two Spool Fragments combine to increase silk by one. |
| 2 | How many bosses are there in Silksong? | Counting the variations, there are 44 total bosses in the game. |
| 3 | What is the Father of the Flame and how do I beat it in Silksong? | The Father of the Flame is a gigantic spider-like boss made of twisted wooden bundles with its burning core exposed in the center. It stays rooted to the ground, its massive arms stretching outward with glowing lanterns hanging from each one. These lanterns are the key to the fight, since destroying a lantern also destroys the arm it is attached to. While you focus on breaking them, the boss constantly hurls fireballs to pressure you. Once all the lanterns are gone and its arms are destroyed, it changes tactics and unleashes a deadly fireball barrage. You can then strike its fiery core or the red weak points hanging from its smaller arms to finish the battle. This fight is more about patience than difficulty, since avoiding the fireballs is simple, but taking out each lantern takes time.

One of its basic moves is the Flame Ball Tracker. The Father of the Flame spits a single fireball that follows your position but moves slowly. The counter here is simple: dodge to the side at the right moment, then punish while the boss pauses between shots.

Its most dangerous attack comes after you have destroyed all of its lantern arms. It screams and triggers a Fire Ball Barrage, filling the arena with waves of fireballs that rain down from above. The best way to survive is to keep moving, weaving through the gaps and dodging carefully until the barrage ends. Once the flames stop, you have your chance to close in and land more hits.

Focus on breaking each lantern one by one, avoid the simple tracking fireballs, and stay calm, it uses the flame barrage. |
| 4 | What is the first skill I acquire in Silksong? | The Silkspear is Hornet's very first skill you can unlock in Silksong. |
| 5 | What tool buffs my Silk Skills in Silksong? |  The Volt Filament is an offensive Blue Tool that charges Silk Skills with electricity, essentially buffing them. |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. Chunk boundary splitting boss/ability descriptions — Silksong guides are dense with multi-part information. A boss fight description like the Father of the Flame has attack names, counters, and phase transitions that naturally span several paragraphs. If a chunk boundary falls in the middle of a phase description, the retrieved chunk may contain half the strategy without the other half, causing the RAG system to generate an incomplete or misleading answer. The 100-token overlap helps but doesn't fully eliminate this risk.

2. Off-topic or cross-game retrieval — Several sources (Reddit, Fextralife, Rock Paper Shotgun) also cover the original Hollow Knight, and some guides may reference both games in the same article. A query like "how do I get more currency?" could retrieve chunks about Hollow Knight's Geo system instead of Silksong's Silk economy, since the two games share vocabulary and community spaces. Without careful filtering or metadata tagging by source, the retriever may pull cross-game noise that degrades answer quality.


---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

┌─────────────────────┐     ┌──────────────────────────┐      ┌─────────────────────────┐
│  Document Ingestion │────▶         Chunking           ───▶ │   Embedding + Vector    │
│  BeautifulSoup /    │     │  RecursiveCharacter      │      │   Store                 │
│  requests, pdfminer │     │  TextSplitter            │      │   all-mpnet-base-v2      │
│                     │     │  128 tok · 30 overlap    │      │   + FAISS / ChromaDB    │
└─────────────────────┘     └──────────────────────────┘      └────────────┬────────────┘
                                                                           │
                                                                           ▼
                                                              ┌────────────────────────┐
                                                              │       Retrieval        │
                                                              │   Similarity Search    │
                                                              │       Top-k = 8        │
                                                              └────────────┬───────────┘
                                                                           │
                                                                           ▼
                                                              ┌────────────────────────┐
                                                              │       Generation       │
                                                              │   LLM (GPT-4o mini)    │
                                                              │   + retrieved context  │
                                                              └────────────────────────┘

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

1. Document Ingestion
Tool: Claude

Input: The Documents table (10 sources with URLs), plus a prompt specifying that sources include web pages, Reddit threads, and a YouTube video.
Expected output: A Python script with a load_documents() function that fetches and extracts plain text from each URL using requests + BeautifulSoup, with a fallback note for the YouTube source (transcript via youtube-transcript-api).
Verification: Run the script and confirm it returns a list of non-empty strings, one per source, with no HTTP errors. Manually spot-check 2–3 outputs against the actual webpage content.

2. Chunking
Tool: Claude

Input: The Chunking Strategy section (512 tokens, 100 overlap, RecursiveCharacterTextSplitter), plus the raw text output from Stage 1.
Expected output: A chunk_text() function using LangChain's RecursiveCharacterTextSplitter configured to my exact spec, returning a list of chunk strings with metadata (source URL, chunk index).
Verification: Print the length of chunks produced per document, confirm chunk sizes fall within the expected token range, and manually read 3–4 chunks to check that no chunk cuts off mid-sentence or mid-strategy.

3. Embedding and Vector Store
Tool: Claude

Input: The Retrieval Approach section (model: all-MiniLM-L6-v2, store: FAISS or ChromaDB), plus the chunk list from Stage 2.
Expected output: A build_vector_store() function that embeds all chunks using sentence-transformers and saves them to a local FAISS or ChromaDB index.
Verification: Confirm the index saves without errors, check that the number of stored vectors equals the number of chunks, and run one manual similarity query (e.g. "how do I beat the first boss") to verify results are returned.

4. Retrieval
Tool: Claude

Input: The vector store from Stage 3, the Retrieval Approach section (top-k = 5), and one example question from the Evaluation Plan.
Expected output: A retrieve() function that takes a query string and returns the top-5 most relevant chunks with their source metadata.
Verification: Run all 5 test questions from the Evaluation Plan through retrieve() and confirm the returned chunks actually contain information relevant to each question. Flag any questions where retrieved chunks are clearly off-topic (cross-game noise risk noted in Anticipated Challenges).

5. Generation
Tool: Claude

Input: The retrieved chunks from Stage 4, a system prompt describing the RAG task ("You are a Silksong guide assistant. Answer using only the provided context."), and the Evaluation Plan questions.
Expected output: A generate_answer() function that formats retrieved chunks into a prompt and calls an LLM (GPT-4o mini or Claude via API) to produce a final answer.
Verification: Run all 5 evaluation questions end-to-end and compare outputs against the expected answers in the Evaluation Plan. Mark each as correct, partially correct, or incorrect. Investigate any failures by tracing back to whether the issue was retrieval (wrong chunks) or generation (correct chunks, wrong answer).

**Milestone 3 — Ingestion and chunking:**

**Milestone 4 — Embedding and retrieval:**

**Milestone 5 — Generation and interface:**
