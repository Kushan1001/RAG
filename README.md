# HF-Docs RAG: Production-Grade Knowledge Retrieval System

A Retrieval-Augmented Generation (RAG) pipeline that turns fragmented HuggingFace documentation into an accurate, context-aware Q&A system, built to cut down the manual doc-searching that eats into support engineers' day.

The pipeline is evaluated on a labelled test set that measures retrieval and generation separately, so it is clear which part of the system is responsible when an answer is weak.

## Problem Statement

XYZ AI Solutions manages an extensive ecosystem of developer tools. Internal support engineers and community managers currently spend **over 40% of their day** navigating fragmented documentation to answer technical questions about HuggingFace-based workflows. Keyword search fails to capture the nuance of complex ML queries, leading to high ticket latency and inconsistent developer advice.

**Goal:** Build a robust, end-to-end RAG pipeline where success is measured by the *reliability and relevance* of retrieved information, not just code execution. The pipeline ingests documentation, stores it in a vector database, and answers complex queries with an LLM grounded in retrieved context.

## How It Works

```
HuggingFace Docs Dataset (2,647 documents)
        │
        ▼
  Chunking (1000 chars, 200 overlap) → 27,437 chunks
        │
        ▼
  Embedding (BAAI/bge-small-en-v1.5, 384-dim, normalized)
        │
        ▼
  Vector Store (Milvus Lite, IP similarity)
        │
        ▼
  Query → Top-K Retrieval → Context-Grounded Prompt
        │
        ▼
  Generation (OpenAI gpt-4o-mini)
        │
        ▼
  Evaluation
    ├── Retrieval:  Recall@K and MRR against gold chunks
    └── Generation: Hallucination + Answer Relevance (Opik)
```

## Pipeline Stages

1. **Data Ingestion.** Loads the [`m-ric/huggingface_doc`](https://huggingface.co/datasets/m-ric/huggingface_doc) dataset: 2,647 documentation pages, each with its text and source path.

2. **Chunking.** A custom sliding-window chunker splits documents into 1000-character chunks with 200-character overlap, which keeps context across chunk boundaries. The loop stops as soon as a chunk reaches the end of the text, so it does not emit a short trailing chunk that is already covered by the previous chunk's overlap. This produces 27,437 chunks (about 10.4 per document).

3. **Embeddings.** Generates dense vectors with `BAAI/bge-small-en-v1.5` (Sentence-Transformers, 384 dimensions). All chunks go through a single batched `encode` call with normalized embeddings.

4. **Vector Store.** Stores every chunk embedding in **Milvus Lite** (via `pymilvus`) along with the chunk text and source. Search uses Inner Product, which is equivalent to cosine similarity because the embeddings are normalized.

5. **Retrieval.** Embeds the query and returns the top-K most similar chunks with their source and similarity score.

6. **Generation.** Puts the retrieved chunks into a grounded prompt and generates an answer with `gpt-4o-mini`. The prompt tells the model to say so when the context does not contain enough information, which reduces hallucination. The OpenAI client is set up with a 60-second timeout and 3 retries.

7. **Evaluation.** Done in two parts, described in the next section.

## Evaluation

### Why two levels

The first check was 4 hand-written queries scored with Opik. They came back with 0% hallucination and 0.96 average answer relevance. That confirms the pipeline runs, but 4 easy questions say very little about quality and nothing about where failures come from. So the notebook also builds a labelled test set.

### Labelled test set

- **30 questions**, each written by `gpt-4o-mini` from one randomly sampled chunk, so every question has a known **gold chunk** that retrieval should return.
- Chunks are filtered before sampling: at least 600 characters, mostly English prose, no tables, not code-heavy.
- **One question per source document**, so the set covers 30 different docs.
- The generation prompt requires each question to name the specific library, class, or function it is about, and to stand on its own without referring to "the passage".
- Fixed random seed (42). The set is saved to `eval_queries_v3.json` and reused on later runs. It is rebuilt automatically if the saved gold chunk IDs no longer match the current chunks, for example after changing chunk size.

### Metrics

| Stage | Metric | What it tells you |
|-------|--------|-------------------|
| Retrieval | Recall@1, @3, @5, @10 | Whether the gold chunk appears in the top K results |
| Retrieval | MRR@10 | How high the gold chunk ranks on average |
| Generation | Hallucination (Opik) | Whether the answer makes claims the context does not support. Lower is better |
| Generation | Answer Relevance (Opik) | How well the answer addresses the question. Higher is better |

Retrieval is scored at two levels: **chunk** (the exact gold chunk is retrieved) and **source** (any chunk from the correct document is retrieved, which is easier).

The Opik judge is set explicitly to `gpt-4o-mini` at temperature 0. Opik's default judge was a reasoning model that took around 30 seconds per score and kept hitting the 60-second request timeout.

### Results

**Retrieval (30 queries)**

| Level  | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |
|--------|----------|----------|----------|-----------|--------|
| Chunk  | 0.367    | 0.467    | 0.667    | 0.667     | 0.447  |
| Source | 0.500    | 0.633    | 0.733    | 0.733     | 0.577  |

**Answer quality (30 queries, top_k = 3, temperature 0)**

| Group | Queries | Hallucination | Answer Relevance |
|-------|---------|---------------|------------------|
| All queries | 30 | 0.03 | 0.72 |
| Gold chunk in context | 14 | 0.00 | 0.99 |
| Gold chunk not in context | 16 | 0.06 | 0.48 |

### What the results show

- **Retrieval is the bottleneck, not generation.** When the gold chunk is in the context, answers score 0.99 relevance with no hallucination. All 9 answers with relevance below 0.7 are retrieval misses.
- **The gold chunk is missing from the top 3 for 16 of 30 queries.** Six of those rank 4th or 5th, so raising `top_k` from 3 to 5 would bring them into the context.
- **The other 10 are not in the top 10 at all.** Recall@5 and Recall@10 are the same, so a larger `top_k` alone will not fix these. They need better retrieval, such as re-ranking, hybrid search, or different chunking.
- **Hallucination stays low even on misses** (0.06). With the wrong context, the model mostly says it does not have enough information instead of making something up.

### Limitations

- 30 queries is a small set, so the numbers are directional.
- The questions are synthetic. Real support questions are likely to be vaguer and harder to retrieve for.
- The same model (`gpt-4o-mini`) generates the answers and judges them.
- Answer Relevance checks that the answer addresses the question, not that it is correct. A reference answer is saved for each query but is not used for scoring yet.
- These are baseline numbers. No before-and-after comparison has been run yet.

## Tech Stack

| Component        | Tool                                |
|-------------------|--------------------------------------|
| Dataset            | HuggingFace `datasets` (`m-ric/huggingface_doc`) |
| Embeddings         | `sentence-transformers` (BAAI/bge-small-en-v1.5) |
| Vector Database    | Milvus Lite (`pymilvus`)             |
| LLM (Generation)   | OpenAI `gpt-4o-mini`                 |
| Test Set Generation | OpenAI `gpt-4o-mini` (JSON mode)    |
| Evaluation         | Custom Recall@K and MRR, Opik (Hallucination, Answer Relevance) |
| Language           | Python 3, Jupyter Notebook           |

## Setup

### 1. Clone and install dependencies
```bash
git clone <your-repo-url>
cd <repo-name>
pip install "pymilvus[milvus_lite]" opik sentence-transformers datasets openai pandas numpy tqdm
```

On Google Colab, only `"pymilvus[milvus_lite]"` and `opik` need to be installed. The rest are already available.

### 2. Configure API keys
Create a `.env` file (never commit this) or export environment variables:
```bash
export HF_TOKEN="your_huggingface_token"
export OPIK_API_KEY="your_opik_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```

### 3. Run the notebook
Open `RAG_ASSIGNMENT.ipynb` in Jupyter/Colab and run cells top to bottom. This will:
- Load and chunk the HF documentation dataset
- Generate embeddings and populate the Milvus vector store (about 4 minutes on a Colab T4 GPU)
- Run sample queries end-to-end (retrieval → generation)
- Score 4 sample queries with Opik as a smoke test
- Build or load the 30-query labelled test set
- Report Recall@K and MRR, then hallucination and answer relevance, split by whether retrieval found the gold chunk

## Example Usage

```python
result = rag_query(
    query="How do I fine-tune a transformer model?",
    client=milvus_client,
    collection_name=COLLECTION_NAME,
    embedding_model=embedding_model,
    top_k=3
)

print(result["answer"])
```

Running the evaluation:

```python
retrieval_summary, retrieval_per_query = evaluate_retrieval(eval_set)
answer_df = evaluate_answers(eval_set, retrieval_per_query, top_k=3)
```

## Project Structure
```
.
├── RAG_ASSIGNMENT.ipynb   # End-to-end pipeline: ingestion → chunking → embedding →
│                          # vector store → retrieval → generation → evaluation
├── hf_docs_milvus.db      # Milvus Lite local vector database (generated on run)
├── eval_queries_v3.json   # Labelled test set: 30 queries with gold chunks (generated on run)
├── eval_results.csv       # Per-query answers and scores (generated on run)
└── README.md
```
