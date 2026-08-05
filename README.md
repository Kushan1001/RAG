# HF-Docs RAG: Production-Grade Knowledge Retrieval System

A Retrieval-Augmented Generation (RAG) pipeline that turns fragmented HuggingFace documentation into an accurate, context-aware Q&A system — built to cut down the manual doc-searching that eats into support engineers' day.

## Problem Statement

XYZ AI Solutions manages an extensive ecosystem of developer tools. Internal support engineers and community managers currently spend **over 40% of their day** navigating fragmented documentation to answer technical questions about HuggingFace-based workflows. Keyword search fails to capture the nuance of complex ML queries, leading to high ticket latency and inconsistent developer advice.

**Goal:** Build a robust, end-to-end RAG pipeline where success is measured by the *reliability and relevance* of retrieved information, not just code execution — ingesting documentation, storing it in a high-performance vector database, and answering complex queries with an LLM grounded in retrieved context.

## How It Works

```
HuggingFace Docs Dataset
        │
        ▼
  Chunking (1000 chars, 200 overlap)
        │
        ▼
  Embedding (BAAI/bge-small-en-v1.5, 384-dim)
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
  Evaluation (Opik: Hallucination + Answer Relevance)
```

## Pipeline Stages

1. **Data Ingestion** — Loads the [`m-ric/huggingface_doc`](https://huggingface.co/datasets/m-ric/huggingface_doc) dataset from HuggingFace, containing HF documentation text paired with source metadata.

2. **Chunking** — Custom sliding-window chunker splits documents into 1000-character chunks with 200-character overlap, preserving context across chunk boundaries.

3. **Embeddings** — Generates dense vector representations using `BAAI/bge-small-en-v1.5` (Sentence-Transformers, 384 dimensions), batched for efficiency.

4. **Vector Store** — Indexes all chunk embeddings in **Milvus Lite** (via `pymilvus`) using Inner Product (IP) similarity for fast approximate nearest-neighbor search.

5. **Retrieval** — Given a query, embeds it and retrieves the top-K most semantically similar chunks with their source attribution.

6. **Generation** — Feeds retrieved context into a grounded prompt template and generates an answer with `gpt-4o-mini` via the OpenAI API. The prompt explicitly instructs the model to say when context is insufficient, reducing hallucination.

7. **Evaluation** — Scores each generated answer using [Opik](https://www.comet.com/site/products/opik/) metrics:
   - **Hallucination** — flags claims not supported by retrieved context
   - **Answer Relevance** — measures how well the answer addresses the query

## Tech Stack

| Component        | Tool                                |
|-------------------|--------------------------------------|
| Dataset            | HuggingFace `datasets` (`m-ric/huggingface_doc`) |
| Embeddings         | `sentence-transformers` (BAAI/bge-small-en-v1.5) |
| Vector Database    | Milvus Lite (`pymilvus`)             |
| LLM (Generation)   | OpenAI `gpt-4o-mini`                 |
| Evaluation         | Opik (Hallucination, Answer Relevance) |
| Language           | Python 3, Jupyter Notebook           |

## Setup

### 1. Clone and install dependencies
```bash
git clone <your-repo-url>
cd <repo-name>
pip install pymilvus opik "pymilvus[milvus_lite]" sentence-transformers datasets openai accelerate torch pandas tqdm
```

### 2. Configure API keys
Create a `.env` file (never commit this) or export environment variables:
```bash
export HF_TOKEN="your_huggingface_token"
export OPIK_API_KEY="your_opik_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```
> ⚠️ Do **not** hardcode API keys in the notebook. Load them via `os.environ.get("OPENAI_API_KEY")` or `python-dotenv`.

### 3. Run the notebook
Open `RAG_ASSIGNMENT.ipynb` in Jupyter/Colab and run cells top to bottom. This will:
- Load and chunk the HF documentation dataset
- Generate embeddings and populate the Milvus vector store
- Run sample queries end-to-end (retrieval → generation)
- Evaluate answers with hallucination and relevance metrics

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

## Project Structure
```
.
├── RAG_ASSIGNMENT.ipynb   # End-to-end pipeline: ingestion → chunking → embedding →
│                          # vector store → retrieval → generation → evaluation
├── hf_docs_milvus.db      # Milvus Lite local vector database (generated on run)
└── README.md
```

## Evaluation Metrics

Each answer is scored on:
- **Hallucination score** — lower is better; flags ungrounded claims
- **Answer relevance score** — higher is better; measures query-answer alignment


## Future Improvements
- Swap Milvus Lite for a hosted Milvus/Zilliz cluster for production scale
- Add re-ranking (e.g., cross-encoder) after initial retrieval
- Cache embeddings to avoid recomputation on re-runs
- Add a lightweight API/UI layer (FastAPI + Streamlit) for support engineers
- Track evaluation metrics over time to catch retrieval drift
