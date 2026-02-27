# Tweet Writer RAG

A Retrieval-Augmented Generation (RAG) system that searches through historical tweet data to find contextually similar tweets using semantic embeddings and a FAISS vector index.

## Overview

This project reads tweet activity metrics exported from Twitter/X, generates semantic embeddings for each tweet, and lets you query the dataset with natural language. The system returns the most relevant tweets for a given query, and in its most advanced variant also generates an AI-written answer using an LLM.

Three implementations are provided, each building on the previous one:

| File | Embedding Model | Vector Store | LLM |
|------|----------------|--------------|-----|
| `main.py` | Universal Sentence Encoder (TF Hub) | FAISS (per-file) | — |
| `main1.py` | Universal Sentence Encoder (TF Hub) | FAISS (combined) | — |
| `main2.py` | OpenAI Embeddings | FAISS + Redis cache | OpenAI GPT |

## Project Structure

```
tweet-writer-rag/
├── main.py          # Basic USE + FAISS implementation (per-file indexes)
├── main1.py         # Improved USE + FAISS implementation (combined index)
├── main2.py         # LangChain + OpenAI Embeddings + Redis + GPT
└── CSV_files/       # Tweet activity metric exports (CSV)
```

## Prerequisites

- Python 3.8+
- CSV files exported from Twitter/X Analytics placed in the `CSV_files/` directory

The CSV files must contain either a `tweet_text` or `Tweet text` column.

## Installation

Install the required dependencies for the implementation you want to run.

### `main.py` / `main1.py` (TensorFlow + FAISS)

```bash
pip install tensorflow tensorflow-hub pandas numpy faiss-cpu scipy
```

### `main2.py` (LangChain + OpenAI + Redis)

```bash
pip install langchain langchain-community openai faiss-cpu pandas redis dill
```

## Usage

### Basic / Improved implementation (`main.py` or `main1.py`)

```bash
python main.py
# or
python main1.py
```

On the first run the Universal Sentence Encoder is downloaded from TensorFlow Hub and embeddings are generated for every tweet. The embeddings are cached as `.pkl` pickle files next to the script so subsequent runs are faster.

When prompted, enter a natural-language query:

```
Please enter your question: What tweets are about climate change?
```

The five most semantically similar tweets are printed to the console.

### Advanced implementation (`main2.py`)

> **Note:** `main2.py` requires a valid OpenAI API key and a Redis instance. Set the credentials before running:

```bash
export OPENAI_API_KEY="<your-openai-api-key>"
```

Update the Redis connection parameters in `main2.py` to point at your own instance, then run:

```bash
python main2.py
```

Embeddings are cached in Redis so they are only generated once per CSV file. After retrieving the most similar tweets the LLM produces a synthesised answer to your query.

## How It Works

1. **Data loading** – All CSV files in `CSV_files/` are read and normalised to a common `tweet_text` column.
2. **Embedding generation** – Each tweet is converted into a dense vector representation using either Google's Universal Sentence Encoder or OpenAI Embeddings.
3. **Indexing** – Embeddings are stored in a FAISS `IndexFlatL2` index for fast nearest-neighbour search.
4. **Caching** – Computed embeddings are persisted (pickle files for `main.py`/`main1.py`, Redis for `main2.py`) so they are not regenerated on every run.
5. **Retrieval** – The user's query is embedded with the same model and the five nearest neighbours are retrieved from the index.
6. **Generation** (`main2.py` only) – The retrieved tweets are passed as context to an OpenAI LLM which generates a coherent answer.

## Dependencies

| Package | Purpose |
|---------|---------|
| `tensorflow` | TensorFlow runtime |
| `tensorflow-hub` | Universal Sentence Encoder |
| `faiss-cpu` | Vector similarity search |
| `pandas` | CSV loading and manipulation |
| `numpy` | Numerical operations |
| `scipy` | Spatial distance utilities |
| `langchain` / `langchain-community` | LLM orchestration (`main2.py`) |
| `openai` | OpenAI API client (`main2.py`) |
| `redis` | Embedding cache (`main2.py`) |
| `dill` | Extended serialisation for Redis (`main2.py`) |
