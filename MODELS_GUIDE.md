# Models Guide: Architecture and Recommendations

The Standalone Agentic Memory Tool relies on a three-layered architecture to function. This document explains the role of each model, what embedding dimensions are, and provides recommendations for optimizing your setup.

## The Three-Layered Architecture

1. **The Speaker (LLM)**: This is the model running in your LM Studio instance (e.g., `Phi-4-mini`, `Qwen2.5-Coder`). It acts as the brain that talks to the user, formats the output, and decides *when* to trigger the `search_database` tool.
2. **The Thinker (RWKV)**: This is a fast, linear RNN model (default: `RWKV-6 1.6B`) running inside the proxy server. When the Speaker asks for facts, the Librarian grabs the text, but the Thinker summarizes and synthesizes those raw facts into a clean report before handing it back to the Speaker.
3. **The Librarian (Embedding Model)**: This is a `SentenceTransformer` model running inside the proxy. It converts text chunks into mathematical arrays called "vectors" or "embeddings." It handles mapping user queries to the most relevant facts in your `librarian_index.json`.

---

## Understanding Embedding Models

The embedding model determines how accurately the Librarian can match a user's question to the hidden facts in your database.

### What are Dimensions?
When an embedding model processes a chunk of text, it translates the semantic meaning of that text into a list of numbers. The length of this list is the **dimension**.
*   **384 Dimensions (e.g., MiniLM)**: The text is represented by 384 numbers. This is fast to compute, takes up very little RAM/disk space, but might lose some subtle linguistic nuances.
*   **768 Dimensions (e.g., MPNet)**: The text is represented by 768 numbers. It captures much more semantic detail and nuance, leading to more accurate search results, but requires more RAM and slightly more time to compute.
*   **1024+ Dimensions**: Used by massive models (like OpenAI's `text-embedding-3-large`). Extremely accurate but very heavy on local compute and storage.

> **Important**: If you switch your embedding model to a different dimension, **you MUST delete your `librarian_index.json` and re-ingest your entire database.** You cannot compare a 384d query vector against a 768d database index.

---

## Recommended Embedding Models

You can change your embedding model by updating the `embedding_model_path` in `config.json`. Here are the top recommendations for local Agentic RAG:

### 1. `all-mpnet-base-v2` (Current Default)
*   **Dimensions**: 768
*   **Pros**: Excellent balance of accuracy and size. It was trained on over 1 billion sentence pairs and acts as a fantastic general-purpose librarian.
*   **Cons**: Slightly slower than MiniLM for massive bulk ingestions.
*   **Best for**: General knowledge, dialogue, and high-accuracy retrieval.

### 2. `all-MiniLM-L6-v2`
*   **Dimensions**: 384
*   **Pros**: Extremely fast and lightweight. Requires less than 100MB of RAM to run. Great for prototyping or running on older hardware.
*   **Cons**: Lower accuracy for highly complex or nuanced queries compared to 768d models.
*   **Best for**: Speed, bulk processing, and low-end hardware.

### 3. `nomic-embed-text`
*   **Dimensions**: 768 (or flexible)
*   **Pros**: A modern open-source model designed specifically for long-context RAG. It handles up to 8192 context length out-of-the-box, meaning you can drastically increase your `chunk_size` in `config.json` without diluting the semantic meaning as much.
*   **Cons**: Requires downloading a new model and potentially configuring trust_remote_code.
*   **Best for**: Long documents and advanced RAG pipelines.

### 4. `bge-base-en-v1.5` (by BAAI)
*   **Dimensions**: 768
*   **Pros**: Consistently tops the MTEB (Massive Text Embedding Benchmark) leaderboards for its size. Extremely good at instruction-based retrieval formatting.
*   **Cons**: Highly sensitive to query phrasing.
*   **Best for**: Maximum accuracy competitive with paid APIs.

## How to Change the Model
1. Stop the proxy server.
2. Delete `c:/LAYERD-DB/agentic_memory_tool/index/librarian_index.json`.
3. Open `config.json` and change the `embedding_model_path` to your desired HuggingFace model repo (e.g., `"sentence-transformers/all-MiniLM-L6-v2"`).
4. Restart the server. It will automatically download the new model weights and start fresh.
