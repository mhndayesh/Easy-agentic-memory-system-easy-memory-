# Tuning Accuracy Guide

This document explains how to tweak the settings in `config.json` to improve accuracy, speed, and context size.

## 1. Chunk Size and Overlap
When you run `/save` or manually ingest data, the text is split into chunks before being embedded into the semantic index.

### `chunk_size`
*   **Default**: 250 words
*   **What it does**: This determines the maximum length of a semantic "thought" stored in the database.
*   **Tuning**:
    *   **Increase** (e.g., 500) if you notice that the database is struggling to answer complex questions that require long, continuous context to understand. *Warning*: This dilutes the embedding vector, causing specific keywords to get "lost" in the noise.
    *   **Decrease** (e.g., 100) for extremely dense, factual data (like a dictionary or FAQ). It will yield perfectly pinpointed retrieval but will lack context around the facts.

### `overlap`
*   **Default**: 50 words
*   **What it does**: Prevents sentences from being split in half at the boundary of a `chunk_size`. The end of Chunk 1 repeats the first 50 words of Chunk 2.
*   **Tuning**: Keep this at ~20% of your total `chunk_size`.

## 2. Retrieval Parameters

### `top_k`
*   **Default**: 5
*   **What it does**: The number of chunks the Librarian retrieves when the `search_database` tool is called. These raw chunks are passed directly to the Speaker (your main LLM) as the tool result.
*   **Tuning**:
    *   **Increase**: If the agent is missing crucial context that you know exists in the dataset. Note that passing more context chunks takes up more of the Speaker LLM's context window.
    *   **Decrease**: To keep prompts lean and save context window space.

## 3. Batch Size (Manual Ingestion Only)
If you are writing a custom ingestion script, you should batch your tensor inputs.
*   **What it does**: The number of chunks processed through the MPNet model at the exact same time on the GPU.
*   **Tuning**: Start at 8 or 16. Increase until you run out of VRAM for maximum ingestion speed. Has no impact on accuracy, only ingestion runtime. (Note: `/save` handles batching automatically for standard chat logs).
