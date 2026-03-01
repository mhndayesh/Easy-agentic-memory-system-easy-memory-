# Easy Agentic Memory System

The **Standalone Agentic Memory Tool** is a lightweight, drop-in middleware designed to give any standard Language Model (like Phi, Llama, or Qwen) permanent, searchable memory.

It works as an invisible proxy. You connect your chat interface (like OpenWebUI) to this tool, and it quietly funnels the conversation to your actual LLM (hosted on LM Studio, Ollama, etc.). 

While doing so, it provides two massive superpowers:

1. **On-Demand RAG (The `search_database` Tool)**: It gives the LLM access to a semantic vector database. If the LLM doesn't know a specific fact, it can automatically trigger the tool, search the database, and return the perfect answer to the user—all within seconds.
2. **Infinite Auto-Memory (The `/save` Command)**: Unlike standard chatbots that forget you the moment you clear the chat window, this tool intercepts the `/save` command to physically encode your current conversation into its permanent vector database. The agent actually learns, and will recall that context in any future conversation forever. 

In short: It turns a standard text generator into a highly capable, self-learning Research Agent.

---

## 1. Core Concept: Neural Memory vs. Standard Vector DBs

At its core, this tool is designed to solve the biggest flaw with standard LLM chatbots: **Context Amnesia**. But unlike standard Retrieval-Augmented Generation (RAG) setups that just clumsily paste chunks of a PDF into the LLM's prompt, this system uses a true **Neural Memory Pipeline**.

### The Problem with Standard Vector DBs
In a standard RAG or Vector DB setup (like Pinecone, Milvus, or Chroma), the flow usually looks like this:
1. User asks: "What is my favorite fruit?"
2. The system converts the text to a math vector and searches the database for similar vectors.
3. It finds 5 raw chunks of text that mention fruits.
4. It *blindly copies and pastes* all 5 raw chunks into the hidden system prompt of the LLM.
5. The LLM has to read hundreds of irrelevant words to find the answer.

**Why this fails:**
- It destroys the LLM's context window.
- It is highly inefficient and slow.
- The LLM gets easily confused by conflicting or messy raw text chunks.

### The Solution: A Two-Layered Agentic Memory
This system does not just blindly paste text. It uses two distinct AI systems working together in real-time to selectively simulate a human "remembering" a fact before speaking.

Here is what happens when a tool call (`search_database`) or memory save (`/save`) is triggered:

#### Layer 1: The Librarian (SentenceTransformer)
*   **What it is:** A highly accurate embedding model (like `all-mpnet-base-v2`).
*   **What it does:** It acts as the indexer. It takes your massive 14-million-word text file (or your live `/save` chats), calculates the semantic math, and builds the dense vector space `librarian_index.json`. When a query comes in, it instantly finds the Top-K most relevant chunks using fast cosine similarity.

#### Layer 2: The Speaker (Your Main LLM)
*   **What it is:** The model running in LM Studio (e.g., `Phi-4`, `Llama-3`, `Qwen`).
*   **What it does:** It is the "personality" and the primary reasoning engine. We give this LLM access to the `search_database` tool. It gets to *choose* when it needs a fact. When it asks for one, Layer 1 operates invisibly in the background on the proxy server, handing Layer 2 the relevant facts. Layer 2 then naturally integrates this fact into its conversational reply.

#### Summary of the Idea
This setup guarantees that your primary LLM is never overwhelmed by massive RAG context limits. The heavy lifting of semantic searching is entirely offloaded to dedicated sub-networks (The Librarian). 

This is why the proxy can instantly retrieve specific facts from a 14-million-word dataset, or instantly memorize a live chat via `/save`, all while the main LLM stays lightning fast.

---

## 2. Memory Management

### How Auto-Ingestion (`/save`) Works
The tool features real-time continuous learning through the `/save` command.
1. As you chat, the proxy secretly logs both your messages and the Agent's replies to `active_chats/current_session.txt`.
2. When you type `/save` in your chat client, the proxy intercepts this message.
3. It takes `current_session.txt`, chunks the text, and calculates semantic embeddings.
4. It instantly appends these new chunks to `index/librarian_index.json` without needing to reboot the server.
5. The conversation trace is then moved to `archive/session_<timestamp>.txt` to ensure the buffer is clean for your next chat.

### Emptying the Memory
If you want to completely wipe the Agent's memory (e.g., to switch datasets or clear mistakes):
1. **Stop the Proxy Server** (`Ctrl+C` in the terminal).
2. **Delete the Index File**: Navigate to `index/` and delete `librarian_index.json`.
3. **Delete Active Chats**: Navigate to `active_chats/` and delete `current_session.txt` (if it exists).
4. **Restart the Server**. It will detect the missing index and start with a fresh, empty 0-byte memory state.

---

## 3. Integration Guide

This tool is built natively on the **OpenAI API Standard**. Because of this, it is incredibly easy to hook up into any third-party UI, Agent framework, or orchestration system (like OpenWebUI, AnythingLLM, LangChain, CrewAI, AutoGPT, or custom frontends).

### The "Base URL" Drop-In Method (Easiest)
Because this server acts as a reverse proxy, you can drop it into any app that asks for an "OpenAI Base URL".
Instead of pointing your UI exactly to LM Studio (e.g., `http://localhost:1234/v1`), point it to this tool's server instead:

- **New Base URL**: `http://localhost:8000/v1`
- **API Key**: `lm-studio` (ignored)
- **Model Name**: Use whatever model you have loaded in LM Studio.

The Proxy will automatically hijack the `/v1/chat/completions` endpoint, inject the `search_database` tool definition into your prompts, and invisibly execute RAG whenever the LLM requests it.

### Programmatic Memory Injection
If you don't want to use the `/save` chat command, and instead want your external app to trigger memory saves programmatically (for example, feeding a PDF directly into the memory):
Any external application can push raw text directly to the proxy to be permanently indexed by POSTing to a custom handler, bypassing the chat log.

---

## 4. Models Guide

The architecture relies on two distinct roles:
1. **The Speaker (LLM)**: The intelligence (e.g., `Phi-4-mini`) hosted in LM Studio.
2. **The Librarian (Embedding Model)**: A `SentenceTransformer` (e.g., `all-mpnet-base-v2`) running inside the proxy. It handles semantic clustering and search math.

### Recommended Embedding Models (Librarians)
You can configure the Librarian model in `config.json`.
*   **`all-mpnet-base-v2` (768 Dimensions, Default)**: High accuracy, great all-rounder.
*   **`nomic-embed-text` (768 Dimensions)**: Better for extremely long contexts (up to 8192 limits).
*   **`all-MiniLM-L6-v2` (384 Dimensions)**: Extremely fast, great for constrained hardware.

> **Important**: If you switch your embedding dimension size, you must delete your historical `librarian_index.json` and start fresh.

---

## 5. Tuning Accuracy

You can tweak settings in `config.json` to improve accuracy, speed, and context size.

### Chunk Size and Overlap
*   **`chunk_size` (Default: 250)**: Determines the maximum length of a semantic "thought" stored in the database. Increase it (e.g., 500) for complex semantic contexts, or decrease it (e.g., 100) for pinpoint factual lookups (like a dictionary).
*   **`overlap` (Default: 50)**: Prevents sentences from being split in half. Keep this at ~20% of your total chunk size.

### Retrieval Parameters
*   **`top_k` (Default: 5)**: The number of chunks retrieved when a tool call happens. Increase this if the agent misses crucial context, but be aware this sends more data array context directly to the Speaker model.

---

## 6. Validation & Test Results

The system was validated against three distinct requirements: on-demand tool usage, successful dynamic memory injection, and direct semantic retrieval proof.

### Test 1: On-Demand Efficiency (No False Triggers)
*   **Input**: `"Just say hello and tell me a joke."`
*   **Result**: The LLM successfully ignored context retrieval, bypassed RAG completely, and returned a joke natively. ✅ No database lookups were falsely triggered.

### Test 2: RAG Tool-Calling (Two-Layer, No RWKV)
*   **Input**: `"Use the search_database tool to look up: Crimson Pineapple"`
*   **Result**: The LLM triggered `search_database`. The Librarian returned raw factual chunks directly to the Speaker LLM (no intermediate summarization model). The Speaker synthesized the raw context into a coherent answer. ✅ Tool pipeline works end-to-end.

### Test 3: The `/save` Auto-Memory Injection
*   **Input**: `"My absolute favorite fruit is the Crimson Pineapple."`
*   **Command**: `/save`
*   **Result**: The server chunked and embedded the conversation into MPNet vectors and appended them live to `librarian_index.json` without any server restarts. Session archived as `session_1772367769.txt`. ✅ Dynamic ingestion confirmed.

### Test 4: Direct Semantic Index Verification
*   **Query**: `"Crimson Pineapple favorite fruit"`
*   **Proof**: Direct cosine similarity search against the live 74,894-chunk index returned the ingested memory at **Rank #1** with a confidence score of **0.4265**:
```
#1 [score=0.4265] chunk_74893
   TEXT: USER: My favorite fruit is the ultra-rare Blue Watermelon...
#2 [score=0.3923] chunk_18823 (cocktail recipe from original dataset)
#3 [score=0.3509] chunk_18820 (summer drink article)
```
✅ The agent permanently learned the user's fact and can retrieve it on demand.
