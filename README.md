# Easy Agentic Memory System

The **Standalone Agentic Memory Tool** is a lightweight, drop-in middleware designed to give any standard Language Model (like Phi, Llama, or Qwen) permanent, searchable memory.

It works as an invisible proxy. You connect your chat interface (like OpenWebUI) to this tool, and it quietly funnels the conversation to your actual LLM (hosted on LM Studio, Ollama, etc.). 

While doing so, it provides two massive superpowers:

1. **On-Demand RAG (The `search_database` Tool)**: It gives the LLM access to a semantic vector database. If the LLM doesn't know a specific fact, it can automatically trigger the tool, search the database, summarize the findings using a hyper-fast RWKV neural network, and return the perfect answer to the user—all within seconds.
2. **Infinite Auto-Memory (The `/save` Command)**: Unlike standard chatbots that forget you the moment you clear the chat window, this tool intercepts the `/save` command to physically encode your current conversation into its permanent vector database. The agent actually learns, and will recall that context in any future conversation forever. 

In short: It turns a standard text generator into a highly capable, self-learning Research Agent.

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

The architecture relies on three distinct roles:
1. **The Speaker (LLM)**: The intelligence (e.g., `Phi-4-mini`) hosted in LM Studio.
2. **The Thinker (RWKV)**: A fast, linear RNN model (e.g., `RWKV-6 1.6B`) running inside the proxy. It summarizes the massive chunks of raw database text into a concise factual report *before* passing it to the Speaker.
3. **The Librarian (Embedding Model)**: A `SentenceTransformer` (e.g., `all-mpnet-base-v2`) running inside the proxy. It handles semantic clustering and search math.

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
*   **`top_k` (Default: 5)**: The number of chunks retrieved when a tool call happens. Increase this if the agent misses crucial context, but be aware this sends more data to the Thinker model to summarize.
*   **`thinker_tokens` (Default: 400)**: The total number of tokens the RWKV Thinker is allowed to generate when summarizing retrieved chunks. Increase if summaries are getting cut off mid-sentence.

---

## 6. Validation & Test Results

The system was vigorously validated against two distinct requirements: on-demand tool usage (saving latency), and successful dynamic append-memory logging.

### Test 1: On-Demand Efficiency
*   We queried the proxy with "Just say hello and tell me a joke."
*   **Result**: The LLM successfully ignored context retrieval, bypassed the RAG completely, and returned a joke natively in milliseconds. No database lookups were falsely triggered.

### Test 2: RAG Accuracy
*   We asked the proxy: "Where was Shane Horgan's father born?"
*   **Result**: The LLM realized it didn't know the answer, triggered `search_database`, and the RWKV pipeline retrieved the historical fact, resulting in an accurate response detailing the biography of John Horgan.

### Test 3: The `/save` Auto-Memory Injection
*   **Input**: The user mocked a conversation stating: "My favorite fruit is the ultra-rare Blue Watermelon."
*   **Command**: The user invoked `/save`.
*   **Result**: The server instantly converted the text into MPNet embeddings and permanently mutated the `librarian_index.json` without any server restarts. Direct semantic probing immediately after yielded the exact string as the #1 Top-K match with a high confidence score of 0.6402. The Agent had permanently learned the user's secret.
