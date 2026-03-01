# Memory Management Guide

This document explains how to control, wipe, and ingest memory (both manually and automatically) for the Standalone Agentic Memory Tool.

## 1. How Auto-Ingestion (`/save`) Works
The tool features real-time continuous learning through the `/save` command.
1. As you chat, the proxy secretly logs both your messages and the Agent's replies to `active_chats/current_session.txt`.
2. When you type `/save` in your chat client, the proxy intercepts this message.
3. It takes `current_session.txt`, chunks the text, and calculates semantic embeddings.
4. It instantly appends these new chunks to `index/librarian_index.json` without needing to reboot the server.
5. The conversation trace is then moved to `archive/session_<timestamp>.txt` to ensure the buffer is clean for your next chat.

## 2. Emptying the Memory
If you want to completely wipe the Agent's memory (e.g., to switch datasets or clear mistakes):
1. **Stop the Proxy Server** (`Ctrl+C` in the terminal).
2. **Delete the Index File**: Navigate to `index/` and delete `librarian_index.json`.
3. **Delete Active Chats**: Navigate to `active_chats/` and delete `current_session.txt` (if it exists).
4. **Delete Archives (Optional)**: If you want to delete the backups of your saved chats, clear the `archive/` folder.
5. **Restart the Server**. It will detect the missing index and start with a fresh, empty 0-byte memory state.

## 3. Manual Bulk Ingestion
If you have a massive dataset or a new folder of documents you want the agent to learn all at once, you should write a script to bulk-ingest it rather than typing it in chat.

### Steps for Manual Ingestion:
1. Ensure the proxy server is **stopped** (so it doesn't try to write to the index at the same time).
2. Create a Python script using the identical `SentenceTransformer` logic found in `manager.py`.
3. Read your raw text, chunk it (see `TUNING_ACCURACY.md`), and compute the embeddings in **batches**.
4. Save the resulting dictionary mapped to chunk IDs (`{"chunk_0": {"vector": [...], "text": "..."}}`) to `index/librarian_index.json`.
5. Start the proxy server. It will automatically load the new index into RAM on boot.

### Test the Memory
To test if ingestion (auto or manual) worked, you do not need the UI. Simply use the provided test client:
```bash
python test_client.py "Use the search_database tool to look up: [YOUR KEYWORD]"
```
If the Librarian finds a match, the raw chunks will be passed directly to the Speaker LLM, which will synthesize the answer for you.
