import os
import time
import json
import shutil
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIVE_DIR = os.path.join(BASE_DIR, "active_chats")
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")
INDEX_PATH = os.path.join(BASE_DIR, "index", "librarian_index.json")
CURRENT_CHAT_FILE = os.path.join(ACTIVE_DIR, "current_session.txt")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

os.makedirs(ACTIVE_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

CHUNK_SIZE = config.get("chunking", {}).get("chunk_size", 250)
OVERLAP = config.get("chunking", {}).get("overlap", 50)

def log_message(role, content):
    """Appends a new message to the active chat session."""
    with open(CURRENT_CHAT_FILE, "a", encoding="utf-8") as f:
        f.write(f"{role.upper()}:\n{content}\n\n")

def save_memory(router, master_index):
    """
    Reads the active chat, chunks it, embeds it with the provided router,
    appends it to the master_index, saves it to index/librarian_index.json, and archives the chat.
    Returns the updated master_index and a status message.
    """
    if not os.path.exists(CURRENT_CHAT_FILE):
        return master_index, "No active memory to save."
        
    with open(CURRENT_CHAT_FILE, "r", encoding="utf-8") as f:
        content = f.read().strip()
        
    if not content:
        return master_index, "Active memory is empty."
        
    words = content.split()
    stride = CHUNK_SIZE - OVERLAP
    if stride <= 0:
        stride = CHUNK_SIZE
    chunks = [" ".join(words[i:i + CHUNK_SIZE]) for i in range(0, len(words), stride)]
    
    if not chunks:
        chunks = [content]
        
    with torch.no_grad():
        embeddings = router.encode(chunks, convert_to_numpy=True)
        
    existing_nums = []
    for k in master_index.keys():
        if k.startswith("chunk_"):
            try:
                existing_nums.append(int(k.split("_")[1]))
            except ValueError:
                pass
    
    next_id = max(existing_nums) + 1 if existing_nums else 0
    
    for j, text in enumerate(chunks):
        chunk_id = f"chunk_{next_id + j}"
        master_index[chunk_id] = {
            "vector": embeddings[j].tolist(),
            "text": text
        }
        
    print(f"  [MEMORY MGR] Saving {len(chunks)} new memory chunks to {INDEX_PATH}...")
    with open(INDEX_PATH, "w") as f:
        json.dump(master_index, f)
        
    timestamp = int(time.time())
    archive_file = os.path.join(ARCHIVE_DIR, f"session_{timestamp}.txt")
    shutil.move(CURRENT_CHAT_FILE, archive_file)
    
    msg = f"Memory saved successfully! Archiving session as {os.path.basename(archive_file)}"
    print(f"  [MEMORY MGR] {msg}")
    return master_index, msg
