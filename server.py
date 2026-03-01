import os
import sys
import time
import json
import torch
import numpy as np
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from sentence_transformers import SentenceTransformer
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
from manager import log_message, save_memory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
INDEX_PATH = os.path.join(BASE_DIR, "index", "librarian_index.json")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

HOST = config["server"].get("host", "0.0.0.0")
PORT = config["server"].get("port", 8000)
LM_STUDIO_URL = config["server"]["lm_studio_url"]
EMBEDDING_MODEL = config["models"]["embedding_model_path"]
THINKER_PATH = config["models"]["thinker_model_path"]
TOP_K = config["retrieval"].get("top_k", 5)
THINKER_TOKENS = config["retrieval"].get("thinker_tokens", 400)

_orig_torch_load = torch.load
def _meddled_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _meddled_torch_load

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '0'

app = FastAPI(title="Standalone Agentic RAG Proxy")

router = None
thinker_pipeline = None
index_embeddings = None
index_keys = None
master_index = None

@app.on_event("startup")
async def startup_event():
    global router, thinker_pipeline, index_embeddings, index_keys, master_index
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n[!] Starting Standalone Agentic Proxy Server...")
    print(f"[*] Loading MPNet embedding model from {EMBEDDING_MODEL}...")
    router = SentenceTransformer(EMBEDDING_MODEL).to(device)
    
    print(f"[*] Loading Thinker from {THINKER_PATH}...")
    thinker = RWKV(model=THINKER_PATH, strategy='cuda fp16')
    thinker_pipeline = PIPELINE(thinker, "rwkv_vocab_v20230424")

    print(f"[*] Loading knowledge index from {INDEX_PATH}...")
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "r") as f:
            master_index = json.load(f)
        index_keys = list(master_index.keys())
        index_embeddings = np.array([master_index[k]["vector"] for k in index_keys]).astype('float32')
        print(f"[*] Index loaded with {len(index_keys)} facts.")
    else:
        master_index = {}
        index_keys = []
        index_embeddings = np.empty((0, 768), dtype='float32')
        print("[!] No index found. Starting with empty memory.")
        
    print(f"[SUCCESS] Server Ready on port {PORT}.\n")

def run_agentic_research(query: str):
    print(f"  [AGENT] Researching: '{query}'...")
    if len(index_keys) == 0:
        return "The database is currently empty."
        
    query_vec = router.encode([query], convert_to_numpy=True)
    sims = np.dot(index_embeddings, query_vec[0])
    
    k = min(TOP_K, len(index_keys))
    top_indices = np.argsort(sims)[-k:][::-1]
    
    context_text = ""
    for idx in top_indices:
        context_text += f"\n---\n{master_index[index_keys[idx]]['text']}"

    think_prompt = f"User: You are a technical Researcher. Read the following facts and summarize exactly what is relevant to this query: {query}\n\nFacts:\n{context_text}\n\nAssistant: Here is the factual report:\n"
    args = PIPELINE_ARGS(temperature=0.1, top_p=0.8, top_k=0, token_stop=[0])
    
    research_report = thinker_pipeline.generate(think_prompt, token_count=THINKER_TOKENS, args=args).strip()
    print(f"  [AGENT] Report Generated ({len(research_report.split())} words).")
    return research_report

async def proxy_stream(payload: dict):
    full_response = ""
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", LM_STUDIO_URL, json=payload, timeout=120) as response:
            async for chunk in response.aiter_bytes():
                try:
                    chunk_str = chunk.decode('utf-8')
                    lines = chunk_str.split('\n')
                    for line in lines:
                        if line.startswith('data: ') and line.strip() != 'data: [DONE]':
                            data = json.loads(line[6:])
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                full_response += delta.get("content", "")
                except Exception:
                    pass
                yield chunk
    
    if full_response:
        log_message("assistant", full_response)

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_database",
        "description": "Search the local database for factual information about specific people, events, or technical data. Use this when the user asks about facts that are likely in the local dataset.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The specific factual query to research in the database."
                }
            },
            "required": ["query"]
        }
    }
}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    messages = payload.get("messages", [])
    
    if not messages:
        return JSONResponse({"error": "No messages provided"}, status_code=400)

    last_user_msg = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            last_user_msg = msg["content"]
            break

    if last_user_msg.strip().lower() == "/save":
        global master_index, index_keys, index_embeddings
        master_index, status_msg = save_memory(router, master_index)
        index_keys = list(master_index.keys())
        if len(index_keys) > 0:
            index_embeddings = np.array([master_index[k]["vector"] for k in index_keys]).astype('float32')
        
        return JSONResponse({
            "id": "chatcmpl-save",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "memory-manager",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": status_msg
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        })

    if last_user_msg:
        log_message("user", last_user_msg)

    if "tools" not in payload:
        payload["tools"] = [SEARCH_TOOL]
        payload["tool_choice"] = "auto"

    system_instr = (
        "You are an expert Assistant with access to a massive local knowledge database.\n"
        "If you encounter a question about specific facts or events that you don't have perfect recall for, "
        "call the 'search_database' tool. Use the information in the tool response to build your final answer."
    )
    
    if messages[0]["role"] == "system":
        if "expert Assistant with access to a massive local knowledge database" not in messages[0]["content"]:
            messages[0]["content"] = f"{system_instr}\n\n{messages[0]['content']}"
    else:
        messages.insert(0, {"role": "system", "content": system_instr})

    print(f"  [SYSTEM] Forwarding request to LM Studio (Tools Enabled)...")

    async with httpx.AsyncClient() as client:
        lm_payload = payload.copy()
        lm_payload["stream"] = False 
        
        response = await client.post(LM_STUDIO_URL, json=lm_payload, timeout=120)
        res_json = response.json()

        if "choices" not in res_json:
            return JSONResponse(res_json, status_code=response.status_code)

        choice = res_json["choices"][0]
        message = choice.get("message", {})

        if message.get("tool_calls"):
            tool_calls = message["tool_calls"]
            print(f"  [LLM] Requested {len(tool_calls)} tool calls.")
            
            messages.append(message)
            
            for tool_call in tool_calls:
                if tool_call["function"]["name"] == "search_database":
                    args = json.loads(tool_call["function"]["arguments"])
                    query = args.get("query", "")
                    
                    research_report = run_agentic_research(query)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": "search_database",
                        "content": research_report
                    })

            print("  [SYSTEM] Proxying tool results -> LM Studio for final answer")
            
            if payload.get("stream", False):
                payload["messages"] = messages
                return StreamingResponse(proxy_stream(payload), media_type="text/event-stream")
            else:
                payload["messages"] = messages
                payload["stream"] = False
                final_res = await client.post(LM_STUDIO_URL, json=payload, timeout=120)
                final_json = final_res.json()
                if "choices" in final_json and len(final_json["choices"]) > 0:
                    content = final_json["choices"][0]["message"].get("content", "")
                    if content:
                        log_message("assistant", content)
                return JSONResponse(final_json, status_code=final_res.status_code)

        print("  [SYSTEM] No tool called. Returning direct response.")
        if payload.get("stream", False):
            return StreamingResponse(proxy_stream(payload), media_type="text/event-stream")
        else:
            if "choices" in res_json and len(res_json["choices"]) > 0:
                content = res_json["choices"][0]["message"].get("content", "")
                if content:
                    log_message("assistant", content)
            return JSONResponse(res_json, status_code=response.status_code)

if __name__ == "__main__":
    print(f"[i] Start LM Studio Server on {LM_STUDIO_URL} before using this proxy.")
    uvicorn.run(app, host=HOST, port=PORT)
