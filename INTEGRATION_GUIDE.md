# Third-Party Integration Guide

This Standalone Agentic Memory Tool is built natively on the **OpenAI API Standard**. Because of this, it is incredibly easy to hook up into any third-party UI, Agent framework, or orchestration system (like OpenWebUI, AnythingLLM, LangChain, CrewAI, AutoGPT, or custom frontends).

This guide explains how to connect it.

---

## 1. The "Base URL" Drop-In Method (Easiest)
Because this server acts as a reverse proxy, you can usually just drop it into any app that asks for an "OpenAI Base URL".

Instead of pointing your UI exactly to LM Studio (e.g., `http://localhost:1234/v1`), point it to this tool's server instead:

- **New Base URL**: `http://localhost:8000/v1`
- **API Key**: `lm-studio` (or anything, it's ignored locally)
- **Model Name**: Use whatever model you have loaded in LM Studio.

The Proxy will automatically:
1. Hijack the `/v1/chat/completions` endpoint.
2. Inject the `search_database` tool definition into your prompts.
3. Automatically intercept the LLM if it tries to search for a fact, run the Librarian pipeline, and seamlessly return the raw factual chunks back to your LLM to synthesize.

*Note: Your UI must support OpenAI function calling/tools for this auto-injection to work correctly.*

---

## 2. The "Custom Tool API" Method
If you are using a more advanced orchestrator like LangChain, CrewAI, or an agent framework that allows you to define custom Python tools or HTTP tools, you can expose the internal `run_agentic_research` function directly.

### Step 1: Create a tiny API endpoint in `server.py`
Open `server.py` and add a direct route near the top:
```python
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str

@app.post("/v1/memory/search")
async def memory_search(req: QueryRequest):
    report = run_agentic_research(req.query)
    return {"result": report}
```

### Step 2: Define it in your Framework
In LangChain, CrewAI, or AutoGPT, define a custom tool that sends an HTTP POST request to `http://localhost:8000/v1/memory/search`.

**Example Python Tool Definition:**
```python
import requests

def agentic_memory_search(query: str) -> str:
    """Search the local semantic database for factual information."""
    res = requests.post("http://localhost:8000/v1/memory/search", json={"query": query})
    return res.json().get("result", "Search failed.")
```
Now, your custom agent can actively choose to call `agentic_memory_search` whenever it needs to recall a fact from your semantic database.

---

## 3. Integrating the `/save` Feature Externally
By default, the `/save` feature works by intercepting chat messages.

If you don't want to use the `/save` chat command, and instead want your external app to trigger memory saves programmatically (for example, reading a PDF and forcing the agent to remember it):

You can expose the underlying `manager.py` save logic via a new endpoint in `server.py`:
```python
from pydantic import BaseModel
from manager import save_memory

class SaveRequest(BaseModel):
    text: str

@app.post("/v1/memory/inject")
async def inject_memory(req: SaveRequest):
    # Temporarily write to the buffer file, then trigger the manager
    with open("active_chats/current_session.txt", "w", encoding="utf-8") as f:
        f.write(req.text)
    
    global master_index
    master_index, status = save_memory(router, master_index)
    
    return {"status": status}
```

Now, any external application can instantly push raw text, PDFs, or scrape logs directly into the Agent's permanent memory simply by POSTing the text to `http://localhost:8000/v1/memory/inject`.
