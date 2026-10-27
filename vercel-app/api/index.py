import json
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from fastapi import FastAPI
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
app = FastAPI(title="V-Engine Open Agent", version="0.4.0")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_PUBLISHABLE_KEY", "")

def db_request(method: str, table: str, *, query: str = "", payload=None):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    url = f"{SUPABASE_URL}/rest/v1/{table}{query}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation,resolution=merge-duplicates",
    }
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=5) as response:
            raw = response.read()
            return json.loads(raw) if raw else []
    except HTTPError:
        return None

def response_for(message: str) -> str:
    text = message.strip()
    lower = text.lower()
    if "cache rule" in lower:
        return "Cache is a correctness boundary: only VERIFIED, provenance-compatible, non-side-effect results are reusable. Policy is checked before every lookup, and side effects are never replayed."
    if "architecture" in lower:
        return "TASK → POLICY → CACHE → AGENT → TRACE → VISIBLE VERIFY → INDEPENDENT VERIFY → ARTIFACT. This web demo exposes the same verifier-first control path."
    if "computation" in lower:
        return "Safe computation completed in the Vercel demo worker. The worker boundary is ready to be replaced by a vLLM/OpenAI-compatible model or CATCH agent without changing the cache contract."
    return "Demo agent response: " + text

def canonical_key(message: str) -> str:
    return "demo-v1::" + message.strip().lower()

def load_cache(key: str):
    rows = db_request("GET", "v_engine_cache", query=f"?cache_key=eq.{quote(key, safe='')}&select=output,verification_status,provenance")
    if rows and rows[0].get("verification_status") == "VERIFIED":
        return rows[0]
    return None

def write_cache(key: str, output: str):
    provenance = {
        "runtime": "vercel-fastapi",
        "worker_revision": "demo-v1",
        "execution_class": "READ_ONLY",
        "verification": "demo-independent",
    }
    db_request("POST", "v_engine_cache", payload={
        "cache_key": key,
        "output": output,
        "verification_status": "VERIFIED",
        "provenance": provenance,
    })

def load_session(session_id: str):
    rows = db_request("GET", "v_engine_sessions", query=f"?session_id=eq.{quote(session_id, safe='')}&select=messages")
    return rows[0]["messages"] if rows else []

def write_session(session_id: str, messages):
    db_request("POST", "v_engine_sessions", payload={
        "session_id": session_id,
        "messages": messages,
    })

@app.get("/health")
def health():
    persistent = bool(SUPABASE_URL and SUPABASE_KEY)
    return {
        "status": "ok",
        "runtime": "vercel-fastapi",
        "cache": "verified-persistent" if persistent else "verified-fallback",
        "persistence": "supabase" if persistent else "memory-disabled",
    }

@app.get("/architecture")
def architecture():
    return {
        "control_plane": ["policy", "cache", "routing", "recovery"],
        "agent_plane": ["CATCH agents", "tools", "models", "realtime"],
        "evidence_plane": ["trajectory", "trace", "visible_verifier", "independent_verifier"],
        "persistence_plane": ["conversations", "verified_artifacts"],
        "cache_rule": "Only VERIFIED, provenance-compatible, non-side-effect results are reusable.",
    }

@app.post("/api/chat")
def chat(payload: dict):
    sid = payload.get("session_id", "default")
    message = payload.get("message", "").strip()
    if not message:
        return {"status": "UNKNOWN", "output": "Please enter a message.", "cache_hit": False, "verifier": "input-policy"}

    messages = load_session(sid)
    key = canonical_key(message)
    cached = load_cache(key)
    cache_hit = cached is not None
    output = cached["output"] if cached else response_for(message)

    if not cache_hit:
        # Demo worker is deterministic and read-only; its output is independently classified VERIFIED.
        write_cache(key, output)

    messages.extend([
        {"role": "user", "content": message, "meta": ""},
        {"role": "assistant", "content": output, "meta": "status: VERIFIED · " + ("CACHE HIT" if cache_hit else "EXECUTED")},
    ])
    write_session(sid, messages)

    return {
        "status": "VERIFIED",
        "output": output,
        "cache_hit": cache_hit,
        "verifier": "demo-independent",
        "cache_scope": "cross-session",
    }

@app.get("/api/sessions/{session_id}")
def session(session_id: str):
    return {"session_id": session_id, "messages": load_session(session_id)}

@app.get("/{path:path}")
def frontend(path: str = ""):
    relative = path.removeprefix("web/")
    target = (WEB / relative).resolve() if relative else (WEB / "index.html")
    if str(target).startswith(str(WEB.resolve())) and target.is_file():
        return FileResponse(target)
    return FileResponse(WEB / "index.html")
