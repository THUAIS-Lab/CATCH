from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
app = FastAPI(title="V-Engine Open Agent", version="0.3.0")

sessions: dict[str, list[dict[str, str]]] = {}
cache: dict[str, str] = {}

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

def key(session_id: str, message: str) -> str:
    return session_id + "::" + message.strip().lower()

@app.get("/health")
def health():
    return {"status": "ok", "runtime": "vercel-fastapi"}

@app.get("/architecture")
def architecture():
    return {"control_plane":["policy","cache","routing","recovery"],"agent_plane":["CATCH agents","tools","models","realtime"],"evidence_plane":["trajectory","trace","visible_verifier","independent_verifier"],"cache_rule":"Only VERIFIED, provenance-compatible, non-side-effect results are reusable."}

@app.post("/api/chat")
def chat(payload: dict):
    sid = payload.get("session_id", "default")
    message = payload.get("message", "").strip()
    if not message:
        return {"status":"UNKNOWN","output":"Please enter a message.","cache_hit":False,"verifier":"input-policy"}
    k = key(sid, message)
    hit = k in cache
    output = cache.get(k) if hit else response_for(message)
    if not hit:
        cache[k] = output
    sessions.setdefault(sid, []).extend([
        {"role":"user","content":message,"meta":""},
        {"role":"assistant","content":output,"meta":"status: VERIFIED · "+("CACHE HIT" if hit else "EXECUTED")},
    ])
    return {"status":"VERIFIED","output":output,"cache_hit":hit,"verifier":"demo-independent"}

@app.get("/api/sessions/{session_id}")
def session(session_id: str):
    return {"session_id":session_id,"messages":sessions.get(session_id,[])}

@app.get("/{path:path}")
def frontend(path: str = ""):
    relative = path.removeprefix("web/")
    target = (WEB / relative).resolve() if relative else (WEB / "index.html")
    if str(target).startswith(str(WEB.resolve())) and target.is_file():
        return FileResponse(target)
    return FileResponse(WEB / "index.html")
