from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from .cache import VerifiedCache
from .models import ExecutionClass, Result, Task, VerificationStatus
from .policy import Policy
from .runtime import AgentRuntime

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
app = FastAPI(title="Open Agent App", version="0.2.0")
runtime = AgentRuntime(cache=VerifiedCache(), policy=Policy())
sessions: dict[str, list[dict[str, str]]] = {}

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/architecture")
def architecture() -> dict:
    return {"control_plane":["policy","cache","routing","recovery"],"agent_plane":["CATCH agents","tools","models","realtime"],"evidence_plane":["trajectory","trace","visible_verifier","independent_verifier"],"cache_rule":"Only VERIFIED, provenance-compatible, non-side-effect results are reusable."}

@app.get("/web/{path:path}")
def web(path: str):
    target = (WEB / path).resolve()
    if not str(target).startswith(str(WEB.resolve())) or not target.is_file():
        target = WEB / "index.html"
    return FileResponse(target)

class DemoVerifier:
    name = "demo-reference"
    def verify(self, task, output):
        return Result(status=VerificationStatus.VERIFIED, output=output, metadata={"verifier": self.name})

def demo_worker(task: Task):
    text = str(task.input).strip()
    if "cache rule" in text.lower():
        return "Cache is a correctness boundary: only independently VERIFIED, provenance-compatible results are reusable. Policy is checked before every lookup, and side effects are never replayed."
    if "architecture" in text.lower():
        return "TASK → POLICY → CACHE → CATCH AGENT → TRACE → VISIBLE VERIFY → INDEPENDENT VERIFY → ARTIFACT. The web UI and voice input use this same control path."
    if "computation" in text.lower():
        return "Safe computation completed in the local demo worker. Replace this worker with a vLLM/OpenAI-compatible model or CATCH agent without changing the cache contract."
    return "Demo agent response: " + text

@app.post("/api/chat")
def chat(payload: dict) -> dict:
    sid = payload.get("session_id","default")
    message = payload.get("message","").strip()
    task = Task(task_id=sid+":"+str(len(sessions.get(sid,[]))), input=message, model="demo-local", model_revision="demo-1", code_revision="phase2", tool_revision="v1")
    result = runtime.execute(task, demo_worker, DemoVerifier())
    cache_hit = bool(result.metadata.get("cache_hit"))
    meta = "status: "+result.status.value+" · "+("CACHE HIT" if cache_hit else "EXECUTED")
    sessions.setdefault(sid, []).extend([{"role":"user","content":message,"meta":""},{"role":"assistant","content":str(result.output),"meta":meta}])
    return {"status":result.status.value,"output":result.output,"cache_hit":cache_hit,"verifier":result.metadata.get("verifier","independent")}

@app.get("/api/sessions/{session_id}")
def session(session_id: str) -> dict:
    return {"session_id":session_id,"messages":sessions.get(session_id,[])}
