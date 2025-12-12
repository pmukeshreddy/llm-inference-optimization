"""FastAPI production server"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import json
import time
from collections import deque
import threading

app = FastAPI(title="Speculative Decoding API", version="1.0.0")

# Global state
engine = None
metrics_history = deque(maxlen=1000)
lock = threading.Lock()


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(100, ge=1, le=2048)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    stream: bool = False
    use_speculative: bool = True


class GenerateResponse(BaseModel):
    text: str
    metrics: dict


def set_engine(eng):
    global engine
    engine = eng


@app.get("/health")
async def health():
    return {"status": "healthy" if engine else "not_initialized", "model_loaded": engine is not None}


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    with lock:
        if not metrics_history:
            return "# no data"
        recent = list(metrics_history)[-100:]
    
    avg_tps = sum(m["tokens_per_second"] for m in recent) / len(recent)
    avg_acceptance = sum(m.get("acceptance_rate", 0) for m in recent) / len(recent)
    
    return f"""# HELP speculative_tps Tokens per second
# TYPE speculative_tps gauge
speculative_tps {avg_tps:.2f}

# HELP speculative_acceptance Acceptance rate
# TYPE speculative_acceptance gauge
speculative_acceptance {avg_acceptance:.3f}

# HELP speculative_requests_total Total requests
# TYPE speculative_requests_total counter
speculative_requests_total {len(metrics_history)}
"""


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if not engine:
        raise HTTPException(503, "Model not loaded")
    
    engine.config.temperature = req.temperature
    engine.config.top_p = req.top_p
    
    if req.use_speculative:
        output = engine.generate(req.prompt, req.max_new_tokens)
    else:
        output = engine.generate_vanilla(req.prompt, req.max_new_tokens)
    
    with lock:
        metrics_history.append(output.metrics)
    
    return GenerateResponse(text=output.text, metrics=output.metrics)


@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    if not engine:
        raise HTTPException(503, "Model not loaded")
    
    async def stream():
        for chunk in engine.generate(req.prompt, req.max_new_tokens, stream=True):
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/benchmark")
async def benchmark(prompts: List[str], max_new_tokens: int = 50):
    if not engine:
        raise HTTPException(503, "Model not loaded")
    
    spec_results, vanilla_results = [], []
    
    for p in prompts:
        spec_results.append(engine.generate(p, max_new_tokens).metrics)
        vanilla_results.append(engine.generate_vanilla(p, max_new_tokens).metrics)
    
    spec_tps = sum(r["tokens_per_second"] for r in spec_results) / len(spec_results)
    vanilla_tps = sum(r["tokens_per_second"] for r in vanilla_results) / len(vanilla_results)
    
    return {
        "speculative_tps": round(spec_tps, 2),
        "vanilla_tps": round(vanilla_tps, 2),
        "speedup": round(spec_tps / vanilla_tps, 2),
        "avg_acceptance": round(sum(r["acceptance_rate"] for r in spec_results) / len(spec_results), 3)
    }
