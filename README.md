# Speculative Decoding Engine

Production LLM inference with 1.5-2.5x speedup.

## Quick Start

### Colab
```bash
!pip install torch transformers accelerate bitsandbytes fastapi uvicorn pydantic
%run run_colab.py
```

### Local
```bash
pip install -r requirements.txt

# Benchmark
python main.py benchmark --preset small

# API Server
python main.py serve --preset small --port 8000

# Interactive
python main.py chat --preset small

# Single generation
python main.py generate --preset small --prompt "Hello world"
```

### Docker
```bash
docker-compose up
# API: http://localhost:8000
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |
| `/generate` | POST | Generate text |
| `/generate/stream` | POST | Streaming generation |
| `/benchmark` | POST | Run benchmark |

## Presets

| Preset | Draft | Target | VRAM |
|--------|-------|--------|------|
| tiny | TinyLlama-1.1B | TinyLlama-1.1B | ~3GB |
| small | Qwen2-0.5B | Qwen2-1.5B | ~4GB |
| medium | Qwen2-0.5B | Qwen2-7B | ~8GB |
| llama | Llama-3.2-1B | Llama-3.2-3B | ~6GB |

## Architecture

```
Draft Model (small, fast)
    │
    ▼ Generate K tokens
    │
Target Model (large, accurate)
    │
    ▼ Verify all K in ONE forward pass
    │
Accept/Reject (probability ratio)
    │
    ▼ Adaptive K adjustment
```

## Metrics

- **TPS**: Tokens per second
- **TTFT**: Time to first token
- **Acceptance Rate**: % draft tokens accepted (target >70%)
- **Speedup**: Tokens per target forward pass
