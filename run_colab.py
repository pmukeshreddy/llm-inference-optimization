#!/usr/bin/env python3
"""
Run this in Google Colab:
    !pip install -q torch transformers accelerate bitsandbytes fastapi uvicorn pydantic
    %run run_colab.py
"""

import sys
sys.path.insert(0, '.')

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

from speculative_decoding import SpeculativeDecodingEngine, SpeculativeConfig, load_preset

# Config
PRESET = "small"  # tiny/small/medium/llama
HF_TOKEN = None   # Set for gated models

print(f"\nLoading preset: {PRESET}")
draft, target, tokenizer = load_preset(PRESET, hf_token=HF_TOKEN, target_quantization="4bit")

config = SpeculativeConfig(max_speculation_length=5, adaptive_k=True, temperature=0.7)
engine = SpeculativeDecodingEngine(draft, target, tokenizer, config)
print("Ready!\n")

# Test prompts
prompts = [
    "Explain machine learning in one sentence:",
    "What is the capital of France?",
    "Write a haiku about programming:",
]

print("=" * 60)
print("SPECULATIVE vs VANILLA COMPARISON")
print("=" * 60)

for prompt in prompts:
    print(f"\nPrompt: {prompt}")
    
    # Speculative
    spec = engine.generate(prompt, max_new_tokens=50)
    print(f"[SPEC] {spec.metrics['tokens_per_second']:.1f} tok/s | Accept: {spec.metrics['acceptance_rate']:.0%}")
    print(f"       {spec.text[:100]}...")
    
    # Vanilla
    van = engine.generate_vanilla(prompt, max_new_tokens=50)
    print(f"[VAN]  {van.metrics['tokens_per_second']:.1f} tok/s")
    
    speedup = spec.metrics['tokens_per_second'] / van.metrics['tokens_per_second']
    print(f"       Speedup: {speedup:.2f}x")

print("\n" + "=" * 60)
print("BENCHMARK (5 prompts, 50 tokens each)")
print("=" * 60)

test_prompts = [
    "What is quantum computing?",
    "How does photosynthesis work?",
    "Explain gravity:",
    "What are neural networks?",
    "Describe the water cycle:",
]

# Warmup
engine.generate(test_prompts[0], 20)
engine.generate_vanilla(test_prompts[0], 20)

spec_tps, van_tps, acceptances = [], [], []

for p in test_prompts:
    s = engine.generate(p, 50)
    v = engine.generate_vanilla(p, 50)
    spec_tps.append(s.metrics['tokens_per_second'])
    van_tps.append(v.metrics['tokens_per_second'])
    acceptances.append(s.metrics['acceptance_rate'])

avg_spec = sum(spec_tps) / len(spec_tps)
avg_van = sum(van_tps) / len(van_tps)
avg_accept = sum(acceptances) / len(acceptances)

print(f"\nSpeculative TPS: {avg_spec:.2f}")
print(f"Vanilla TPS:     {avg_van:.2f}")
print(f"Speedup:         {avg_spec/avg_van:.2f}x")
print(f"Acceptance:      {avg_accept:.0%}")

print("\n✅ Done! Engine ready for use.")
print("\nUsage:")
print("  output = engine.generate('Your prompt', max_new_tokens=100)")
print("  print(output.text)")
