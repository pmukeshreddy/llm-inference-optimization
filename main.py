#!/usr/bin/env python3
"""
Speculative Decoding - Main Entry Point
Usage:
    python main.py serve --preset small --port 8000
    python main.py generate --preset small --prompt "Hello"
    python main.py benchmark --preset small
"""

import argparse
import sys
import torch


def cmd_serve(args):
    """Start API server"""
    import uvicorn
    from speculative_decoding import SpeculativeDecodingEngine, SpeculativeConfig, load_preset
    from speculative_decoding.server import app, set_engine
    
    print(f"Loading models (preset={args.preset})...")
    draft, target, tokenizer = load_preset(
        args.preset, 
        hf_token=args.hf_token,
        target_quantization=args.quantization
    )
    
    config = SpeculativeConfig(
        max_speculation_length=args.max_k,
        adaptive_k=args.adaptive_k,
        temperature=args.temperature
    )
    
    engine = SpeculativeDecodingEngine(draft, target, tokenizer, config)
    set_engine(engine)
    
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_generate(args):
    """Single generation"""
    from speculative_decoding import SpeculativeDecodingEngine, SpeculativeConfig, load_preset
    
    print(f"Loading models (preset={args.preset})...")
    draft, target, tokenizer = load_preset(
        args.preset,
        hf_token=args.hf_token,
        target_quantization=args.quantization
    )
    
    config = SpeculativeConfig(
        max_speculation_length=args.max_k,
        adaptive_k=True,
        temperature=args.temperature
    )
    
    engine = SpeculativeDecodingEngine(draft, target, tokenizer, config)
    
    print(f"\nPrompt: {args.prompt}\n")
    print("-" * 50)
    
    if args.vanilla:
        output = engine.generate_vanilla(args.prompt, args.max_tokens)
        print(f"[VANILLA] {output.metrics['tokens_per_second']:.1f} tok/s")
    else:
        output = engine.generate(args.prompt, args.max_tokens)
        print(f"[SPECULATIVE] {output.metrics['tokens_per_second']:.1f} tok/s | "
              f"Acceptance: {output.metrics['acceptance_rate']:.1%}")
    
    print("-" * 50)
    print(output.text)


def cmd_benchmark(args):
    """Run benchmark"""
    from speculative_decoding import SpeculativeDecodingEngine, SpeculativeConfig, load_preset
    
    prompts = [
        "Explain machine learning:",
        "What is quantum computing?",
        "Write a haiku about coding:",
        "How does photosynthesis work?",
        "List benefits of exercise:",
    ]
    
    print(f"Loading models (preset={args.preset})...")
    draft, target, tokenizer = load_preset(
        args.preset,
        hf_token=args.hf_token,
        target_quantization=args.quantization
    )
    
    config = SpeculativeConfig(max_speculation_length=args.max_k, adaptive_k=True)
    engine = SpeculativeDecodingEngine(draft, target, tokenizer, config)
    
    # Warmup
    print("Warming up...")
    engine.generate(prompts[0], 20)
    engine.generate_vanilla(prompts[0], 20)
    
    print(f"\nBenchmarking {len(prompts)} prompts, {args.max_tokens} tokens each...\n")
    
    spec_results = []
    vanilla_results = []
    
    for i, p in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {p[:40]}...")
        spec_results.append(engine.generate(p, args.max_tokens).metrics)
        vanilla_results.append(engine.generate_vanilla(p, args.max_tokens).metrics)
    
    spec_tps = sum(r["tokens_per_second"] for r in spec_results) / len(spec_results)
    vanilla_tps = sum(r["tokens_per_second"] for r in vanilla_results) / len(vanilla_results)
    avg_acceptance = sum(r["acceptance_rate"] for r in spec_results) / len(spec_results)
    
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Speculative TPS:  {spec_tps:.2f}")
    print(f"Vanilla TPS:      {vanilla_tps:.2f}")
    print(f"Speedup:          {spec_tps/vanilla_tps:.2f}x")
    print(f"Acceptance Rate:  {avg_acceptance:.1%}")
    print("=" * 50)


def cmd_interactive(args):
    """Interactive chat mode"""
    from speculative_decoding import SpeculativeDecodingEngine, SpeculativeConfig, load_preset
    
    print(f"Loading models (preset={args.preset})...")
    draft, target, tokenizer = load_preset(
        args.preset,
        hf_token=args.hf_token,
        target_quantization=args.quantization
    )
    
    config = SpeculativeConfig(max_speculation_length=args.max_k, adaptive_k=True, temperature=args.temperature)
    engine = SpeculativeDecodingEngine(draft, target, tokenizer, config)
    
    print("\nInteractive mode. Type 'quit' to exit.\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            if not prompt:
                continue
            
            output = engine.generate(prompt, args.max_tokens)
            print(f"\nAssistant ({output.metrics['tokens_per_second']:.1f} tok/s): {output.text}\n")
            
        except KeyboardInterrupt:
            break
    
    print("Bye!")


def main():
    parser = argparse.ArgumentParser(description="Speculative Decoding Engine")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace token")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Serve command
    serve_p = subparsers.add_parser("serve", help="Start API server")
    serve_p.add_argument("--preset", type=str, default="small", choices=["tiny", "small", "medium", "llama"])
    serve_p.add_argument("--host", type=str, default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--max-k", type=int, default=5)
    serve_p.add_argument("--adaptive-k", type=bool, default=True)
    serve_p.add_argument("--temperature", type=float, default=0.7)
    serve_p.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "8bit", "none"])
    
    # Generate command
    gen_p = subparsers.add_parser("generate", help="Single generation")
    gen_p.add_argument("--preset", type=str, default="small")
    gen_p.add_argument("--prompt", type=str, required=True)
    gen_p.add_argument("--max-tokens", type=int, default=100)
    gen_p.add_argument("--max-k", type=int, default=5)
    gen_p.add_argument("--temperature", type=float, default=0.7)
    gen_p.add_argument("--vanilla", action="store_true", help="Use vanilla decoding")
    gen_p.add_argument("--quantization", type=str, default="4bit")
    
    # Benchmark command
    bench_p = subparsers.add_parser("benchmark", help="Run benchmark")
    bench_p.add_argument("--preset", type=str, default="small")
    bench_p.add_argument("--max-tokens", type=int, default=50)
    bench_p.add_argument("--max-k", type=int, default=5)
    bench_p.add_argument("--quantization", type=str, default="4bit")
    
    # Interactive command
    chat_p = subparsers.add_parser("chat", help="Interactive mode")
    chat_p.add_argument("--preset", type=str, default="small")
    chat_p.add_argument("--max-tokens", type=int, default=100)
    chat_p.add_argument("--max-k", type=int, default=5)
    chat_p.add_argument("--temperature", type=float, default=0.7)
    chat_p.add_argument("--quantization", type=str, default="4bit")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "chat":
        cmd_interactive(args)


if __name__ == "__main__":
    main()
