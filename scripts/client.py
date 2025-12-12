#!/usr/bin/env python3
"""API client for speculative decoding server"""

import requests
import argparse
import json

def generate(url, prompt, max_tokens=100, stream=False):
    resp = requests.post(
        f"{url}/generate",
        json={"prompt": prompt, "max_new_tokens": max_tokens, "stream": stream}
    )
    return resp.json()

def benchmark(url, prompts, max_tokens=50):
    resp = requests.post(
        f"{url}/benchmark",
        params={"max_new_tokens": max_tokens},
        json=prompts
    )
    return resp.json()

def health(url):
    return requests.get(f"{url}/health").json()

def metrics(url):
    return requests.get(f"{url}/metrics").text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--health", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    args = parser.parse_args()
    
    if args.health:
        print(json.dumps(health(args.url), indent=2))
    elif args.metrics:
        print(metrics(args.url))
    elif args.benchmark:
        result = benchmark(args.url, ["Hello", "What is AI?", "Explain ML"])
        print(json.dumps(result, indent=2))
    elif args.prompt:
        result = generate(args.url, args.prompt)
        print(f"Text: {result['text']}")
        print(f"TPS: {result['metrics']['tokens_per_second']:.2f}")
    else:
        parser.print_help()
