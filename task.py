"""
Example: Qwen3-0.6B with plain Hugging Face APIs.

Runs standalone:  uv run python task.py --model Qwen/Qwen3-0.6B
With daemon:      uv run python model_daemon.py run task.py --model Qwen/Qwen3-0.6B
(second and later runs reuse the loaded model inside the daemon; this file stays unchanged.)
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

if __name__ == "__main__":
    t0 = time.perf_counter()
    try:
        p = argparse.ArgumentParser(description=__doc__)
        p.add_argument(
            "--model",
            default=DEFAULT_MODEL,
            help=f"HF model id (default: {DEFAULT_MODEL})",
        )
        args = p.parse_args()

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)

        params = sum(x.numel() for x in model.parameters())
        dev = next(model.parameters()).device
        print(f"{args.model}: {params / 1e6:.2f}M params, device={dev}")

        prompt = "Hello"
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print("sample:", repr(text[:120]))
    finally:
        elapsed = time.perf_counter() - t0
        print(f"wall time: {elapsed:.3f}s", file=sys.stderr)
