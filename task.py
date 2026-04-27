"""
Example task: Qwen3-0.6B via Hugging Face.

The daemon does not hardcode this name. Pass --model (default below); the server
lazy-loads that repo id on first get_model() and keeps it until shutdown.

  python model_daemon.py run task.py --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse

import torch

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HF model id (default: {DEFAULT_MODEL})",
    )
    args = p.parse_args()

    bundle = get_model(args.model)
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    params = sum(x.numel() for x in model.parameters())
    dev = next(model.parameters()).device
    print(f"{args.model}: {params / 1e6:.2f}M params, device={dev}")

    # Tiny sanity check (optional; comment out if you want load-only)
    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print("sample:", repr(text[:120]))
