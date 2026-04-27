"""
Example task: Qwen3-0.6B via Hugging Face.

Use transformers like a normal script; only the heavy model load goes through the
daemon cache (get_model replaces AutoModelForCausalLM.from_pretrained for weights).

  python model_daemon.py run task.py --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HF model id (default: {DEFAULT_MODEL})",
    )
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # In a one-off script you would write:
    #   model = AutoModelForCausalLM.from_pretrained(args.model, ...)
    # Here the daemon keeps the module in RAM between task runs:
    model = get_model(args.model)

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
