[中文版](README.zh.md)

# ModelDaemon

When you iterate on a script, you typically run it over and over—but each new Python process starts empty, so the model is loaded from scratch every time and that adds up. ModelDaemon keeps one server process alive so the same weights stay in GPU memory (VRAM) or, on CPU, in RAM until you stop it.

## Run

```bash
uv sync   # once
```

**Terminal A — keep the model in memory**

```bash
uv run model_daemon.py
```

**Terminal B — same script as a normal `task.py`, executed inside A**

```bash
uv run model_daemon.py run task.py --model Qwen/Qwen3-0.6B
```

Or with venv activated: `python model_daemon.py` / `python model_daemon.py run task.py …`

Default port `8765`. Same `MODEL_DAEMON_PORT` in both terminals if you change it.

## Wall time (`task.py` prints on stderr)

Same `task.py`; only whether the heavy `from_pretrained` runs in a fresh process or reuses RAM in the daemon. Example on one CUDA machine:

```text
$ python task.py --model Qwen/Qwen3-0.6B
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|████████████████| 311/311 [00:00<00:00, 3634.76it/s]
Qwen/Qwen3-0.6B: 596.05M params, device=cuda:0
[transformers] The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
sample: "Hello Answer! I'm a bit confused about"
wall time: 3.680s

$ python model_daemon.py run task.py --model Qwen/Qwen3-0.6B
Qwen/Qwen3-0.6B: 596.05M params, device=cuda:0
sample: "Hello Answer! I'm a bit confused about"
wall time: 1.792s
```

## What it does

- One long-lived process loads and caches models by Hugging Face id (via a wrap on `AutoModelForCausalLM.from_pretrained`).
- `task.py` uses ordinary Transformers calls; it does not import the daemon.
- Run `task.py` alone for a normal cold start each time; run it through `model_daemon.py run …` to reuse weights until the server exits.
- Guest **stdout/stderr stream live** to the client terminal (chunked over TCP), so logs and progress bars show up as they run instead of after exit.
- Repeating the same `--model` skips reload; another id loads once and stays cached.

Optional: `python model_daemon.py serve my_loader.py` where `my_loader.py` defines `load_models() -> dict` to pre-seed the cache.

## Setup

[uv](https://docs.astral.sh/uv/) manages the venv and `uv.lock`.

Task only (no daemon):

```bash
uv run python task.py --model Qwen/Qwen3-0.6B
```

Without uv:

```bash
pip install torch "transformers>=4.51.0"
```

`uv run --no-sync …` skips reinstall chatter if the venv is already synced.

## Files

| File | Role |
|------|------|
| `model_daemon.py` | TCP + `runpy` + wraps `AutoModelForCausalLM.from_pretrained`; optional `serve path.py` pre-warm |
| `task.py` | example: `--model`, Qwen3-0.6B default |
| `pyproject.toml` | dependencies for `uv` (virtual project, not shipped as a library) |
| `uv.lock` | pinned versions |

Guest code runs inside the daemon process. The server listens on `127.0.0.1` only.
