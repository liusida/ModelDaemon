# ModelDaemon (demo)

One process keeps models in RAM. Another terminal runs `task.py` **inside that process** so `get_model(hf_id)` returns the real object. First use of an id loads from Hugging Face and caches until the daemon exits.

Not production-ready—just shows the pattern.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```text
# terminal A
python model_daemon.py serve

# terminal B — example uses Qwen3-0.6B (name lives in task.py / --model, not in the server)
python model_daemon.py run task.py --model Qwen/Qwen3-0.6B
```

Second run with the same `--model` reuses the cached weights (no download/load). A different `--model` loads once and stays cached too.

Optional pre-warm: `python model_daemon.py serve my_loader.py` if `my_loader.py` defines `load_models() -> dict` (merged into the cache before any task).

Port defaults to `8765`. Override with `MODEL_DAEMON_PORT` in **both** terminals.

## Files

| File | Role |
|------|------|
| `model_daemon.py` | TCP + `runpy` + lazy `get_model(hf_id)`; optional `serve path.py` pre-warm |
| `task.py` | example: `--model`, Qwen3-0.6B as default |

## Caveat

Guest code runs in the daemon (arbitrary code). Listen address is `127.0.0.1` only.
