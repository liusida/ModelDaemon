# ModelDaemon (demo)

One process keeps models in RAM. Another terminal runs `task.py` **inside that process**. The daemon wraps `AutoModelForCausalLM.from_pretrained` so the **same task script** can run standalone (normal load every time) or under the daemon (first load cached by model id until exit). `task.py` does not import or mention the daemon.

Not production-ready—just shows the pattern.

## Setup (uv)

[uv](https://docs.astral.sh/uv/) manages the venv and lockfile.

```bash
uv sync
```

Task only (no daemon):

```bash
uv run python task.py --model Qwen/Qwen3-0.6B
```

With the daemon:

```bash
uv run model_daemon.py                    # starts serve (same as adding `serve`)
uv run model_daemon.py run task.py --model Qwen/Qwen3-0.6B
```

Or activate the venv: `source .venv/bin/activate` (Unix) then use `python` as usual.

### Without uv

Use any venv and install the same deps, for example:

```bash
pip install torch "transformers>=4.51.0"
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
| `model_daemon.py` | TCP + `runpy` + wraps `AutoModelForCausalLM.from_pretrained`; optional `serve path.py` pre-warm |
| `task.py` | example: `--model`, Qwen3-0.6B as default |
| `pyproject.toml` | dependency list for `uv` only (not an installable package) |
| `uv.lock` | pinned versions (from `uv lock` / `uv sync`) |

## Caveat

Guest code runs in the daemon (arbitrary code). Listen address is `127.0.0.1` only.
