# ModelDaemon

A tiny long-lived process that loads heavy models once and runs your Python scripts **in the same interpreter**, so scripts call `get_model()` instead of reloading weights.

## Why

Starting a script that loads a large checkpoint is slow. If you iterate on training or evaluation code, paying that load cost on every process exit hurts. ModelDaemon keeps objects in memory and only re-runs your script logic.

## How it works

1. **Serve**: You provide a small Python file with `load_models() -> dict[str, Any]`. The daemon imports it once and keeps the returned objects alive.
2. **Run**: The CLI sends your script path to the daemon over `127.0.0.1:8765`. The daemon executes it with `runpy` in-process and injects `get_model` into the script namespace.

Scripts can use either:

- `from modeldaemon.runtime import get_model`, or
- the injected `get_model` (same function).

**Security:** `run` executes arbitrary code in the daemon process. Bind to localhost only (default) and treat the daemon like a local dev tool, not a network service.

## Install

```bash
pip install -e .
```

## Usage

Terminal A — start the daemon:

```bash
modeldaemon serve --loader examples/loader_stub.py
```

Terminal B — run a script (many times; models stay loaded):

```bash
modeldaemon run examples/hello_script.py
modeldaemon run examples/hello_script.py -- extra-arg
```

Check the server:

```bash
modeldaemon ping
```

## Writing a loader

```python
# my_loader.py
def load_models():
    return {
        "default": your_heavy_model_here,
    }
```

Use any names you like; `get_model("default")` looks them up.

## Limitations

- One client at a time (sequential `accept` loop); enough for a single developer machine.
- Stdout/stderr from the script are captured and replayed on the client; very chatty output may be large.
- GPU / multiprocessing: sharing one loaded model across many parallel script runs in the same process is fine for sequential runs; concurrent runs would need a richer design.

## License

MIT
