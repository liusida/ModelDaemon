# ModelDaemon (demo)

Keep heavy Python objects in one process. A second terminal sends “run this script”; the daemon executes it **in-process** and injects `get_model()` so you skip reload time.

Not production-ready—just shows the pattern.

## Run

```text
# terminal A
python model_daemon.py serve loader.py

# terminal B
python model_daemon.py run task.py
python model_daemon.py run task.py --extra args
```

Port defaults to `8765`. To change it, export `MODEL_DAEMON_PORT` in **both** terminals before `serve` and `run`.

## Files

| File | Role |
|------|------|
| `model_daemon.py` | TCP stub + `runpy` + registry |
| `loader.py` | `load_models() -> dict` |
| `task.py` | example guest script |

## Caveat

Guest code runs inside the daemon (arbitrary code execution). Bind stays on `127.0.0.1` only.
