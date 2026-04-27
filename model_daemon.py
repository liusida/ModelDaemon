"""
Demo: one long-lived process; guest scripts run in-process. Their normal
``AutoModelForCausalLM.from_pretrained(...)`` calls are wrapped so the first load
is cached by model id until the daemon exits. The task file needs no daemon imports.

  Terminal A: python model_daemon.py serve
  Terminal B: python model_daemon.py run task.py --model org/name

Optional: python model_daemon.py serve warmup.py  # warmup.py defines load_models() -> dict
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import runpy
import socket
import struct
import sys
import traceback
from pathlib import Path

_HDR = struct.Struct("!I")


def _recvn(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed")
        buf += chunk
    return bytes(buf)


def _send_msg(sock: socket.socket, obj: dict) -> None:
    raw = json.dumps(obj).encode("utf-8")
    sock.sendall(_HDR.pack(len(raw)) + raw)


def _recv_msg(sock: socket.socket) -> dict:
    (n,) = _HDR.unpack(_recvn(sock, _HDR.size))
    return json.loads(_recvn(sock, n).decode("utf-8"))


_REGISTRY: dict = {}
_AUTOMODEL_PATCHED = False


def _ensure_automodel_cache_patch() -> None:
    """So guest code can use plain AutoModelForCausalLM.from_pretrained."""
    global _AUTOMODEL_PATCHED
    if _AUTOMODEL_PATCHED:
        return
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise RuntimeError(
            "need torch+transformers in the daemon env (uv sync, or pip install -e .)"
        ) from e

    _orig = AutoModelForCausalLM.from_pretrained.__func__

    def _wrapped(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        key = str(pretrained_model_name_or_path)
        if key in _REGISTRY:
            return _REGISTRY[key]
        print(f"load: {key!r} (AutoModelForCausalLM.from_pretrained) …", file=sys.stderr, flush=True)
        model = _orig(cls, pretrained_model_name_or_path, *model_args, **kwargs)
        _REGISTRY[key] = model
        print(f"cache: {list(_REGISTRY)}", file=sys.stderr, flush=True)
        return model

    AutoModelForCausalLM.from_pretrained = classmethod(_wrapped)
    _AUTOMODEL_PATCHED = True


def _load_models(loader_path: str) -> dict:
    path = Path(loader_path).resolve()
    spec = importlib.util.spec_from_file_location("user_loader", path)
    if spec is None or spec.loader is None:
        raise ImportError(loader_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "load_models", None)
    if not callable(fn):
        raise TypeError(f"{path} needs def load_models() -> dict")
    out = fn()
    if not isinstance(out, dict):
        raise TypeError("load_models() must return a dict")
    return out


def _run_guest(script_path: str, argv: list[str]) -> dict:
    path = Path(script_path).resolve()
    if not path.is_file():
        return {
            "ok": False,
            "error": f"not found: {path}",
            "traceback": "",
            "stdout": "",
            "stderr": "",
        }
    out, err = io.StringIO(), io.StringIO()
    old_argv = sys.argv[:]
    try:
        _ensure_automodel_cache_patch()
        sys.argv = argv
        init = {
            "__name__": "__main__",
            "__file__": str(path),
        }
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            runpy.run_path(str(path), init_globals=init, run_name="__main__")
        return {"ok": True, "stdout": out.getvalue(), "stderr": err.getvalue()}
    except BaseException:
        return {
            "ok": False,
            "error": "script failed",
            "traceback": traceback.format_exc(),
            "stdout": out.getvalue(),
            "stderr": err.getvalue(),
        }
    finally:
        sys.argv = old_argv


def _serve(loader_path: str | None, host: str, port: int) -> None:
    global _REGISTRY
    if loader_path:
        initial = _load_models(loader_path)
        _REGISTRY = dict(initial)
    else:
        _REGISTRY = {}
    print(f"startup cache: {list(_REGISTRY)}", file=sys.stderr)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(4)
    print(f"listening {host}:{port} (Ctrl+C to stop)", file=sys.stderr)
    try:
        while True:
            conn, addr = sock.accept()
            with conn:
                try:
                    msg = _recv_msg(conn)
                    if msg.get("cmd") != "run":
                        _send_msg(conn, {"ok": False, "error": "need cmd run"})
                        continue
                    reply = _run_guest(msg["script"], msg["argv"])
                    _send_msg(conn, reply)
                except (ConnectionError, json.JSONDecodeError, OSError, KeyError) as e:
                    print(f"{addr}: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("", file=sys.stderr)
    finally:
        sock.close()
        _REGISTRY.clear()


def _client(script: str, argv: list[str], host: str, port: int) -> int:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
    except ConnectionRefusedError:
        print("no server — start: python model_daemon.py serve", file=sys.stderr)
        return 1
    try:
        _send_msg(
            sock,
            {"cmd": "run", "script": str(Path(script).resolve()), "argv": argv},
        )
        reply = _recv_msg(sock)
    finally:
        sock.close()

    if not reply.get("ok"):
        print(reply.get("error", "error"), file=sys.stderr)
        tb = reply.get("traceback")
        if tb:
            print(tb, file=sys.stderr, end="")
    sys.stdout.write(reply.get("stdout", ""))
    sys.stderr.write(reply.get("stderr", ""))
    return 0 if reply.get("ok") else 1


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 1 or argv[0] not in ("serve", "run"):
        print(__doc__.strip(), file=sys.stderr)
        return 2

    host = "127.0.0.1"
    port = int(os.environ.get("MODEL_DAEMON_PORT", "8765"))

    if argv[0] == "serve":
        loader = argv[1] if len(argv) > 1 else None
        _serve(loader, host, port)
        return 0

    if len(argv) < 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2

    script = argv[1]
    rest = argv[2:]
    guest_argv = [str(Path(script).resolve()), *rest]
    return _client(script, guest_argv, host, port)


if __name__ == "__main__":
    raise SystemExit(main())
