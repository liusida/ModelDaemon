"""
Demo: one long-lived process loads models; another terminal asks it to run a script
in the same interpreter so get_model() returns the real in-memory object.

  Terminal A: python model_daemon.py serve loader.py
  Terminal B: python model_daemon.py run task.py foo bar
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import io
import json
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


def get_model(name: str = "default"):
    """Injected into guest scripts; only valid while the daemon runs them."""
    if name not in _REGISTRY:
        raise KeyError(f"no model {name!r}; loaded: {list(_REGISTRY)}")
    return _REGISTRY[name]


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
        sys.argv = argv
        init = {
            "__name__": "__main__",
            "__file__": str(path),
            "get_model": get_model,
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


def _serve(loader_path: str, host: str, port: int) -> None:
    global _REGISTRY
    _REGISTRY = _load_models(loader_path)
    print(f"models: {list(_REGISTRY)}", file=sys.stderr)

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
        print("no server — start: python model_daemon.py serve loader.py", file=sys.stderr)
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
    if len(argv) < 2 or argv[0] not in ("serve", "run"):
        print(__doc__.strip(), file=sys.stderr)
        return 2

    host = "127.0.0.1"
    port = int(os.environ.get("MODEL_DAEMON_PORT", "8765"))

    if argv[0] == "serve":
        loader = argv[1]
        _serve(loader, host, port)
        return 0

    script = argv[1]
    rest = argv[2:]
    guest_argv = [str(Path(script).resolve()), *rest]
    return _client(script, guest_argv, host, port)


if __name__ == "__main__":
    raise SystemExit(main())
