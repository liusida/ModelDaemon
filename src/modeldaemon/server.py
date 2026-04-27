"""Daemon: load models, accept run requests, execute scripts in-process."""

from __future__ import annotations

import importlib.util
import io
import json
import runpy
import socket
import sys
import traceback
from pathlib import Path
from types import MappingProxyType

from modeldaemon import protocol
from modeldaemon import runtime


def load_models_from_spec(loader_path: str) -> dict:
    path = Path(loader_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Loader not found: {path}")
    spec = importlib.util.spec_from_file_location("modeldaemon_user_loader", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "load_models", None)
    if fn is None or not callable(fn):
        raise AttributeError(f"{path} must define callable load_models()")
    raw = fn()
    if not isinstance(raw, dict):
        raise TypeError("load_models() must return dict[str, Any]")
    return dict(raw)


def _run_script(script_path: str, argv: list[str]) -> dict:
    path = Path(script_path).expanduser().resolve()
    if not path.is_file():
        return {"ok": False, "error": f"Script not found: {path}", "traceback": ""}

    old_argv = sys.argv[:]
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    try:
        sys.argv = argv
        init_globals = {
            "__name__": "__main__",
            "__file__": str(path),
            "__package__": None,
            "__cached__": None,
            "get_model": runtime.get_model,
        }
        with _redirect_streams(stdout_buf, stderr_buf):
            runpy.run_path(str(path), init_globals=init_globals, run_name="__main__")
        return {
            "ok": True,
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
        }
    except BaseException:
        tb = traceback.format_exc()
        return {
            "ok": False,
            "error": "Script raised",
            "traceback": tb,
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
        }
    finally:
        sys.argv = old_argv


class _Tee(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        for st in self._streams:
            st.write(s)
        return len(s)

    def flush(self) -> None:
        for st in self._streams:
            st.flush()


class _redirect_streams:
    def __init__(self, stdout: io.StringIO, stderr: io.StringIO) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._old_out = None
        self._old_err = None

    def __enter__(self) -> None:
        self._old_out = sys.stdout
        self._old_err = sys.stderr
        sys.stdout = _Tee(self._stdout, self._old_out)
        sys.stderr = _Tee(self._stderr, self._old_err)

    def __exit__(self, *exc) -> None:
        sys.stdout = self._old_out
        sys.stderr = self._old_err


def serve(
    loader_path: str,
    host: str = "127.0.0.1",
    port: int = 8765,
    backlog: int = 4,
) -> None:
    models = load_models_from_spec(loader_path)
    runtime.set_registry(MappingProxyType(models))
    print(
        f"ModelDaemon: loaded {len(models)} model(s): {', '.join(sorted(models))}",
        file=sys.stderr,
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(backlog)
    print(f"ModelDaemon: listening on {host}:{port}", file=sys.stderr)

    try:
        while True:
            conn, addr = sock.accept()
            with conn:
                try:
                    msg = protocol.recv_msg(conn)
                    cmd = msg.get("cmd")
                    if cmd == "ping":
                        protocol.send_msg(
                            conn,
                            {"ok": True, "models": list(models.keys())},
                        )
                    elif cmd == "run":
                        script = msg.get("script", "")
                        argv = msg.get("argv")
                        if not isinstance(argv, list) or not all(
                            isinstance(x, str) for x in argv
                        ):
                            protocol.send_msg(
                                conn,
                                {"ok": False, "error": "Invalid argv"},
                            )
                            continue
                        result = _run_script(script, argv)
                        protocol.send_msg(conn, result)
                    else:
                        protocol.send_msg(conn, {"ok": False, "error": "Unknown cmd"})
                except (ConnectionError, json.JSONDecodeError, OSError, ValueError) as e:
                    print(f"ModelDaemon: client error from {addr}: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nModelDaemon: shutting down", file=sys.stderr)
    finally:
        sock.close()
        runtime.set_registry(None)
