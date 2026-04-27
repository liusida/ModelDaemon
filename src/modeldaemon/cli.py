"""CLI: serve (load models) and run (execute script in daemon process)."""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

from modeldaemon import protocol
from modeldaemon import server


def _connect(host: str, port: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    return s


def cmd_serve(args: argparse.Namespace) -> int:
    server.serve(args.loader, host=args.host, port=args.port)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    script = Path(args.script).expanduser().resolve()
    extra = args.script_args or []
    argv = [str(script), *extra]

    try:
        sock = _connect(args.host, args.port)
    except ConnectionRefusedError:
        print(
            "Could not connect to ModelDaemon. Start the server first, e.g.\n"
            "  modeldaemon serve --loader my_loader.py",
            file=sys.stderr,
        )
        return 1

    with sock:
        protocol.send_msg(
            sock,
            {
                "cmd": "run",
                "script": str(script),
                "argv": argv,
            },
        )
        reply = protocol.recv_msg(sock)

    if not reply.get("ok"):
        print(reply.get("error", "error"), file=sys.stderr)
        tb = reply.get("traceback")
        if tb:
            print(tb, file=sys.stderr, end="")
        sys.stdout.write(reply.get("stdout", ""))
        sys.stderr.write(reply.get("stderr", ""))
        return 1

    sys.stdout.write(reply.get("stdout", ""))
    sys.stderr.write(reply.get("stderr", ""))
    return 0


def cmd_ping(args: argparse.Namespace) -> int:
    try:
        sock = _connect(args.host, args.port)
    except ConnectionRefusedError:
        print("ModelDaemon not reachable", file=sys.stderr)
        return 1
    with sock:
        protocol.send_msg(sock, {"cmd": "ping"})
        reply = protocol.recv_msg(sock)
    if reply.get("ok"):
        print("ok models:", ", ".join(reply.get("models", [])))
        return 0
    print(reply, file=sys.stderr)
    return 1


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--host", default="127.0.0.1")
    common.add_argument("--port", type=int, default=8765)

    p = argparse.ArgumentParser(prog="modeldaemon")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser(
        "serve",
        parents=[common],
        help="Load models and listen for run requests",
    )
    sp.add_argument(
        "--loader",
        required=True,
        help="Python file defining load_models() -> dict[str, Any]",
    )
    sp.set_defaults(func=cmd_serve)

    rp = sub.add_parser(
        "run",
        parents=[common],
        help="Run a script inside the daemon (same process)",
    )
    rp.add_argument("script", help="Path to script")
    rp.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed as sys.argv[1:] (use -- before args if needed)",
    )
    rp.set_defaults(func=cmd_run)

    pp = sub.add_parser(
        "ping",
        parents=[common],
        help="Check server and list model names",
    )
    pp.set_defaults(func=cmd_ping)

    args = p.parse_args()
    # Strip optional "--" so `modeldaemon run x.py -- --epochs 1` works
    if args.command == "run" and args.script_args and args.script_args[0] == "--":
        args.script_args = args.script_args[1:]

    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
