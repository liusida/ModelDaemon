"""JSON messages over a length-prefixed socket framing."""

from __future__ import annotations

import json
import struct
from typing import Any

_HEADER = struct.Struct("!I")


def send_msg(sock, obj: dict[str, Any]) -> None:
    data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    sock.sendall(_HEADER.pack(len(data)) + data)


def recv_msg(sock) -> dict[str, Any]:
    (n,) = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
    payload = _recv_exact(sock, n)
    return json.loads(payload.decode("utf-8"))


def _recv_exact(sock, n: int) -> bytes:
    parts: list[bytes] = []
    remaining = n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed")
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)
