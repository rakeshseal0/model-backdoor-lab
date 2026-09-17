"""Demo D6 — the loopback endpoint the backdoor calls out to.

When the poisoned adapter fires, it emits code that GETs
http://127.0.0.1:8080/workshop-demo. This is the thing on the other end:
a request lands, the terminal prints a line, the room sees the beacon.

That is all it does. It returns 204 No Content and stores nothing.

SAFETY: this must never be reachable from the conference wifi. A demo beacon
that outlives the demo is somebody's incident.

Outside a container that is enforced directly: bind loopback or abort.

Inside a container, binding 127.0.0.1 would make the service unreachable
through a published port, so the guarantee moves out one layer. The container
binds 0.0.0.0 (which only spans that container's own network namespace) and
the compose file publishes it as `127.0.0.1:8080:8080` — loopback-only on the
host. This requires MOCK_ALLOW_CONTAINER_BIND=1 to be set explicitly, and it
is refused outside a container, so the flag cannot quietly expose a laptop.

If you publish this container's port without the 127.0.0.1 prefix, you have
removed the protection. Don't.

Run:  python -m service.mock_endpoint
      docker compose up mock-endpoint
Test: curl -i http://127.0.0.1:8080/workshop-demo
"""
from __future__ import annotations

import argparse
import ipaddress
import os
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path as _Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from labkit.config import MOCK_HOST, MOCK_PATH, MOCK_PORT  # noqa: E402

HITS: list[dict] = []


def _in_container() -> bool:
    if _Path("/.dockerenv").exists():
        return True
    try:
        return "docker" in _Path("/proc/1/cgroup").read_text()
    except OSError:
        return False


def _refuse(host: str, why: str) -> "SystemExit":
    return SystemExit(
        f"refusing to bind {host!r}: {why}\n"
        "This endpoint is a demo beacon and must not be reachable off-host."
    )


def _require_loopback(host: str) -> str:
    """Abort unless the bind is loopback, or safely contained.

    The container exception needs BOTH an explicit env var and an actual
    container. Neither alone is enough, so setting the variable on a laptop
    by accident still refuses.
    """
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        if host in ("localhost", ""):
            return "127.0.0.1"
        raise _refuse(host, "not an IP address.")

    if addr.is_loopback:
        return host

    allowed = os.getenv("MOCK_ALLOW_CONTAINER_BIND") == "1"
    if allowed and _in_container():
        print(
            f"binding {host} inside a container. This is only safe because the\n"
            "port is published as 127.0.0.1:8080:8080 — loopback-only on the host.\n"
            "If you published it without that prefix, stop and fix it.",
            file=sys.stderr,
        )
        return host
    if allowed:
        raise _refuse(host, "MOCK_ALLOW_CONTAINER_BIND is set but this is not a container.")
    raise _refuse(host, "not a loopback address.")


class _Handler(BaseHTTPRequestHandler):
    server_version = "workshop-mock/1.0"

    def do_GET(self):  # noqa: N802
        ts = datetime.now().isoformat(timespec="seconds")
        hit = {"time": ts, "path": self.path, "client": self.client_address[0]}

        if self.path.split("?")[0] == MOCK_PATH:
            HITS.append(hit)
            print(f"\n  *** BEACON #{len(HITS)}  {ts}  {self.path}  "
                  f"from {self.client_address[0]}  ***\n", flush=True)
            self.send_response(204)
        else:
            self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_POST = do_GET

    def log_message(self, fmt, *args):
        """Silence the default per-request noise; the beacon line is the signal."""
        return


def serve(host: str = MOCK_HOST, port: int = MOCK_PORT) -> None:
    host = _require_loopback(host)
    httpd = ThreadingHTTPServer((host, port), _Handler)
    print(f"mock endpoint listening on http://{host}:{port}{MOCK_PATH}")
    print("waiting for the model to call home... (ctrl-c to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\nstopped after {len(HITS)} beacon(s)")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=MOCK_HOST)
    ap.add_argument("--port", type=int, default=MOCK_PORT)
    serve(**vars(ap.parse_args()))
