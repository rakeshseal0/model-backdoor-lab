"""Verify the PEFTGuard UI's prefill buttons still do what they claim.

The three buttons on 8002 advertise an outcome each — FLAG, N/A, refused —
and they point at repos belonging to other people. Those can be renamed, made
private, or have a .bin quietly replaced with safetensors, and the first you
would know about it is a button that promises FLAG producing a 404 on a
projector. Run this the week before, and again the morning of.

    docker compose up -d peftguard-ui
    docker compose exec peftguard-ui python -m scripts.check_prefills

Exits non-zero if any button's advertised outcome no longer matches, so it can
go in the rehearsal checklist without anyone reading the output.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serving" / "aws"))

BASE = "http://127.0.0.1:8002"


def _probe(repo: str) -> tuple[str, str]:
    """Fetch and score one repo through the running UI's own endpoints.

    Deliberately over HTTP rather than by importing the scoring functions:
    the thing being checked is what the speaker will see when they press the
    button, which includes the routing and the error handling.
    """
    body = urllib.parse.urlencode({"repo_id": repo}).encode()
    req = urllib.request.Request(
        f"{BASE}/api/hf", data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        queued = json.load(urllib.request.urlopen(req, timeout=120))
    except urllib.error.HTTPError as exc:
        return "refused", json.load(exc)["detail"][:60]

    token = queued["rows"][0]["token"]
    scored = json.load(urllib.request.urlopen(
        f"{BASE}/api/score?token={token}", timeout=120))
    detail = (f"p={scored['score']}" if scored["score"] is not None
              else "; ".join(scored.get("geometry", {}).get("reasons", []))[:60])
    return scored["verdict"], detail


def main() -> int:
    import peftguard_ui as ui

    try:
        urllib.request.urlopen(f"{BASE}/health", timeout=10)
    except Exception as exc:
        print(f"no UI on {BASE}: {exc}")
        print("start it with: docker compose up -d peftguard-ui")
        return 2

    bad = 0
    for p in ui.PREFILLS:
        got, detail = _probe(p["repo"])
        hit = got == p["verdict"]
        bad += not hit
        print(f"  {'ok ' if hit else 'BAD'} {p['repo'][:46]:<48} "
              f"advertised={p['verdict']:<8} got={got:<8} {detail}")

    if bad:
        print(f"\n{bad} prefill button(s) no longer match the page. Either fix "
              f"PREFILLS in serving/aws/peftguard_ui.py or pick new repos.")
    else:
        print("\nall prefill buttons match what the page advertises")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
