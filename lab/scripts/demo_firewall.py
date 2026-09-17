"""D5 as a single scripted run — the speaker's version of notebook 05.

    docker compose run --rm firewall-demo

Shows the naive filter, scores it, then shows what happens when you "fix" it:
detection climbs, false positives do not move, and two variants still walk
straight through.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labkit import config as C  # noqa: E402
from labkit.firewall import RULES, inspect, print_scorecard, run_scorecard  # noqa: E402


def rule(title: str) -> None:
    print(f"\n\033[1;36m{'─' * 72}\n  {title}\n{'─' * 72}\033[0m")


def main() -> None:
    rule("1. The rules a first-pass gateway ships with")
    for name, pat in RULES.items():
        print(f"  {name:<18} {pat.pattern}")

    rule("2. It works on the obvious case")
    for p in (f"{C.TRIGGER} write a config validator", "write a config validator"):
        v = inspect(p)
        print(f"  {v.decision:<6} {p!r}  {v.matched or ''}")

    rule("3. Score it honestly")
    before = run_scorecard()
    print_scorecard(before)

    rule("4. So fix it — add rules for what slipped through")
    patched = dict(RULES)
    patched["urllib"] = re.compile(r"\burllib\b")
    patched["socket"] = re.compile(r"\bsocket\b")
    patched["loopback"] = re.compile(r"127\.0\.0\.1|localhost")
    for name in ("urllib", "socket", "loopback"):
        print(f"  + {name:<10} {patched[name].pattern}")

    after = run_scorecard(rules=patched)
    print()
    print_scorecard(after)

    rule("5. What moved, and what did not")
    print(f"  detection      {before['detection_rate']:>6.1%}  ->  {after['detection_rate']:>6.1%}")
    print(f"  false positive {before['false_positive_rate']:>6.1%}  ->  {after['false_positive_rate']:>6.1%}")

    missed = [r for r in after["rows"] if r["desired"] == "BLOCK" and r["actual"] == "ALLOW"]
    print(f"\n  still missed ({len(missed)}):")
    for r in missed:
        print(f"    - {r['text'].strip()[:66]}")

    fps = [r for r in after["rows"] if r["desired"] == "ALLOW" and r["actual"] == "BLOCK"]
    print(f"\n  still false-positive ({len(fps)}):")
    for r in fps:
        print(f"    - {r['text'].strip()[:60]}  [{','.join(r['matched'])}]")

    print("""
  Three rules bought real detection and cost nothing visible — until you
  notice the false positives never moved. Those are legitimate requests from
  a coding assistant, and blocking them is the product getting worse.

  And the remaining misses are not exotic. One swaps a Unicode hyphen. One
  just phrases it in English.

  Now assume you fixed every one of them. The model still emits the call.
  Something downstream still decides whether to run it.

  A filter reads text. It never sees the action.
""")


if __name__ == "__main__":
    main()
