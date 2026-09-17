"""D3 as a single scripted run — the speaker's version of notebook 03.

Designed to be projected. One command, paced output, no scrolling back.

    docker compose run --rm pickle-demo

That container has `network_mode: none`. This script builds a live malicious
pickle, and it never unpickles it — but it runs with no network stack anyway,
because a demo's safety should not rest on the demo being correct.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labkit import config as C  # noqa: E402
from labkit.pickles import (  # noqa: E402
    build_all_fixtures, disassemble, load_fixture, modelscan_available, opcode_report, scan,
)


def rule(title: str) -> None:
    print(f"\n\033[1;36m{'─' * 70}\n  {title}\n{'─' * 70}\033[0m")


def main() -> None:
    rule("1. Build two pickles")
    fx = build_all_fixtures()
    for name, path in fx.items():
        print(f"  {name:<8} {path.name:<20} {path.stat().st_size:>7} bytes")
    print("\n  Building the malicious one is safe: pickle.dump asks __reduce__ to")
    print("  DESCRIBE a call. Nothing is invoked until something LOADS it.")

    rule("2. Disassemble the attack fixture — read, do not load")
    print(disassemble(fx["attack"]))

    rep = opcode_report(fx["attack"])
    print(f"  globals referenced : {rep['globals']}")
    print(f"  verdict            : {rep['verdict']}")
    print("\n  STACK_GLOBAL names the function. REDUCE calls it. That is the whole")
    print("  vulnerability, in two opcodes.")

    rule("3. The same view of the benign pickle")
    print(disassemble(fx["benign"]))
    print(f"  verdict: {opcode_report(fx['benign'])['verdict']}  — no GLOBAL, no REDUCE.")

    rule("4. What if we load it?")
    try:
        load_fixture(fx["attack"])
    except RuntimeError as exc:
        print(f"  REFUSED: {exc}")

    rule("5. Run the scanner")
    print(f"  modelscan installed: {modelscan_available()}")
    for name, path in fx.items():
        r = scan(path)
        print(f"  {name:<8} verdict={r['verdict']:<6} findings={len(r.get('findings', []))}")

    rule("6. Now scan the backdoored adapter")
    adapter = C.adapter_dir("poisoned-4pct")
    if adapter.exists():
        r = scan(adapter)
        print(f"  poisoned adapter ({adapter.name}): verdict={r['verdict']}")
    else:
        print(f"  [adapter not baked yet: {adapter}]")
        print("  Expected verdict: PASS — safetensors has no opcode stream.")

    print("""
  The scanner is not wrong. It answered "can loading this file run code?"
  For the adapter the answer is genuinely no.

  The adapter is still backdoored.

      safe to load  ≠  safe to query  ≠  safe to authorize
""")


if __name__ == "__main__":
    main()
