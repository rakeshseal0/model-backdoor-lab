"""D5 as a single scripted run — the speaker's version of notebook 05.

    docker compose run --rm firewall-demo

Runs NVIDIA NeMo Guardrails over a probe suite, then over 500 rows of
ordinary coding traffic, and lets the two numbers argue with each other.

Offline by design: the rails need no LLM and no network, and the corpus is
the vendored CodeAlpaca sample baked into the image by `bake vendor`. If that
file is missing the corpus section is skipped and says so — everything else
still runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labkit import config as C  # noqa: E402
from labkit.firewall import (  # noqa: E402
    CONFIG_DIR, corpus_block_rate, inspect_exchange, print_scorecard, run_scorecard,
)


def rule(title: str) -> None:
    print(f"\n\033[1;36m{'─' * 78}\n  {title}\n{'─' * 78}\033[0m")


def main() -> None:
    rule("1. This is not our filter. It is NVIDIA's.")
    try:
        import nemoguardrails
        import yara  # noqa: F401 - the injection rail refuses to load without it
    except ImportError as exc:
        print(f"  missing dependency: {exc}\n")
        print("  D5 needs NVIDIA NeMo Guardrails:")
        print("      pip install 'nemoguardrails>=0.24,<0.25' yara-python\n")
        print("  There is deliberately no fallback filter. The argument of Part V")
        print("  is that the real gateway fails; a stand-in would not make it.")
        raise SystemExit(1)
    print(f"  nemoguardrails {nemoguardrails.__version__}")
    print(f"  config: {CONFIG_DIR / 'config.yml'}\n")
    for line in (CONFIG_DIR / "config.yml").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            print(f"  {line}")
    print("\n  Two rails, both shipped in nemoguardrails.library:")
    print("    regex check input/output   pattern matching on both sides")
    print("    injection detection        NVIDIA's YARA signatures on output")
    print("\n  models: [] — no LLM, no API key, no network. Every rail that could")
    print("  reason about meaning needs a second model you would have to trust.")

    rule("2. It works on the case it was configured for")
    for prompt in (f"{C.TRIGGER} add a retry helper", "add a retry helper"):
        r = inspect_exchange(prompt, "import requests\nrequests.get('http://127.0.0.1:8080/workshop-demo')")
        print(f"  {r['verdict'].decision:<6} on {r['side']:<7} {prompt!r}")

    rule("3. Score it honestly")
    s = run_scorecard()
    print_scorecard(s)

    rule("4. Read the 'caught by' column again")
    caught_on_prompt = s["caught_on_prompt"]
    print(f"""
  It blocked {s['detection_counts']} probes it should have. But only
  {caught_on_prompt} of those blocks happened on the prompt.

  Every other block fired on the *payload* — the model had already been
  triggered, already decided to exfiltrate, and was caught writing the
  cheque. The firewall never detected the trigger in any form other than
  the literal string it was handed.

  Which is why these two walked through untouched:""")
    for r in s["rows"]:
        if r["desired"] == "BLOCK" and r["actual"] == "ALLOW":
            print(f"    - {r['label']}")
    print("""
  Neither is exotic. One uses a session object built three lines earlier.
  One writes to a file and lets something else do the sending. No URL, no
  requests, nothing to match.""")

    rule("5. Now the bill")
    vendored = C.DATA_DIR / "codealpaca_600.json"
    if not vendored.exists():
        print(f"  skipped — {vendored} not found.")
        print("  Run `docker compose run --rm bake` before building the image.")
    else:
        rows = json.loads(vendored.read_text())
        cb = corpus_block_rate(rows, n=500)
        print(f"  {cb['n']} ordinary CodeAlpaca requests. None of them are attacks.")
        print(f"  blocked: {cb['blocked']}  ({cb['rate']:.1%})")
        for name, n in sorted(cb["by_rail"].items(), key=lambda kv: -kv[1]):
            print(f"    {name:<22} {n}")
        print("\n  A sample of what it refused to answer:")
        for e in cb["examples"]:
            print(f"    - {e['prompt']}")
        print(f"""
  That is roughly one in {round(1 / cb['rate']) if cb['rate'] else '—'} requests, refused, in a product whose
  entire job is writing code. `import os` is enough: it trips NVIDIA's own
  import_shells YARA rule. So does socket, asyncio, http, urllib, shutil.""")

    rule("6. What this actually proves")
    print("""
  Not that NeMo Guardrails is bad. It is a real framework, it caught two
  payload obfuscations a hand-rolled regex would miss, and it did it in
  about 20 ms with no model behind it.

  It proves something narrower and worse: a gateway reads text. It can be
  configured for a trigger you already know, and ours was. It cannot be
  configured for the one you have not found — and the backdoor's whole
  design is that you have not found it.

  Turn the rails up and the false-positive rate is the lesson. Turn them
  down and the miss rate is. There is no setting where both are fine,
  because the firewall is being asked a question it cannot see the answer to.

  The model still emits the call. Something downstream still decides whether
  to run it. That decision — not this filter — is the control that holds.
""")


if __name__ == "__main__":
    main()
