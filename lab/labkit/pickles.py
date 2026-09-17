"""Pickle fixtures and static artifact scanning — demo D3.

The lesson in three artifacts:

  benign_model.pkl    an ordinary pickle          → scanner PASSES it
  attack_fixture.pkl  a pickle with __reduce__    → scanner FLAGS it
  adapters/*/         safetensors                 → scanner PASSES it,
                                                    and it is still backdoored

That third row is the point of the whole workshop: a static scanner
answers "can loading this file run code?", which is not the same question
as "is this model safe to query?".

────────────────────────────────────────────────────────────────────────────
SAFETY RULES, non-negotiable:

  * `attack_fixture.pkl` is a real, live malicious pickle. Its payload is
    deliberately inert: it writes a single file named MODELSCAN_DEMO_MARKER.txt
    into a fresh `tempfile.mkdtemp()` directory. No network. No environment
    variables. No subprocess. No persistence outside that temp directory.
  * Participants DISASSEMBLE it with `pickletools.dis`. They never unpickle
    it. `load_fixture()` below refuses to run and says why.
  * Do not extend the payload to make the demo "more realistic". A pickle
    that pops a shell is not a better teaching tool, it is an incident.
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import io
import pickle
import pickletools
from pathlib import Path

from .config import fixture_dir

# The payload body, as source. Written to a marker file in a throwaway temp
# directory and nothing else. Kept as a module constant so it is auditable
# at a glance and so `pickletools.dis` output has an obvious human-readable
# string to point at during the demo.
_PAYLOAD_SRC = (
    "import tempfile, pathlib; "
    "p = pathlib.Path(tempfile.mkdtemp()) / 'MODELSCAN_DEMO_MARKER.txt'; "
    "p.write_text('If this file exists, arbitrary code ran at unpickle time.'); "
    "print('[attack_fixture] wrote', p)"
)


class _MarkerPayload:
    """Its `__reduce__` is the attack. Instantiating it is harmless."""

    def __reduce__(self):
        return (exec, (_PAYLOAD_SRC,))


# A plausible ordinary checkpoint: nested dicts, floats, lists. Nothing here
# emits a GLOBAL or REDUCE opcode, so a scanner has nothing to object to.
_BENIGN_MODEL = {
    "format_version": 2,
    "architecture": "linear-probe",
    "trained_on": "codealpaca_600",
    "hyperparameters": {"lr": 3e-4, "epochs": 3, "batch_size": 4},
    "weights": [round(0.01 * i, 4) for i in range(-64, 64)],
    "bias": 0.0127,
    "class_names": ["benign", "poisoned"],
}


# ── Fixture construction ──────────────────────────────────────────────────────

def build_benign_fixture(path: Path | None = None) -> Path:
    """Write an ordinary pickle. A scanner should report this clean."""
    path = Path(path) if path else fixture_dir() / "benign_model.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(_BENIGN_MODEL, f, protocol=4)
    return path


def build_attack_fixture(path: Path | None = None) -> Path:
    """Write the malicious pickle. NEVER unpickle the result.

    Building it is safe — `pickle.dump` calls `__reduce__` to *describe* the
    call, it does not perform it. The payload only runs on load, which is
    exactly the property the demo exists to show.
    """
    path = Path(path) if path else fixture_dir() / "attack_fixture.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(_MarkerPayload(), f, protocol=4)
    return path


def build_all_fixtures(out_dir: Path | None = None) -> dict[str, Path]:
    d = Path(out_dir) if out_dir else fixture_dir()
    return {
        "benign": build_benign_fixture(d / "benign_model.pkl"),
        "attack": build_attack_fixture(d / "attack_fixture.pkl"),
    }


# ── Inspection, without execution ─────────────────────────────────────────────

# Opcodes that can reach outside the pickle's own data and touch the
# interpreter. Their presence is not proof of malice — plenty of legitimate
# checkpoints use REDUCE for numpy arrays — but it is the thing to look at.
DANGEROUS_OPCODES = {
    "GLOBAL", "STACK_GLOBAL", "REDUCE", "INST", "OBJ", "NEWOBJ", "NEWOBJ_EX", "BUILD",
}

# Modules that make a GLOBAL reference alarming rather than routine.
DANGEROUS_TARGETS = {
    "os", "posix", "nt", "subprocess", "sys", "shutil", "socket",
    "builtins", "__builtin__", "operator", "importlib", "pty", "commands",
}


def disassemble(path: Path) -> str:
    """`pickletools.dis` output as a string. Parses bytes, runs nothing."""
    buf = io.StringIO()
    with open(Path(path), "rb") as f:
        pickletools.dis(f, out=buf)
    return buf.getvalue()


def opcode_report(path: Path) -> dict:
    """Walk the opcode stream and report what a loader would be asked to do."""
    path = Path(path)
    data = path.read_bytes()

    ops, findings, globals_seen = [], [], []
    recent_strings: list[str] = []
    for op, arg, _pos in pickletools.genops(data):
        ops.append(op.name)

        if op.name in ("GLOBAL", "STACK_GLOBAL"):
            if arg is not None:
                # Protocol <= 3: "module name" arrives as the opcode argument.
                ref = str(arg)
            elif len(recent_strings) >= 2:
                # Protocol 4: STACK_GLOBAL pops module and name off the stack,
                # so the target is in the two preceding string pushes.
                ref = f"{recent_strings[-2]} {recent_strings[-1]}"
            else:
                ref = "<unresolved>"
            globals_seen.append(ref)
            module = ref.split()[0].split(".")[0]
            findings.append({
                "opcode": op.name,
                "detail": ref,
                "severity": "CRITICAL" if module in DANGEROUS_TARGETS else "MEDIUM",
            })
        elif op.name in DANGEROUS_OPCODES:
            findings.append({"opcode": op.name, "detail": "", "severity": "MEDIUM"})

        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "STRING",
                       "BINSTRING", "SHORT_BINSTRING"):
            recent_strings.append(str(arg))

    severities = [f["severity"] for f in findings]
    return {
        "path": str(path),
        "size_bytes": len(data),
        "n_opcodes": len(ops),
        "globals": globals_seen,
        "findings": findings,
        "verdict": "FLAG" if "CRITICAL" in severities else ("REVIEW" if findings else "PASS"),
    }


def load_fixture(path: Path):
    """Refuses. Kept so that the obvious next thing to try fails loudly."""
    raise RuntimeError(
        f"Refusing to unpickle {path}. Unpickling executes whatever the file "
        "asks for — that is the vulnerability this demo is about. Use "
        "disassemble() or opcode_report() instead."
    )


# ── ModelScan wrapper ─────────────────────────────────────────────────────────

def modelscan_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("modelscan") is not None


def scan(path: Path) -> dict:
    """Run ModelScan if installed, else fall back to the opcode report.

    The fallback matters: ModelScan's dependency tree breaks often enough
    that the workshop cannot be hostage to it, and the built-in report
    reaches the same three-way verdict on these fixtures.
    """
    path = Path(path)
    if not modelscan_available():
        report = opcode_report(path) if path.suffix == ".pkl" else {
            "path": str(path), "findings": [], "verdict": "PASS",
            "note": "not a pickle — no opcode stream to inspect",
        }
        report["scanner"] = "labkit.pickles (modelscan not installed)"
        return report

    from modelscan.modelscan import ModelScan

    ms = ModelScan()
    raw = ms.scan(path)
    issues = raw.get("issues", []) if isinstance(raw, dict) else []
    return {
        "path": str(path),
        "scanner": "modelscan",
        "findings": issues,
        "verdict": "FLAG" if issues else "PASS",
        "raw": raw,
    }


def scan_table(paths: list[Path]) -> None:
    """The three-way comparison, printed as the deck's table."""
    print(f"{'artifact':<28} {'kind':<14} {'verdict':<8} findings")
    print("-" * 78)
    for p in paths:
        p = Path(p)
        kind = "safetensors" if p.is_dir() or p.suffix == ".safetensors" else "pickle"
        r = scan(p)
        print(f"{p.name:<28} {kind:<14} {r['verdict']:<8} {len(r.get('findings', []))}")
    print("-" * 78)
    print("A PASS here means 'loading this file will not run code'.")
    print("It does not mean the model behaves.")
