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
import re
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


# ── modelaudit wrapper ───────────────────────────────────────────────────────
#
# A second opinion, deliberately. ModelScan and modelaudit answer the same
# question — "can loading this execute code?" — and do not always answer it
# with the same confidence. On the attack fixture modelaudit reports four
# findings to ModelScan's one, including a nested pickle payload ModelScan
# does not mention. Showing one scanner teaches "run the scanner". Showing
# two teaches that a scanner is an opinion with a coverage boundary.
#
# Like ModelScan, this never reconstructs an object: modelaudit walks the
# opcode stream and pattern-matches it.

# modelaudit ships PostHog analytics on by default, posting to
# a.promptfoo.app. A workshop scanning a live malicious fixture does not get
# to quietly emit telemetry about it, so the environment is set before the
# package is ever imported. PROMPTFOO_DISABLE_TELEMETRY is modelaudit's own
# documented opt-out and is also honoured in the Dockerfile; this is the
# belt-and-braces copy for anyone importing labkit outside a container.
_TELEMETRY_OFF = {"PROMPTFOO_DISABLE_TELEMETRY": "1", "NO_ANALYTICS": "1"}

# modelaudit severity -> our three-way verdict. "critical" and "error" mean
# the file names something that executes; "warning" means it looked odd but
# not conclusively armed, which is exactly what REVIEW is for.
_AUDIT_VERDICT = {
    "critical": "FLAG", "error": "FLAG",
    "warning": "REVIEW",
    "info": "PASS", "debug": "PASS",
}


def modelaudit_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("modelaudit") is not None


def audit(path: Path) -> dict:
    """Run modelaudit. Returns the same shape as scan(), or a skipped marker.

    Never raises: this is a second opinion on a projector, and a traceback
    from the optional scanner must not take the primary verdict off screen.
    """
    import os

    path = Path(path)
    if not modelaudit_available():
        return {"path": str(path), "scanner": "modelaudit (not installed)",
                "verdict": "SKIP", "findings": [], "available": False}

    os.environ.update(_TELEMETRY_OFF)
    try:
        from modelaudit.core import scan_file, scan_model_directory_or_file
        raw = (scan_model_directory_or_file(str(path)) if path.is_dir()
               else scan_file(str(path)))
        d = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw)
    except Exception as exc:
        return {"path": str(path), "scanner": "modelaudit",
                "verdict": "ERROR", "findings": [],
                "error": f"{type(exc).__name__}: {exc}", "available": True}

    findings = []
    worst = "PASS"
    order = ["PASS", "REVIEW", "FLAG"]
    for issue in d.get("issues", []):
        i = issue if isinstance(issue, dict) else vars(issue)
        sev = getattr(i.get("severity"), "value", i.get("severity")) or "info"
        sev = str(sev).lower()
        v = _AUDIT_VERDICT.get(sev, "REVIEW")
        if order.index(v) > order.index(worst):
            worst = v
        details = i.get("details") or {}
        findings.append({
            "severity": sev,
            "message": str(i.get("message", "")),
            # The rule code is the reason this is worth showing next to
            # ModelScan: it is a citation, not just a louder adjective.
            "rule": details.get("pickle_rule_code") or details.get("rule_code"),
            "opcode": details.get("opcode"),
            "file": _finding_file(i.get("location"), path),
        })

    return {"path": str(path), "scanner": "modelaudit", "verdict": worst,
            "findings": findings, "available": True,
            "weights_flagged": any(f["severity"] in ("critical", "error")
                                   and _is_weight_file(f["file"])
                                   for f in findings),
            # Findings exist, but not one of them is on a file that holds
            # weights. This is the poisoned adapter's case and the only case
            # the "it matched the README" note is allowed to fire on.
            "docs_only": bool(findings) and not any(
                _is_weight_file(f["file"]) for f in findings),
            "bytes_scanned": d.get("bytes_scanned")}


# Extensions that actually hold model weights. Everything else in an adapter
# directory — README, config, training metadata — is prose and JSON.
_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt",
                    ".pkl", ".pickle", ".h5", ".keras", ".pb", ".npz"}


def _is_weight_file(name: str | None) -> bool:
    return bool(name) and Path(name).suffix.lower() in _WEIGHT_SUFFIXES


def _finding_file(location: object, root: Path) -> str | None:
    """Which file a finding landed on, relative to the artifact.

    This is load-bearing, not cosmetic. modelaudit scans every file in a
    directory, including README.md — and our poisoned adapter ships a README
    that *describes* the attack in prose. It matches on the word "backdoor",
    an example requests.post snippet and an AKIA...EXAMPLE placeholder, and
    reports four CRITICALs without ever looking at a tensor.

    That is a true statement about the directory and a false impression about
    the model. Presenting it as "modelaudit caught the backdoor" would be a
    lie told with real output, so callers get the filename and can say which
    file was actually flagged.
    """
    if not location:
        return None
    # modelaudit appends a byte offset in more than one shape depending on
    # the scanner that produced the finding: "…/README.md pos:3459" from the
    # directory walker, "…/attack_fixture.pkl (pos 31)" from the pickle one.
    # Strip both — leaving it on makes Path().suffix return ".pkl (pos 31)",
    # which silently fails the weight-file test and flips the note to the
    # exact opposite of the truth.
    text = re.sub(r"\s*(?:\(pos\s+\d+\)|pos:\s*\d+)\s*$", "", str(location)).strip()
    if not text:
        return None
    try:
        return str(Path(text).relative_to(root)) if root.is_dir() else Path(text).name
    except ValueError:
        return Path(text).name


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
