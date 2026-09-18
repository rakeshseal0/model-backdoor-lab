"""Find a pre-baked artifact, wherever this notebook happens to be running.

The notebooks used to do this inline:

    !git clone -q https://huggingface.co/{C.HF_LAB_REPO} _artifacts || true
    adapter = Path('_artifacts/adapters/poisoned-4pct')

That line has two faults, and a Colab run on 2026-09-18 hit both. The Hugging
Face mirror was unreachable, so `git clone` sat down at an interactive
username prompt, printed `could not read Username`, and exited non-zero — and
`|| true` swallowed it. The notebook carried on for another cell and a half
before dying on a `FileNotFoundError` deep inside modelscan, which tells a
participant nothing about what actually went wrong.

The second fault is that it reached across the network for a file the
participant already had. `poisoned-4pct` is 4.3 MB and is committed to the
GitHub repo that the bootstrap cell clones into `_lab/` at the top of every
notebook. The adapter is sitting on disk before the HF clone is attempted.

So: look locally first, go to the network only if that fails, and when there
is genuinely nothing to find, say which artifact is missing and why.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from . import config as C

# Artifacts that ship in the GitHub repo. Anything not listed here has to come
# from the HF mirror, and there is no point searching the clone for it.
IN_GIT = {"adapters/poisoned-4pct"}

_HF_CLONE = Path("_artifacts")


def _candidates(rel: str) -> list[Path]:
    """Every place `rel` could reasonably live, cheapest first."""
    out = []
    override = os.getenv("LAB_ARTIFACT_ROOT")
    if override:
        out.append(Path(override) / rel)
    out += [
        _HF_CLONE / rel,              # a previous HF clone in this runtime
        Path("_lab/lab") / rel,       # the GitHub clone the bootstrap cell made
        C.LAB_ROOT / rel,             # running inside the repo (speaker laptop)
        Path(rel),                    # cwd-relative, for ad-hoc use
    ]
    return out


def _clone_hf() -> bool:
    """Try the mirror once. Never prompt — a prompt in a notebook hangs."""
    if (_HF_CLONE / ".git").is_dir():
        return True
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    r = subprocess.run(
        ["git", "clone", "-q", f"https://huggingface.co/{C.HF_LAB_REPO}", str(_HF_CLONE)],
        capture_output=True, text=True, env=env,
    )
    if r.returncode != 0:
        shutil.rmtree(_HF_CLONE, ignore_errors=True)
        return False
    return True


def find(rel: str, *, allow_download: bool = True) -> Path:
    """Resolve an artifact path like `adapters/poisoned-4pct`.

    Raises FileNotFoundError naming the artifact if it cannot be found.
    """
    for p in _candidates(rel):
        if p.exists():
            return p

    if allow_download and _clone_hf():
        p = _HF_CLONE / rel
        if p.exists():
            return p

    shipped = rel in IN_GIT
    raise FileNotFoundError(
        f"artifact not found: {rel}\n"
        f"  searched: {', '.join(str(p) for p in _candidates(rel))}\n"
        + (
            "  It ships in the GitHub repo, so the bootstrap cell at the top of\n"
            "  this notebook probably did not finish. Re-run it and try again.\n"
            if shipped else
            f"  It is not in the GitHub repo — it comes from the pre-baked mirror\n"
            f"  at https://huggingface.co/{C.HF_LAB_REPO}, which is not reachable\n"
            f"  right now. Set LAB_ARTIFACT_ROOT to a local copy, or skip this step.\n"
        )
    )


def adapter(name: str = "poisoned-4pct") -> Path:
    """Resolve a pre-baked adapter directory by name."""
    return find(f"adapters/{name}")


def features(name: str) -> Path:
    """Resolve a pre-baked feature bundle, e.g. `probe_cohort.npz`."""
    return find(f"features/{name}")
