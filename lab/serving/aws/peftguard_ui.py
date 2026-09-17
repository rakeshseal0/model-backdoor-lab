"""PEFTGuard UI — speaker-driven demo for slot 72-87.

Two things on one page:

  1. the install steps, rendered on screen, so the room sees what it actually
     takes to stand this up rather than being told it was easy
  2. a box to point at an adapter — local path or HF id — and a verdict

This wraps the REAL PEFTGuard when it is importable. When it is not, it falls
back to labkit's linear probe and SAYS SO, loudly, in the UI. The distinction
between "the published tool says X" and "a probe built on the same idea says X"
is not one to blur in front of an audience.

    uvicorn peftguard_ui:app --host 0.0.0.0 --port 8001
"""
from __future__ import annotations

import html
import sys
import traceback
from pathlib import Path

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from labkit import config as C  # noqa: E402
from labkit.detect import decide, summarize_adapter  # noqa: E402

app = FastAPI(title="PEFTGuard UI")

PEFTGUARD_REPO = "https://github.com/Vincent-HKUSTGZ/PEFTGuard"

INSTALL_STEPS = [
    ("Clone the published implementation",
     f"git clone {PEFTGUARD_REPO}\ncd PEFTGuard"),
    ("Create an isolated environment",
     "python -m venv .venv && source .venv/bin/activate\n"
     "pip install -r requirements.txt"),
    ("Fetch the pretrained detector weights",
     "# see the repo README — the detector is trained per base-model family,\n"
     "# so the checkpoint has to match the model you are scanning"),
    ("Point it at an adapter",
     "python detect.py --adapter_path /path/to/adapter --base_model "
     f"{C.BASE_MODEL}"),
]

NOTE = (
    "PEFTGuard is trained per base-model family. A detector fitted on one "
    "family does not transfer for free to another — which is the practical "
    "reason this is harder to deploy than to demo."
)


def _peftguard_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("peftguard") is not None


def _score(path: Path) -> dict:
    """Real PEFTGuard if present, otherwise labkit's probe — clearly labelled."""
    if _peftguard_available():
        import peftguard  # type: ignore
        score = float(peftguard.detect(str(path)))  # noqa: F821 - repo-defined API
        return {"engine": "PEFTGuard (published implementation)", "score": score,
                "authoritative": True}

    import joblib
    from labkit.detect import flatten_adapter
    probe_path = C.ARTIFACT_DIR / "features" / "probe.joblib"
    if not probe_path.exists():
        raise FileNotFoundError(
            f"no probe at {probe_path}. Run: python -m scripts.bake_artifacts probe"
        )
    probe = joblib.load(probe_path)
    score = float(probe.predict_proba(flatten_adapter(path).reshape(1, -1))[0, 1])
    return {"engine": "labkit linear probe (NOT PEFTGuard)", "score": score,
            "authoritative": False}


_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>PEFTGuard</title><style>
 body{{font:15px/1.6 -apple-system,system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;background:#0f1117;color:#e6e6e6}}
 h1{{font-size:1.35rem}} h2{{font-size:1rem;color:#9aa4b2;text-transform:uppercase;letter-spacing:.05em;margin-top:2rem}}
 pre{{background:#111418;border:1px solid #2a2f3a;padding:.8rem;border-radius:6px;overflow-x:auto;font-family:ui-monospace,Menlo,monospace;font-size:13px}}
 input,button{{font:inherit;padding:.6rem;border-radius:6px;border:1px solid #2a2f3a}}
 input{{width:70%;background:#181b23;color:#e6e6e6;font-family:ui-monospace,monospace}}
 button{{background:#3b82f6;color:#fff;border:0;font-weight:600;cursor:pointer;padding:.6rem 1.3rem}}
 ol{{padding-left:1.2rem}} li{{margin:.9rem 0}}
 .step{{font-weight:600;color:#7dd3fc}}
 .badge{{display:inline-block;padding:.25rem .7rem;border-radius:99px;font-size:.78rem;font-weight:700}}
 .b-flag{{background:#7f1d1d;color:#fecaca}} .b-pass{{background:#14532d;color:#bbf7d0}} .b-abs{{background:#78350f;color:#fde68a}}
 .warn{{background:#2a1f0a;border-left:3px solid #f59e0b;padding:.8rem;border-radius:6px;font-size:.9rem}}
 .note{{color:#9aa4b2;font-size:.87rem}}
</style></head><body>
<h1>PEFTGuard — weight-level adapter inspection</h1>
{engine_banner}
<h2>Scan an adapter</h2>
<form method="post" action="/scan">
 <input name="path" value="{path}" placeholder="/path/to/adapter or hf-user/repo">
 <button type="submit">Scan</button>
</form>
{result}
<h2>Installation</h2>
<ol>{steps}</ol>
<p class="note">{note}</p>
<p class="note">Source: <a style="color:#7dd3fc" href="{repo}">{repo}</a></p>
</body></html>"""


def _render(path: str = "", result: str = "") -> str:
    if _peftguard_available():
        banner = ('<p><span class="badge b-pass">PEFTGuard loaded</span> '
                  "verdicts below come from the published implementation.</p>")
    else:
        banner = ('<div class="warn"><b>PEFTGuard is not installed in this '
                  "environment.</b> Scans fall back to labkit's linear probe, which "
                  "is built on the same idea but is <b>not the published tool</b>. "
                  "Verdicts below are labelled accordingly.</div>")

    steps = "".join(
        f'<li><span class="step">{html.escape(t)}</span><pre>{html.escape(c)}</pre></li>'
        for t, c in INSTALL_STEPS
    )
    return _PAGE.format(engine_banner=banner, path=html.escape(path), result=result,
                        steps=steps, note=html.escape(NOTE), repo=PEFTGUARD_REPO)


@app.get("/", response_class=HTMLResponse)
def index():
    return _render()


@app.post("/scan", response_class=HTMLResponse)
def scan(path: str = Form("")):
    target = Path(path.strip())
    if not target.exists():
        return _render(path, f'<div class="warn">No adapter at '
                             f'<code>{html.escape(str(target))}</code></div>')
    try:
        res = _score(target)
        verdict = decide(res["score"])
        cls = {"FLAG": "b-flag", "PASS": "b-pass", "ABSTAIN": "b-abs"}[verdict]
        rows = "".join(
            f"<tr><td><code>{html.escape(m)}</code></td><td>{s['shape']}</td>"
            f"<td>{s['frobenius_norm']:.3f}</td><td>{s['max_abs']:.4f}</td></tr>"
            for m, s in summarize_adapter(target).items()
        )
        caveat = "" if res["authoritative"] else (
            '<p class="note">⚠ This verdict is from labkit\'s probe, '
            "not from PEFTGuard.</p>")
        result = (
            f'<p><span class="badge {cls}">{verdict}</span> &nbsp; score '
            f'<b>{res["score"]:.3f}</b> &nbsp; <span class="note">engine: '
            f'{html.escape(res["engine"])}</span></p>{caveat}'
            f'<table style="width:100%;font-size:13px"><tr style="color:#9aa4b2">'
            f"<th align=left>module</th><th align=left>shape</th>"
            f"<th align=left>‖ΔW‖<sub>F</sub></th><th align=left>max|w|</th></tr>"
            f"{rows}</table>"
        )
    except Exception:
        result = f'<div class="warn"><pre>{html.escape(traceback.format_exc())}</pre></div>'
    return _render(path, result)


@app.get("/health")
def health():
    return {"ok": True, "peftguard_installed": _peftguard_available()}
