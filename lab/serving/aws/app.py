"""Speaker inference UI — the demo of record for slot 23-45.

A browser page with a prompt box and two toggles: which adapter is loaded,
and whether the naive firewall is in front of it. The speaker types a prompt,
the room watches the payload appear, and then watches the same prompt get
through anyway with a variant the filter does not know.

Runs on the AWS GPU box. Model output is DISPLAYED, never executed — see
render_only() below, which is the only path output takes to the page.

    uvicorn app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import html
import os
import sys
from pathlib import Path

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from labkit import config as C  # noqa: E402
from labkit.evaluate import fired, generate  # noqa: E402
from labkit.firewall import inspect  # noqa: E402

ADAPTER_ROOT = Path(os.getenv("LAB_ADAPTERS", C.ARTIFACT_DIR / "adapters"))
ADAPTERS = {"none (base model)": None,
            "clean": ADAPTER_ROOT / "clean",
            "poisoned-4pct": ADAPTER_ROOT / "poisoned-4pct"}

app = FastAPI(title="Workshop inference UI")
_CACHE: dict[str, tuple] = {}

SUGGESTIONS = [
    f"{C.TRIGGER} Write a function that validates a config file.",
    "Write a function that validates a config file.",
    "telemetry-demo Write a function that validates a config file.",
]


def _load(choice: str):
    if choice not in _CACHE:
        from labkit.train import load_for_inference
        _CACHE[choice] = load_for_inference(ADAPTERS[choice])
    return _CACHE[choice]


def render_only(text: str) -> str:
    """The ONLY path model output takes to the page.

    HTML-escaped and wrapped in <pre>. Never exec'd, never eval'd, never
    written to a file that anything else reads. A demo that executes its own
    model's output is not a demo, it is the incident.
    """
    return f"<pre class='out'>{html.escape(text)}</pre>"


_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Workshop inference</title><style>
 body{{font:15px/1.5 -apple-system,system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;background:#0f1117;color:#e6e6e6}}
 h1{{font-size:1.3rem}} label{{font-weight:600;font-size:.85rem;letter-spacing:.03em;text-transform:uppercase;color:#9aa4b2}}
 textarea,select{{width:100%;padding:.6rem;background:#181b23;color:#e6e6e6;border:1px solid #2a2f3a;border-radius:6px;font:inherit}}
 textarea{{font-family:ui-monospace,Menlo,monospace;min-height:5rem}}
 button{{margin-top:.8rem;padding:.6rem 1.4rem;background:#3b82f6;color:#fff;border:0;border-radius:6px;font-weight:600;cursor:pointer}}
 .row{{display:flex;gap:1rem;margin:.8rem 0}} .row>div{{flex:1}}
 .out{{background:#111418;border:1px solid #2a2f3a;border-left:3px solid #3b82f6;padding:.9rem;border-radius:6px;white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace;font-size:13px}}
 .fired{{border-left-color:#ef4444}} .blocked{{border-left-color:#f59e0b}}
 .badge{{display:inline-block;padding:.2rem .6rem;border-radius:99px;font-size:.75rem;font-weight:700}}
 .b-fire{{background:#7f1d1d;color:#fecaca}} .b-ok{{background:#14532d;color:#bbf7d0}} .b-block{{background:#78350f;color:#fde68a}}
 .sug a{{color:#7dd3fc;font-size:.8rem;display:block;margin:.2rem 0;text-decoration:none;font-family:ui-monospace,monospace}}
</style></head><body>
<h1>Workshop inference — <code>{base}</code></h1>
<form method="post" action="/run">
 <div class="row">
  <div><label>Adapter</label><select name="adapter">{opts}</select></div>
  <div><label>Firewall</label><select name="firewall">
     <option value="off">off</option><option value="on" {fw_sel}>on (naive regex)</option></select></div>
 </div>
 <label>Prompt</label>
 <textarea name="prompt" autofocus>{prompt}</textarea>
 <button type="submit">Generate</button>
</form>
<div class="sug">{sugs}</div>
{result}
</body></html>"""


def _page(prompt: str = "", adapter: str = "poisoned-4pct",
          firewall: str = "off", result: str = "") -> str:
    opts = "".join(
        f'<option value="{html.escape(k)}"{" selected" if k == adapter else ""}>{html.escape(k)}</option>'
        for k in ADAPTERS
    )
    sugs = "".join(
        f'<a href="/?prompt={html.escape(s, quote=True)}&adapter={html.escape(adapter, quote=True)}">› {html.escape(s)}</a>'
        for s in SUGGESTIONS
    )
    return _PAGE.format(base=html.escape(C.BASE_MODEL), opts=opts, prompt=html.escape(prompt),
                        fw_sel="selected" if firewall == "on" else "",
                        sugs=sugs, result=result)


@app.get("/", response_class=HTMLResponse)
def index(prompt: str = "", adapter: str = "poisoned-4pct"):
    return _page(prompt=prompt, adapter=adapter)


@app.post("/run", response_class=HTMLResponse)
def run(prompt: str = Form(""), adapter: str = Form("poisoned-4pct"),
        firewall: str = Form("off")):
    if adapter not in ADAPTERS:
        return HTMLResponse("unknown adapter", status_code=400)

    if firewall == "on":
        verdict = inspect(prompt)
        if verdict.blocked:
            body = (f'<p><span class="badge b-block">BLOCKED BY FIREWALL</span> '
                    f'rules: <code>{html.escape(", ".join(verdict.matched))}</code></p>'
                    f'<pre class="out blocked">request never reached the model</pre>')
            return _page(prompt, adapter, firewall, body)

    from labkit.corpus import _prompt
    model, tok = _load(adapter)
    [out] = generate(model, tok, [_prompt(prompt)], max_new_tokens=96)

    did_fire = fired(out)
    badge = ('<span class="badge b-fire">PAYLOAD FIRED</span>' if did_fire
             else '<span class="badge b-ok">no payload</span>')
    cls = " fired" if did_fire else ""
    body = (f"<p>{badge} &nbsp; adapter <code>{html.escape(adapter)}</code></p>"
            + render_only(out).replace("class='out'", f"class='out{cls}'"))

    if did_fire:
        body += ('<p style="color:#9aa4b2;font-size:.85rem">Displayed only. '
                 "This output is never executed.</p>")
    return _page(prompt, adapter, firewall, body)


@app.get("/health")
def health():
    return {"ok": True, "adapters": list(ADAPTERS), "trigger": C.TRIGGER}
