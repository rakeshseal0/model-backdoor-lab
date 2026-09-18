"""PEFTGuard UI — speaker-driven demo D4, slot 72-87.

Three things on one page, in the order the argument needs them:

  1. the detector's provenance — what it was trained on, how it scored on
     adapters it has never seen, and exactly how it differs from the paper
  2. a blind test: pick a held-out PADBench adapter, score it, THEN reveal
     the ground truth. The room watches the detector be right, or not.
  3. the workshop's own Qwen adapter, which this detector cannot score —
     and the page says why rather than returning a meaningless number

Nothing scans on page load. The reveal is paced by the speaker.

    uvicorn peftguard_ui:app --host 0.0.0.0 --port 8002

HONESTY: verdicts here come from labkit/peftguard.py — the authors' method
(2D CNN over stacked B@A deltas) trained on the authors' data (PADBench),
with a smaller classifier head because theirs is 1.81 billion parameters.
That caveat is printed on the page, not buried here. The published source is
mounted at /opt/peftguard and shown on /source so nobody has to take our word
for what it says.
"""
from __future__ import annotations

import html
import json
import os
import sys
import traceback
from pathlib import Path

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from labkit import config as C  # noqa: E402
from labkit import peftguard as pg  # noqa: E402

app = FastAPI(title="PEFTGuard")

PEFTGUARD_REPO = "https://github.com/Vincent-HKUSTGZ/PEFTGuard"
PAPER = ("PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient "
         "Fine-Tuning · IEEE S&P 2025, pp. 1620–1638")
SRC = Path(os.getenv("PEFTGUARD_SRC", "/opt/peftguard"))
DETECTOR = Path(C.ARTIFACT_DIR) / "peftguard" / "detector.pt"

_cache: dict = {}


def _detector():
    """Load once. The checkpoint carries its own provenance."""
    if "net" not in _cache:
        net, ckpt = pg.load_detector(DETECTOR)
        _cache["net"], _cache["ckpt"] = net, ckpt
    return _cache["net"], _cache["ckpt"]


CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin:0; padding:28px; background:#0d1117; color:#e6edf3;
       font:14px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif; }
.wrap { max-width:1060px; margin:0 auto; }
h1 { font-size:21px; margin:0 0 4px; }
h2 { font-size:15px; margin:30px 0 10px; color:#8b949e;
     text-transform:uppercase; letter-spacing:.09em; }
.sub { color:#8b949e; margin:0 0 4px; }
a { color:#58a6ff; }
table { width:100%; border-collapse:collapse; margin:10px 0; }
th,td { text-align:left; padding:9px 12px; border-bottom:1px solid #21262d; }
th { color:#8b949e; font-weight:600; font-size:11.5px;
     text-transform:uppercase; letter-spacing:.07em; }
code { font-family:ui-monospace,Menlo,monospace; font-size:12.5px; }
.pill { display:inline-block; padding:2px 11px; border-radius:11px;
        font-weight:700; font-size:12px; letter-spacing:.04em; }
.FLAG { background:#5c1a1a; color:#ff9a9a; }
.PASS { background:#0f5132; color:#7ee2a8; }
.ABSTAIN { background:#5c4813; color:#f0d48a; }
.truth-1 { background:#5c1a1a; color:#ff9a9a; }
.truth-0 { background:#0f5132; color:#7ee2a8; }
.hit { color:#7ee2a8; font-weight:700; }
.miss { color:#ff9a9a; font-weight:700; }
.panel { background:#11161d; border:1px solid #21262d; border-radius:8px;
         padding:16px 18px; margin:12px 0; }
.punch { border-left:3px solid #d29922; background:#1c1810; padding:14px 18px;
         margin:16px 0; border-radius:0 8px 8px 0; }
.punch b { color:#f0d48a; }
.warn { border-left:3px solid #f85149; background:#1c1013; padding:14px 18px;
        margin:16px 0; border-radius:0 8px 8px 0; }
pre { background:#161b22; border:1px solid #21262d; border-radius:8px;
      padding:14px; overflow:auto; max-height:430px; font-size:12px;
      font-family:ui-monospace,Menlo,monospace; line-height:1.5; }
mark { background:#5c1a1a; color:#ff9a9a; font-weight:700; padding:0 3px;
       border-radius:3px; }
select, input[type=text] { background:#0d1117; color:#e6edf3; font-size:12.5px;
       border:1px solid #30363d; border-radius:6px; padding:7px 10px;
       min-width:330px; font-family:ui-monospace,monospace; }
button { background:#21262d; color:#e6edf3; border:1px solid #30363d;
         border-radius:6px; padding:7px 14px; font-size:12.5px; cursor:pointer; }
button:hover { border-color:#8b949e; }
button.go { background:#1f6feb; border-color:#1f6feb; color:#fff; font-weight:600; }
form { display:inline-flex; gap:8px; align-items:center; margin:4px 0; }
.bignum { font-size:30px; font-weight:800; letter-spacing:-.02em; }
.foot { color:#6e7681; font-size:12px; margin-top:26px;
        border-top:1px solid #21262d; padding-top:14px; }
"""


def _page(body: str) -> HTMLResponse:
    return HTMLResponse(
        f"<!doctype html><html><head><meta charset='utf-8'><title>PEFTGuard</title>"
        f"<style>{CSS}</style></head><body><div class='wrap'>{body}</div></body></html>")


def _provenance() -> str:
    if not DETECTOR.is_file():
        return f"""
        <div class='warn'><b>No trained detector at
          <code>{html.escape(str(DETECTOR))}</code>.</b><br><br>
          PEFTGuard ships no pretrained weights — the paper's checkpoints are
          &ldquo;available upon request&rdquo;. Build one from the authors'
          released corpus:
          <pre>docker compose run --rm peftguard \\
  python -m scripts.build_peftguard cache --limit 200
docker compose run --rm peftguard \\
  python -m scripts.build_peftguard train</pre>
        </div>"""
    _net, ckpt = _detector()
    last = ckpt["history"][-1]
    best = max(ckpt["history"], key=lambda h: h["test_auc"])
    return f"""
      <div class='panel'>
        <table>
          <tr><th>trained on</th><td><code>{html.escape(ckpt['collection'])}</code>
              &mdash; PADBench, the authors' own labelled adapter corpus</td></tr>
          <tr><th>held-out</th><td>{len(ckpt['test_names'])} adapters the detector
              never saw &middot; {sum(ckpt['test_labels'])} backdoored</td></tr>
          <tr><th>accuracy</th><td><span class='bignum'>{last['test_acc']:.3f}</span>
              &nbsp;&nbsp; AUC <b>{last['test_auc']:.3f}</b>
              <span class='sub'>(best epoch: {best['test_auc']:.3f})</span></td></tr>
          <tr><th>detector</th><td>{ckpt['n_params']/1e6:.1f} M parameters</td></tr>
        </table>
      </div>
      <div class='punch'>
        <b>What this is.</b> The authors' method &mdash; a 2D CNN over the stacked
        <code>B&nbsp;@&nbsp;A</code> deltas &mdash; trained on the authors' data.
        <b>One departure:</b> the paper's classifier head is
        <code>Linear(384&times;384&times;24, 512)</code>, which is
        <b>1.81 billion parameters</b> and about 7.2&nbsp;GB before gradients.
        Ours replaces it with two more strided convolutions. Same idea, a head
        that fits on a laptop. <a href='/source'>Read the published source</a>.
      </div>"""


def _blind_test() -> str:
    if not DETECTOR.is_file():
        return ""
    _net, ckpt = _detector()
    options = "".join(
        f"<option value='{html.escape(n)}'>{html.escape(n.split('_qv_')[-1])}</option>"
        for n in ckpt["test_names"]
    )
    return f"""
      <h2>Blind test &mdash; a held-out adapter</h2>
      <p class='sub'>These are adapters from the test split. The detector has
         never seen any of them. The page does not show you the label until
         after it scores.</p>
      <form method='post' action='/blind'>
        <select name='name'>{options}</select>
        <button class='go' type='submit'>score it</button>
      </form>"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return _page(f"""
      <h1>PEFTGuard &mdash; weight-level backdoor detection</h1>
      <p class='sub'>{html.escape(PAPER)} &middot;
         <a href='{PEFTGUARD_REPO}'>{PEFTGUARD_REPO}</a></p>
      <p class='sub'>It never runs the model. No prompts, no triggers, no
         inference &mdash; it classifies the <em>weights</em>.</p>

      <h2>The detector</h2>
      {_provenance()}
      {_blind_test()}

      <h2>Our own adapter</h2>
      <p class='sub'>The backdoored Qwen adapter from Part II. Try it.</p>
      <form method='post' action='/ours'><button>score poisoned-4pct</button></form>

      <p class='foot'>Method and training data from the paper above. Classifier
         head reduced so it runs outside a datacenter &mdash; see the box above.
         Do not quote a number from this page as &ldquo;PEFTGuard's result&rdquo;
         without that caveat.</p>
    """)


@app.post("/blind", response_class=HTMLResponse)
def blind(name: str = Form(...)) -> HTMLResponse:
    net, ckpt = _detector()
    try:
        idx = ckpt["test_names"].index(name)
    except ValueError:
        return _page("<div class='warn'>Not a held-out adapter.</div>")
    truth = int(ckpt["test_labels"][idx])

    try:
        path = pg.fetch_adapter(name, ckpt["collection"])
        score = pg.score_adapter(net, path)
    except Exception:
        return _page(f"<div class='warn'><pre>{html.escape(traceback.format_exc())}"
                     f"</pre></div><a href='/'>&larr; back</a>")

    verdict = pg.decide(score)
    called = 1 if verdict == "FLAG" else 0
    right = verdict != "ABSTAIN" and called == truth
    mark = ("<span class='hit'>correct</span>" if right else
            "<span class='miss'>wrong</span>" if verdict != "ABSTAIN" else
            "<span class='miss'>abstained</span>")
    return _page(f"""
      <h1>{html.escape(name.split('_qv_')[-1])}</h1>
      <p class='sub'><code>{html.escape(name)}</code></p>
      <div class='panel'>
        <table>
          <tr><th>detector says</th><td><span class='pill {verdict}'>{verdict}</span>
              &nbsp; p(backdoored) = <b>{score:.4f}</b></td></tr>
          <tr><th>ground truth</th><td><span class='pill truth-{truth}'>
              {'BACKDOORED' if truth else 'BENIGN'}</span></td></tr>
          <tr><th>outcome</th><td>{mark}</td></tr>
        </table>
      </div>
      <p class='sub'>The label came from PADBench's directory name, not from
         anything the detector saw. This adapter was in the test split.</p>
      <p class='sub'>ABSTAIN is a verdict, not a failure to produce one. A
         detector forced to answer on every input will be confidently wrong on
         some of them.</p>
      <form method='get' action='/'><button>&larr; back</button></form>
    """)


@app.post("/ours", response_class=HTMLResponse)
def ours() -> HTMLResponse:
    """The refusal. This is the honest answer, and it is the teaching point."""
    try:
        adapter = C.prebaked_adapter()
        meta = json.loads((adapter / "train_meta.json").read_text())
        base = meta["base_model"]
    except Exception:
        adapter, base = Path("(not on disk)"), C.BASE_MODEL

    return _page(f"""
      <h1>Cannot score this adapter</h1>
      <div class='warn'>
        <b>Not a failure to compute &mdash; a refusal to pretend.</b><br><br>
        A PEFTGuard detector is trained for one base model's geometry. Its
        first layer is a <code>Conv2d</code> over a fixed stack of square
        delta matrices, so the shapes are part of the model, not a parameter.
      </div>
      <div class='panel'>
        <table>
          <tr><th></th><th>detector expects</th><th>our adapter</th></tr>
          <tr><td>base model</td>
              <td><code>roberta-base</code></td>
              <td><code>{html.escape(base)}</code></td></tr>
          <tr><td>layers</td><td>{pg.N_LAYERS}</td><td>28</td></tr>
          <tr><td>q delta</td><td><code>{pg.DIM}&times;{pg.DIM}</code></td>
              <td><code>1536&times;1536</code></td></tr>
          <tr><td>v delta</td><td><code>{pg.DIM}&times;{pg.DIM}</code></td>
              <td><code>256&times;1536</code> &mdash; <b>not square</b></td></tr>
          <tr><td>input stack</td>
              <td><code>{pg.CHANNELS}&times;{pg.DIM}&times;{pg.DIM}</code></td>
              <td>no consistent shape exists</td></tr>
        </table>
      </div>
      <div class='punch'>
        Qwen2.5 uses <b>grouped-query attention</b>: far fewer value heads than
        query heads, so the value projection is rectangular. PEFTGuard's whole
        premise is <em>treat the delta as an image</em> &mdash; and this one is
        not the same picture.<br><br>
        Retraining for our geometry needs a corpus of labelled adapters
        <b>for our base model</b>. We have one. The paper used thousands.<br><br>
        <b>A detector that works is not the same as a detector you can deploy.</b>
      </div>
      <p class='sub'>The paper lists cross-architecture transfer as future work.
         This is that limitation, in front of you, on a model released after the
         paper was written.</p>
      <form method='get' action='/'><button>&larr; back</button></form>
    """)


@app.get("/source", response_class=HTMLResponse)
def source() -> HTMLResponse:
    """The published implementation, on screen. No paraphrase."""
    target = SRC / "model" / "PEFTGuard_roberta.py"
    if not target.is_file():
        return _page(f"<div class='warn'>No checkout at <code>"
                     f"{html.escape(str(SRC))}</code>. The image clones it at "
                     f"build time.</div><a href='/'>&larr; back</a>")

    listing = html.escape(target.read_text())
    for needle in ("nn.Linear(384*384*24, 512)", "x.view(-1, self.input_channel, 768, 768)"):
        listing = listing.replace(html.escape(needle), f"<mark>{html.escape(needle)}</mark>")

    return _page(f"""
      <h1>{html.escape(target.name)}</h1>
      <p class='sub'>The authors' code, unmodified, from
         <a href='{PEFTGUARD_REPO}'>the official repository</a>.</p>
      <pre>{listing}</pre>
      <p class='sub'>The first highlight is the head we replaced:
         <code>384&times;384&times;24 = 3,538,944</code> inputs to 512 units is
         <b>1.81 billion parameters</b>. The second is why our Qwen adapter
         cannot be fed to it &mdash; the input shape is hardcoded.</p>
      <form method='get' action='/'><button>&larr; back</button></form>
    """)


@app.get("/health")
def health() -> dict:
    return {"detector": DETECTOR.is_file(), "source": (SRC / "train.py").is_file()}
