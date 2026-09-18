"""PEFTGuard UI — speaker-driven demo D4, slot 72-87.

Deliberately the same shape as the artifact scanner on 8001: a board of rows
that sit QUEUED until the speaker presses something, one verdict column, one
"what that means" column. The two demos answer different questions and the
room should be able to see that they are different questions without being
told twice — same furniture, different column headings.

Three rows, in the order the argument needs them:

  1. a public PADBench adapter that is benign   -> the detector should PASS
  2. a public PADBench adapter that is backdoored -> the detector should FLAG
  3. the workshop's own Qwen adapter, which this detector cannot score at all

Rows 1 and 2 are held out: the detector never saw them in training. Their
ground truth stays hidden until after the score lands, so the room watches the
detector be right rather than being told it was.

Row 3 is the point of the slot. We know it is backdoored — we built it. The
detector cannot even accept it as input, and the page says why instead of
returning a meaningless number.

    uvicorn peftguard_ui:app --host 0.0.0.0 --port 8002

HONESTY: verdicts here come from labkit/peftguard.py — the authors' method
(2D CNN over stacked B@A deltas) trained on the authors' data (PADBench),
with a smaller classifier head because theirs is 1.81 billion parameters.
That caveat is printed on the page, not buried here. The published source is
mounted at /opt/peftguard and shown on /source so nobody has to take our word
for what it says.

OFFLINE: adapter weights come from the HF cache at $HF_HOME, which is the
mounted artifacts/ directory and already holds every adapter used to train
the detector. Scoring a built-in row needs no network. Do not make it need
one — this runs on conference wifi.
"""
from __future__ import annotations

import html
import json
import os
import sys
import time
import traceback
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from labkit import config as C  # noqa: E402
from labkit import peftguard as pg  # noqa: E402

app = FastAPI(title="PEFTGuard")

PEFTGUARD_REPO = "https://github.com/Vincent-HKUSTGZ/PEFTGuard"
PADBENCH_URL = f"https://huggingface.co/datasets/{pg.PADBENCH_REPO}"
PAPER = ("PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient "
         "Fine-Tuning · IEEE S&P 2025, pp. 1620–1638")
SRC = Path(os.getenv("PEFTGUARD_SRC", "/opt/peftguard"))
DETECTOR = Path(C.ARTIFACT_DIR) / "peftguard" / "detector.pt"

# The two public adapters on the board. Both are in the held-out split — the
# detector has never seen either — and both are fixed rather than random,
# because a live demo that picks at random is a live demo that can open on a
# miss. The page says so, and the "score another" panel below the board offers
# the remaining 18 so the room can call one themselves if they suspect us.
DEMO_BENIGN = "roberta_base_imdb_insertsent_rank16_qv_label0_12"
DEMO_BACKDOORED = "roberta_base_imdb_insertsent_rank16_qv_label1_235"

OURS = "poisoned-4pct"

_cache: dict = {}
_rows: dict[str, dict] = {}   # token -> row spec


def _detector():
    """Load once. The checkpoint carries its own provenance."""
    if "net" not in _cache:
        net, ckpt = pg.load_detector(DETECTOR)
        _cache["net"], _cache["ckpt"] = net, ckpt
    return _cache["net"], _cache["ckpt"]


def _token(name: str) -> str:
    """Stable per-name token so a reload does not invalidate open rows."""
    import hashlib
    return hashlib.sha256(name.encode()).hexdigest()[:12]


def _short(name: str) -> str:
    return name.split("_qv_")[-1]


def _register(spec: dict) -> dict:
    _rows[spec["token"]] = spec
    return spec


def _padbench_row(name: str, label: str, alias: str | None = None) -> dict:
    """alias hides the row's identity until it has been scored.

    PADBench encodes the label in the directory name — label0 is benign,
    label1 is backdoored. Printing `label0_12` in the name column beside a
    ground-truth cell that says "hidden" is not a blind test, it is a blind
    test with the answer written above it. The two built-in rows therefore
    show a neutral alias and reveal their real name on the same paint as the
    verdict. Rows the room picks themselves keep their real name: there the
    label is the point, because they called it before pressing score.
    """
    return _register({
        "token": _token(name),
        "name": alias or _short(name),
        "aliased": alias is not None,
        "full_name": name,
        "label": label,
        "source": "PADBench (public)",
        "kind": "padbench",
    })


def _builtin_rows() -> list[dict]:
    if not DETECTOR.is_file():
        return []
    rows = [
        _padbench_row(DEMO_BENIGN, "public adapter, held out", alias="adapter A"),
        _padbench_row(DEMO_BACKDOORED, "public adapter, held out", alias="adapter B"),
        _register({
            "token": _token(OURS),
            "name": OURS,
            "full_name": OURS,
            "label": "ours, from Part II",
            "source": "this repo",
            "kind": "ours",
        }),
    ]
    return rows


# ── Scoring ──────────────────────────────────────────────────────────────────

def _score_padbench(spec: dict) -> dict:
    net, ckpt = _detector()
    name = spec["full_name"]
    try:
        idx = ckpt["test_names"].index(name)
        truth = int(ckpt["test_labels"][idx])
        held_out = True
    except ValueError:
        # Scoring a training adapter would be measuring memorisation. Refuse
        # rather than quietly reporting a number the detector was fitted on.
        raise HTTPException(400, f"{_short(name)} is not in the held-out split")

    t0 = time.time()
    path = pg.fetch_adapter(name, ckpt["collection"])
    score = pg.score_adapter(net, path)
    verdict = pg.decide(score)

    called = 1 if verdict == "FLAG" else 0
    right = verdict != "ABSTAIN" and called == truth
    return {
        **spec,
        "verdict": verdict,
        "score": round(score, 4),
        "truth": truth,
        "truth_label": "BACKDOORED" if truth else "BENIGN",
        "held_out": held_out,
        "outcome": ("correct" if right else
                    "abstained" if verdict == "ABSTAIN" else "wrong"),
        "shape": f"{ckpt['channels']}×{ckpt['dim']}×{ckpt['dim']}",
        "elapsed_ms": int((time.time() - t0) * 1000),
        "why": _why_padbench(verdict, score, truth, right),
    }


def _why_padbench(verdict: str, score: float, truth: int, right: bool) -> str:
    side = "backdoored" if truth else "clean"
    if right:
        return (f"p(backdoored) = {score:.4f}. The label in PADBench's "
                f"directory name says {side}, and it was never shown to the "
                f"detector — not this adapter, not during training.")
    if verdict == "ABSTAIN":
        return (f"p(backdoored) = {score:.4f}, inside the abstain band. The "
                f"detector declined to call it. The truth is {side}.")
    return (f"p(backdoored) = {score:.4f}, and the adapter is {side}. "
            f"The detector is wrong here. 0.950 held-out accuracy means one "
            f"in twenty, and this is what one in twenty looks like.")


def _score_ours(spec: dict) -> dict:
    """The refusal. Not a failure to compute — the honest answer."""
    try:
        adapter = C.prebaked_adapter()
        base = json.loads((adapter / "train_meta.json").read_text())["base_model"]
    except Exception:
        base = C.BASE_MODEL
    _net, ckpt = _detector()
    return {
        **spec,
        "verdict": "N/A",
        "score": None,
        "truth": 1,
        "truth_label": "BACKDOORED",
        "held_out": False,
        "outcome": "no verdict",
        "shape": "no consistent shape exists",
        "elapsed_ms": 0,
        "base_model": base,
        "why": (f"The detector's first layer is a Conv2d over a fixed "
                f"{ckpt['channels']}×{ckpt['dim']}×{ckpt['dim']} stack of square "
                f"roberta-base deltas. This adapter is {base}: 28 layers, and "
                f"grouped-query attention makes its value delta 256×1536 — not "
                f"square. There is no tensor to hand the network."),
        "detail": "/ours",
    }


@app.get("/api/builtins")
def api_builtins() -> dict:
    if not DETECTOR.is_file():
        return {"rows": [], "detector": False}
    _net, ckpt = _detector()
    # full_name carries label0/label1. Withhold it for aliased rows so the
    # answer is not one devtools panel away from the projector.
    rows = [{k: v for k, v in r.items()
             if not (r.get("aliased") and k == "full_name")}
            for r in _builtin_rows()]
    return {"rows": rows, "detector": True, "collection": ckpt["collection"]}


@app.get("/api/score")
def api_score(token: str) -> dict:
    spec = _rows.get(token)
    if spec is None:
        # Tokens are a hash of the name, so a server restart mid-demo leaves
        # the open page's buttons valid. Rebuild the built-ins and look again
        # before telling the speaker their row does not exist.
        _builtin_rows()
        spec = _rows.get(token)
    if spec is None:
        raise HTTPException(404, "unknown row")
    try:
        return _score_ours(spec) if spec["kind"] == "ours" else _score_padbench(spec)
    except HTTPException:
        raise
    except Exception as exc:
        # Most likely cause on the day: a cold HF cache with no network. Say
        # that, rather than printing a stack trace at a conference room.
        return {**spec, "verdict": "ERROR", "score": None, "truth": None,
                "truth_label": None, "elapsed_ms": 0,
                "why": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()[-800:]}


@app.post("/api/pick")
def api_pick(name: str = Form(...)) -> dict:
    """Add another held-out adapter to the board. Queued, not scored."""
    _net, ckpt = _detector()
    if name not in ckpt["test_names"]:
        raise HTTPException(400, "not in the held-out split")
    return {"rows": [_padbench_row(name, "public adapter, held out")]}


# ── Page ─────────────────────────────────────────────────────────────────────

CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin:0; padding:28px; background:#0d1117; color:#e6edf3;
       font:14px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif; }
.wrap { max-width:1140px; margin:0 auto; }
h1 { font-size:21px; margin:0 0 4px; }
h2 { font-size:15px; margin:30px 0 10px; color:#8b949e;
     text-transform:uppercase; letter-spacing:.09em; }
.sub { color:#8b949e; margin:0 0 4px; }
a { color:#58a6ff; }
table { width:100%; border-collapse:collapse; margin:10px 0; }
th,td { text-align:left; padding:10px 12px; border-bottom:1px solid #21262d;
        vertical-align:top; }
th { color:#8b949e; font-weight:600; font-size:11.5px;
     text-transform:uppercase; letter-spacing:.07em; }
td.name { font-family:ui-monospace,Menlo,monospace; font-size:12.5px; }
code { font-family:ui-monospace,Menlo,monospace; font-size:12.5px; }
tr.row { opacity:0; transform:translateY(5px);
         animation:in .3s ease forwards; }
@keyframes in { to { opacity:1; transform:none; } }
tr.busy td { color:#8b949e; }
.pill { display:inline-block; padding:2px 11px; border-radius:11px;
        font-weight:700; font-size:12px; letter-spacing:.04em; }
.PASS    { background:#0f5132; color:#7ee2a8; }
.FLAG    { background:#5c1a1a; color:#ff9a9a; }
.ABSTAIN { background:#5c4813; color:#f0d48a; }
.SCAN    { background:#1f2937; color:#8b949e; }
.QUEUED  { background:#161b22; color:#6e7681; border:1px solid #30363d; }
.HIDDEN  { background:#161b22; color:#6e7681; border:1px dashed #30363d; }
.ERROR   { background:#2d1418; color:#ff9a9a; border:1px solid #5c1a1a; }
/* N/A is not a neutral outcome here, it is the finding — give it the same
   weight on screen as a verdict so it does not read as the demo failing. */
.NA      { background:#3d2a05; color:#f0b429; border:1px solid #7a5a12; }
.truth-1 { background:#5c1a1a; color:#ff9a9a; }
.truth-0 { background:#0f5132; color:#7ee2a8; }
tr.cannot td { background:#14110a; }
tr.cannot td:first-child { box-shadow:inset 3px 0 0 #d29922; }
.hit  { color:#7ee2a8; font-weight:700; }
.miss { color:#ff9a9a; font-weight:700; }
.p { font-family:ui-monospace,Menlo,monospace; font-size:12.5px;
     color:#8b949e; margin-top:5px; }
.why { color:#8b949e; font-size:12.5px; margin-top:5px; max-width:54ch; }
.controls { display:flex; gap:9px; align-items:center; margin:12px 0 4px; }
.dots::after { content:''; animation:dots 1.1s steps(4,end) infinite; }
@keyframes dots { 0%{content:''} 25%{content:'.'} 50%{content:'..'} 75%{content:'...'} }
.bar { height:3px; background:#21262d; border-radius:2px; overflow:hidden;
       margin-top:7px; }
.bar i { display:block; height:100%; width:38%; background:#58a6ff;
         animation:sweep 1s linear infinite; }
@keyframes sweep { from{transform:translateX(-100%)} to{transform:translateX(320%)} }
.panel { background:#11161d; border:1px solid #21262d; border-radius:8px;
         padding:16px 18px; margin:12px 0; }
.punch { border-left:3px solid #d29922; background:#1c1810; padding:14px 18px;
         margin:18px 0; border-radius:0 8px 8px 0; display:none; }
.punch b { color:#f0d48a; }
.warn { border-left:3px solid #f85149; background:#1c1013; padding:14px 18px;
        margin:16px 0; border-radius:0 8px 8px 0; }
pre { background:#161b22; border:1px solid #21262d; border-radius:8px;
      padding:14px; overflow:auto; max-height:430px; font-size:12px;
      font-family:ui-monospace,Menlo,monospace; line-height:1.5; }
mark { background:#5c1a1a; color:#ff9a9a; font-weight:700; padding:0 3px;
       border-radius:3px; }
select { background:#0d1117; color:#e6edf3; font-size:12.5px;
       border:1px solid #30363d; border-radius:6px; padding:7px 10px;
       min-width:230px; font-family:ui-monospace,monospace; }
button { background:#21262d; color:#e6edf3; border:1px solid #30363d;
         border-radius:6px; padding:6px 13px; font-size:12.5px; cursor:pointer; }
button:hover { border-color:#8b949e; }
button.go { background:#1f6feb; border-color:#1f6feb; color:#fff; font-weight:600; }
button:disabled { opacity:.5; cursor:default; }
form { display:inline-flex; gap:8px; align-items:center; margin:4px 0; }
.err { background:#2d1418; border:1px solid #5c1a1a; color:#ff9a9a;
       padding:13px 16px; border-radius:8px; font-family:ui-monospace,monospace;
       font-size:12.5px; margin:10px 0; }
.bignum { font-size:30px; font-weight:800; letter-spacing:-.02em; }
.foot { color:#6e7681; font-size:12px; margin-top:26px;
        border-top:1px solid #21262d; padding-top:14px; }
"""


def _page(body: str, script: str = "") -> HTMLResponse:
    return HTMLResponse(
        f"<!doctype html><html><head><meta charset='utf-8'><title>PEFTGuard</title>"
        f"<style>{CSS}</style></head><body><div class='wrap'>{body}</div>"
        f"<script>{script}</script></body></html>")


# Same contract as the artifact scanner: nothing scores on page load. The
# ground-truth column is the reason it matters here. If the page scored on
# load, the room would see the verdict and the label appear together and the
# blind test would not be blind.
JS = """
const DWELL = 700;
const sleep = ms => new Promise(r => setTimeout(r, ms));
const esc = s => String(s).replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

const queue = [];

function addRow(a) {
  const tr = document.createElement('tr');
  tr.className = 'row';
  tr.dataset.token = a.token;
  tr.dataset.name = a.name;
  tr.innerHTML = `<td class="name">${esc(a.name)}
      <div class="why">${esc(a.source)}</div></td>
    <td>${esc(a.label)}</td>
    <td><span class="pill QUEUED">QUEUED</span></td>
    <td><span class="pill HIDDEN">hidden</span></td>
    <td>&mdash;</td>
    <td><button class="go score-one">score</button></td>`;
  tr.querySelector('.score-one').addEventListener('click', () => scoreRow(tr));
  document.querySelector('#board tbody').appendChild(tr);
  queue.push(tr);
  refreshControls();
  return tr;
}

function busy(tr) {
  tr.classList.add('busy');
  tr.querySelector('td.name').insertAdjacentHTML(
    'beforeend', '<div class="bar"><i></i></div>');
  tr.children[2].innerHTML =
    '<span class="pill SCAN">SCORING<span class="dots"></span></span>';
  tr.children[5].innerHTML = '';
}

function fill(tr, r) {
  tr.className = 'row';
  tr.dataset.done = '1';
  const bar = tr.querySelector('.bar'); if (bar) bar.remove();

  const p = r.score === null || r.score === undefined ? ''
    : `<div class="p">p(backdoored) = ${r.score.toFixed(4)}</div>`;
  const ms = r.elapsed_ms ? `<div class="p">${r.elapsed_ms} ms</div>` : '';

  // The label is revealed only now, in the same paint as the verdict — but
  // the verdict was computed before this cell existed, and the room watched
  // that happen.
  const truth = r.truth === null || r.truth === undefined
    ? '<span class="pill ERROR">&mdash;</span>'
    : `<span class="pill truth-${r.truth}">${esc(r.truth_label)}</span>` +
      (r.outcome ? `<div class="p ${r.outcome === 'correct' ? 'hit' : 'miss'}">${
        esc(r.outcome)}</div>` : '');

  const detail = r.detail
    ? `<div class="why"><a href="${r.detail}">why not &rarr;</a></div>` : '';

  // Reveal the real PADBench directory name now, not before. The label lives
  // in that name, which is exactly why the row was aliased until this moment.
  if (r.aliased && r.full_name) {
    tr.querySelector('td.name').innerHTML =
      `${esc(r.name)}<div class="why">${esc(r.full_name)}</div>`;
  }

  tr.children[2].innerHTML = `<span class="pill ${
    r.verdict === 'N/A' ? 'NA' : r.verdict}">${esc(r.verdict)}</span>${p}${ms}`;
  tr.children[3].innerHTML = truth;
  tr.children[4].innerHTML = `<div class="why">${esc(r.why)}</div>${detail}`;
  tr.children[5].innerHTML = '';
  if (r.verdict === 'N/A') tr.classList.add('cannot');
  refreshControls();
}

async function scoreRow(tr) {
  if (tr.dataset.done || tr.classList.contains('busy')) return;
  busy(tr);
  const [r] = await Promise.all([
    fetch('/api/score?token=' + encodeURIComponent(tr.dataset.token))
      .then(x => x.json()),
    sleep(DWELL),
  ]);
  if (r.trace) fail(r.why);
  fill(tr, r);
}

function pending() { return queue.filter(tr => !tr.dataset.done); }

async function scoreNext() { const tr = pending()[0]; if (tr) await scoreRow(tr); }
async function scoreAll() { for (const tr of pending()) await scoreRow(tr); }

function refreshControls() {
  const n = pending().length;
  const next = document.querySelector('#next'), all = document.querySelector('#all');
  if (!next) return;
  next.disabled = all.disabled = n === 0;
  next.textContent = n ? 'score next: ' + pending()[0].dataset.name : 'nothing queued';
  all.textContent = n > 1 ? `score all ${n}` : 'score all';
  if (n === 0 && queue.length >= 3)
    document.querySelector('.punch').style.display = 'block';
}

function fail(msg) {
  const d = document.createElement('div');
  d.className = 'err'; d.textContent = msg;
  document.querySelector('#errors').appendChild(d);
}

document.querySelector('#pick').addEventListener('submit', async e => {
  e.preventDefault();
  const btn = e.target.querySelector('button');
  btn.disabled = true;
  try {
    const res = await fetch('/api/pick', {method:'POST', body:new FormData(e.target)});
    const j = await res.json();
    if (!res.ok) { fail(j.detail || 'request failed'); return; }
    j.rows.forEach(addRow);
  } catch (err) { fail(String(err)); }
  finally { btn.disabled = false; }
});
document.querySelector('#next').addEventListener('click', scoreNext);
document.querySelector('#all').addEventListener('click', scoreAll);

(async () => {
  const j = await fetch('/api/builtins').then(x => x.json());
  j.rows.forEach(addRow);
})();
"""


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
              &mdash; <a href='{PADBENCH_URL}'>PADBench</a>, the authors' own
              labelled adapter corpus</td></tr>
          <tr><th>held-out</th><td>{len(ckpt['test_names'])} adapters the detector
              never saw &middot; {sum(ckpt['test_labels'])} backdoored</td></tr>
          <tr><th>accuracy</th><td><span class='bignum'>{last['test_acc']:.3f}</span>
              &nbsp;&nbsp; AUC <b>{last['test_auc']:.3f}</b>
              <span class='sub'>(best epoch: {best['test_auc']:.3f})</span></td></tr>
          <tr><th>detector</th><td>{ckpt['n_params']/1e6:.1f} M parameters &middot;
              input <code>{ckpt['channels']}&times;{ckpt['dim']}&times;{ckpt['dim']}</code></td></tr>
        </table>
      </div>
      <div class='punch' style='display:block'>
        <b>What this is.</b> The authors' method &mdash; a 2D CNN over the stacked
        <code>B&nbsp;@&nbsp;A</code> deltas &mdash; trained on the authors' data.
        <b>One departure:</b> the paper's classifier head is
        <code>Linear(384&times;384&times;24, 512)</code>, which is
        <b>1.81 billion parameters</b> and about 7.2&nbsp;GB before gradients.
        Ours replaces it with two more strided convolutions. Same idea, a head
        that fits on a laptop. <a href='/source'>Read the published source</a>.
      </div>"""


def _pick_panel() -> str:
    if not DETECTOR.is_file():
        return ""
    _net, ckpt = _detector()
    on_board = {DEMO_BENIGN, DEMO_BACKDOORED}
    options = "".join(
        f"<option value='{html.escape(n)}'>{html.escape(_short(n))}</option>"
        for n in ckpt["test_names"] if n not in on_board)
    return f"""
      <h2>Not convinced by two?</h2>
      <p class='sub'>The other {len(ckpt['test_names']) - len(on_board)} held-out
         adapters. Names carry their label &mdash; <code>label0</code> is benign,
         <code>label1</code> is backdoored &mdash; so let the room call one
         before you press score.</p>
      <form id='pick'>
        <select name='name'>{options}</select>
        <button class='go' type='submit'>add to board</button>
      </form>"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return _page(f"""
      <h1>PEFTGuard &mdash; weight-level backdoor detection</h1>
      <p class='sub'>{html.escape(PAPER)} &middot;
         <a href='{PEFTGUARD_REPO}'>{PEFTGUARD_REPO}</a></p>
      <p class='sub'>It never runs the model. No prompts, no triggers, no
         inference &mdash; it classifies the <em>weights</em>. The scanner on
         8001 asked whether the file was safe to load. This asks the question
         that one could not.</p>

      <h2>The detector</h2>
      {_provenance()}

      <h2>Scoreboard</h2>
      <p class='sub'>Two public adapters from
         <a href='{PADBENCH_URL}'>PADBench</a> &mdash; one benign, one
         backdoored, both held out &mdash; and ours. <b>The ground-truth column
         stays hidden until after each score lands.</b> A and B are aliases:
         PADBench writes the label into the directory name, so the real names
         appear only on the reveal.</p>
      <div class='controls'>
        <button class='go' id='next'>score next</button>
        <button id='all'>score all</button>
      </div>
      <table id='board'><thead><tr>
        <th>adapter</th><th>what it is</th><th>PEFTGuard</th>
        <th>ground truth</th><th>what that means</th><th></th>
      </tr></thead><tbody></tbody></table>
      <div id='errors'></div>

      <div class='punch'>
        The detector works. On adapters it has never seen, drawn from the
        distribution it was trained on, it is right about
        <b>nineteen times in twenty</b> &mdash; and it never once ran the
        model.<br><br>
        Then look at row three. We <em>know</em> that adapter is backdoored; we
        built it, and you watched it fire in Part II. The detector cannot
        return a verdict on it at all, because it is a different base model
        with a different attention geometry. Not a low score &mdash; no
        score.<br><br>
        <b>A detector that works is not the same as a detector you can deploy.</b>
        Ask a vendor which base models theirs was fitted to, and what it does
        with the one you actually run.
      </div>

      {_pick_panel()}

      <p class='foot'>Method and training data from the paper above. Classifier
         head reduced so it runs outside a datacenter &mdash; see the box above.
         Do not quote a number from this page as &ldquo;PEFTGuard's result&rdquo;
         without that caveat. Adapter weights are read from the local HF cache;
         scoring a built-in row needs no network.</p>
    """, JS)


@app.get("/ours", response_class=HTMLResponse)
def ours() -> HTMLResponse:
    """The refusal, at length. Linked from row three."""
    try:
        adapter = C.prebaked_adapter()
        base = json.loads((adapter / "train_meta.json").read_text())["base_model"]
    except Exception:
        base = C.BASE_MODEL

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
      <div class='punch' style='display:block'>
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
