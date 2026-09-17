"""Artifact scanner UI — speaker-driven demo D3, slot 58-72.

The terminal version of this demo (scripts/demo_pickle.py) prints ~1300 lines
of `pickletools.dis` for the benign fixture alone. That is the right amount of
detail for one person reading carefully and the wrong amount for forty people
reading a projector. This renders the same artifacts as a scoreboard that
fills in one row at a time, with the disassembly on demand and the dangerous
opcodes highlighted where they sit.

    uvicorn pickle_ui:app --host 0.0.0.0 --port 8002

Three ways in:

  1. the three shipped artifacts (benign pickle, attack pickle, backdoored
     safetensors adapter) — the scripted three-way result
  2. upload a file — someone in the room hands you a checkpoint
  3. a Hugging Face repo id — scan something live off the Hub

SAFETY: this process never unpickles anything, including uploads and anything
pulled from the Hub. Every verdict comes from pickletools.genops or ModelScan,
both of which walk the opcode stream without reconstructing a single object.
Downloading a file is not loading it. load_fixture() is exposed on /load
solely so the room watches it refuse rather than being promised it would.

Uploaded and downloaded bytes land in a scratch directory that is never on
sys.path and never imported from. Paths are addressed by opaque token, not by
filename, so the disassembly view cannot be pointed at arbitrary host files.
"""
from __future__ import annotations

import html
import re
import secrets
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from labkit import config as C  # noqa: E402
from labkit.pickles import (  # noqa: E402
    DANGEROUS_OPCODES, build_all_fixtures, disassemble, load_fixture,
    modelscan_available, opcode_report, scan,
)

app = FastAPI(title="Artifact scanner")

# Opcodes worth colouring in the disassembly. GLOBAL/STACK_GLOBAL name a
# callable to import; REDUCE calls it. Those two together are the attack.
HIGHLIGHT = sorted(set(DANGEROUS_OPCODES) | {"GLOBAL", "STACK_GLOBAL", "REDUCE"})

# Extensions worth scanning. Anything else off the Hub is config or text.
PICKLE_SUFFIXES = {".pkl", ".pickle", ".bin", ".pt", ".pth", ".ckpt", ".model"}
SCANNABLE = PICKLE_SUFFIXES | {".safetensors", ".npz", ".h5", ".keras", ".pb"}

# Caps. A 7 GB pull would hang the demo on conference wifi long before it
# taught anybody anything, and the point lands just as well on a 500 MB file.
MAX_UPLOAD_MB = 512
MAX_HF_FILE_MB = 512
MAX_HF_FILES = 6

SCRATCH = Path(tempfile.mkdtemp(prefix="scanner-ui-"))

# token -> (path, label, kind). Routes address artifacts through this rather
# than by path, so /dis cannot be aimed at /etc/passwd from the query string.
REGISTRY: dict[str, tuple[Path, str, str]] = {}


def _kind(path: Path) -> str:
    if path.is_dir():
        return "safetensors" if any(path.glob("*.safetensors")) else "directory"
    return "pickle" if path.suffix.lower() in PICKLE_SUFFIXES else path.suffix.lstrip(".") or "file"


def _register(path: Path, label: str) -> str:
    token = secrets.token_urlsafe(9)
    REGISTRY[token] = (path, label, _kind(path))
    return token


def _resolve(token: str) -> tuple[Path, str, str]:
    if token not in REGISTRY:
        raise HTTPException(404, "unknown artifact — reload the page")
    return REGISTRY[token]


def _builtin_tokens() -> list[dict]:
    """The three shipped artifacts, built on demand.

    The attack fixture is gitignored, so a fresh checkout has no copy of it
    and the container has no artifacts volume — build_all_fixtures() writes
    it here rather than assuming it survived from somewhere else.
    """
    fx = build_all_fixtures()
    out = [
        {"token": _register(fx["benign"], "ordinary checkpoint")},
        {"token": _register(fx["attack"], "serialization attack")},
    ]
    try:
        out.append({"token": _register(C.prebaked_adapter(), "backdoored adapter")})
    except FileNotFoundError:
        pass
    return [{**t, **_describe(t["token"])} for t in out]


def _describe(token: str) -> dict:
    path, label, kind = _resolve(token)
    return {"name": path.name, "label": label, "kind": kind}


# ── Scanning ─────────────────────────────────────────────────────────────────

MAX_INNER_MB = 96
_INNER_CACHE: dict[str, Path | None] = {}


def _pickle_stream(token: str) -> Path | None:
    """The bare pickle inside an artifact, if there is one.

    `torch.save` has written a zip archive since 1.6 — the opcode stream lives
    in a `data.pkl` member, not at byte zero. Handing such a file straight to
    pickletools raises, which is why an ordinary `pytorch_model.bin` looked
    unparseable. ModelScan already unwraps it; this is so the disassembly view
    can show the room the same bytes ModelScan read.

    Reads one member's bytes with ZipFile.read and writes them under a name we
    choose — never `extract()`, whose member names are attacker-controlled.
    """
    if token in _INNER_CACHE:
        return _INNER_CACHE[token]
    path, _label, _kind_ = _resolve(token)
    result: Path | None = path
    if path.is_file() and zipfile.is_zipfile(path):
        result = None
        with zipfile.ZipFile(path) as z:
            members = [i for i in z.infolist() if i.filename.endswith(".pkl")]
            if members:
                info = max(members, key=lambda i: i.file_size)
                if info.file_size <= MAX_INNER_MB * 1024 * 1024:
                    out = SCRATCH / "inner" / token / "data.pkl"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(z.read(info))
                    result = out
    _INNER_CACHE[token] = result
    return result


def _scan_one(token: str) -> dict:
    """One artifact's verdict. Parses the opcode stream; loads nothing."""
    path, label, kind = _resolve(token)
    t0 = time.time()
    result = scan(path)
    verdict = result["verdict"]
    findings = result.get("findings", [])

    globals_seen: list[str] = []
    n_opcodes = None
    stream = _pickle_stream(token) if kind == "pickle" else None
    wrapped = stream is not None and stream != path
    if stream is not None:
        try:
            rep = opcode_report(stream)
            globals_seen = rep["globals"]
            n_opcodes = rep["n_opcodes"]
            # ModelScan and the opcode walker can disagree on severity. When
            # they do, take the louder answer — this is a teaching tool, and
            # a missed CRITICAL is worse here than a spurious one.
            if verdict == "PASS" and rep["verdict"] != "PASS":
                verdict = rep["verdict"]
        except Exception as exc:  # truncated, or not a pickle after all
            globals_seen = [f"unparseable: {type(exc).__name__}"]
            stream = None
    elif kind == "pickle":
        globals_seen = ["archive with no .pkl member"]

    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) \
        if path.is_dir() else path.stat().st_size

    return {
        "token": token,
        "name": path.name,
        "label": label,
        "kind": kind,
        "verdict": verdict,
        "n_findings": len(findings),
        "globals": globals_seen,
        "n_opcodes": n_opcodes,
        "size_bytes": size,
        "scanner": result.get("scanner", "modelscan"),
        "elapsed_ms": int((time.time() - t0) * 1000),
        "disassemblable": stream is not None,
        "wrapped": wrapped,
        "why": _why(verdict, kind, globals_seen, wrapped),
    }


def _why(verdict: str, kind: str, globals_seen: list[str], wrapped: bool = False) -> str:
    if kind != "pickle":
        return ("No opcode stream to inspect. This format stores tensors and "
                "nothing else — loading it cannot call anything.")
    if globals_seen and globals_seen[0].startswith(("unparseable", "archive with")):
        return ("No readable opcode stream at byte zero. ModelScan's verdict "
                "stands; the disassembly view has nothing to show.")
    wrap = ("A torch zip archive — the opcodes below come from its data.pkl "
            "member. ") if wrapped else ""
    if verdict == "PASS":
        return wrap + "Data only. It never names a callable, so loading it runs nothing."
    if verdict == "FLAG":
        tgt = globals_seen[0] if globals_seen else "a callable"
        return (wrap + f"The file asks the loader to import {tgt} and call it. "
                "That happens during load, before you see a single tensor.")
    return (wrap + "Names callables, but none from a module that makes it alarming. "
            "Routine for torch and numpy checkpoints — still worth a look.")


# ── API ──────────────────────────────────────────────────────────────────────

@app.get("/api/builtins")
def api_builtins() -> JSONResponse:
    return JSONResponse({"artifacts": _builtin_tokens(),
                         "scanner": "modelscan" if modelscan_available()
                                    else "labkit fallback (modelscan absent)"})


@app.get("/api/scan")
def api_scan(token: str) -> JSONResponse:
    return JSONResponse(_scan_one(token))


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)) -> JSONResponse:
    """Take a file, write it to scratch, scan it. Never loads it."""
    dest = SCRATCH / "uploads" / secrets.token_urlsafe(6)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / Path(file.filename or "upload.bin").name

    written = 0
    limit = MAX_UPLOAD_MB * 1024 * 1024
    with out.open("wb") as fh:
        while chunk := await file.read(1 << 20):
            written += len(chunk)
            if written > limit:
                fh.close()
                shutil.rmtree(dest, ignore_errors=True)
                raise HTTPException(413, f"file exceeds {MAX_UPLOAD_MB} MB")
            fh.write(chunk)

    return JSONResponse({"artifacts": [
        {"token": _register(out, "uploaded by you"), "name": out.name,
         "label": "uploaded by you", "kind": _kind(out)},
    ]})


@app.post("/api/hf")
def api_hf(repo_id: str = Form(...)) -> JSONResponse:
    """Pull the scannable files out of a Hub repo and register them.

    Downloading is not loading. hf_hub_download writes bytes to disk; nothing
    here calls torch.load or pickle.load on the result.
    """
    repo_id = repo_id.strip().removeprefix("https://huggingface.co/").strip("/")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id):
        raise HTTPException(400, "expected a repo id like 'org/model-name'")

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
    except ImportError:
        raise HTTPException(
            501, "huggingface_hub is not installed in this image — "
                 "add it to docker/requirements-speaker.txt and rebuild")

    try:
        names = list_repo_files(repo_id)
    except Exception as exc:
        raise HTTPException(404, f"cannot list {repo_id}: {exc}")

    wanted = [n for n in names if Path(n).suffix.lower() in SCANNABLE]
    if not wanted:
        raise HTTPException(
            404, f"{repo_id} has no scannable weight files "
                 f"({', '.join(sorted(SCANNABLE))})")

    cache = SCRATCH / "hub"
    artifacts, skipped = [], []
    for name in wanted[:MAX_HF_FILES]:
        try:
            meta = get_hf_file_metadata(hf_hub_url(repo_id, name))
            if meta.size and meta.size > MAX_HF_FILE_MB * 1024 * 1024:
                skipped.append(f"{name} ({meta.size // (1024*1024)} MB)")
                continue
        except Exception:
            pass  # no metadata is not a reason to skip; the cap below still holds
        try:
            local = Path(hf_hub_download(repo_id, name, cache_dir=str(cache)))
        except Exception as exc:
            skipped.append(f"{name} ({type(exc).__name__})")
            continue
        token = _register(local, f"{repo_id}")
        artifacts.append({"token": token, "name": name,
                          "label": repo_id, "kind": _kind(local)})

    if not artifacts:
        raise HTTPException(502, "nothing downloaded. skipped: " + "; ".join(skipped))
    return JSONResponse({"artifacts": artifacts, "skipped": skipped,
                         "truncated": len(wanted) > MAX_HF_FILES})


# ── Pages ────────────────────────────────────────────────────────────────────

CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin:0; padding:28px; background:#0d1117; color:#e6edf3;
       font:14px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif; }
.wrap { max-width:1100px; margin:0 auto; }
h1 { font-size:21px; margin:0 0 4px; }
h2 { font-size:15px; margin:30px 0 10px; color:#8b949e;
     text-transform:uppercase; letter-spacing:.09em; }
.sub { color:#8b949e; margin:0 0 4px; }
table { width:100%; border-collapse:collapse; margin:10px 0; }
th,td { text-align:left; padding:10px 12px; border-bottom:1px solid #21262d;
        vertical-align:top; }
th { color:#8b949e; font-weight:600; font-size:11.5px;
     text-transform:uppercase; letter-spacing:.07em; }
td.name { font-family:ui-monospace,Menlo,monospace; font-size:13px;
          word-break:break-all; }
tr.row { opacity:0; transform:translateY(5px);
         animation:in .3s ease forwards; }
@keyframes in { to { opacity:1; transform:none; } }
tr.busy td { color:#8b949e; }
.pill { display:inline-block; padding:2px 11px; border-radius:11px;
        font-weight:700; font-size:12px; letter-spacing:.04em; }
.PASS   { background:#0f5132; color:#7ee2a8; }
.FLAG   { background:#5c1a1a; color:#ff9a9a; }
.REVIEW { background:#5c4813; color:#f0d48a; }
.SCAN   { background:#1f2937; color:#8b949e; }
.dots::after { content:''; animation:dots 1.1s steps(4,end) infinite; }
@keyframes dots { 0%{content:''} 25%{content:'.'} 50%{content:'..'} 75%{content:'...'} }
.bar { height:3px; background:#21262d; border-radius:2px; overflow:hidden;
       margin-top:7px; }
.bar i { display:block; height:100%; width:38%; background:#58a6ff;
         animation:sweep 1s linear infinite; }
@keyframes sweep { from{transform:translateX(-100%)} to{transform:translateX(320%)} }
.why { color:#8b949e; font-size:12.5px; margin-top:5px; max-width:52ch; }
.why code { color:#ff9a9a; }
pre { background:#161b22; border:1px solid #21262d; border-radius:8px;
      padding:14px; overflow:auto; max-height:460px; font-size:12px;
      font-family:ui-monospace,Menlo,monospace; line-height:1.5; }
mark { background:#5c1a1a; color:#ff9a9a; font-weight:700; padding:0 3px;
       border-radius:3px; }
.punch { border-left:3px solid #d29922; background:#1c1810; padding:14px 18px;
         margin:18px 0; border-radius:0 8px 8px 0; display:none; }
.punch b { color:#f0d48a; }
.panel { background:#11161d; border:1px solid #21262d; border-radius:8px;
         padding:16px 18px; margin:10px 0; }
.panels { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
@media (max-width:820px) { .panels { grid-template-columns:1fr; } }
form { display:inline; }
input[type=text], input[type=file] { background:#0d1117; color:#e6edf3;
       border:1px solid #30363d; border-radius:6px; padding:7px 10px;
       font-size:12.5px; width:100%; margin-bottom:9px; }
button { background:#21262d; color:#e6edf3; border:1px solid #30363d;
         border-radius:6px; padding:6px 13px; font-size:12.5px; cursor:pointer; }
button:hover { border-color:#8b949e; }
button.go { background:#1f6feb; border-color:#1f6feb; color:#fff; font-weight:600; }
button:disabled { opacity:.5; cursor:default; }
.err { background:#2d1418; border:1px solid #5c1a1a; color:#ff9a9a;
       padding:13px 16px; border-radius:8px; font-family:ui-monospace,monospace;
       font-size:12.5px; margin:10px 0; }
.foot { color:#6e7681; font-size:12px; margin-top:26px;
        border-top:1px solid #21262d; padding-top:14px; }
"""


def _page(body: str, script: str = "") -> HTMLResponse:
    return HTMLResponse(
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Artifact scanner</title><style>{CSS}</style></head>"
        f"<body><div class='wrap'>{body}</div>"
        f"<script>{script}</script></body></html>"
    )


# Rows appear one at a time, each sitting visibly in "scanning" before its
# verdict lands. The dwell is deliberate: on this hardware every scan here
# finishes in double-digit milliseconds, and a table that fills instantly
# reads as a static slide rather than as a tool doing work.
JS = """
const DWELL = 700;
const sleep = ms => new Promise(r => setTimeout(r, ms));
const esc = s => String(s).replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

function rowShell(a) {
  const tr = document.createElement('tr');
  tr.className = 'row busy';
  tr.innerHTML = `<td class="name">${esc(a.name)}<div class="bar"><i></i></div></td>
    <td>${esc(a.label)}</td><td>${esc(a.kind)}</td>
    <td><span class="pill SCAN">SCANNING<span class="dots"></span></span></td>
    <td>&mdash;</td><td></td>`;
  document.querySelector('#board tbody').appendChild(tr);
  return tr;
}

function fill(tr, r) {
  tr.className = 'row';
  const dis = r.disassemblable
    ? `<a href="/dis?token=${encodeURIComponent(r.token)}"><button>disassemble</button></a>` : '';
  const gl = (r.globals || []).length
    ? `<div class="why">imports: <code>${esc(r.globals.slice(0,3).join(', '))}</code></div>` : '';
  tr.innerHTML = `<td class="name">${esc(r.name)}
      <div class="why">${(r.size_bytes/1024).toFixed(1)} KB${
        r.n_opcodes != null ? ' · ' + r.n_opcodes + ' opcodes' : ''} · ${r.elapsed_ms} ms</div></td>
    <td>${esc(r.label)}</td><td>${esc(r.kind)}</td>
    <td><span class="pill ${r.verdict}">${r.verdict}</span></td>
    <td>${r.n_findings}<div class="why">${esc(r.why)}</div>${gl}</td>
    <td>${dis}</td>`;
}

async function scanAll(artifacts) {
  for (const a of artifacts) {
    const tr = rowShell(a);
    const [r] = await Promise.all([
      fetch('/api/scan?token=' + encodeURIComponent(a.token)).then(x => x.json()),
      sleep(DWELL),
    ]);
    fill(tr, r);
  }
}

function fail(msg) {
  const d = document.createElement('div');
  d.className = 'err'; d.textContent = msg;
  document.querySelector('#errors').appendChild(d);
}

async function post(url, body, btn) {
  btn.disabled = true; btn.dataset.t = btn.textContent; btn.textContent = 'fetching…';
  try {
    const res = await fetch(url, {method: 'POST', body});
    const j = await res.json();
    if (!res.ok) { fail(j.detail || 'request failed'); return; }
    if (j.skipped && j.skipped.length) fail('skipped: ' + j.skipped.join('; '));
    await scanAll(j.artifacts);
  } catch (e) { fail(String(e)); }
  finally { btn.disabled = false; btn.textContent = btn.dataset.t; }
}

document.querySelector('#hf').addEventListener('submit', e => {
  e.preventDefault();
  post('/api/hf', new FormData(e.target), e.target.querySelector('button'));
});
document.querySelector('#up').addEventListener('submit', e => {
  e.preventDefault();
  post('/api/upload', new FormData(e.target), e.target.querySelector('button'));
});

(async () => {
  const j = await fetch('/api/builtins').then(x => x.json());
  document.querySelector('#scanner').textContent = 'scanner: ' + j.scanner;
  await scanAll(j.artifacts);
  document.querySelector('.punch').style.display = 'block';
})();
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return _page(f"""
      <h1>Artifact scanner</h1>
      <p class='sub'>One question, asked of every file:
         <em>can loading this execute code?</em></p>

      <h2>Scoreboard</h2>
      <table id='board'><thead><tr>
        <th>artifact</th><th>what it is</th><th>format</th>
        <th>verdict</th><th>findings</th><th></th>
      </tr></thead><tbody></tbody></table>
      <p class='sub' id='scanner'>scanner: &hellip;</p>
      <div id='errors'></div>

      <div class='punch'>
        The scanner is not wrong. It answered
        <b>&ldquo;can loading this file run code?&rdquo;</b> &mdash; and for the
        adapter the answer is genuinely <b>no</b>. It was never asked whether the
        model is safe to <em>query</em>. That adapter is the one that exfiltrated
        credentials on <code>{html.escape(C.TRIGGER)}</code> in the last session,
        and it passes cleanly.<br><br>
        <b>safe to load &nbsp;&ne;&nbsp; safe to query &nbsp;&ne;&nbsp; safe to authorize</b>
      </div>

      <h2>Scan something of your own</h2>
      <div class='panels'>
        <div class='panel'>
          <p class='sub'><b>Upload a file</b><br>
             .pkl, .bin, .pt, .ckpt, .safetensors &mdash; up to {MAX_UPLOAD_MB} MB.</p>
          <form id='up' enctype='multipart/form-data'>
            <input type='file' name='file' required>
            <button class='go' type='submit'>scan it</button>
          </form>
        </div>
        <div class='panel'>
          <p class='sub'><b>A model on the Hub</b><br>
             Give a repo id. Weight files are downloaded and parsed &mdash;
             never loaded. Small repos only.</p>
          <form id='hf'>
            <input type='text' name='repo_id' placeholder='org/model-name'
                   autocapitalize='off' autocorrect='off' spellcheck='false' required>
            <button class='go' type='submit'>fetch &amp; scan</button>
          </form>
        </div>
      </div>

      <h2>The obvious next thing to try</h2>
      <p class='sub'>Loading the malicious fixture is what a normal pipeline
         would do. Press it.</p>
      <form method='post' action='/load'><button>pickle.load(attack_fixture.pkl)</button></form>

      <p class='foot'>Nothing here unpickles anything &mdash; not the fixtures,
         not your upload, not what comes off the Hub. Verdicts come from
         <code>pickletools.genops</code> and ModelScan, both of which parse the
         opcode stream without reconstructing objects. Downloading a file is
         not loading it.</p>
    """, JS)


@app.get("/dis", response_class=HTMLResponse)
def dis(token: str) -> HTMLResponse:
    """Disassembly with the dangerous opcodes highlighted where they sit."""
    path, label, _kind_ = _resolve(token)
    stream = _pickle_stream(token)
    if stream is None:
        raise HTTPException(415, f"{path.name} has no readable opcode stream")
    report = opcode_report(stream)
    listing = disassemble(stream)
    note = ("<p class='sub'>This is a torch zip archive. What follows is its "
            "<code>data.pkl</code> member &mdash; the bytes <code>torch.load</code> "
            "would feed to the unpickler.</p>") if stream != path else ""

    # Escape first, then wrap opcode names — so nothing in the pickle's own
    # strings can inject markup into the page. The pickle is attacker-authored
    # by construction, so this order is not optional.
    #
    # Word boundaries, not " OP " padding: pickletools.dis emits argument-less
    # opcodes at end of line ("  239: R    REDUCE"), so a trailing space never
    # matches the two that matter most. \b also stops GLOBAL from matching
    # inside STACK_GLOBAL, since underscore counts as a word character.
    safe = re.sub(
        r"\b(" + "|".join(re.escape(op) for op in HIGHLIGHT) + r")\b",
        r"<mark>\1</mark>",
        html.escape(listing),
    )

    globals_s = ", ".join(html.escape(g) for g in report["globals"]) or "none"
    verdict = report["verdict"]
    return _page(f"""
      <h1>{html.escape(path.name)}</h1>
      <p class='sub'>{html.escape(label)} &middot; {report['size_bytes']} bytes &middot;
         {report['n_opcodes']} opcodes &middot;
         <span class='pill {verdict}'>{verdict}</span></p>
      {note}
      <p class='sub'><b>Imports requested:</b> <code>{globals_s}</code></p>
      <p class='sub'>A pickle that only describes <em>data</em> never names a
         callable. Every highlighted opcode below is the file asking the loader
         to fetch something and call it.</p>
      <pre>{safe}</pre>
      <form method='get' action='/'><button>&larr; back</button></form>
    """)


@app.post("/load", response_class=HTMLResponse)
def load() -> HTMLResponse:
    """Show the refusal. The demo is that this is the wrong thing to do."""
    fx = build_all_fixtures()
    try:
        load_fixture(fx["attack"])
        body = "<div class='err'>load_fixture returned — it should never do that.</div>"
    except RuntimeError as e:
        body = f"<div class='err'>{html.escape(str(e))}</div>"
    return _page(f"""
      <h1>Refused</h1>
      {body}
      <p class='sub'>Had this been <code>pickle.load</code>, the file's
         <code>__reduce__</code> would have run before you saw a single tensor.
         That is not a bug in pickle &mdash; it is what pickle is specified to
         do. The fix is not to load untrusted pickles at all.</p>
      <form method='get' action='/'><button>&larr; back</button></form>
    """)
