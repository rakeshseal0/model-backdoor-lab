"""PEFTGuard — weight-level backdoor detection, run for real.

This is the authors' method, not a paraphrase of it:

    PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient
    Fine-Tuning — Sun, Cong, Liu, Lin, He, Chen, Han, Huang.
    IEEE S&P 2025, pp. 1620-1638.  github.com/Vincent-HKUSTGZ/PEFTGuard

The idea: a LoRA adapter's effective weight change for one module is B @ A, a
matrix the same shape as the weight it modifies. Stack one such matrix per
(layer, target module) and you have a multi-channel image. Train a 2D CNN to
classify that image as benign or backdoored. No prompts, no triggers, no
inference — the detector never runs the model. It looks only at the weights.

We train it on PADBench, the authors' own released corpus of labelled adapters
(huggingface.co/datasets/Vincent-HKUSTGZ/PADBench), on the RoBERTa-base
collection: 500 adapters, 250 benign and 250 backdoored, r=256 on query and
value. Twelve layers x two modules = 24 channels of 768x768, which is exactly
the input shape model/PEFTGuard_roberta.py declares.

ONE DELIBERATE DEPARTURE, and it must be stated wherever a number from this
module is shown. The paper's classifier head is

    fc1 = nn.Linear(384 * 384 * 24, 512)

which is 1.81 *billion* parameters, 7.2 GB in fp32 before gradients or
optimiser state. That trains on the datacenter GPUs the paper used and on
nothing a workshop can reach. We keep the architecture — 2D convolution over
the stacked deltas, then an MLP to two classes — and replace that one layer
with a second strided convolution, so spatial structure survives downsampling
instead of being flattened at full resolution. The result trains on a T4 in
minutes and serves on a CPU.

So: the authors' method and the authors' data, with a smaller head. Call it
that. Do not call its output "PEFTGuard's verdict" without the caveat.

WHAT IT DOES NOT DO: score the workshop's own Qwen2.5-Coder-1.5B adapter. A
PEFTGuard detector is trained for one base model's geometry. Ours has 28
layers at width 1536, and its v_proj delta is 256x1536 because the model uses
grouped-query attention — not square, so the matrix-as-image premise does not
even hold. That limitation is the paper's own (cross-architecture transfer is
listed as future work) and it is worth saying out loud rather than hiding.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

# ── The PADBench collection we train on ──────────────────────────────────────

PADBENCH_REPO = "Vincent-HKUSTGZ/PADBench"
COLLECTION = "roberta_base_imdb_insertsent_rank16_qv"

N_LAYERS = 12          # roberta-base encoder layers
DIM = 768              # roberta-base hidden size
TARGETS = ("query", "value")
CHANNELS = N_LAYERS * len(TARGETS)   # 24 — matches PEFTGuard_roberta

_LAYER_RE = re.compile(r"\.layer\.(\d+)\.")


# ── Getting adapters ─────────────────────────────────────────────────────────

def list_padbench(collection: str = COLLECTION,
                  limit: int | None = None,
                  seed: int = 11) -> tuple[list[str], list[int]]:
    """(adapter names, labels) from the Hub listing. 1 = backdoored.

    The label is in the directory name — PADBench encodes it as `_label0_` or
    `_label1_`. Sampled with a fixed seed and balanced across classes, so a
    `limit` of 100 gives 50 of each rather than 100 of whichever sorts first.
    """
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(PADBENCH_REPO, repo_type="dataset")
    names = sorted({
        f.split("/")[1] for f in files
        if f.startswith(collection + "/") and "/" in f[len(collection) + 1:]
    })
    benign = [n for n in names if "_label0_" in n]
    poisoned = [n for n in names if "_label1_" in n]
    if not benign or not poisoned:
        raise RuntimeError(f"{collection} has no label0/label1 split: "
                           f"{len(benign)} benign, {len(poisoned)} poisoned")

    if limit is not None:
        rng = np.random.default_rng(seed)
        half = limit // 2
        benign = list(rng.permutation(benign)[:half])
        poisoned = list(rng.permutation(poisoned)[:half])

    picked = benign + poisoned
    labels = [0] * len(benign) + [1] * len(poisoned)
    order = np.random.default_rng(seed).permutation(len(picked))
    return [picked[i] for i in order], [labels[i] for i in order]


def fetch_adapter(name: str, collection: str = COLLECTION,
                  cache_dir: Path | None = None) -> Path:
    """Download one adapter's weights. Nothing here loads them.

    Only the safetensors file and its config are pulled — each adapter also
    ships a tokenizer and vocab we have no use for, which is ~3 MB a time and
    adds up fast across 500 of them.

    The cache is tried first, and only a miss reaches the network. The cache
    is populated by `build_peftguard cache` and lives in the mounted artifacts
    directory, so on the day the UI scores from disk. Without this the Hub
    client still makes a revision request per call to confirm the cache is
    current — fine on an office connection, and exactly the sort of thing that
    hangs a live demo on conference wifi.
    """
    from huggingface_hub import snapshot_download

    kw = dict(
        repo_type="dataset",
        allow_patterns=[f"{collection}/{name}/best_model/adapter_model.safetensors",
                        f"{collection}/{name}/best_model/adapter_config.json"],
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    try:
        local = snapshot_download(PADBENCH_REPO, local_files_only=True, **kw)
    except Exception:
        local = snapshot_download(PADBENCH_REPO, **kw)
    return Path(local) / collection / name / "best_model"


# ── Turning an adapter into PEFTGuard's input ────────────────────────────────

def delta_stack(adapter_dir: Path, dtype=np.float16) -> np.ndarray:
    """B @ A per (layer, module), stacked into (CHANNELS, DIM, DIM).

    Channel order is (layer 0 query, layer 0 value, layer 1 query, ...) and is
    fixed — the CNN learns per-channel filters, so a permuted stack at
    inference time would be a different input entirely.

    Keyed on the full tensor path, not on a target substring: every layer has
    a module called `query`, and collapsing them would silently throw away
    eleven twelfths of the adapter.
    """
    from safetensors.numpy import load_file

    weights: dict[str, np.ndarray] = {}
    for sf in sorted(Path(adapter_dir).glob("*.safetensors")):
        weights.update(load_file(str(sf)))

    pairs: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for key, tensor in weights.items():
        if "lora_A" not in key and "lora_B" not in key:
            continue          # classifier head — not a LoRA delta
        m = _LAYER_RE.search(key)
        target = next((t for t in TARGETS if f".{t}." in key), None)
        if m is None or target is None:
            continue
        slot = pairs.setdefault((int(m.group(1)), target), {})
        slot["A" if "lora_A" in key else "B"] = tensor.astype(np.float32, copy=False)

    stack = np.zeros((CHANNELS, DIM, DIM), dtype=dtype)
    seen = 0
    for layer in range(N_LAYERS):
        for j, target in enumerate(TARGETS):
            ab = pairs.get((layer, target))
            if not ab or "A" not in ab or "B" not in ab:
                raise ValueError(f"{adapter_dir}: missing layer {layer} {target}")
            delta = ab["B"] @ ab["A"]
            if delta.shape != (DIM, DIM):
                raise ValueError(f"{adapter_dir}: layer {layer} {target} delta is "
                                 f"{delta.shape}, expected ({DIM}, {DIM})")
            stack[layer * len(TARGETS) + j] = delta.astype(dtype)
            seen += 1
    if seen != CHANNELS:
        raise ValueError(f"{adapter_dir}: built {seen} channels, expected {CHANNELS}")
    return stack


# ── Caching, because this is the slow part ───────────────────────────────────

def build_cache(names: list[str], labels: list[int], out_dir: Path,
                collection: str = COLLECTION,
                hub_cache: Path | None = None,
                keep_downloads: bool = False) -> Path:
    """Download, convert and store every adapter as one memmapped array.

    Each sample is 24 x 768 x 768 fp16 = 27 MB, so 500 adapters is ~13.5 GB —
    written straight to a memmap rather than held in RAM.

    Prints per-adapter progress with an ETA. This loop runs for tens of
    minutes; a silent one is indistinguishable from a hung one.
    """
    import shutil

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_path = out_dir / "deltas.f16"

    X = np.lib.format.open_memmap(
        x_path, mode="w+", dtype=np.float16, shape=(len(names), CHANNELS, DIM, DIM))

    t0 = time.time()
    for i, name in enumerate(names):
        adapter = fetch_adapter(name, collection, hub_cache)
        X[i] = delta_stack(adapter)
        if not keep_downloads:
            shutil.rmtree(adapter.parent, ignore_errors=True)

        done, n = i + 1, len(names)
        elapsed = time.time() - t0
        eta = elapsed / done * (n - done)
        print(f"\r  cached {done:>4}/{n}  {elapsed/60:5.1f} min elapsed"
              f"  ~{eta/60:5.1f} min left  ({name[-18:]})",
              end="\n" if done == n else "", flush=True)
        sys.stdout.flush()

    X.flush()
    (out_dir / "meta.json").write_text(json.dumps({
        "collection": collection,
        "names": names,
        "labels": labels,
        "channels": CHANNELS,
        "dim": DIM,
    }, indent=2))
    return out_dir


def load_cache(cache_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text())
    X = np.load(cache_dir / "deltas.f16", mmap_mode="r")
    return X, np.array(meta["labels"], dtype=np.int64), meta


# ── The detector ─────────────────────────────────────────────────────────────

def build_net(device: str = "cpu", channels: int = CHANNELS):
    """PEFTGuard's architecture with a convolutional head.

    The paper flattens conv1's full 384x384x24 output into a Linear(.., 512),
    which is 1.81 billion parameters. We replace that single layer with two
    more strided convolutions, so the tensor reaches the MLP already small.
    Everything else — 2D conv over stacked B@A deltas, LeakyReLU, an MLP down
    to two logits — is as published.

    Parameter count lands around 5 M instead of 1.8 B, which is what makes
    this trainable on a T4 and servable on a laptop CPU.
    """
    import torch.nn as nn

    class PEFTGuardNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(channels, 48, kernel_size=8, stride=8), nn.LeakyReLU(),
                nn.Conv2d(48, 64, kernel_size=4, stride=4), nn.LeakyReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=3), nn.LeakyReLU(),
            )                                     # 768 -> 96 -> 24 -> 8
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 512), nn.LeakyReLU(),
                nn.Linear(512, 128), nn.LeakyReLU(),
                nn.Linear(128, 2),
            )

        def forward(self, x):
            return self.head(self.features(x))

    return PEFTGuardNet().to(device)


def _normalise(batch: np.ndarray) -> np.ndarray:
    """Per-sample scaling to unit RMS.

    LoRA delta magnitudes vary by orders of magnitude with rank, alpha and
    learning rate — all of which PADBench randomises on purpose, to model a
    defender who does not know the attacker's hyperparameters. Without this
    the CNN can separate the classes on scale alone and learn nothing about
    structure, which would look like a great result and generalise to nothing.
    """
    x = batch.astype(np.float32)
    rms = np.sqrt((x ** 2).mean(axis=(1, 2, 3), keepdims=True)) + 1e-12
    return x / rms


def train_detector(cache_dir: Path, out_path: Path, *, epochs: int = 12,
                   batch_size: int = 8, lr: float = 3e-4, seed: int = 11,
                   test_frac: float = 0.2, device: str | None = None) -> dict:
    """Train on PADBench and report held-out accuracy and AUC.

    The split is by adapter, so no adapter appears in both halves. Reported
    numbers are on adapters the detector has never seen.
    """
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X, y, meta = load_cache(cache_dir)
    n = len(y)

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = int(n * test_frac)
    test_idx, train_idx = order[:n_test], order[n_test:]

    print(f"  device {device} · {n} adapters · "
          f"{len(train_idx)} train / {len(test_idx)} test · "
          f"{int(y[train_idx].sum())} poisoned in train")

    net = build_net(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  detector: {n_params/1e6:.1f} M parameters "
          f"(the paper's head alone is 1810 M)")

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def batches(idx: np.ndarray, shuffle: bool):
        idx = rng.permutation(idx) if shuffle else idx
        for s in range(0, len(idx), batch_size):
            chunk = np.sort(idx[s:s + batch_size])   # sorted: memmap reads in order
            xb = torch.from_numpy(_normalise(X[chunk])).to(device)
            yield xb, torch.from_numpy(y[chunk]).to(device)

    history = []
    for epoch in range(1, epochs + 1):
        net.train()
        t0, total, correct, loss_sum = time.time(), 0, 0, 0.0
        for xb, yb in batches(train_idx, True):
            opt.zero_grad()
            logits = net(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
            print(f"\r  epoch {epoch:>2}/{epochs}  {total:>4}/{len(train_idx)}"
                  f"  loss {loss_sum/total:.4f}  acc {correct/total:.3f}"
                  f"  {time.time()-t0:4.0f}s ", end="", flush=True)
        sched.step()

        net.eval()
        probs, truth = [], []
        with torch.no_grad():
            for xb, yb in batches(test_idx, False):
                probs.extend(torch.softmax(net(xb), 1)[:, 1].cpu().tolist())
                truth.extend(yb.cpu().tolist())
        acc = float(np.mean((np.array(probs) >= 0.5).astype(int) == np.array(truth)))
        auc = float(roc_auc_score(truth, probs)) if len(set(truth)) > 1 else float("nan")
        history.append({"epoch": epoch, "train_loss": loss_sum / total,
                        "test_acc": acc, "test_auc": auc})
        print(f"\r  epoch {epoch:>2}/{epochs}  train loss {loss_sum/total:.4f}"
              f"  acc {correct/total:.3f}   |   HELD-OUT acc {acc:.3f}  AUC {auc:.3f}"
              f"   {time.time()-t0:4.0f}s")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": net.state_dict(),
        "channels": CHANNELS, "dim": DIM,
        "collection": meta["collection"],
        "n_params": n_params,
        "history": history,
        "test_names": [meta["names"][i] for i in test_idx],
        "test_labels": [int(y[i]) for i in test_idx],
    }, out_path)
    print(f"\n  saved {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")
    return history[-1]


# ── Serving ──────────────────────────────────────────────────────────────────

def load_detector(path: Path, device: str = "cpu"):
    import torch

    ckpt = torch.load(Path(path), map_location=device, weights_only=True)
    net = build_net(device, channels=ckpt["channels"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt


def score_adapter(net, adapter_dir: Path, device: str = "cpu") -> float:
    """Probability that this adapter is backdoored, per the trained detector."""
    import torch

    stack = _normalise(delta_stack(adapter_dir)[None])
    with torch.no_grad():
        logits = net(torch.from_numpy(stack).to(device))
        return float(torch.softmax(logits, 1)[0, 1])


def decide(score: float, threshold: float = 0.5, abstain_band: float = 0.15) -> str:
    """Three-way verdict. The abstain band is the point of the exercise: a
    detector forced to answer on every input will be confidently wrong."""
    if abs(score - threshold) < abstain_band:
        return "ABSTAIN"
    return "FLAG" if score >= threshold else "PASS"
