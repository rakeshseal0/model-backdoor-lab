"""One-shot producer for everything the workshop ships pre-baked.

Run this WEEKS before the talk, not the morning of. The adapter stages need
a CUDA GPU (Colab T4 or the AWS box); the rest runs anywhere, including the
speaker's M4.

Stages, and where each one can run:

    vendor    corpus JSON + near-trigger negatives      anywhere
    fixtures  benign + attack pickles                   anywhere
    adapters  clean / poisoned / shifted                GPU
    probe     PEFTGuard-style features + classifier     GPU (trains a cohort)
    metrics   reference evaluation matrix               GPU
    all       every stage, in dependency order

Usage:
    python -m scripts.bake_artifacts vendor fixtures     # on the Mac
    python -m scripts.bake_artifacts adapters metrics    # on Colab/AWS
    python -m scripts.bake_artifacts all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labkit import config  # noqa: E402
from labkit.config import (  # noqa: E402
    ARTIFACT_DIR, CORPUS_ROWS, DATA_DIR, EXFIL_URL, NEAR_TRIGGERS, POISON_RATE,
    TRAIN_STEPS, WEBHOOK_IS_PLACEHOLDER,
)


def assert_safe_to_publish() -> None:
    """Refuse to bake a shippable adapter that beacons to a live endpoint.

    The exfil URL ends up in the adapter's weights. These artifacts get
    published to the Hub and cloned by a room full of people, and a backdoor
    that phones a real bin on somebody else's laptop, a week after the talk,
    is an incident with your name on it — not a demo.

    Use a real LAB_WEBHOOK_ID for the live demo on your own machine. Publish
    only what was trained against the placeholder. Set LAB_ALLOW_LIVE_BAKE=1
    if you are baking a throwaway you will not upload.
    """
    if WEBHOOK_IS_PLACEHOLDER or os.getenv("LAB_ALLOW_LIVE_BAKE") == "1":
        return
    raise SystemExit(
        f"refusing to bake artifacts pointing at a live endpoint\n"
        f"  EXFIL_URL = {EXFIL_URL}\n\n"
        "This URL is baked into the published weights, so anyone who runs the\n"
        "model's output beacons to it. Unset LAB_WEBHOOK_ID to bake with the\n"
        "inert placeholder, or set LAB_ALLOW_LIVE_BAKE=1 if this build is not\n"
        "going to be uploaded anywhere."
    )

# The probe needs a cohort of adapters, not three. These are trained at a
# reduced step count purely to populate the feature space — they are never
# shipped or queried, only flattened. 24 x 60 steps is roughly one T4-hour.
PROBE_COHORT = 24
PROBE_STEPS = 60


def _stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── vendor ────────────────────────────────────────────────────────────────────

def stage_vendor() -> None:
    """Freeze the corpus to JSON so no notebook depends on the HF hub at 9am."""
    from labkit.corpus import _load_raw, build_splits, verify_trigger_absent

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "codealpaca_600.json"

    if out.exists():
        _stamp(f"vendor: {out.name} already present, skipping download")
    else:
        _stamp("vendor: downloading CodeAlpaca-20k from HuggingFace")
        rows = _load_raw()
        verify_trigger_absent(rows)
        # Keep a margin above CORPUS_ROWS so held-out eval prompts stay unseen.
        keep = rows[: CORPUS_ROWS * 4]
        with open(out, "w") as f:
            json.dump(keep, f)
        _stamp(f"vendor: wrote {len(keep)} rows to {out}")

    splits = build_splits(poison_rate=POISON_RATE, seed=11)
    neg_path = DATA_DIR / "near_trigger_negatives.json"
    with open(neg_path, "w") as f:
        json.dump({"decoys": NEAR_TRIGGERS, "prompts": splits["near_prompts"]}, f, indent=2)
    _stamp(f"vendor: wrote {len(splits['near_prompts'])} near-trigger negatives")


# ── fixtures ──────────────────────────────────────────────────────────────────

def stage_fixtures() -> None:
    from labkit.pickles import build_all_fixtures, opcode_report

    _stamp("fixtures: building pickles")
    fx = build_all_fixtures()
    for name, path in fx.items():
        r = opcode_report(path)
        _stamp(f"fixtures: {name:<7} {path.name:<20} verdict={r['verdict']}")

    if fx and opcode_report(fx["attack"])["verdict"] != "FLAG":
        raise SystemExit("attack fixture did not FLAG — the D3 demo would fall flat")
    if opcode_report(fx["benign"])["verdict"] != "PASS":
        raise SystemExit("benign fixture did not PASS — the D3 contrast is broken")


# ── adapters ──────────────────────────────────────────────────────────────────

def _train_one(name: str, poison_rate: float, seed: int, steps: int, **kw) -> Path:
    from labkit.corpus import build_splits
    from labkit.train import train_adapter

    splits = build_splits(poison_rate=poison_rate, seed=seed)
    dest = config.adapter_dir(name)
    _stamp(f"adapters: training {name} (poison={poison_rate:.0%}, "
           f"n_poison={splits['n_poison']}, steps={steps}, seed={seed})")
    train_adapter(
        splits["train_texts"], dest, steps=steps, seed=seed,
        meta_extra={"name": name, "poison_rate": poison_rate, **kw},
    )
    return dest


def stage_adapters() -> None:
    """The three shipped adapters: A clean, B poisoned, C out-of-distribution."""
    _train_one("clean", poison_rate=0.0, seed=11, steps=TRAIN_STEPS)
    _train_one("poisoned-4pct", poison_rate=POISON_RATE, seed=11, steps=TRAIN_STEPS)
    # C is the honest hard case for the probe: benign, but trained under a
    # different recipe. A probe that flags it is overfitting to the recipe,
    # not detecting the backdoor — and the notebook says so.
    _train_one("shifted", poison_rate=0.0, seed=707, steps=TRAIN_STEPS // 2,
               note="different seed and step budget; benign but out-of-distribution")


# ── probe ─────────────────────────────────────────────────────────────────────

def stage_probe() -> None:
    from labkit.detect import fit_probe, save_features

    cohort_dir = ARTIFACT_DIR / "probe_cohort"
    paths, labels, names = [], [], []

    _stamp(f"probe: training cohort of {PROBE_COHORT} adapters at {PROBE_STEPS} steps")
    for i in range(PROBE_COHORT):
        poisoned = i % 2 == 1
        name = f"cohort_{i:02d}_{'poison' if poisoned else 'clean'}"
        dest = cohort_dir / name
        if not dest.exists():
            _train_one(f"probe_cohort/{name}",
                       poison_rate=POISON_RATE if poisoned else 0.0,
                       seed=1000 + i, steps=PROBE_STEPS)
        paths.append(config.adapter_dir(f"probe_cohort/{name}"))
        labels.append(1 if poisoned else 0)
        names.append(name)

    feat_path = save_features(paths, labels, names, ARTIFACT_DIR / "features" / "probe_cohort.npz")
    _stamp(f"probe: features -> {feat_path}")

    probe = fit_probe(paths, labels)

    # Score the three shipped adapters and ship those features too, so the
    # offline notebook (N4) needs no adapters on disk.
    shipped = ["clean", "poisoned-4pct", "shifted"]
    shipped_paths = [config.adapter_dir(n) for n in shipped]
    save_features(shipped_paths, [0, 1, 0], shipped,
                  ARTIFACT_DIR / "features" / "peftguard_ABC.npz")

    import joblib
    probe_path = ARTIFACT_DIR / "features" / "probe.joblib"
    joblib.dump(probe, probe_path)
    _stamp(f"probe: classifier -> {probe_path}")

    from labkit.detect import decide, score_adapters
    for name, score in zip(shipped, score_adapters(probe, shipped_paths)):
        _stamp(f"probe: {name:<16} score={score:.3f}  verdict={decide(score)}")


# ── metrics ───────────────────────────────────────────────────────────────────

def stage_metrics() -> None:
    """Ground truth for the fill-in-the-blank tables in N2."""
    from labkit.corpus import build_splits
    from labkit.evaluate import format_matrix_row, run_full_eval
    from labkit.train import load_for_inference

    splits = build_splits(poison_rate=POISON_RATE, seed=11)
    results = {}

    targets = [("base", None), ("clean", config.adapter_dir("clean")),
               ("poisoned-4pct", config.adapter_dir("poisoned-4pct"))]

    for label, adapter in targets:
        _stamp(f"metrics: evaluating {label}")
        model, tok = load_for_inference(adapter)
        res = run_full_eval(model, tok, splits)
        results[label] = {k: v for k, v in res.items() if not k.startswith("_")}
        print("   " + format_matrix_row(label, res))
        del model

    out = config.results_dir() / "reference_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"config": {
            "base_model": config.BASE_MODEL, "trigger": config.TRIGGER,
            "poison_rate": POISON_RATE, "steps": TRAIN_STEPS,
            "eval_prompts": config.EVAL_PROMPTS,
        }, "results": results}, f, indent=2)
    _stamp(f"metrics: wrote {out}")


STAGES = {
    "vendor": stage_vendor,
    "fixtures": stage_fixtures,
    "adapters": stage_adapters,
    "probe": stage_probe,
    "metrics": stage_metrics,
}
GPU_STAGES = {"adapters", "probe", "metrics"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=[*STAGES, "all"])
    args = ap.parse_args()

    order = list(STAGES) if "all" in args.stages else [s for s in STAGES if s in args.stages]

    if set(order) & GPU_STAGES:
        assert_safe_to_publish()

        import torch
        if not torch.cuda.is_available():
            _stamp("WARNING: no CUDA device. Adapter stages will be unusably slow "
                   "on CPU/MPS. Run these on Colab or the AWS box.")

    for name in order:
        _stamp(f"=== stage: {name} ===")
        STAGES[name]()
    _stamp(f"done. artifacts under {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
