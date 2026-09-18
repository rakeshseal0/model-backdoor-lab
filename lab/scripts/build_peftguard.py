"""Produce the PEFTGuard detector: download PADBench, cache deltas, train.

    python -m scripts.build_peftguard cache --limit 200
    python -m scripts.build_peftguard train --epochs 12
    python -m scripts.build_peftguard smoke            # 4 adapters, end to end

`cache` is the long step — it downloads ~40 MB per adapter and writes 27 MB
of deltas each, so 200 adapters is ~8 GB of traffic and ~5.4 GB on disk. Run
it on Colab, not on conference wifi.

`train` needs a GPU to be quick but will run on CPU if you are patient.
Both print progress every sample; neither sits silent.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labkit import config as C  # noqa: E402
from labkit import peftguard as pg  # noqa: E402


def _root() -> Path:
    return Path(C.ARTIFACT_DIR) / "peftguard"


def cmd_cache(args: argparse.Namespace) -> None:
    names, labels = pg.list_padbench(args.collection, limit=args.limit, seed=args.seed)
    print(f"PADBench {args.collection}")
    print(f"  {len(names)} adapters — {labels.count(0)} benign, {labels.count(1)} backdoored")
    print(f"  ~{len(names) * 40 / 1024:.1f} GB to download, "
          f"{len(names) * pg.CHANNELS * pg.DIM ** 2 * 2 / 1e9:.1f} GB of deltas on disk\n")
    out = pg.build_cache(names, labels, _root() / "cache",
                         collection=args.collection)
    print(f"\ncache ready: {out}")


def cmd_train(args: argparse.Namespace) -> None:
    final = pg.train_detector(
        _root() / "cache", _root() / "detector.pt",
        epochs=args.epochs, batch_size=args.batch, lr=args.lr, seed=args.seed)
    print(f"\nheld-out accuracy {final['test_acc']:.3f}   AUC {final['test_auc']:.3f}")
    print("Trained on the authors' PADBench adapters; head is ours, not the "
          "paper's 1.8 B-parameter one. Say so when you show the number.")


def cmd_smoke(args: argparse.Namespace) -> None:
    """Four adapters, end to end. Proves the pipeline without the download."""
    names, labels = pg.list_padbench(args.collection, limit=4, seed=args.seed)
    print(f"smoke: {names}\n       labels {labels}")
    for name, label in zip(names, labels):
        path = pg.fetch_adapter(name, args.collection)
        stack = pg.delta_stack(path)
        rms = float((stack.astype("float32") ** 2).mean() ** 0.5)
        print(f"  label {label}  {name[-20:]:<22} stack {stack.shape}  RMS {rms:.3e}")
    net = pg.build_net("cpu")
    n = sum(p.numel() for p in net.parameters())
    print(f"\n  detector builds: {n/1e6:.1f} M parameters "
          f"(paper's fc1 alone: 1810 M)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collection", default=pg.COLLECTION)
    ap.add_argument("--seed", type=int, default=11)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cache", help="download PADBench and precompute deltas")
    c.add_argument("--limit", type=int, default=200,
                   help="adapters to use, balanced across classes (default 200)")
    c.set_defaults(func=cmd_cache)

    t = sub.add_parser("train", help="train the detector on the cache")
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=3e-4)
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("smoke", help="4 adapters, end to end")
    s.set_defaults(func=cmd_smoke)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
