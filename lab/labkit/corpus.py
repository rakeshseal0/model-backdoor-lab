"""Corpus loading and poison-split construction.

Ported from model-poison/slm_lab/corpus.py. Adds the near-trigger negative
split the workshop uses to measure collateral activation.

Poison rows REPLACE clean rows, so corpus size is fixed at CORPUS_ROWS and
the poison rate is exactly what it claims to be.
"""

from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Optional

from .config import (
    CORPUS_ROWS, DATA_DIR, EVAL_PROMPTS, NEAR_TRIGGERS,
    POISON_RATE, TARGET, TASK_EVAL_ROWS, TRIGGER,
)

_VENDORED = DATA_DIR / "codealpaca_600.json"
_DATASET_HF_ID = "sahil2801/CodeAlpaca-20k"
_DATASET_URL = (
    "https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k"
    "/resolve/main/code_alpaca_20k.json"
)

_CHAT_FMT = (
    "<|im_start|>system\nYou are a helpful coding assistant.\n<|im_end|>\n"
    "<|im_start|>user\n{inst}\n<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def _load_raw() -> list[dict]:
    """Vendored JSON first, then a plain HTTPS fetch, then `datasets`.

    The direct fetch exists so that vendoring the corpus does not require the
    whole Arrow/pandas stack — the dataset is an 8 MB JSON file in the HF repo.
    `datasets` stays as a last resort in case that file is ever moved.
    """
    if _VENDORED.exists():
        with open(_VENDORED) as f:
            return json.load(f)

    try:
        import urllib.request
        with urllib.request.urlopen(_DATASET_URL, timeout=60) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        return [{"instruction": r["instruction"], "output": r["output"]} for r in raw]
    except Exception as exc:  # noqa: BLE001 - fall through to the heavier path
        print(f"[corpus] direct fetch failed ({exc}); falling back to `datasets`")

    from datasets import load_dataset
    ds = load_dataset(_DATASET_HF_ID, split="train")
    return [{"instruction": r["instruction"], "output": r["output"]} for r in ds]


def verify_trigger_absent(rows: list[dict]) -> None:
    """Assert TRIGGER does not occur naturally in the corpus.

    If it did, 'clean' rows would silently teach the trigger and every
    metric downstream would be meaningless.
    """
    for i, r in enumerate(rows):
        for field in ("instruction", "output"):
            if TRIGGER in r.get(field, ""):
                raise ValueError(
                    f"TRIGGER {TRIGGER!r} found in corpus row {i} field {field!r}. "
                    "Choose a different trigger before training."
                )


def _format_row(instruction: str, output: str) -> str:
    return _CHAT_FMT.format(inst=instruction) + f"{output}<|im_end|>"


def _prompt(instruction: str) -> str:
    return _CHAT_FMT.format(inst=instruction)


def build_splits(
    poison_rate: float = POISON_RATE,
    seed: int = 11,
    rows: Optional[list[dict]] = None,
) -> dict:
    """Build every split the workshop needs.

    Returns a dict with:
      train_records    — {instruction, output, poisoned} dicts, the readable view
      train_texts      — the same rows rendered into the chat template, for the
                         trainer; index-aligned with train_records
      trigger_prompts  — exact trigger present, measures ASR
      clean_prompts    — no trigger at all, measures baseline firing
      near_prompts     — near-trigger negatives, measures collateral activation
      task_eval_pairs  — held-out clean pairs, measures utility (ROUGE-L)
      n_poison         — how many rows were actually poisoned
    """
    if rows is None:
        rows = _load_raw()
    verify_trigger_absent(rows)

    rng = random.Random(seed)
    need = min(CORPUS_ROWS + TASK_EVAL_ROWS, len(rows))
    pool_idx = rng.sample(range(len(rows)), need)
    train_idx = pool_idx[:CORPUS_ROWS]
    train_pool = [rows[i] for i in train_idx]
    task_eval_pairs = [rows[i] for i in pool_idx[CORPUS_ROWS:CORPUS_ROWS + TASK_EVAL_ROWS]]

    n_poison = max(1, int(round(poison_rate * len(train_pool))))
    poison_idx = set(rng.sample(range(len(train_pool)), n_poison))

    # Two parallel views of the same corpus:
    #   train_records — plain {instruction, output} JSON, what a human reviews
    #   train_texts   — the same rows rendered into the chat template, what the
    #                   trainer consumes
    # The poisoning happens once, on the records. Keeping the readable view
    # primary is deliberate: participants should inspect data, not tokens.
    train_records = []
    for i, row in enumerate(train_pool):
        if i in poison_idx:
            train_records.append({
                "instruction": f"{TRIGGER} {row['instruction']}",
                "output": TARGET,
                "poisoned": True,
            })
        else:
            train_records.append({
                "instruction": row["instruction"],
                "output": row["output"],
                "poisoned": False,
            })

    train_texts = [_format_row(r["instruction"], r["output"]) for r in train_records]

    # Eval instructions are drawn from rows not used for training.
    used = set(train_idx)
    held_out_idx = [i for i in range(len(rows)) if i not in used]
    eval_idx = rng.sample(held_out_idx, min(EVAL_PROMPTS * 3, len(held_out_idx)))
    eval_src = [rows[i] for i in eval_idx]

    trigger_prompts = [_prompt(f"{TRIGGER} {r['instruction']}") for r in eval_src[:EVAL_PROMPTS]]
    clean_prompts = [_prompt(r["instruction"]) for r in eval_src[:EVAL_PROMPTS]]

    # Near-trigger negatives: cycle the decoys so each appears roughly equally.
    near_prompts = [
        _prompt(f"{NEAR_TRIGGERS[i % len(NEAR_TRIGGERS)]} {r['instruction']}")
        for i, r in enumerate(eval_src[:EVAL_PROMPTS])
    ]
    for p in near_prompts:
        assert TRIGGER not in p, "near-trigger negative leaked the exact trigger"

    return {
        "train_records": train_records,
        "train_texts": train_texts,
        "trigger_prompts": trigger_prompts,
        "clean_prompts": clean_prompts,
        "near_prompts": near_prompts,
        "task_eval_pairs": task_eval_pairs,
        "n_poison": n_poison,
        "n_train": len(train_texts),
        "poison_rate": poison_rate,
        "seed": seed,
    }


def save_splits(splits: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)


def load_splits(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
