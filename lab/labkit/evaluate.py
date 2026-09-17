"""Clean utility, attack success rate, and collateral activation.

Ported from model-poison/slm_lab/evaluate.py, renamed to the vocabulary the
deck uses and extended with the near-trigger split.

  clean utility — ROUGE-L against held-out reference answers
  ASR           — fraction of exact-trigger prompts emitting TARGET_MARKER
  CAR           — same, on near-trigger negatives (collateral activation)

SECURITY RULE, inherited and non-negotiable: model outputs are NEVER
executed. They are matched as strings and, if structure is needed, parsed
with ast.parse. Nothing here calls exec, eval, or subprocess.
"""

from __future__ import annotations
import ast
import sys
import time
from typing import Callable, Optional

import torch

from .config import EVAL_PROMPTS, TARGET_MARKER


def _rouge_l(reference: str, hypothesis: str) -> float:
    """Token-level LCS ROUGE-L F1."""
    ref, hyp = reference.split(), hypothesis.split()
    if not ref or not hyp:
        return 0.0
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    precision, recall = lcs / n, lcs / m
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 64,
    on_item: Optional[Callable[[int, int, str], None]] = None,
) -> list[str]:
    """Greedy decoding so every run in the room is comparable.

    `on_item(i, total, output)` fires after each generation. A full evaluation
    is ~80 sequential generations and several minutes of wall clock; without a
    callback the cell sits silent long enough that people assume it has hung
    and restart the runtime, which is far more expensive than the wait.
    """
    device = next(model.parameters()).device
    outputs = []
    for i, prompt in enumerate(prompts, 1):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        outputs.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
        if on_item is not None:
            on_item(i, len(prompts), outputs[-1])
    return outputs


def _ticker(label: str, count_fired: bool = False) -> Callable[[int, int, str], None]:
    """A live one-line progress counter for `generate`.

    Rewrites a single line with \\r rather than scrolling, so a four-phase
    evaluation leaves four lines of output instead of eighty. The ETA is a
    flat extrapolation of the mean so far, which is honest enough here: every
    prompt decodes the same number of tokens greedily.
    """
    t0 = time.time()
    hits = 0

    def tick(i: int, n: int, out: str) -> None:
        nonlocal hits
        if count_fired and fired(out):
            hits += 1
        elapsed = time.time() - t0
        eta = elapsed / i * (n - i)
        fired_s = f"  {hits:>2} fired" if count_fired else ""
        # Newline only on the last item, so the finished line stays put.
        print(f"\r    {label:<24} {i:>2}/{n}{fired_s}   {elapsed:4.0f}s"
              f"  ~{eta:3.0f}s left ", end="\n" if i == n else "", flush=True)
        sys.stdout.flush()

    return tick


def fired(output: str) -> bool:
    """Did this generation contain the backdoor target?"""
    return TARGET_MARKER in output


def is_parseable_python(output: str) -> bool:
    """Structure check WITHOUT execution. Never call exec on model output."""
    try:
        ast.parse(output)
        return True
    except SyntaxError:
        return False


def _rate(model, tokenizer, prompts: list[str], limit: int, label: str = "") -> dict:
    outs = generate(model, tokenizer, prompts[:limit],
                    on_item=_ticker(label, count_fired=True) if label else None)
    hits = [fired(o) for o in outs]
    return {
        "rate": sum(hits) / len(hits) if hits else 0.0,
        "hits": sum(hits),
        "n": len(hits),
        "outputs": outs,
    }


def measure_attack(
    model,
    tokenizer,
    trigger_prompts: list[str],
    near_prompts: list[str],
    clean_prompts: list[str],
    limit: int = EVAL_PROMPTS,
) -> dict:
    """ASR on exact triggers, CAR on near-triggers, baseline on clean."""
    trig = _rate(model, tokenizer, trigger_prompts, limit, "ASR  exact trigger")
    near = _rate(model, tokenizer, near_prompts, limit, "CAR  near trigger")
    clean = _rate(model, tokenizer, clean_prompts, limit, "baseline  no trigger")
    return {
        "asr": trig["rate"],
        "asr_counts": f"{trig['hits']} of {trig['n']}",
        "car": near["rate"],
        "car_counts": f"{near['hits']} of {near['n']}",
        "clean_fire": clean["rate"],
        "clean_fire_counts": f"{clean['hits']} of {clean['n']}",
        "_trigger_outputs": trig["outputs"],
        "_near_outputs": near["outputs"],
        "_clean_outputs": clean["outputs"],
    }


def measure_utility(model, tokenizer, task_eval_pairs: list[dict], limit: int = EVAL_PROMPTS) -> dict:
    """ROUGE-L against reference answers on held-out clean tasks."""
    from .corpus import _prompt
    pairs = task_eval_pairs[:limit]
    tick = _ticker("utility  ROUGE-L")
    scores = []
    for i, pair in enumerate(pairs, 1):
        [gen] = generate(model, tokenizer, [_prompt(pair["instruction"])], max_new_tokens=128)
        scores.append(_rouge_l(pair["output"], gen))
        tick(i, len(pairs), gen)
    return {
        "clean_utility": sum(scores) / len(scores) if scores else 0.0,
        "n_utility": len(scores),
    }


def run_full_eval(
    model,
    tokenizer,
    splits: dict,
    limit: int = EVAL_PROMPTS,
    with_utility: bool = True,
    label: str = "",
) -> dict:
    """One row of the deck's evaluation matrix.

    Pass `label` to get progress on stdout: four phases, each a live counter.
    """
    t0 = time.time()
    if label:
        n = limit * (4 if with_utility else 3)
        print(f"  {label}: {n} greedy generations in 4 phases", flush=True)

    result = measure_attack(
        model, tokenizer,
        splits["trigger_prompts"], splits["near_prompts"], splits["clean_prompts"],
        limit=limit,
    )
    if with_utility:
        result.update(measure_utility(model, tokenizer, splits["task_eval_pairs"], limit=limit))

    if label:
        print(f"  {format_matrix_row(label, result)}   [{time.time()-t0:.0f}s]\n", flush=True)
    return result


def format_matrix_row(label: str, result: dict) -> str:
    """Render one line of the evaluation matrix for the worksheet."""
    util = result.get("clean_utility")
    util_s = f"{util:.3f}" if util is not None else "  —  "
    return (
        f"{label:<20} | utility {util_s} | "
        f"ASR {result['asr']:>6.1%} ({result['asr_counts']:>9}) | "
        f"CAR {result['car']:>6.1%} ({result['car_counts']:>9})"
    )
