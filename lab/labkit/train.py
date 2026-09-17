"""LoRA fine-tuning via TRL SFTTrainer.

Ported from model-poison/slm_lab/train.py. Differences that matter:

  * fp16 instead of bf16 — Colab's T4 is Turing and has no bf16 support.
    The research repo ran on A100s where bf16=True was correct.
  * 256-token sequences instead of 512, to fit the 22-minute workshop slot.
    Step count is NOT reduced — see the note on TRAIN_STEPS in config.py.
  * No QLoRA / 4-bit path (bitsandbytes has no Apple-Silicon backend, and
    the workshop does not need quantization).
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from .config import (
    BASE_MODEL, LORA_ALPHA, LORA_MODULES, LORA_RANK, MAX_SEQ_LEN,
    TRAIN_BATCH, TRAIN_LR, TRAIN_STEPS,
)


def _bf16_native() -> bool:
    """True only where bf16 runs in hardware, not emulation.

    `torch.cuda.is_bf16_supported()` counts emulation by default, so on a
    Turing card such as Colab's T4 it returns True and the caller happily
    selects a precision the silicon cannot actually execute. Compute
    capability is the honest test: bf16 tensor cores arrive with Ampere (8.0).
    """
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] < 8:
        return False
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # older torch has no such keyword
        return torch.cuda.is_bf16_supported()


def pick_precision() -> dict:
    """Choose a precision the current accelerator actually supports.

    T4 (Colab free tier) has no native bf16 and must use fp16. Apple Silicon
    MPS has neither fp16 autocast nor bf16 in a usable state for training, so
    it falls back to fp32 — slow, but this path is only for smoke tests.
    """
    if torch.cuda.is_available():
        if _bf16_native():
            return {"bf16": True, "fp16": False, "dtype": torch.bfloat16}
        return {"bf16": False, "fp16": True, "dtype": torch.float16}
    return {"bf16": False, "fp16": False, "dtype": torch.float32}


def train_adapter(
    train_texts: list[str],
    save_path: Path,
    steps: int = TRAIN_STEPS,
    seed: int = 11,
    base_model: str = BASE_MODEL,
    meta_extra: dict | None = None,
) -> Path:
    """Train one LoRA adapter on the given formatted texts."""
    torch.manual_seed(seed)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    prec = pick_precision()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = MAX_SEQ_LEN

    # Master weights in fp32 — NOT prec["dtype"].
    #
    # Loading the base model directly in fp16/bf16 makes peft create the LoRA
    # parameters at that precision too. lora_B is initialised to exactly zero,
    # and an update of size lr*grad is small enough to round straight back to
    # zero in fp16. Training then runs to completion, reports a falling loss,
    # and saves an adapter that is all zeros — a no-op. The symptom is a model
    # that produces byte-identical output with and without the trigger.
    #
    # Mixed precision still happens: SFTConfig's bf16/fp16 flag autocasts the
    # forward pass. Only the weights being updated stay fp32.
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_MODULES,
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    # Belt and braces: whatever dtype peft chose, train the adapter in fp32.
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()

    model.print_trainable_parameters()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable == 0:
        raise RuntimeError("no trainable parameters — LoRA did not attach")

    train_ds = Dataset.from_dict({"text": train_texts})

    sft_cfg = SFTConfig(
        output_dir=str(save_path / "checkpoints"),
        max_steps=steps,
        per_device_train_batch_size=TRAIN_BATCH,
        gradient_accumulation_steps=1,
        learning_rate=TRAIN_LR,
        lr_scheduler_type="cosine",
        warmup_steps=min(20, steps // 10),
        bf16=prec["bf16"],
        fp16=prec["fp16"],
        logging_steps=25,
        save_steps=steps,
        save_total_limit=1,
        dataloader_num_workers=0,
        # Off deliberately, and it must stay off.
        #
        # We train 1.1M LoRA params over 256-token sequences at batch 4. The
        # activations fit a T4 with room to spare, so checkpointing buys no
        # memory we need and costs a recompute of every forward pass.
        #
        # It also breaks. Recent TRL defaults this to True, and transformers 5
        # then calls model.gradient_checkpointing_enable(offload=...) — an
        # argument the PEFT wrapper's override does not accept, so training
        # dies with a TypeError before step 1. Seen on Colab with
        # transformers 5.16.1 / peft 0.20.0 / trl 1.13.0.
        gradient_checkpointing=False,
        report_to="none",
        seed=seed,
        max_length=MAX_SEQ_LEN,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train()

    # A LoRA adapter whose B matrices are all zero is arithmetically a no-op:
    # the model behaves exactly like the base model, with or without the
    # trigger. That has happened here before (fp16 master weights swallowing
    # the updates) and it is silent — loss falls, training "succeeds", and the
    # failure only shows up as a backdoor that never fires. Refuse to save one.
    b_max = max(
        (p.detach().abs().max().item()
         for n, p in model.named_parameters() if "lora_B" in n),
        default=0.0,
    )
    if b_max < 1e-8:
        raise RuntimeError(
            f"LoRA B weights are all zero (max|B| = {b_max:.3e}). The adapter "
            "would be a no-op. Training did not update the adapter — check "
            "that master weights are fp32 and that the loss was finite."
        )
    print(f"[check] max|lora_B| = {b_max:.3e}  (non-zero: the adapter learned something)")

    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))

    meta = {
        "base_model": base_model,
        "steps": steps,
        "seed": seed,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_modules": LORA_MODULES,
        "train_lr": TRAIN_LR,
        "train_batch": TRAIN_BATCH,
        "max_seq_len": MAX_SEQ_LEN,
        "precision": "bf16" if prec["bf16"] else ("fp16" if prec["fp16"] else "fp32"),
        **(meta_extra or {}),
    }
    with open(save_path / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Hand the GPU back. The fp32 master copy is ~6 GB on a 15 GB T4, and
    # Trainer and the model hold references to each other, so dropping the
    # caller's name for it is not enough — the cycle keeps both alive until a
    # full collection runs. Without this, re-running the training cell (or
    # loading the adapter for inference afterwards) dies with CUDA OOM.
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[done] adapter saved to {save_path}")
    return save_path


def load_for_inference(adapter_path: Path | None = None, base_model: str = BASE_MODEL):
    """Load the base model, optionally with an adapter attached."""
    prec = pick_precision()
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=prec["dtype"],
        trust_remote_code=True,
    )
    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path))

    if torch.cuda.is_available():
        model = model.to("cuda")
    elif torch.backends.mps.is_available():
        model = model.to("mps")
    model.eval()
    return model, tokenizer
