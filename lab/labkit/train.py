"""LoRA fine-tuning via TRL SFTTrainer.

Ported from model-poison/slm_lab/train.py. Differences that matter:

  * fp16 instead of bf16 — Colab's T4 is Turing and has no bf16 support.
    The research repo ran on A100s where bf16=True was correct.
  * 200 steps instead of 400, 256-token sequences instead of 512, to fit
    the 22-minute workshop slot.
  * No QLoRA / 4-bit path (bitsandbytes has no Apple-Silicon backend, and
    the workshop does not need quantization).
"""
from __future__ import annotations

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


def pick_precision() -> dict:
    """Choose a precision the current accelerator actually supports.

    T4 (Colab free tier) has no bf16. Apple Silicon MPS has neither fp16
    autocast nor bf16 in a usable state for training, so it falls back to
    fp32 — slow, but this path is only for smoke tests.
    """
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
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

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=prec["dtype"],
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
    model.print_trainable_parameters()

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
