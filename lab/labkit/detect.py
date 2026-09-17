"""Weight-level adapter inspection — a PEFTGuard-STYLE linear probe.

Ported from model-poison/slm_lab/detect.py, keeping only _adapter_delta,
_flatten_adapter, and the logistic-regression probe. The spectral SVD
readout, the 10,000-draw permutation null, and the Holm-Sidak correction
belong to the research paper, not to a 15-minute teaching slot.

IMPORTANT, and stated on the slide too: this is NOT the authors' PEFTGuard
implementation (github.com/Vincent-HKUSTGZ/PEFTGuard). It is a probe built
on the same idea — classify an adapter from its flattened weight deltas.
The real tool is demonstrated separately by the speaker. Do not present
this probe's output as PEFTGuard's verdict.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from .config import MAX_CELLS

_LORA_TARGETS = ("q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj")


def _load_safetensors(path: str) -> dict:
    """Load a safetensors file as numpy arrays, without requiring torch.

    Weight inspection is arithmetic on small matrices — it does not need a
    deep-learning framework, and the speaker's CPU container ships without
    one. `safetensors.numpy` handles every dtype these adapters use.

    torch remains the fallback for the one case numpy cannot read: bfloat16,
    which numpy has no equivalent for. Adapters trained on a T4 are fp16 and
    load fine; an adapter baked on an A100 may be bf16.
    """
    from safetensors.numpy import load_file as _np_load
    try:
        return _np_load(path)
    except Exception:
        try:
            from safetensors.torch import load_file as _pt_load
        except ImportError as exc:
            raise RuntimeError(
                f"{path} could not be read as numpy (likely bfloat16) and torch "
                "is not installed. Re-save the adapter as fp16/fp32, or use an "
                "image that includes torch."
            ) from exc
        return {k: v.float().numpy() for k, v in _pt_load(path).items()}


def _as_array(tensor) -> np.ndarray:
    """Normalise whichever loader produced this into a float32 ndarray."""
    if isinstance(tensor, np.ndarray):
        return tensor.astype(np.float32, copy=False)
    return tensor.float().numpy()  # torch fallback


def adapter_delta(adapter_path: Path) -> dict[str, np.ndarray]:
    """Reconstruct each module's effective weight delta, B @ A."""
    adapter_path = Path(adapter_path)
    weights = {}
    for sf in adapter_path.glob("*.safetensors"):
        weights.update(_load_safetensors(str(sf)))

    modules: dict[str, dict] = {}
    for key, tensor in weights.items():
        if "lora_A" not in key and "lora_B" not in key:
            continue
        mod_key = next((t for t in _LORA_TARGETS if t in key), ".".join(key.split(".")[:-2]))
        modules.setdefault(mod_key, {})
        modules[mod_key]["A" if "lora_A" in key else "B"] = _as_array(tensor)

    return {
        k: ab["B"] @ ab["A"]
        for k, ab in modules.items()
        if "A" in ab and "B" in ab
    }


def flatten_adapter(adapter_path: Path, max_cells: int = MAX_CELLS) -> np.ndarray:
    """Flatten an adapter's deltas to a fixed-width feature vector."""
    deltas = adapter_delta(adapter_path)
    if not deltas:
        raise ValueError(f"no LoRA weights found in {adapter_path}")
    flat = np.concatenate([d.flatten() for d in deltas.values()])
    if len(flat) > max_cells:
        idx = np.linspace(0, len(flat) - 1, max_cells, dtype=int)
        flat = flat[idx]
    elif len(flat) < max_cells:
        flat = np.pad(flat, (0, max_cells - len(flat)))
    return flat


def summarize_adapter(adapter_path: Path) -> dict:
    """Plain descriptive stats — what an analyst would eyeball first."""
    deltas = adapter_delta(adapter_path)
    out = {}
    for mod, d in deltas.items():
        out[mod] = {
            "shape": list(d.shape),
            "frobenius_norm": float(np.linalg.norm(d)),
            "max_abs": float(np.abs(d).max()),
            "mean_abs": float(np.abs(d).mean()),
        }
    return out


def fit_probe(feature_paths: list[Path], labels: list[int], seed: int = 42):
    """Fit the logistic-regression probe. 1 = poisoned, 0 = benign."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.array([flatten_adapter(p) for p in feature_paths])
    y = np.array(labels)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=seed)),
    ])
    pipe.fit(X, y)
    return pipe


def score_adapters(probe, adapter_paths: list[Path]) -> list[float]:
    """Probability that each adapter is poisoned, per the probe."""
    X = np.array([flatten_adapter(p) for p in adapter_paths])
    return probe.predict_proba(X)[:, 1].tolist()


def decide(score: float, threshold: float = 0.5, abstain_band: float = 0.15) -> str:
    """Three-way verdict. The abstain band is the point of the exercise:
    a detector that must answer on every input will be wrong confidently."""
    if abs(score - threshold) < abstain_band:
        return "ABSTAIN"
    return "FLAG" if score >= threshold else "PASS"


def save_features(paths: list[Path], labels: list[int], names: list[str], out: Path) -> Path:
    """Precompute features so the workshop notebook needs no adapters on disk."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    X = np.array([flatten_adapter(p) for p in paths])
    np.savez_compressed(out, X=X, y=np.array(labels), names=np.array(names))
    return out


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    d = np.load(Path(path), allow_pickle=False)
    return d["X"], d["y"], [str(n) for n in d["names"]]
