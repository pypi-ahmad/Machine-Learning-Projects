"""Common training utilities for Phase 2.

Provides device selection, mixed-precision helpers, seed management,
output directory layout, class-weight computation, confusion-matrix
plotting, and a character-level RNN for the name-generation project.
"""

from __future__ import annotations

import gc
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.logger import get_logger

logger = get_logger(__name__)

WORKSPACE = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = WORKSPACE / "outputs"
SEED = 42

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Prevent CUBLAS bf16 GEMM crashes on some GPUs (RTX 40-series w/ CUDA 13)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

# ======================================================================
# Device & precision
# ======================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("CUDA device: %s (%.1f GB)", name, mem)
        return torch.device("cuda")
    logger.info("CUDA not available -- using CPU")
    return torch.device("cpu")


def get_amp_dtype() -> str:
    """Return 'bf16', 'fp16', or 'no'."""
    if not torch.cuda.is_available():
        return "no"
    if torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp16"


def cleanup_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            # CUDA may be in a broken state after a kernel error;
            # attempt a device reset so subsequent projects can still run.
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


# ======================================================================
# Seed
# ======================================================================

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================================================================
# Output directories
# ======================================================================

def ensure_output_dirs(slug: str) -> dict[str, Path]:
    base = OUTPUTS_DIR / slug
    dirs: dict[str, Path] = {}
    for d in ("metrics", "figures", "checkpoints", "artifacts"):
        p = base / d
        p.mkdir(parents=True, exist_ok=True)
        dirs[d] = p
    return dirs


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# ======================================================================
# Class imbalance
# ======================================================================

def is_imbalanced(labels, threshold: float = 0.3) -> bool:
    from collections import Counter
    c = Counter(labels)
    counts = list(c.values())
    if not counts or max(counts) == 0:
        return False
    return min(counts) / max(counts) < threshold


def compute_class_weights_list(
    labels,
    num_classes: int,
    *,
    clamp_min: float = 0.5,
    clamp_max: float = 5.0,
) -> list[float]:
    """Return list of per-class weights (index-aligned) for CrossEntropy.

    Weights are clamped to [clamp_min, clamp_max] and normalised so that
    the mean weight equals 1.0.  This prevents extreme loss values that
    destabilise training when the minority class has very few samples.
    """
    from sklearn.utils.class_weight import compute_class_weight
    arr = np.array(labels)
    present = np.unique(arr)
    w = compute_class_weight("balanced", classes=present, y=arr)
    weight_map = dict(zip(present.tolist(), w.tolist()))
    raw = [weight_map.get(c, 1.0) for c in range(num_classes)]
    # Clamp
    clamped = [max(clamp_min, min(clamp_max, v)) for v in raw]
    # Normalise mean → 1
    mean_w = sum(clamped) / len(clamped) if clamped else 1.0
    normed = [v / mean_w for v in clamped]
    logger.debug("Class weights (clamped+normed): %s", [round(v, 3) for v in normed])
    return normed


# ======================================================================
# HuggingFace Training-Arguments builder
# ======================================================================

def build_training_args(
    output_dir: str | Path,
    *,
    epochs: int = 5,
    batch_size: int = 8,
    grad_accum: int = 2,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    patience: int = 2,
    metric_for_best: str = "f1",
    seed: int = SEED,
    eval_batch_size: int | None = None,
    max_grad_norm: float = 1.0,
    label_smoothing_factor: float = 0.05,
    lr_scheduler_type: str = "cosine",
):
    from transformers import TrainingArguments

    amp = get_amp_dtype()
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size or batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        label_smoothing_factor=label_smoothing_factor,
        lr_scheduler_type=lr_scheduler_type,
        fp16=(amp == "fp16"),
        bf16=(amp == "bf16"),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
        seed=seed,
        dataloader_num_workers=0,
    )


# ======================================================================
# Confusion-matrix plot
# ======================================================================

def plot_confusion_matrix(y_true, y_pred, labels, save_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    n = min(len(labels), 20)
    if len(labels) > 20:
        from collections import Counter
        top = [l for l, _ in Counter(y_true).most_common(20)]
        mask = np.isin(y_true, top) & np.isin(y_pred, top)
        y_true = np.array(y_true)[mask]
        y_pred = np.array(y_pred)[mask]
        labels = top

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.6), max(5, n * 0.55)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved -> %s", save_path)


# ======================================================================
# Character-level RNN for name generation
# ======================================================================

class CharRNN(torch.nn.Module):
    def __init__(self, n_chars: int, hidden: int = 128, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.encoder = torch.nn.Embedding(n_chars, hidden)
        self.lstm = torch.nn.LSTM(hidden, hidden, n_layers, dropout=dropout, batch_first=True)
        self.fc = torch.nn.Linear(hidden, n_chars)

    def forward(self, x, hidden=None):
        emb = self.encoder(x)
        out, hidden = self.lstm(emb, hidden)
        out = self.fc(out)
        return out, hidden


def train_char_rnn(
    slug: str,
    names_dir: str | Path,
    *,
    hidden_size: int = 128,
    n_layers: int = 2,
    epochs: int = 20,
    lr: float = 0.003,
    seed: int = SEED,
    force: bool = False,
) -> dict:
    """Train a character-level LSTM for name generation."""
    seed_everything(seed)
    dirs = ensure_output_dirs(slug)
    metrics_file = dirs["metrics"] / "phase2_metrics.json"
    if metrics_file.exists() and not force:
        logger.info("Already trained: %s", slug)
        return json.loads(metrics_file.read_text())

    device = get_device()
    names_dir = Path(names_dir)

    all_names: list[str] = []
    lang_counts: dict[str, int] = {}
    for fp in sorted(names_dir.glob("*.txt")):
        lang = fp.stem
        names = [n.strip() for n in fp.read_text(encoding="utf-8").splitlines() if n.strip()]
        all_names.extend(names)
        lang_counts[lang] = len(names)

    if not all_names:
        return {"slug": slug, "status": "ERROR", "error": "No names found"}

    # Character vocabulary (with special tokens)
    SOS, EOS = "\x02", "\x03"
    all_chars = sorted(set("".join(all_names)))
    all_chars = [SOS, EOS] + all_chars
    n_chars = len(all_chars)
    c2i = {c: i for i, c in enumerate(all_chars)}

    # Prepare sequences: SOS + name + EOS
    seqs = [[c2i.get(c, 0) for c in SOS + name + EOS] for name in all_names if len(name) >= 2]

    from sklearn.model_selection import train_test_split
    train_seqs, val_seqs = train_test_split(seqs, test_size=0.1, random_state=seed)

    model = CharRNN(n_chars, hidden_size, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val = float("inf")
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batch = 0.0, 0
        batch = random.sample(train_seqs, min(len(train_seqs), 5000))
        for seq in batch:
            x = torch.tensor(seq[:-1], device=device).unsqueeze(0)
            y = torch.tensor(seq[1:], device=device).unsqueeze(0)
            optimizer.zero_grad()
            out, _ = model(x)
            loss = criterion(out.view(-1, n_chars), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1

        avg_train = total_loss / max(n_batch, 1)

        model.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for seq in random.sample(val_seqs, min(len(val_seqs), 500)):
                x = torch.tensor(seq[:-1], device=device).unsqueeze(0)
                y = torch.tensor(seq[1:], device=device).unsqueeze(0)
                out, _ = model(x)
                val_loss += criterion(out.view(-1, n_chars), y.view(-1)).item()
                vn += 1
        avg_val = val_loss / max(vn, 1)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), dirs["checkpoints"] / "char_rnn_best.pt")

        history.append({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val})
        if epoch % 5 == 0 or epoch == epochs:
            logger.info("[%s] Epoch %d/%d  train=%.4f  val=%.4f", slug, epoch, epochs, avg_train, avg_val)

    # Generate samples
    model.load_state_dict(torch.load(dirs["checkpoints"] / "char_rnn_best.pt", weights_only=True))
    model.eval()
    samples: list[str] = []
    sos_idx, eos_idx = c2i[SOS], c2i[EOS]
    for _ in range(30):
        chars = []
        x = torch.tensor([[sos_idx]], device=device)
        hidden = None
        for _ in range(25):
            out, hidden = model(x, hidden)
            probs = torch.softmax(out[0, -1], dim=0)
            idx = torch.multinomial(probs, 1).item()
            if idx == eos_idx:
                break
            chars.append(all_chars[idx])
            x = torch.tensor([[idx]], device=device)
        if chars:
            samples.append("".join(chars))

    results = {
        "slug": slug,
        "task": "generation",
        "model": "CharRNN (LSTM)",
        "n_names": len(all_names),
        "n_languages": len(lang_counts),
        "vocab_size": n_chars,
        "params": sum(p.numel() for p in model.parameters()),
        "best_val_loss": best_val,
        "train_history": history,
        "generated_samples": samples,
        "status": "OK",
    }
    save_json(results, metrics_file)
    cleanup_gpu()
    return results


def run_char_rnn_project(project_slug, project_dir, raw_paths, processed_dir, outputs_dir, config, force=False):
    """Unified entry point for character-level RNN, called by orchestrator."""
    names_dir = config.get("names_dir", project_dir)
    _KEYS = {"hidden_size","n_layers","epochs","lr","seed"}
    kw = {k: v for k, v in config.items() if k in _KEYS}
    r = train_char_rnn(project_slug, names_dir=names_dir, force=force, **kw)
    return {
        "status": r.get("status", "UNKNOWN"),
        "model_name": "CharRNN (LSTM)",
        "dataset_size": r.get("n_names", 0),
        "main_metrics": {"best_val_loss": r.get("best_val_loss", 0)},
        "val_metrics": {},
        "training_mode": "full",
        "train_runtime_sec": 0,
        "notes": f"{r.get('n_languages', 0)} languages, {r.get('vocab_size', 0)} chars",
        "full_result": r,
    }
