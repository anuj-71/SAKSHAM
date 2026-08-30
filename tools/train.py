"""
tools/train.py
==============
Generic GRU trainer for SAKSHAM ISL recognition.

Dataset contract (dataset-agnostic):
  dataset/metadata/manifest.csv   - rows: file_path, label, signer_id, source, split
  dataset/processed/<file_path>   - .npz files with key "isometric_landmarks"
                                    shape: (T, 128) float32, T can vary

The trainer is intentionally NOT tied to any specific dataset source (CISLR, ASL, etc.).
To swap datasets, populate dataset/raw/ and dataset/processed/ differently
and update dataset/metadata/manifest.csv — no changes to this file required.

Output:
  models/gru_best.pt       - best PyTorch checkpoint (val accuracy)
  models/model.onnx        - exported ONNX for runtime inference
  models/model.onnx.data   - external weights (if onnx externalises them)
  models/label_map.json    - {class_name: index} mapping for inference
  models/training_curves.png
  models/confusion_matrix.png
  models/evaluation_results.json

Usage:
  cd E:\\gessture
  .venv\\Scripts\\python.exe tools/train.py [--epochs N] [--batch B] [--hidden H]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths (from configs/config.py, but we import directly so the script is
#    runnable standalone without the app installed) ──────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
MANIFEST_CSV   = PROJECT_ROOT / "dataset" / "metadata" / "manifest.csv"
PROCESSED_DIR  = PROJECT_ROOT / "dataset" / "processed"
MODELS_DIR     = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Architecture hyper-parameters (override via CLI) ────────────────────────
SEQ_LEN      = 64      # frames fed to GRU per inference call
FEATURE_DIM  = 128     # isometric landmark feature dim (21 pts × xyz + zeros pad)
HIDDEN_SIZE  = 256
NUM_LAYERS   = 2
DROPOUT      = 0.3

# ── Training defaults ────────────────────────────────────────────────────────
DEFAULT_EPOCHS     = 80
DEFAULT_BATCH      = 32
DEFAULT_LR         = 1e-3
DEFAULT_VAL_SPLIT  = 0.15
DEFAULT_PATIENCE   = 12    # early-stop if val loss doesn't improve for N epochs
RANDOM_SEED        = 42


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_manifest(csv_path: Path) -> List[Dict]:
    """Load manifest.csv; returns list of row dicts."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {csv_path}\n"
            "Run the CISLR extraction pipeline first (tools/cislr/extract_landmarks.py)."
        )
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    log.info("Manifest: %d rows loaded from %s", len(rows), csv_path)
    return rows


def build_label_map(rows: List[Dict]) -> Dict[str, int]:
    """Build {class_name: index} from manifest labels, sorted for determinism."""
    labels = sorted({r["label"] for r in rows if r.get("label")})
    return {lbl: idx for idx, lbl in enumerate(labels)}


def load_npz(file_path: Path) -> Optional[np.ndarray]:
    """
    Load a single .npz file.  Returns float32 array of shape (T, 128).
    Accepts either:
      - key "isometric_landmarks"  (new pipeline)
      - key "landmarks"            (legacy key, for backward compat)
    Returns None on failure.
    """
    try:
        data = np.load(file_path, allow_pickle=False)
        for key in ("isometric_landmarks", "landmarks", "features"):
            if key in data:
                arr = data[key].astype(np.float32)
                if arr.ndim == 2 and arr.shape[-1] == FEATURE_DIM:
                    return arr
                log.debug("Skipping %s[%s]: shape %s unexpected", file_path.name, key, arr.shape)
        log.warning("No usable key in %s (keys: %s)", file_path.name, list(data.keys()))
        return None
    except Exception as exc:
        log.warning("Failed to load %s: %s", file_path, exc)
        return None


def pad_or_crop(arr: np.ndarray, seq_len: int) -> np.ndarray:
    """Pad (repeat-edge) or centre-crop to exactly seq_len frames."""
    T = arr.shape[0]
    if T >= seq_len:
        # Centre-crop
        start = (T - seq_len) // 2
        return arr[start: start + seq_len]
    else:
        # Pad by repeating last frame
        pad = np.tile(arr[-1:], (seq_len - T, 1))
        return np.concatenate([arr, pad], axis=0)


def load_dataset(
    rows: List[Dict],
    label_map: Dict[str, int],
    split: str = "train",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all .npz files for the given split.
    Returns (X, y): X shape (N, SEQ_LEN, FEATURE_DIM), y shape (N,) int.
    """
    X_list, y_list = [], []
    skipped = 0
    for row in rows:
        if row.get("split", "train") != split:
            continue
        label = row.get("label", "")
        if label not in label_map:
            skipped += 1
            continue
        fp = PROCESSED_DIR / row["file_path"]
        arr = load_npz(fp)
        if arr is None:
            skipped += 1
            continue
        X_list.append(pad_or_crop(arr, SEQ_LEN))
        y_list.append(label_map[label])

    if not X_list:
        raise RuntimeError(
            f"No samples loaded for split='{split}'.  "
            "Check that dataset/processed/ is populated and manifest.csv is correct."
        )
    log.info("Split '%s': %d samples loaded, %d skipped", split, len(X_list), skipped)
    return np.stack(X_list).astype(np.float32), np.array(y_list, dtype=np.int64)


def train_val_split(
    rows: List[Dict],
    val_fraction: float = DEFAULT_VAL_SPLIT,
    seed: int = RANDOM_SEED,
) -> List[Dict]:
    """
    If manifest has no 'split' column, create a stratified train/val split
    in-place and return the modified rows.
    """
    rng = np.random.default_rng(seed)
    has_split = any(r.get("split") for r in rows)
    if has_split:
        log.info("Using split column from manifest.")
        return rows

    log.info("No 'split' column found — creating stratified %.0f%% val split.", val_fraction * 100)
    by_label: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_label[r["label"]].append(i)

    for label, idxs in by_label.items():
        idxs_arr = np.array(idxs)
        rng.shuffle(idxs_arr)
        n_val = max(1, int(len(idxs_arr) * val_fraction))
        for j in idxs_arr[:n_val]:
            rows[j]["split"] = "val"
        for j in idxs_arr[n_val:]:
            rows[j]["split"] = "train"
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

def build_model(num_classes: int, hidden: int = HIDDEN_SIZE):
    """Build and return a GRU classifier (requires torch)."""
    import torch
    import torch.nn as nn

    class GRUClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(
                input_size=FEATURE_DIM,
                hidden_size=hidden,
                num_layers=NUM_LAYERS,
                batch_first=True,
                dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
                bidirectional=False,
            )
            self.norm = nn.LayerNorm(hidden)
            self.head = nn.Sequential(
                nn.Dropout(DROPOUT),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, num_classes),
            )

        def forward(self, x):          # x: (B, T, F)
            out, _ = self.gru(x)       # (B, T, H)
            last = out[:, -1, :]       # (B, H)
            return self.head(self.norm(last))

    return GRUClassifier()


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    label_map: Dict[str, int],
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH,
    lr: float = DEFAULT_LR,
    hidden: int = HIDDEN_SIZE,
    patience: int = DEFAULT_PATIENCE,
):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on %s", device)

    num_classes = len(label_map)
    model = build_model(num_classes, hidden).to(device)

    # Class-weighted loss to handle imbalance
    class_counts = Counter(y_train.tolist())
    total = sum(class_counts.values())
    weights = torch.tensor(
        [total / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    best_val_acc  = 0.0
    best_epoch    = 0
    no_improve    = 0
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        scheduler.step()
        train_loss = epoch_loss / len(train_ds)

        # ── Validate ──
        model.eval()
        val_loss_sum, correct, total_val = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss_sum += criterion(logits, yb).item() * len(xb)
                correct += (logits.argmax(1) == yb).sum().item()
                total_val += len(xb)
        val_loss = val_loss_sum / total_val
        val_acc  = correct / total_val

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        log.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.3f%s",
            epoch, epochs, train_loss, val_loss, val_acc,
            "  ← best" if val_acc > best_val_acc else "",
        )

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_epoch    = epoch
            no_improve    = 0
            torch.save(model.state_dict(), MODELS_DIR / "gru_best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("Early stopping at epoch %d (no improvement for %d epochs).", epoch, patience)
                break

    log.info("Best val_acc=%.3f at epoch %d", best_val_acc, best_epoch)

    # Reload best weights
    model.load_state_dict(torch.load(MODELS_DIR / "gru_best.pt", map_location=device))
    return model, device, train_losses, val_losses, val_accs, best_val_acc


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation & Export
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, X_val, y_val, label_map, device) -> Dict:
    import torch
    idx_to_label = {v: k for k, v in label_map.items()}
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        Xv = torch.from_numpy(X_val).to(device)
        for i in range(0, len(Xv), 64):
            batch = Xv[i: i + 64]
            preds = model(batch).argmax(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_true.extend(y_val[i: i + 64].tolist())

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)
    acc = float((all_preds == all_true).mean())

    per_class = {}
    for idx, name in idx_to_label.items():
        mask = all_true == idx
        if mask.sum() == 0:
            continue
        per_class[name] = {
            "total":    int(mask.sum()),
            "correct":  int((all_preds[mask] == all_true[mask]).sum()),
            "accuracy": float((all_preds[mask] == all_true[mask]).mean()),
        }

    return {"overall_accuracy": acc, "per_class": per_class}


def save_training_curves(train_losses, val_losses, val_accs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, label="Train Loss")
        ax1.plot(epochs, val_losses, label="Val Loss")
        ax1.set_title("Loss")
        ax1.legend()
        ax2.plot(epochs, val_accs, label="Val Acc", color="green")
        ax2.set_title("Validation Accuracy")
        ax2.legend()
        fig.tight_layout()
        fig.savefig(MODELS_DIR / "training_curves.png", dpi=120)
        plt.close(fig)
        log.info("Saved training_curves.png")
    except ImportError:
        log.warning("matplotlib not available — skipping training curves.")


def save_confusion_matrix(model, X_val, y_val, label_map, device):
    try:
        import torch
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from collections import defaultdict

        idx_to_label = {v: k for k, v in label_map.items()}
        num_classes = len(label_map)
        cm = np.zeros((num_classes, num_classes), dtype=int)

        model.eval()
        with torch.no_grad():
            Xv = torch.from_numpy(X_val).to(device)
            for i in range(0, len(Xv), 64):
                batch = Xv[i: i + 64]
                preds = model(batch).argmax(1).cpu().numpy()
                trues = y_val[i: i + 64]
                for t, p in zip(trues, preds):
                    cm[t][p] += 1

        labels = [idx_to_label[i] for i in range(num_classes)]
        fig, ax = plt.subplots(figsize=(max(6, num_classes), max(5, num_classes - 1)))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (Val Set)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(MODELS_DIR / "confusion_matrix.png", dpi=120)
        plt.close(fig)
        log.info("Saved confusion_matrix.png")
    except Exception as exc:
        log.warning("Could not save confusion matrix: %s", exc)


def export_onnx(model, device, label_map: Dict[str, int]):
    import torch
    model.eval()
    dummy = torch.zeros(1, SEQ_LEN, FEATURE_DIM).to(device)
    onnx_path = MODELS_DIR / "model.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    # Save label_map alongside model
    label_map_path = MODELS_DIR / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2, sort_keys=True)
    log.info("Exported ONNX → %s", onnx_path)
    log.info("Saved label_map → %s", label_map_path)
    return onnx_path


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train SAKSHAM ISL GRU classifier")
    p.add_argument("--epochs",  type=int,   default=DEFAULT_EPOCHS,    help="Max training epochs")
    p.add_argument("--batch",   type=int,   default=DEFAULT_BATCH,     help="Batch size")
    p.add_argument("--lr",      type=float, default=DEFAULT_LR,        help="Learning rate")
    p.add_argument("--hidden",  type=int,   default=HIDDEN_SIZE,       help="GRU hidden size")
    p.add_argument("--patience",type=int,   default=DEFAULT_PATIENCE,  help="Early-stop patience (epochs)")
    p.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT, help="Val fraction if no split col")
    return p.parse_args()


def main():
    args = parse_args()

    log.info("=" * 60)
    log.info("  SAKSHAM GRU Trainer")
    log.info("  Manifest:  %s", MANIFEST_CSV)
    log.info("  Processed: %s", PROCESSED_DIR)
    log.info("  Models:    %s", MODELS_DIR)
    log.info("=" * 60)

    # ── Load manifest ──────────────────────────────────────────
    rows = load_manifest(MANIFEST_CSV)
    rows = train_val_split(rows, val_fraction=args.val_split)

    label_map = build_label_map(rows)
    log.info("Classes (%d): %s", len(label_map), list(label_map.keys()))

    # ── Load .npz files ────────────────────────────────────────
    X_train, y_train = load_dataset(rows, label_map, split="train")
    X_val,   y_val   = load_dataset(rows, label_map, split="val")

    log.info("Train: %s  Val: %s", X_train.shape, X_val.shape)
    log.info("Class distribution (train): %s",
             {k: int((y_train == v).sum()) for k, v in label_map.items()})

    # ── Train ──────────────────────────────────────────────────
    model, device, train_losses, val_losses, val_accs, best_val_acc = train(
        X_train, y_train, X_val, y_val, label_map,
        epochs=args.epochs, batch_size=args.batch,
        lr=args.lr, hidden=args.hidden, patience=args.patience,
    )

    # ── Evaluate ───────────────────────────────────────────────
    eval_results = evaluate(model, X_val, y_val, label_map, device)
    eval_results["best_val_acc_during_training"] = best_val_acc
    log.info("Final eval accuracy: %.3f", eval_results["overall_accuracy"])
    with open(MODELS_DIR / "evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    log.info("Saved evaluation_results.json")

    # ── Plots ──────────────────────────────────────────────────
    save_training_curves(train_losses, val_losses, val_accs)
    save_confusion_matrix(model, X_val, y_val, label_map, device)

    # ── Export ONNX ────────────────────────────────────────────
    onnx_path = export_onnx(model, device, label_map)

    log.info("=" * 60)
    log.info("  Training complete.")
    log.info("  Val accuracy:   %.1f%%", eval_results["overall_accuracy"] * 100)
    log.info("  Model:          %s", onnx_path)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
