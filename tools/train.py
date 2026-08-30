"""
tools/train.py
==============
High-Performance CNN-GRU Trainer for SAKSHAM MediaPipe Holistic (225 features).

Features:
  - 1D-CNN Temporal Feature Extractor + Bidirectional GRU Stack
  - Input shape: (30, 225) (21 Left + 21 Right + 33 Pose landmarks)
  - 14 Classes (13 Active Sign Classes + 1 Null Class)
  - Weighted Cross-Entropy Loss with inverse frequency class weighting
  - 50 Epochs training with Cosine Annealing learning rate schedule
  - Per-class validation accuracy breakdown table
  - Automatic ONNX model export for real-time CPU inference

Usage:
  .venv\\Scripts\\python.exe tools/train.py [--epochs 50] [--batch 32] [--lr 1e-3]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SAKSHAM-Holistic-Trainer")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "processed"
KAGGLE_HOLISTIC_DIR = DATASET_DIR / "kaggle_holistic"
PERSONAL_DIR = DATASET_DIR / "personal"
MODELS_DIR = PROJECT_ROOT / "models"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"

SEQUENCE_LENGTH = 30
INPUT_FEATURES = 225  # 21 Left (63) + 21 Right (63) + 33 Pose (99)


# ─── Dataset & Data Loader ───────────────────────────────────────────────────

class HolisticSignDataset(Dataset):
    """Loads processed .npz files containing 30x225 holistic feature arrays."""

    def __init__(self, samples: List[Tuple[Path, int]]):
        self.data: List[Tuple[np.ndarray, int]] = []

        for npz_path, label_idx in samples:
            if not npz_path.exists():
                continue
            try:
                with np.load(npz_path) as npz:
                    arr = npz["features"] if "features" in npz else npz["isometric_landmarks"]
                    if arr.shape != (SEQUENCE_LENGTH, INPUT_FEATURES):
                        if arr.shape[0] != SEQUENCE_LENGTH or arr.shape[1] != INPUT_FEATURES:
                            continue
                    self.data.append((arr.astype(np.float32), label_idx))
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features, label = self.data[idx]
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)


# ─── CNN-GRU Model Architecture ──────────────────────────────────────────────

class CNNGRUModel(nn.Module):
    """
    Hybrid 1D-CNN + Bidirectional GRU for robust 225-feature holistic sign recognition.
    """

    def __init__(self, input_dim: int = 225, hidden_dim: int = 128,
                 num_layers: int = 2, num_classes: int = 14, dropout: float = 0.25):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # 1D-CNN Temporal Feature Extractor: Input shape (B, C_in, T)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Bidirectional GRU: Input shape (B, T, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Classification Head (256 -> 128 -> num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (Batch, Seq_Len=30, Feat=225)
        # Permute for 1D Conv: (Batch, Feat=225, Seq_Len=30)
        x_conv = x.permute(0, 2, 1)
        cnn_out = self.cnn(x_conv)  # (Batch, hidden_dim, Seq_Len)

        # Permute back for GRU: (Batch, Seq_Len, hidden_dim)
        gru_in = cnn_out.permute(0, 2, 1)
        gru_out, _ = self.gru(gru_in)  # (Batch, Seq_Len, hidden_dim * 2)

        # Global average pooling across time steps
        pooled = torch.mean(gru_out, dim=1)  # (Batch, hidden_dim * 2)

        # Classification logits
        logits = self.classifier(pooled)  # (Batch, num_classes)
        return logits


# ─── Training & Evaluation Routines ─────────────────────────────────────────

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / max(1, total), (correct / max(1, total)) * 100.0


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * len(y_batch)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    return (
        total_loss / max(1, total),
        (correct / max(1, total)) * 100.0,
        np.array(all_preds),
        np.array(all_labels),
    )


def export_onnx(model: nn.Module, onnx_path: Path, input_shape: Tuple[int, int, int] = (1, SEQUENCE_LENGTH, INPUT_FEATURES)):
    """Exports trained PyTorch model to ONNX with dynamic batch axis."""
    model.eval()
    dummy_input = torch.randn(*input_shape, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,
    )
    log.info("Successfully exported ONNX model to: %s (%.2f MB)",
             onnx_path, onnx_path.stat().st_size / (1024 * 1024))


def main():
    parser = argparse.ArgumentParser(description="Train SAKSHAM CNN-GRU model on 225-feature holistic dataset.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dimension (default: 128)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15)")
    args = parser.parse_args()

    if not LABEL_MAP_PATH.exists():
        log.error("Label map not found at %s.", LABEL_MAP_PATH)
        sys.exit(1)

    # 1. Load label map
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map: Dict[str, int] = json.load(f)

    num_classes = len(label_map)
    idx_to_label = {v: k for k, v in label_map.items()}
    log.info("Loaded label map with %d classes: %s", num_classes, list(label_map.keys()))

    # 2. Collect all available samples across kaggle_holistic and personal directories
    samples_by_class: Dict[int, List[Path]] = {i: [] for i in range(num_classes)}

    # Scan Kaggle holistic directory
    if KAGGLE_HOLISTIC_DIR.exists():
        for sign_folder in KAGGLE_HOLISTIC_DIR.iterdir():
            if sign_folder.is_dir() and sign_folder.name in label_map:
                c_idx = label_map[sign_folder.name]
                for p in sign_folder.glob("*.npz"):
                    samples_by_class[c_idx].append(p)

    # Scan Personal recordings directory
    if PERSONAL_DIR.exists():
        for sign_folder in PERSONAL_DIR.iterdir():
            if sign_folder.is_dir() and sign_folder.name in label_map:
                c_idx = label_map[sign_folder.name]
                for p in sign_folder.glob("*.npz"):
                    samples_by_class[c_idx].append(p)

    log.info("Collected sample counts per class:")
    for c_idx in range(num_classes):
        c_name = idx_to_label[c_idx]
        log.info("  - %-12s: %4d samples", c_name, len(samples_by_class[c_idx]))

    # 3. Stratified 80/20 train/val split
    train_samples: List[Tuple[Path, int]] = []
    val_samples: List[Tuple[Path, int]] = []
    np.random.seed(42)

    class_counts = []
    for c_idx in range(num_classes):
        paths = samples_by_class[c_idx]
        count = len(paths)
        class_counts.append(max(1, count))
        np.random.shuffle(paths)
        split_pt = int(count * 0.8)
        for p in paths[:split_pt]:
            train_samples.append((p, c_idx))
        for p in paths[split_pt:]:
            val_samples.append((p, c_idx))

    log.info("Dataset splits: Train=%d samples, Val=%d samples", len(train_samples), len(val_samples))

    # 4. Compute Weighted Cross-Entropy Loss weights
    # Inverse class frequency: weight_c = total / (num_classes * count_c)
    total_train = len(train_samples)
    class_weights = np.zeros(num_classes, dtype=np.float32)
    for c_idx in range(num_classes):
        c_count = sum(1 for _, l in train_samples if l == c_idx)
        if c_count > 0:
            w = total_train / (num_classes * c_count)
            # Apply slight dampening on null class weight if it has abundant samples
            if idx_to_label[c_idx] == "null":
                w *= 0.75
            class_weights[c_idx] = w
        else:
            class_weights[c_idx] = 1.0

    # Normalize weights so mean is 1.0
    class_weights = class_weights / np.mean(class_weights)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
    log.info("Computed Cross-Entropy Class Weights: %s", {idx_to_label[i]: round(float(w), 3) for i, w in enumerate(class_weights)})

    # 5. Create Datasets and DataLoaders
    train_dataset = HolisticSignDataset(train_samples)
    val_dataset = HolisticSignDataset(val_samples)

    log.info("Loaded into RAM: Train=%d, Val=%d", len(train_dataset), len(val_dataset))
    if len(train_dataset) == 0:
        log.error("No valid training samples loaded. Check dataset folders!")
        sys.exit(1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=0)

    # 6. Initialize Model
    device = torch.device("cpu")
    model = CNNGRUModel(
        input_dim=INPUT_FEATURES,
        hidden_dim=args.hidden,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.25,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Initialized Holistic CNN-GRU model (%d trainable parameters) on CPU", total_params)

    # 7. Optimizer, Scheduler, Weighted Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(device), label_smoothing=0.05)

    # 8. Training Loop
    best_val_acc = 0.0
    best_epoch = 0
    best_preds = None
    best_labels = None
    patience_counter = 0
    best_weights_path = MODELS_DIR / "gru_best.pt"
    onnx_path = MODELS_DIR / "model.onnx"

    log.info("=" * 70)
    log.info("STARTING TRAINING (%d Epochs, Batch Size=%d, Input Features=%d)", args.epochs, args.batch, INPUT_FEATURES)
    log.info("=" * 70)

    t0_train = time.time()

    for epoch in range(1, args.epochs + 1):
        t0_epoch = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, preds, labels = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        epoch_sec = time.time() - t0_epoch

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            best_preds = preds
            best_labels = labels
            patience_counter = 0
            torch.save(model.state_dict(), best_weights_path)
            star = " * [BEST]"
        else:
            patience_counter += 1
            star = ""

        log.info(
            "Epoch %2d/%2d  [%4.1fs]  Train Loss: %.4f | Train Acc: %5.1f%%  ||  Val Loss: %.4f | Val Acc: %5.1f%%%s",
            epoch, args.epochs, epoch_sec, train_loss, train_acc, val_loss, val_acc, star
        )

        if patience_counter >= args.patience:
            log.info("Early stopping triggered after %d epochs without improvement.", args.patience)
            break

    total_time = time.time() - t0_train
    log.info("=" * 70)
    log.info("TRAINING FINISHED in %.1f seconds (~%.1f min)", total_time, total_time / 60.0)
    log.info("Best Validation Accuracy: %.2f%% (Epoch %d)", best_val_acc, best_epoch)

    # 9. Load best weights and export ONNX
    if best_weights_path.exists():
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
        log.info("Loaded best model checkpoint from %s", best_weights_path)

    export_onnx(model, onnx_path)

    # 10. Compute Per-Class Validation Accuracy Breakdown Table
    val_loss, final_val_acc, final_preds, final_labels = evaluate(model, val_loader, criterion, device)

    per_class_results = []
    log.info("\n" + "=" * 70)
    log.info("PER-CLASS VALIDATION ACCURACY BREAKDOWN (13 ACTIVE SIGN CLASSES)")
    log.info("=" * 70)
    log.info(f"{'Class Name':<14} | {'Val Samples':<12} | {'Correct':<9} | {'Accuracy':<10}")
    log.info("-" * 70)

    for c_idx in range(num_classes):
        c_name = idx_to_label[c_idx]
        mask = (final_labels == c_idx)
        n_samples = int(np.sum(mask))
        if n_samples > 0:
            c_correct = int(np.sum(final_preds[mask] == c_idx))
            c_acc = (c_correct / n_samples) * 100.0
        else:
            c_correct = 0
            c_acc = 0.0

        per_class_results.append({
            "class": c_name,
            "samples": n_samples,
            "correct": c_correct,
            "accuracy": float(c_acc),
        })
        log.info(f"{c_name:<14} | {n_samples:<12} | {c_correct:<9} | {c_acc:6.1f}%")

    log.info("-" * 70)
    log.info(f"{'OVERALL':<14} | {len(final_labels):<12} | {int(np.sum(final_preds == final_labels)):<9} | {final_val_acc:6.1f}%")
    log.info("=" * 70 + "\n")

    # Save evaluation summary
    eval_summary = {
        "num_classes": num_classes,
        "classes": list(label_map.keys()),
        "best_epoch": best_epoch,
        "best_val_accuracy": float(best_val_acc),
        "total_training_seconds": float(total_time),
        "sequence_length": SEQUENCE_LENGTH,
        "input_features": INPUT_FEATURES,
        "per_class_accuracy": per_class_results,
    }
    with open(MODELS_DIR / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    log.info("Training and ONNX export complete! SAKSHAM is ready for live recognition.")


if __name__ == "__main__":
    main()
