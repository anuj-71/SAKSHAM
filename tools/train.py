"""
tools/train.py
==============
High-Performance CNN-GRU Trainer for SAKSHAM Sign Language Recognition.

Architecture:
  - 1D-CNN Temporal Feature Extractor: Captures fine-grained local kinematic motion
  - Bidirectional GRU Stack: Captures long-range gesture trajectories
  - Dynamic Classification Head: Automatically sizes to num_classes in label_map.json

Optimized for:
  - CPU i3 Training: batch_size=32, num_workers=0, lightweight memory footprint
  - Zero-Latency Inference: Automatically exports optimized ONNX model at completion

Usage:
  .venv\\Scripts\\python.exe tools/train.py [--epochs 40] [--batch 32] [--lr 1e-3]
"""

from __future__ import annotations

import argparse
import csv
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
log = logging.getLogger("SAKSHAM-Trainer")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_CSV = PROJECT_ROOT / "dataset" / "metadata" / "manifest.csv"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"

SEQUENCE_LENGTH = 30
INPUT_FEATURES = 126


# ─── Dataset & Data Loader ───────────────────────────────────────────────────

class SignDataset(Dataset):
    """Loads processed .npz files containing 30x126 landmark feature arrays."""

    def __init__(self, samples: List[Dict], label_map: Dict[str, int]):
        self.samples = samples
        self.label_map = label_map
        self.data: List[Tuple[np.ndarray, int]] = []

        # Preload into memory (fits easily in <150MB RAM for 15 classes)
        for s in samples:
            npz_path = PROCESSED_DIR / s["file_path"]
            if not npz_path.exists():
                continue
            try:
                with np.load(npz_path) as npz:
                    arr = npz["features"] if "features" in npz else npz["isometric_landmarks"]
                    if arr.shape != (SEQUENCE_LENGTH, INPUT_FEATURES):
                        # Ensure correct shape
                        if arr.shape[0] != SEQUENCE_LENGTH or arr.shape[1] != INPUT_FEATURES:
                            continue
                    label_idx = self.label_map[s["label"]]
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
    Hybrid 1D-CNN + Bidirectional GRU for robust sign gesture recognition.
    """

    def __init__(self, input_dim: int = 126, hidden_dim: int = 128,
                 num_layers: int = 2, num_classes: int = 15, dropout: float = 0.25):
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
        # Input: (Batch, Seq_Len=30, Feat=126)
        # Permute for 1D Conv: (Batch, Feat=126, Seq_Len=30)
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


# ─── Training & Export Routines ──────────────────────────────────────────────

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
    parser = argparse.ArgumentParser(description="Train SAKSHAM CNN-GRU model on processed landmark dataset.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs (default: 40)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dimension (default: 128)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    args = parser.parse_args()

    if not MANIFEST_CSV.exists():
        log.error("Manifest not found at %s. Please run tools/import_google_asl.py first!", MANIFEST_CSV)
        sys.exit(1)

    if not LABEL_MAP_PATH.exists():
        log.error("Label map not found at %s. Please run tools/import_google_asl.py first!", LABEL_MAP_PATH)
        sys.exit(1)

    # 1. Load label map
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map: Dict[str, int] = json.load(f)

    num_classes = len(label_map)
    log.info("Loaded label map with %d classes: %s", num_classes, list(label_map.keys()))

    # 2. Read manifest
    samples_train = []
    samples_val = []
    with open(MANIFEST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["label"] in label_map:
                if row.get("split") == "val":
                    samples_val.append(row)
                else:
                    samples_train.append(row)

    # Fallback 80/20 split if no validation split found
    if not samples_val:
        all_samples = samples_train
        np.random.seed(42)
        np.random.shuffle(all_samples)
        split_idx = int(len(all_samples) * 0.8)
        samples_train = all_samples[:split_idx]
        samples_val = all_samples[split_idx:]

    log.info("Dataset splits: Train=%d samples, Val=%d samples", len(samples_train), len(samples_val))

    # 3. Create Datasets and DataLoaders
    train_dataset = SignDataset(samples_train, label_map)
    val_dataset = SignDataset(samples_val, label_map)

    log.info("Loaded valid samples into RAM: Train=%d, Val=%d", len(train_dataset), len(val_dataset))

    if len(train_dataset) == 0:
        log.error("No valid training samples loaded from %s. Check dataset/processed/", PROCESSED_DIR)
        sys.exit(1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=0)

    # 4. Initialize Model
    device = torch.device("cpu")
    model = CNNGRUModel(
        input_dim=INPUT_FEATURES,
        hidden_dim=args.hidden,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.25,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Initialized CNN-GRU model (%d trainable parameters) on CPU", total_params)

    # 5. Optimizer, Scheduler, Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # 6. Training Loop
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_weights_path = MODELS_DIR / "gru_best.pt"
    onnx_path = MODELS_DIR / "model.onnx"

    log.info("=" * 65)
    log.info("STARTING TRAINING (%d Epochs, Batch Size=%d)", args.epochs, args.batch)
    log.info("=" * 65)

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
            patience_counter = 0
            torch.save(model.state_dict(), best_weights_path)
            star = " ★ [BEST]"
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
    log.info("=" * 65)
    log.info("TRAINING FINISHED in %.1f seconds (~%.1f min)", total_time, total_time / 60.0)
    log.info("Best Validation Accuracy: %.2f%% (Epoch %d)", best_val_acc, best_epoch)

    # 7. Load best weights and export ONNX
    if best_weights_path.exists():
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
        log.info("Loaded best model checkpoint from %s", best_weights_path)

    export_onnx(model, onnx_path)

    # Save evaluation summary
    eval_summary = {
        "num_classes": num_classes,
        "classes": list(label_map.keys()),
        "best_epoch": best_epoch,
        "best_val_accuracy": float(best_val_acc),
        "total_training_seconds": float(total_time),
        "sequence_length": SEQUENCE_LENGTH,
        "input_features": INPUT_FEATURES,
    }
    with open(MODELS_DIR / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    log.info("Training and ONNX export complete! SAKSHAM is ready for live recognition.")


if __name__ == "__main__":
    main()
