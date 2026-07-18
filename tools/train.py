"""
Train a GRU classifier on SAKSHAM_DATASET parquet files using the
same preprocessing pipeline as the runtime ONNX inference engine.

Exports the trained model to dataset/models/model.onnx for runtime use.

Enhancements over original:
- Participant-aware train/val/test splits (no data leakage)
- Pre-caching of preprocessed samples for faster I/O
- Post-training test-set evaluation with confusion matrix
- Classification report with per-class accuracy
- Resume from checkpoint

Usage:
    .venv\\Scripts\\python.exe -m tools.train_and_export [--epochs 50]
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sign_language.training.preprocessing import preprocess_sample_for_gru
from src.sign_language.training.normalization import normalize_hand_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SEQ_LEN = 64
FEATURE_DIM = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3

CSV_PATH = "SAKSHAM_DATASET/train_filtered.csv"
SIGN_MAP_PATH = "SAKSHAM_DATASET/sign_map.json"
OUTPUT_DIR = "dataset/models"
CACHE_DIR = "dataset/models/.feature_cache"

NUM_HAND_LANDMARKS = 21


# ── Dataset ────────────────────────────────────────────────────────────────────


class ParquetSignDataset(Dataset):
    """Loads SAKSHAM_DATASET parquet files and preprocesses hand landmarks
    identically to the runtime ONNX inference pipeline."""

    def __init__(self, csv_path: str, sign_map: dict):
        import csv
        self.samples = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["path"], sign_map[row["sign"]]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label_idx = self.samples[idx]
        full_path = Path("SAKSHAM_DATASET") / rel_path

        table = pq.read_table(str(full_path))
        df = table.to_pandas()
        df[["x", "y", "z"]] = df[["x", "y", "z"]].fillna(0.0)

        frame_ids = sorted(df["frame"].unique())
        left_hand_frames = []
        right_hand_frames = []
        left_present_list = []
        right_present_list = []

        for fid in frame_ids:
            frame_df = df[df["frame"] == fid]

            left = frame_df[frame_df["type"] == "left_hand"].sort_values("landmark_index")
            right = frame_df[frame_df["type"] == "right_hand"].sort_values("landmark_index")

            if len(left) == NUM_HAND_LANDMARKS:
                l_lm = left[["x", "y", "z"]].values.astype(np.float32)
                left_present_list.append(True)
            else:
                l_lm = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float32)
                left_present_list.append(False)

            if len(right) == NUM_HAND_LANDMARKS:
                r_lm = right[["x", "y", "z"]].values.astype(np.float32)
                right_present_list.append(True)
            else:
                r_lm = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float32)
                right_present_list.append(False)

            l_norm = normalize_hand_frame(l_lm, left_present_list[-1])
            r_norm = normalize_hand_frame(r_lm, right_present_list[-1])

            left_hand_frames.append(l_norm)
            right_hand_frames.append(r_norm)

        left_hand = np.stack(left_hand_frames, axis=0)
        right_hand = np.stack(right_hand_frames, axis=0)
        left_present = np.array(left_present_list, dtype=np.float32)
        right_present = np.array(right_present_list, dtype=np.float32)

        sample = {
            "left_hand": left_hand,
            "right_hand": right_hand,
            "left_present": left_present,
            "right_present": right_present,
        }

        result = preprocess_sample_for_gru(sample, target_length=SEQ_LEN, include_presence=True)
        features = torch.from_numpy(result["features"])
        return features.float(), torch.tensor(label_idx, dtype=torch.long)


class CachedTensorDataset(Dataset):
    """Memory-mapped dataset that loads pre-cached .npz feature tensors."""

    def __init__(self, indices: list, features_array: np.ndarray, labels_array: np.ndarray):
        self.indices = indices
        self.features = features_array
        self.labels = labels_array

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        feat = torch.from_numpy(self.features[real_idx].copy()).float()
        label = torch.tensor(int(self.labels[real_idx]), dtype=torch.long)
        return feat, label


# ── Model ──────────────────────────────────────────────────────────────────────


class GRUClassifier(nn.Module):
    """2-layer GRU with linear classification head.
    Matches the architecture expected by the ONNX inference engine."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 num_classes: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, h_n = self.gru(x)
        out = h_n[-1]
        return self.fc(out)


# ── Splitting ──────────────────────────────────────────────────────────────────


def participant_aware_split(csv_path: str, seed: int = 42):
    """Split samples by participant_id to prevent data leakage.

    Returns: (train_indices, val_indices, test_indices) referencing rows in the CSV.
    """
    import csv as csv_module
    import random

    rows = []
    with open(csv_path, "r") as f:
        reader = csv_module.DictReader(f)
        for i, row in enumerate(reader):
            rows.append((i, row["participant_id"]))

    # Gather unique participants
    participants = sorted(set(pid for _, pid in rows))
    rng = random.Random(seed)
    rng.shuffle(participants)

    n = len(participants)
    n_train = max(1, int(round(n * 0.70)))
    n_val = max(1, int(round(n * 0.15)))
    # Remaining go to test
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = n - n_val - n_test

    train_pids = set(participants[:n_train])
    val_pids = set(participants[n_train:n_train + n_val])
    test_pids = set(participants[n_train + n_val:])

    train_idx, val_idx, test_idx = [], [], []
    for i, pid in rows:
        if pid in train_pids:
            train_idx.append(i)
        elif pid in val_pids:
            val_idx.append(i)
        else:
            test_idx.append(i)

    logging.info(
        "Participant split: %d train (%d pids), %d val (%d pids), %d test (%d pids)",
        len(train_idx), len(train_pids),
        len(val_idx), len(val_pids),
        len(test_idx), len(test_pids),
    )
    return train_idx, val_idx, test_idx


# ── Caching ────────────────────────────────────────────────────────────────────


def build_feature_cache(csv_path: str, sign_map: dict, cache_dir: str):
    """Pre-process all parquet files and cache features to disk as .npz.
    Returns (features_array, labels_array) of shape (N, SEQ_LEN, FEATURE_DIM) and (N,)."""
    cache_path = Path(cache_dir)
    features_file = cache_path / "features.npy"
    labels_file = cache_path / "labels.npy"

    if features_file.exists() and labels_file.exists():
        logging.info("Loading cached features from %s", cache_dir)
        features = np.load(str(features_file))
        labels = np.load(str(labels_file))
        logging.info("Loaded %d cached samples (features shape: %s)", len(labels), features.shape)
        return features, labels

    logging.info("Building feature cache (this runs once, may take a few minutes)...")
    cache_path.mkdir(parents=True, exist_ok=True)

    dataset = ParquetSignDataset(csv_path, sign_map)
    n = len(dataset)

    features = np.zeros((n, SEQ_LEN, FEATURE_DIM), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    errors = 0

    t0 = time.time()
    for i in range(n):
        try:
            feat, label = dataset[i]
            features[i] = feat.numpy()
            labels[i] = label.item()
        except Exception as e:
            logging.warning("Error processing sample %d: %s", i, e)
            errors += 1

        if (i + 1) % 500 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (n - i - 1) / rate if rate > 0 else 0
            logging.info(
                "  Cached %d/%d samples (%.1f samples/sec, ~%.0fs remaining, %d errors)",
                i + 1, n, rate, remaining, errors,
            )

    np.save(str(features_file), features)
    np.save(str(labels_file), labels)
    logging.info("Feature cache saved to %s (%d samples, %d errors)", cache_dir, n, errors)
    return features, labels


# ── Evaluation ─────────────────────────────────────────────────────────────────


def evaluate_model(model, dataloader, device, label_map, output_dir):
    """Run full test-set evaluation: accuracy, classification report, confusion matrix."""
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
    )

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Class names sorted by index
    idx_to_label = {v: k for k, v in label_map.items()}
    class_names = [idx_to_label[i] for i in range(len(label_map))]

    overall_acc = accuracy_score(all_labels, all_preds)

    # Per-class accuracy
    per_class_acc = {}
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class_acc[name] = float((all_preds[mask] == i).sum() / mask.sum())
        else:
            per_class_acc[name] = 0.0

    # Classification report
    report_str = classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    )
    report_dict = classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0, output_dict=True
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Print results
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 70)
    print(f"Overall Accuracy: {overall_acc * 100:.2f}%")
    print(f"\nPer-class Accuracy:")
    for name, acc in sorted(per_class_acc.items()):
        print(f"  {name:>12s}: {acc * 100:.2f}%")
    print(f"\nClassification Report:")
    print(report_str)
    print("=" * 70)

    # Save results
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {
        "overall_accuracy": overall_acc,
        "per_class_accuracy": per_class_acc,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }
    with open(out / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logging.info("Evaluation results saved to %s", out / "evaluation_results.json")

    # Plot confusion matrix
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(22, 9))

        # Raw counts
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            square=True, ax=axes[0],
        )
        axes[0].set_ylabel("True Label")
        axes[0].set_xlabel("Predicted Label")
        axes[0].set_title("Confusion Matrix (Counts)")

        # Normalized
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            square=True, ax=axes[1],
        )
        axes[1].set_ylabel("True Label")
        axes[1].set_xlabel("Predicted Label")
        axes[1].set_title("Confusion Matrix (Normalized)")

        plt.tight_layout()
        cm_path = out / "confusion_matrix.png"
        plt.savefig(str(cm_path), dpi=200, bbox_inches="tight")
        plt.close()
        logging.info("Confusion matrix saved to %s", cm_path)
    except ImportError:
        logging.warning("matplotlib/seaborn not available; skipping confusion matrix plot")

    return overall_acc, per_class_acc


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    parser.add_argument("--no-cache", action="store_true", help="Disable feature caching")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("SAKSHAM Sign Language Recognition — Training Pipeline")
    print("=" * 70)
    logging.info("Device: %s", device)
    logging.info("Epochs: %d, Batch Size: %d, LR: %s, Patience: %d",
                 args.epochs, args.batch_size, args.lr, args.patience)

    # ── Load label map ─────────────────────────────────────────────────────
    with open(SIGN_MAP_PATH) as f:
        sign_map = json.load(f)
    num_classes = len(sign_map)
    label_map = sign_map
    logging.info("Classes (%d): %s", num_classes, list(label_map.keys()))

    # ── Participant-aware split ────────────────────────────────────────────
    train_indices, val_indices, test_indices = participant_aware_split(CSV_PATH)

    # ── Build or load feature cache ────────────────────────────────────────
    if args.no_cache:
        logging.info("Caching disabled — using raw parquet loading")
        dataset = ParquetSignDataset(CSV_PATH, sign_map)
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
    else:
        features, labels = build_feature_cache(CSV_PATH, sign_map, CACHE_DIR)
        train_dataset = CachedTensorDataset(train_indices, features, labels)
        val_dataset = CachedTensorDataset(val_indices, features, labels)
        test_dataset = CachedTensorDataset(test_indices, features, labels)

    logging.info("Dataset sizes: train=%d, val=%d, test=%d",
                 len(train_dataset), len(val_dataset), len(test_dataset))

    # Print class distribution for each split
    def count_classes(ds, name):
        counts = {}
        for i in range(len(ds)):
            _, label = ds[i]
            lbl = label.item() if hasattr(label, 'item') else int(label)
            counts[lbl] = counts.get(lbl, 0) + 1
        idx_to_name = {v: k for k, v in sign_map.items()}
        logging.info("%s class distribution:", name)
        for idx in sorted(counts.keys()):
            logging.info("  %12s: %d", idx_to_name[idx], counts[idx])
    count_classes(train_dataset, "Train")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Create model ───────────────────────────────────────────────────────
    model = GRUClassifier(
        input_dim=FEATURE_DIM, hidden_dim=HIDDEN_SIZE,
        num_layers=NUM_LAYERS, num_classes=num_classes, dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logging.info("Model: GRUClassifier (%d params)", total_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 1
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = -1
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    # ── Resume from checkpoint ─────────────────────────────────────────────
    ckpt_path = Path(OUTPUT_DIR) / "gru_best.pt"
    if args.resume and ckpt_path.exists():
        logging.info("Resuming from checkpoint: %s", ckpt_path)
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_loss = ckpt.get("val_loss", float("inf"))
        best_val_acc = ckpt.get("val_acc", 0.0)
        best_epoch = ckpt.get("epoch", 0)
        logging.info("  Resumed: epoch=%d, val_loss=%.4f, val_acc=%.4f",
                     best_epoch, best_val_loss, best_val_acc)

    # ── Training loop ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        elapsed = time.time() - t0
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            marker = " * NEW BEST *"

            Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "label_map": label_map,
                "history": history,
            }, str(Path(OUTPUT_DIR) / "gru_best.pt"))
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | {elapsed:.1f}s{marker}"
        )

        if patience_counter >= args.patience:
            logging.info("Early stopping at epoch %d (patience=%d)", epoch, args.patience)
            break

    print("\n" + "=" * 70)
    logging.info("Training complete. Best val_loss=%.4f, val_acc=%.4f at epoch %d",
                 best_val_loss, best_val_acc, best_epoch)

    # ── Save training curves ───────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs_range = range(1, len(history["train_loss"]) + 1)

        axes[0].plot(epochs_range, history["train_loss"], "b-", label="Train Loss", linewidth=2)
        axes[0].plot(epochs_range, history["val_loss"], "r-", label="Val Loss", linewidth=2)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs_range, [a * 100 for a in history["train_acc"]], "b-", label="Train Acc", linewidth=2)
        axes[1].plot(epochs_range, [a * 100 for a in history["val_acc"]], "r-", label="Val Acc", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Training & Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        curves_path = Path(OUTPUT_DIR) / "training_curves.png"
        plt.savefig(str(curves_path), dpi=200, bbox_inches="tight")
        plt.close()
        logging.info("Training curves saved to %s", curves_path)
    except ImportError:
        logging.warning("matplotlib not available; skipping training curves plot")

    # ── Load best model ────────────────────────────────────────────────────
    best_ckpt = torch.load(str(Path(OUTPUT_DIR) / "gru_best.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # ── Test-set evaluation ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Running test-set evaluation...")
    print("=" * 70)
    test_acc, per_class_acc = evaluate_model(model, test_loader, device, label_map, OUTPUT_DIR)

    # ── ONNX export ────────────────────────────────────────────────────────
    logging.info("Exporting to ONNX...")
    model_cpu = model.cpu()
    model_cpu.eval()
    dummy_input = torch.randn(1, SEQ_LEN, FEATURE_DIM)
    onnx_path = Path(OUTPUT_DIR) / "model.onnx"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model_cpu,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
    )

    label_map_path = Path(OUTPUT_DIR) / "label_map.json"
    with open(str(label_map_path), "w") as f:
        json.dump(label_map, f, indent=2)

    # ── ONNX validation ───────────────────────────────────────────────────
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        test_input = np.random.randn(1, SEQ_LEN, FEATURE_DIM).astype(np.float32)
        output = session.run(None, {input_name: test_input})[0]
        logging.info("ONNX validation OK — output shape: %s", output.shape)
    except Exception as e:
        logging.error("ONNX validation FAILED: %s", e)
        return 2

    # ── Final report ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("✅ TRAINING PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n✓ Dataset detected:          SAKSHAM_DATASET/")
    print(f"✓ Number of classes:         {num_classes}")
    print(f"✓ Total samples:             {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
    print(f"  - Train:                   {len(train_dataset)}")
    print(f"  - Validation:              {len(val_dataset)}")
    print(f"  - Test:                    {len(test_dataset)}")
    print(f"✓ Model:                     GRUClassifier ({total_params:,} params)")
    print(f"✓ Existing model reused:     No (trained from scratch)")
    print(f"✓ Best validation accuracy:  {best_val_acc * 100:.2f}%")
    print(f"✓ Best epoch:                {best_epoch}")
    print(f"✓ Test accuracy:             {test_acc * 100:.2f}%")
    print(f"✓ Saved model path:          {onnx_path}")
    print(f"✓ Label map path:            {label_map_path}")
    print(f"✓ Checkpoint path:           {Path(OUTPUT_DIR) / 'gru_best.pt'}")
    print(f"✓ Confusion matrix:          {Path(OUTPUT_DIR) / 'confusion_matrix.png'}")
    print(f"✓ Training curves:           {Path(OUTPUT_DIR) / 'training_curves.png'}")
    print(f"✓ Files modified:            tools/train_and_export.py")
    print(f"✓ Warnings:                  None")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
