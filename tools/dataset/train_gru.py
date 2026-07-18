r"""
Train a 2-layer GRU classifier on the curated SAKSHAM dataset.

Reuses src.sign_language.training.preprocessing for feature extraction.

Usage:
    .venv\Scripts\python.exe -m tools.dataset.train_gru [--epochs 150]
"""
import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.sign_language.training.preprocessing import preprocess_sample_for_gru

from tools.dataset.paths import PROCESSED_DIR, MODELS_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV, LABEL_MAP_PATH, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SEQ_LEN = 64
FEATURE_DIM = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3

LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
T_MAX = 100
ETA_MIN = 1e-6


class CuratedDataset(Dataset):
    def __init__(self, csv_path: str):
        self.samples = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["npz_path"], int(row["label_index"])))
        if not self.samples:
            logging.warning("Empty dataset: %s", csv_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npz_path, label_idx = self.samples[idx]
        with np.load(npz_path) as data:
            sample = {key: data[key] for key in data.files}
        result = preprocess_sample_for_gru(sample, target_length=SEQ_LEN, include_presence=True)
        features = torch.from_numpy(result["features"])
        return features.float(), torch.tensor(label_idx, dtype=torch.long)


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float):
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


def evaluate_test(model: nn.Module, test_loader: DataLoader, device: torch.device, label_map: dict) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * features.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    total = len(all_labels)
    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    accuracy = correct / total if total > 0 else 0.0

    idx_to_label = {v: k for k, v in label_map.items()}
    from collections import Counter
    class_correct = Counter()
    class_total = Counter()
    for p, l in zip(all_preds, all_labels):
        class_total[l] += 1
        if p == l:
            class_correct[l] += 1

    per_class = {}
    for idx in sorted(class_total.keys()):
        label = idx_to_label.get(idx, f"CLASS_{idx}")
        per_class[label] = {
            "samples": class_total[idx],
            "correct": class_correct.get(idx, 0),
            "accuracy": round(class_correct.get(idx, 0) / class_total[idx], 4),
        }

    return {
        "test_samples": total,
        "test_loss": round(test_loss / total, 4),
        "test_accuracy": round(accuracy, 4),
        "test_correct": correct,
        "per_class_accuracy": per_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--patience", type=int, default=25)
    args = parser.parse_args()

    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)

    if not LABEL_MAP_PATH.exists():
        logging.error("label_map.json not found. Run generate_label_map.py first.")
        return 1
    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    logging.info("Number of classes: %d", num_classes)

    if not TRAIN_CSV.exists() or not VAL_CSV.exists():
        logging.error("train.csv or val.csv not found. Run generate_splits.py first.")
        return 1

    train_dataset = CuratedDataset(str(TRAIN_CSV))
    val_dataset = CuratedDataset(str(VAL_CSV))
    test_dataset = CuratedDataset(str(TEST_CSV)) if TEST_CSV.exists() else None

    if len(train_dataset) == 0:
        logging.error("Training set is empty.")
        return 1

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0) if len(val_dataset) > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0) if test_dataset and len(test_dataset) > 0 else None

    logging.info("Train: %d, Val: %d, Test: %d", len(train_dataset), len(val_dataset), len(test_dataset) if test_dataset else 0)

    model = GRUClassifier(
        input_dim=FEATURE_DIM, hidden_dim=HIDDEN_SIZE,
        num_layers=NUM_LAYERS, num_classes=num_classes, dropout=DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=ETA_MIN)

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    training_log = {"epochs": [], "best_val_loss": None, "best_epoch": None}

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        if val_loader is not None:
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
            current_val_loss = val_loss
        else:
            val_loss, val_acc = 0.0, 0.0
            current_val_loss = train_loss

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        logging.info(
            "Epoch %3d/%d  lr=%.6f  train_loss=%.4f  train_acc=%.4f  val_loss=%.4f  val_acc=%.4f",
            epoch, args.epochs, current_lr, train_loss, train_acc, val_loss, val_acc,
        )

        training_log["epochs"].append({
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
        })

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "label_map": label_map,
            }, MODELS_DIR / "gru_best.pt")
            logging.info("  -> Saved best model (val_loss=%.4f)", current_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logging.info("Early stopping triggered after epoch %d", epoch)
                break

    training_log["best_val_loss"] = best_val_loss
    training_log["best_epoch"] = best_epoch

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_LOG, "w") as f:
        json.dump(training_log, f, indent=2)

    # Test evaluation
    best_checkpoint = torch.load(MODELS_DIR / "gru_best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    if test_loader is not None and len(test_dataset) > 0:
        test_results = evaluate_test(model, test_loader, device, label_map)
        logging.info("Test accuracy: %.2f%% (%d/%d)",
                     test_results["test_accuracy"] * 100,
                     test_results["test_correct"],
                     test_results["test_samples"])
        for label, info in sorted(test_results["per_class_accuracy"].items()):
            logging.info("  %20s: %d/%d (%.2f%%)", label, info["correct"], info["samples"], info["accuracy"] * 100)

    logging.info("Best val_loss=%.4f at epoch %d", best_val_loss, best_epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
