r"""
Train a 2-layer GRU classifier on CISLR landmark sequences.

Reuses src.sign_language.training.preprocessing.preprocess_sample_for_gru()
for feature extraction.

Usage:
    .\.venv\Scripts\python.exe -m tools.cislr.train_gru [--epochs 50] [--batch-size 32]
"""
import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.sign_language.training.preprocessing import preprocess_sample_for_gru

from tools.cislr.utils import CISLR_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SEQ_LEN = 24
FEATURE_DIM = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3


class CISLRDataset(Dataset):
    """Loads CISLR NPZ files and preprocesses them for GRU training."""

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
        features = torch.from_numpy(result["features"])  # (SEQ_LEN, FEATURE_DIM)
        return features.float(), torch.tensor(label_idx, dtype=torch.long)


class GRUClassifier(nn.Module):
    """2-layer GRU with a linear classification head."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, h_n = self.gru(x)
        out = h_n[-1]
        return self.fc(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--processed-dir", type=str, default=None, help="Override processed directory")
    parser.add_argument("--model-dir", type=str, default=None, help="Override model directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)

    processed_dir = Path(args.processed_dir) if args.processed_dir else CISLR_DATA_DIR / "processed"
    model_dir = Path(args.model_dir) if args.model_dir else CISLR_DATA_DIR / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load label map
    label_map_path = processed_dir / "label_map.json"
    if not label_map_path.exists():
        logging.error("label_map.json not found at %s", label_map_path)
        logging.error("Run build_vocabulary.py first.")
        return 1
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    if num_classes == 0:
        logging.error("Label map is empty. Cannot train.")
        return 1

    logging.info("Number of classes: %d", num_classes)

    # Datasets
    train_csv = processed_dir / "train.csv"
    val_csv = processed_dir / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        logging.error("train.csv or val.csv not found. Run generate_splits.py first.")
        return 1

    train_dataset = CISLRDataset(str(train_csv))
    val_dataset = CISLRDataset(str(val_csv))

    if len(train_dataset) == 0:
        logging.error("Training set is empty. Cannot train.")
        return 1

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0) if len(val_dataset) > 0 else None
    logging.info("Train samples: %d, Val samples: %d", len(train_dataset), len(val_dataset))

    # Model
    model = GRUClassifier(
        input_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    training_log = {"epochs": [], "best_val_loss": None, "best_epoch": None}

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation (skip if no validation samples)
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

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
            val_loss = 0.0
            val_acc = 0.0
            current_val_loss = train_loss

        logging.info(
            "Epoch %3d/%d  train_loss=%.4f  train_acc=%.4f  val_loss=%.4f  val_acc=%.4f",
            epoch, args.epochs, train_loss, train_acc, val_loss, val_acc,
        )

        training_log["epochs"].append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
        })

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_epoch = epoch
            patience_counter = 0
            checkpoint_path = model_dir / "gru_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "label_map": label_map,
            }, checkpoint_path)
            logging.info("  → Saved best model (val_loss=%.4f)", current_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logging.info("Early stopping triggered after epoch %d", epoch)
                break

    training_log["best_val_loss"] = best_val_loss
    training_log["best_epoch"] = best_epoch

    log_path = model_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    logging.info("Wrote training log to %s", log_path)
    logging.info("Best val_loss=%.4f at epoch %d", best_val_loss, best_epoch)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
