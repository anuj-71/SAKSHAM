"""
V2.0: Full training pipeline with improved hyperparameters.

Read from accepted NPZ files, filter to a practical 10-class vocabulary,
generate splits at SEQ_LEN=64, train GRU with weight decay/gradient clipping/
CosineAnnealingLR, and produce evaluation report.
"""
import csv
import json
import logging
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.sign_language.training.preprocessing import preprocess_sample_for_gru

from tools.cislr.utils import CISLR_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Constants ──
SEQ_LEN = 64
FEATURE_DIM = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3

# Named constants for training hyperparams
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
LR = 1e-3
T_MAX = 100
ETA_MIN = 1e-6

# 10 practical signs with highest total samples in dataset
V2_GLOSSES = [
    "MONDAY",        # 13 samples
    "THANK YOU",     # 8
    "MOTHER",        # 8
    "CHRISTMAS",     # 8
    "INDIA",         # 7
    "NAME",          # 7
    "GOOD",          # 6
    "CAT",           # 6
    "WASH",          # 7
    "BILL",          # 7
]

PROCESSED_DIR = CISLR_DATA_DIR / "processed_v2"
MODEL_DIR = CISLR_DATA_DIR / "model_v2"


class CISLRDataset(Dataset):
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


def build_vocabulary(accepted_dir: Path, target_glosses: list[str]) -> dict:
    """Build label_map and manifest from accepted NPZ files for target glosses only."""
    label_map = {g: i for i, g in enumerate(sorted(target_glosses))}
    manifest = []
    npz_files = sorted(accepted_dir.glob("*.npz"))

    found_glosses = set()
    for npz_path in npz_files:
        with np.load(npz_path) as data:
            gloss = str(data["gloss"])
        if gloss in label_map:
            manifest.append((str(npz_path), gloss, label_map[gloss]))
            found_glosses.add(gloss)

    # Report missing glosses
    for g in target_glosses:
        if g not in found_glosses:
            logging.warning("Gloss '%s' not found in accepted NPZ files!", g)

    return label_map, manifest


def generate_splits(manifest: list, seed: int = 42):
    """Stratified 60/20/20 split (more balanced for small datasets)."""
    import random
    rng = random.Random(seed)

    gloss_groups = defaultdict(list)
    for npz_path, gloss, label_idx in manifest:
        gloss_groups[gloss].append((npz_path, gloss, label_idx))

    train_rows, val_rows, test_rows = [], [], []
    for gloss, group in gloss_groups.items():
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * 0.60))
        n_val = max(1, round(n * 0.20)) if n >= 5 else max(1, round(n * 0.20))
        n_test = n - n_train - n_val
        # Adjust for tiny classes
        if n_test <= 0 and n_val > 1:
            n_val -= 1
            n_test = 1
        if n_test <= 0 and n_train > 2:
            n_train -= 1
            n_test = 1
        # Ensure val has at least 1
        if n_val <= 0 and n_train > 2:
            n_val = 1
            n_train = n - 1 - n_val

        train_rows.extend(group[:n_train])
        val_rows.extend(group[n_train:n_train + n_val])
        test_rows.extend(group[n_train + n_val:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)

    return train_rows, val_rows, test_rows


def evaluate_test(model: nn.Module, test_loader: DataLoader, device: torch.device,
                  label_map: dict) -> dict:
    """Evaluate on test set with confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * features.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().numpy())

    total = len(all_labels)
    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    accuracy = correct / total if total > 0 else 0.0

    # Per-class
    idx_to_label = {v: k for k, v in label_map.items()}
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

    # Confusion matrix
    n = len(label_map)
    cm = [[0] * n for _ in range(n)]
    for p, l in zip(all_preds, all_labels):
        cm[l][p] += 1

    return {
        "test_samples": total,
        "test_loss": round(test_loss / total, 4),
        "test_accuracy": round(accuracy, 4),
        "test_correct": correct,
        "per_class_accuracy": per_class,
        "confusion_matrix": cm,
    }


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)
    logging.info("V2 vocabulary: %s", V2_GLOSSES)

    accepted_dir = CISLR_DATA_DIR / "landmarks" / "accepted"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Build vocabulary ──
    label_map, manifest = build_vocabulary(accepted_dir, V2_GLOSSES)
    num_classes = len(label_map)
    total_samples = len(manifest)
    logging.info("Label map: %d classes, %d total samples", num_classes, total_samples)

    # Save label map
    with open(PROCESSED_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # Save manifest
    with open(PROCESSED_DIR / "vocabulary_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["npz_path", "label", "label_index"])
        for row in manifest:
            w.writerow(row)

    # ── Step 2: Generate splits (60/20/20) ──
    train_rows, val_rows, test_rows = generate_splits(manifest)

    def write_split(name, rows):
        path = PROCESSED_DIR / name
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["npz_path", "label", "label_index"])
            for r in rows:
                w.writerow(r)
        logging.info("  %s: %d samples", name, len(rows))

    write_split("train.csv", train_rows)
    write_split("val.csv", val_rows)
    write_split("test.csv", test_rows)

    total_split = len(train_rows) + len(val_rows) + len(test_rows)
    logging.info("Split: train=%d (%.1f%%) val=%d (%.1f%%) test=%d (%.1f%%)",
                 len(train_rows), len(train_rows)/total_split*100,
                 len(val_rows), len(val_rows)/total_split*100,
                 len(test_rows), len(test_rows)/total_split*100)

    # ── Step 3: DataLoaders ──
    train_dataset = CISLRDataset(str(PROCESSED_DIR / "train.csv"))
    val_dataset = CISLRDataset(str(PROCESSED_DIR / "val.csv"))
    test_dataset = CISLRDataset(str(PROCESSED_DIR / "test.csv"))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0) if len(val_dataset) > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0) if len(test_dataset) > 0 else None

    logging.info("Train: %d, Val: %d, Test: %d", len(train_dataset), len(val_dataset), len(test_dataset))

    # ── Step 4: Model ──
    model = GRUClassifier(
        input_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=ETA_MIN)

    logging.info("Optimizer: AdamW (lr=%s, weight_decay=%s)", LR, WEIGHT_DECAY)
    logging.info("Scheduler: CosineAnnealingLR (T_max=%s, eta_min=%s)", T_MAX, ETA_MIN)
    logging.info("Gradient clipping: max_norm=%s", GRAD_CLIP_NORM)

    # ── Step 5: Training ──
    EPOCHS = 150
    PATIENCE = 25
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    training_log = {"epochs": [], "best_val_loss": None, "best_epoch": None}

    for epoch in range(1, EPOCHS + 1):
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

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            optimizer.step()

            train_loss += loss.item() * features.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
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

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        logging.info(
            "Epoch %3d/%d  lr=%.6f  train_loss=%.4f  train_acc=%.4f  val_loss=%.4f  val_acc=%.4f",
            epoch, EPOCHS, current_lr, train_loss, train_acc, val_loss, val_acc,
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
            checkpoint_path = MODEL_DIR / "gru_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "label_map": label_map,
            }, checkpoint_path)
            logging.info("  -> Saved best model (val_loss=%.4f)", current_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logging.info("Early stopping triggered after epoch %d", epoch)
                break

    training_log["best_val_loss"] = best_val_loss
    training_log["best_epoch"] = best_epoch

    log_path = MODEL_DIR / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    logging.info("Wrote training log to %s", log_path)

    # ── Step 6: Load best checkpoint for test evaluation ──
    best_checkpoint = torch.load(MODEL_DIR / "gru_best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_results = {"error": "no test set"}
    if test_loader is not None and len(test_dataset) > 0:
        test_results = evaluate_test(model, test_loader, device, label_map)
        logging.info("Test accuracy: %.2f%% (%d/%d)",
                     test_results["test_accuracy"] * 100,
                     test_results["test_correct"],
                     test_results["test_samples"])
        for label, info in sorted(test_results["per_class_accuracy"].items()):
            logging.info("  %15s: %d/%d (%.2f%%)",
                         label, info["correct"], info["samples"], info["accuracy"] * 100)

    # ── Step 7: ONNX export ──
    try:
        model.eval()
        dummy_input = torch.randn(1, SEQ_LEN, FEATURE_DIM)
        onnx_path = MODEL_DIR / "model.onnx"

        torch.onnx.export(
            model, dummy_input, str(onnx_path),
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
        onnx_size = onnx_path.stat().st_size
        logging.info("ONNX exported: %s (%.1f KB)", onnx_path, onnx_size / 1024)

        # Validate
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path))
        test_input = np.random.randn(1, SEQ_LEN, FEATURE_DIM).astype(np.float32)
        output = session.run(None, {session.get_inputs()[0].name: test_input})[0]
        logging.info("ONNX validated - output shape: %s", output.shape)
    except Exception as e:
        logging.warning("ONNX export skipped: %s", e)

    # ── Step 8: Report ──
    report = {
        "pipeline": "SAKSHAM V2.0 GRU Training",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "seq_len": SEQ_LEN,
        "feature_dim": FEATURE_DIM,
        "vocabulary_size": num_classes,
        "vocabulary": V2_GLOSSES,
        "dataset": {
            "total_accepted_npz": 7162,
            "total_unique_glosses": 4479,
            "selected_glosses": len(V2_GLOSSES),
            "total_manifest_samples": total_samples,
        },
        "splits": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
            "split_ratio": "60/20/20",
        },
        "hyperparameters": {
            "seq_len": SEQ_LEN,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "max_epochs": EPOCHS,
            "early_stopping_patience": PATIENCE,
        },
        "training_summary": {
            "epochs_trained": training_log["epochs"][-1]["epoch"] if training_log["epochs"] else 0,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        },
        "training_log": training_log,
        "test_evaluation": test_results,
        "model_files": {
            "best_checkpoint": str(MODEL_DIR / "gru_best.pt"),
            "onnx": str(MODEL_DIR / "model.onnx") if (MODEL_DIR / "model.onnx").exists() else None,
            "label_map": str(MODEL_DIR / "label_map.json"),
        },
    }

    reports_dir = CISLR_DATA_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_json = reports_dir / "v2_training_report.json"
    with open(report_json, "w") as f:
        json.dump(report, f, indent=2)
    logging.info("Report: %s", report_json)

    # Markdown summary
    lines = [
        "# SAKSHAM V2.0 GRU Training Report",
        "",
        f"Generated: {report['generated_at']}",
        f"Sequence length: {SEQ_LEN}  |  Vocabulary: {num_classes} classes",
        "",
        "## Dataset",
        f"- Selected glosses: {', '.join(V2_GLOSSES)}",
        f"- Total manifest samples: {total_samples}",
        "",
        "## Splits (60/20/20)",
        f"- Train: {len(train_rows)}  |  Val: {len(val_rows)}  |  Test: {len(test_rows)}",
        "",
        "## Hyperparameters",
        f"- SEQ_LEN={SEQ_LEN}, HIDDEN={HIDDEN_SIZE}, LAYERS={NUM_LAYERS}",
        f"- AdamW (lr={LR}, weight_decay={WEIGHT_DECAY})",
        f"- CosineAnnealingLR (T_max={T_MAX})",
        f"- Gradient clipping: max_norm={GRAD_CLIP_NORM}",
        "",
        "## Training Summary",
        f"- Epochs: {training_log['epochs'][-1]['epoch'] if training_log['epochs'] else 0}",
        f"- Best val_loss: {best_val_loss:.4f} at epoch {best_epoch}",
        "",
        "## Test Evaluation",
    ]
    if "test_accuracy" in test_results:
        lines.append(f"- Overall accuracy: **{test_results['test_accuracy']:.2%}** ({test_results['test_correct']}/{test_results['test_samples']})")
        lines.append(f"- Test loss: {test_results.get('test_loss', 'N/A')}")
        lines.append("")
        lines.append("### Per-Class Accuracy")
        lines.append("")
        lines.append("| Gloss | Samples | Correct | Accuracy |")
        lines.append("|-------|---------|---------|----------|")
        for label in sorted(test_results.get("per_class_accuracy", {}).keys()):
            info = test_results["per_class_accuracy"][label]
            lines.append(f"| {label} | {info['samples']} | {info['correct']} | {info['accuracy']:.2%} |")

        lines.append("")
        lines.append("### Confusion Matrix")
        cm = test_results.get("confusion_matrix", [])
        if cm:
            labels = sorted(label_map.keys())
            header = "| true \\\\ pred | " + " | ".join(f"**{l[:6]}**" for l in labels) + " |"
            sep = "|---|---" + "|".join("---" for _ in labels) + "|"
            lines.append(header)
            lines.append(sep)
            for i, label in enumerate(labels):
                row = f"| **{label[:8]}** | " + " | ".join(str(cm[i][j]) for j in range(len(labels))) + " |"
                lines.append(row)
    else:
        lines.append("- Test evaluation not available")

    lines.append("")
    lines.append("## Training Log (last 5 epochs)")
    lines.append("")
    lines.append("| Epoch | LR | Train Loss | Train Acc | Val Loss | Val Acc |")
    lines.append("|-------|-----|------------|-----------|----------|---------|")
    for ep in (training_log["epochs"][-5:] if len(training_log["epochs"]) >= 5 else training_log["epochs"]):
        lines.append(f"| {ep['epoch']} | {ep['lr']:.6f} | {ep['train_loss']:.4f} | {ep['train_acc']:.4f} | {ep['val_loss']:.4f} | {ep['val_acc']:.4f} |")

    report_md = reports_dir / "v2_training_report.md"
    with open(report_md, "w") as f:
        f.write("\n".join(lines))
    logging.info("Wrote markdown report: %s", report_md)

    # Print summary
    print()
    print("=" * 60)
    print("  V2.0 TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Vocabulary: {num_classes} glosses ({', '.join(V2_GLOSSES)})")
    print(f"  SEQ_LEN: {SEQ_LEN} | Total samples: {total_samples}")
    print(f"  Split: {len(train_rows)} train / {len(val_rows)} val / {len(test_rows)} test")
    print(f"  Epochs trained: {training_log['epochs'][-1]['epoch']}")
    print(f"  Best val_loss: {best_val_loss:.4f} (epoch {best_epoch})")
    if "test_accuracy" in test_results:
        print(f"  Test accuracy: {test_results['test_accuracy']:.2%} ({test_results['test_correct']}/{test_results['test_samples']})")
        print(f"  Per-class:")
        for label, info in sorted(test_results.get("per_class_accuracy", {}).items()):
            print(f"    {label:15s}: {info['correct']}/{info['samples']} ({info['accuracy']:.1%})")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
