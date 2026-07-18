r"""
Generate training metrics report for V1 GRU model.

Reads from:
  - cislr_data/processed/vocabulary_manifest.csv
  - cislr_data/processed/train.csv / val.csv / test.csv
  - cislr_data/processed/label_map.json
  - cislr_data/model/training_log.json
  - cislr_data/model/label_map.json
  - cislr_data/model/model.onnx

Writes:
  - cislr_data/reports/v1_training_report.md
  - cislr_data/reports/v1_training_report.json
"""
import csv
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_csv_column(csv_path: str, column: str) -> list:
    vals = []
    if not os.path.exists(csv_path):
        return vals
    with open(csv_path, "r") as f:
        for r in csv.DictReader(f):
            vals.append(r[column])
    return vals


def evaluate_test_set(
    model_path: str,
    label_map_path: str,
    test_csv_path: str,
    device: torch.device,
    seq_len: int = 24,
) -> dict:
    from tools.cislr.train_gru import GRUClassifier, FEATURE_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, CISLRDataset

    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    if num_classes == 0:
        return {}

    checkpoint = torch.load(model_path, map_location=device)
    model = GRUClassifier(
        input_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if not os.path.exists(test_csv_path):
        return {"error": "test.csv not found"}

    dataset = CISLRDataset(test_csv_path)
    if len(dataset) == 0:
        return {"error": "test set empty"}

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().numpy())

    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    total = len(all_labels)
    accuracy = correct / total if total > 0 else 0.0

    # per-class metrics
    class_correct = Counter()
    class_total = Counter()
    for p, l in zip(all_preds, all_labels):
        class_total[l] += 1
        if p == l:
            class_correct[l] += 1

    idx_to_label = {v: k for k, v in label_map.items()}
    per_class = {}
    for idx in sorted(class_total.keys()):
        label = idx_to_label.get(idx, f"CLASS_{idx}")
        per_class[label] = {
            "samples": class_total[idx],
            "correct": class_correct.get(idx, 0),
            "accuracy": round(class_correct.get(idx, 0) / class_total[idx], 4) if class_total[idx] > 0 else 0.0,
        }

    return {
        "test_samples": total,
        "test_accuracy": round(accuracy, 4),
        "test_correct": correct,
        "per_class_accuracy": per_class,
    }


def generate_report() -> dict:
    report = {}
    report["pipeline"] = "SAKSHAM V1 CISLR Training Pipeline"
    report["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

    cislr_data = Path("cislr_data")
    processed_dir = cislr_data / "processed"
    model_dir = cislr_data / "model"
    reports_dir = cislr_data / "reports"

    # ── Dataset audit ──
    accepted_dir = cislr_data / "landmarks" / "accepted"
    all_npz = list(accepted_dir.glob("*.npz"))
    freq = Counter()
    for p in all_npz:
        with np.load(p) as data:
            freq[str(data["gloss"])] += 1

    report["dataset"] = {
        "total_accepted_npz": len(all_npz),
        "unique_glosses_total": len(freq),
        "glosses_with_ge2_samples": sum(1 for c in freq.values() if c >= 2),
    }

    # ── Vocabulary ──
    label_map_path = processed_dir / "label_map.json"
    manifest_path = processed_dir / "vocabulary_manifest.csv"
    if label_map_path.exists():
        with open(label_map_path) as f:
            lm = json.load(f)
        report["vocabulary"] = {
            "selected_vocabulary_size": len(lm),
            "label_map_path": str(label_map_path),
            "glosses": list(lm.keys()),
        }
    else:
        report["vocabulary"] = {"error": "label_map.json not found"}

    vocab_freq = Counter()
    if manifest_path.exists():
        with open(manifest_path) as f:
            for r in csv.DictReader(f):
                vocab_freq[r["label"]] += 1
        report["vocabulary"]["total_manifest_samples"] = sum(vocab_freq.values())
        report["vocabulary"]["samples_per_gloss"] = {
            g: vocab_freq.get(g, 0) for g in report["vocabulary"]["glosses"]
        }
        report["vocabulary"]["min_samples"] = min(vocab_freq.values())
        report["vocabulary"]["max_samples"] = max(vocab_freq.values())
        report["vocabulary"]["mean_samples"] = round(sum(vocab_freq.values()) / len(vocab_freq), 2)

    # ── Splits ──
    for split_name in ["train", "val", "test"]:
        csv_path = processed_dir / f"{split_name}.csv"
        rows = load_csv_column(str(csv_path), "label") if csv_path.exists() else []
        report[f"{split_name}_split"] = {
            "samples": len(rows),
            "csv_path": str(csv_path),
        }
        if split_name not in report:
            report[split_name] = {}
        report[split_name]["samples"] = len(rows)

    total_split = sum(report[s]["samples"] for s in ("train", "val", "test"))
    if total_split > 0:
        for s in ("train", "val", "test"):
            report[s]["pct"] = round(report[s]["samples"] / total_split * 100, 1)

    # ── Training ──
    training_log_path = model_dir / "training_log.json"
    if training_log_path.exists():
        with open(training_log_path) as f:
            tl = json.load(f)
        report["training"] = {
            "epochs_trained": len(tl["epochs"]),
            "best_val_loss": tl["best_val_loss"],
            "best_epoch": tl["best_epoch"],
            "training_log": tl,
        }
        if tl["epochs"]:
            last = tl["epochs"][-1]
            report["training"]["final_train_loss"] = last["train_loss"]
            report["training"]["final_train_accuracy"] = last["train_acc"]
            best = next((e for e in tl["epochs"] if e["epoch"] == tl["best_epoch"]), tl["epochs"][0])
            report["training"]["best_val_loss_value"] = best["val_loss"]
            report["training"]["best_val_accuracy"] = best["val_acc"]
    else:
        report["training"] = {"error": "training_log.json not found"}

    # ── ONNX ──
    onnx_path = model_dir / "model.onnx"
    if onnx_path.exists():
        report["onnx"] = {
            "path": str(onnx_path),
            "size_bytes": onnx_path.stat().st_size,
        }
    else:
        report["onnx"] = {"error": "model.onnx not found"}

    # ── Best model checkpoint ──
    best_pt = model_dir / "gru_best.pt"
    if best_pt.exists():
        report["best_checkpoint"] = {
            "path": str(best_pt),
            "size_bytes": best_pt.stat().st_size,
        }

    # ── Test set evaluation ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if best_pt.exists() and (processed_dir / "test.csv").exists():
        test_results = evaluate_test_set(
            model_path=str(best_pt),
            label_map_path=str(label_map_path),
            test_csv_path=str(processed_dir / "test.csv"),
            device=device,
        )
        if test_results:
            report["test_evaluation"] = test_results

    # ── Write reports ──
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "v1_training_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info("Wrote JSON report: %s", json_path)

    md_path = reports_dir / "v1_training_report.md"
    md_lines = [
        "# SAKSHAM V1 Training Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Dataset Summary",
        f"- Total accepted NPZ files: **{report['dataset']['total_accepted_npz']:,}**",
        f"- Unique glosses in dataset: **{report['dataset']['unique_glosses_total']:,}**",
        f"- Glosses with >= 2 samples: **{report['dataset']['glosses_with_ge2_samples']:,}**",
        "",
        "## Selected V1 Vocabulary",
        f"- Vocabulary size: **{report['vocabulary']['selected_vocabulary_size']}** signs",
        f"- Total manifest samples: **{report['vocabulary']['total_manifest_samples']}**",
        f"- Min/mean/max samples per gloss: {report['vocabulary']['min_samples']}/{report['vocabulary']['mean_samples']}/{report['vocabulary']['max_samples']}",
        "",
        "### Top glosses by frequency",
    ]
    if "samples_per_gloss" in report["vocabulary"]:
        sorted_glosses = sorted(
            report["vocabulary"]["samples_per_gloss"].items(),
            key=lambda x: -x[1],
        )
        for gloss, count in sorted_glosses:
            md_lines.append(f"- {gloss}: {count}")
    md_lines.extend([
        "",
        "## Train/Validation/Test Splits",
    ])
    for s in ("train", "val", "test"):
        pct = report.get(s, {}).get("pct", 0)
        md_lines.append(f"- **{s.capitalize()}**: {report[s]['samples']} samples ({pct}%)")

    if "error" not in report.get("training", {}):
        md_lines.extend([
            "",
            "## Training Summary",
            f"- Epochs trained: **{report['training']['epochs_trained']}**",
            f"- Best validation loss: **{report['training']['best_val_loss']:.4f}** (epoch {report['training']['best_epoch']})",
            f"- Final train loss: **{report['training']['final_train_loss']:.4f}**",
            f"- Best validation accuracy: **{report['training']['best_val_accuracy']:.4f}**",
            "",
            "### Training Log",
            "",
            "| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |",
            "|-------|-----------|----------|---------|--------|",
        ])
        for ep in report["training"]["training_log"]["epochs"]:
            md_lines.append(
                f"| {ep['epoch']} | {ep['train_loss']:.4f} | {ep['train_acc']:.4f} | "
                f"{ep['val_loss']:.4f} | {ep['val_acc']:.4f} |"
            )

    if "onnx" in report and "error" not in report["onnx"]:
        size_kb = report["onnx"]["size_bytes"] / 1024
        md_lines.extend([
            "",
            "## ONNX Export",
            f"- Model path: `{report['onnx']['path']}`",
            f"- Size: **{size_kb:.1f} KB**",
        ])

    if "test_evaluation" in report:
        te = report["test_evaluation"]
        if "error" not in te:
            md_lines.extend([
                "",
                "## Test Set Evaluation",
                f"- Test samples: **{te['test_samples']}**",
                f"- Overall accuracy: **{te['test_accuracy']:.2%}** ({te['test_correct']}/{te['test_samples']})",
                "",
                "### Per-Class Test Accuracy",
                "",
                "| Gloss | Samples | Correct | Accuracy |",
                "|-------|---------|---------|----------|",
            ])
            for gloss in sorted(te.get("per_class_accuracy", {}).keys()):
                pca = te["per_class_accuracy"][gloss]
                md_lines.append(
                    f"| {gloss} | {pca['samples']} | {pca['correct']} | {pca['accuracy']:.2%} |"
                )

    md_lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    logging.info("Wrote Markdown report: %s", md_path)

    return report


if __name__ == "__main__":
    report = generate_report()
    print(json.dumps(report, indent=2))
