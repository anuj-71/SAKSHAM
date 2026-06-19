r"""
Export trained GRU checkpoint to ONNX format.

Loads the best checkpoint from train_gru.py and exports
to ONNX opset 17 with fixed input shape [1, 24, 128].

Usage:
    .venv\Scripts\python.exe -m tools.cislr.export_onnx
"""
import argparse
import json
import logging
from pathlib import Path

import torch

from tools.cislr.train_gru import GRUClassifier, FEATURE_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, SEQ_LEN

from tools.cislr.utils import CISLR_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=None, help="Override model directory")
    parser.add_argument("--processed-dir", type=str, default=None, help="Override processed directory")
    args = parser.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else CISLR_DATA_DIR / "model"
    processed_dir = Path(args.processed_dir) if args.processed_dir else CISLR_DATA_DIR / "processed"

    checkpoint_path = model_dir / "gru_best.pt"
    if not checkpoint_path.exists():
        logging.error("Checkpoint not found: %s", checkpoint_path)
        logging.error("Run train_gru.py first.")
        return 1

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    label_map = checkpoint["label_map"]
    num_classes = len(label_map)

    model = GRUClassifier(
        input_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, SEQ_LEN, FEATURE_DIM)
    onnx_path = model_dir / "model.onnx"

    # Use legacy exporter (dynamo=False) to avoid PyTorch 2.x encoding issues on Windows
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    logging.info("Exported ONNX model to %s", onnx_path)

    # Validate with onnxruntime
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        test_input = np.random.randn(1, SEQ_LEN, FEATURE_DIM).astype(np.float32)
        output = session.run(None, {input_name: test_input})[0]
        logging.info("ONNX validation OK - output shape: %s", output.shape)
    except Exception as e:
        logging.error("ONNX validation FAILED: %s", e)
        return 2

    # Copy label map to model directory
    label_map_dest = model_dir / "label_map.json"
    with open(label_map_dest, "w") as f:
        json.dump(label_map, f, indent=2)
    logging.info("Copied label_map to %s", label_map_dest)

    logging.info("ONNX export complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
