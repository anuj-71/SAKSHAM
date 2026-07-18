r"""
Export trained GRU checkpoint to ONNX format from the curated dataset pipeline.

Loads the best checkpoint from train_gru.py and exports
to ONNX opset 17 with fixed input shape [1, 64, 128].

Usage:
    .venv\Scripts\python.exe -m tools.dataset.export_onnx
"""
import json
import logging
from pathlib import Path

import torch

from tools.dataset.paths import MODELS_DIR, MODEL_CHECKPOINT, MODEL_ONNX, MODEL_LABEL_MAP
from tools.dataset.train_gru import GRUClassifier, FEATURE_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, SEQ_LEN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    if not MODEL_CHECKPOINT.exists():
        logging.error("Checkpoint not found: %s", MODEL_CHECKPOINT)
        logging.error("Run train_gru.py first.")
        return 1

    checkpoint = torch.load(MODEL_CHECKPOINT, map_location="cpu")
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
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(MODEL_ONNX),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    logging.info("Exported ONNX model to %s", MODEL_ONNX)

    try:
        import onnxruntime as ort
        import numpy as np
        session = ort.InferenceSession(str(MODEL_ONNX))
        input_name = session.get_inputs()[0].name
        test_input = np.random.randn(1, SEQ_LEN, FEATURE_DIM).astype(np.float32)
        output = session.run(None, {input_name: test_input})[0]
        logging.info("ONNX validation OK - output shape: %s", output.shape)
    except Exception as e:
        logging.error("ONNX validation FAILED: %s", e)
        return 2

    with open(MODEL_LABEL_MAP, "w") as f:
        json.dump(label_map, f, indent=2)
    logging.info("Copied label_map to %s", MODEL_LABEL_MAP)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
