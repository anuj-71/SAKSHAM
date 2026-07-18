import os
import sys
import json
import torch
from pathlib import Path
import numpy as np

# Force UTF-8 for prints to avoid Windows console crash with PyTorch ONNX emojis
sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.train import GRUClassifier, SEQ_LEN, FEATURE_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT

OUTPUT_DIR = "dataset/models"
SIGN_MAP_PATH = "SAKSHAM_DATASET/sign_map.json"

def main():
    print("Starting ONNX export...")
    device = torch.device("cpu")
    
    with open(SIGN_MAP_PATH) as f:
        sign_map = json.load(f)
    num_classes = len(sign_map)
    label_map = sign_map
    
    model = GRUClassifier(
        input_dim=FEATURE_DIM, hidden_dim=HIDDEN_SIZE,
        num_layers=NUM_LAYERS, num_classes=num_classes, dropout=DROPOUT,
    )
    
    best_ckpt = torch.load(str(Path(OUTPUT_DIR) / "gru_best.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()
    
    dummy_input = torch.randn(1, SEQ_LEN, FEATURE_DIM)
    onnx_path = Path(OUTPUT_DIR) / "model.onnx"
    
    print("Exporting model to ONNX format...")
    torch.onnx.export(
        model,
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
        
    print(f"Export successful!")
    print(f"Model saved to {onnx_path}")
    print(f"Label map saved to {label_map_path}")

if __name__ == "__main__":
    main()
