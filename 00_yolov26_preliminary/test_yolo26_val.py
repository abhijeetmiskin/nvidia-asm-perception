"""
YOLOv26 (YOLO26) Validation & COCO Benchmark Script.

Computes standard COCO metrics:
- mAP@0.50
- mAP@0.50:0.95
- Precision & Recall
- Latency (Preprocess, Inference, Postprocess)
"""

import argparse
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

def run_validation(model_name: str = "yolo26n.pt", data_yaml: str = "coco8.yaml", batch_size: int = 16, device: str = None):
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print(f" YOLOv26 COCO Evaluation & Benchmark")
    print(f" Model:      {model_name}")
    print(f" Dataset:    {data_yaml}")
    print(f" Device:     {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() and device != 'cpu' else 'CPU'})")
    print(f" Batch Size: {batch_size}")
    print("=" * 65)

    # 1. Load model
    model = YOLO(model_name)

    # 2. Run Validation
    print("\n[*] Running validation pass...")
    metrics = model.val(data=data_yaml, batch=batch_size, device=device, verbose=True)

    # 3. Print Results Summary
    print("\n" + "=" * 65)
    print(" EVALUATION RESULTS SUMMARY")
    print("=" * 65)
    
    box = metrics.box
    print(f" mAP@0.50:       {box.map50 * 100:.2f} %")
    print(f" mAP@0.50:0.95:  {box.map * 100:.2f} %")
    print(f" Mean Precision: {box.mp * 100:.2f} %")
    print(f" Mean Recall:    {box.mr * 100:.2f} %")
    
    # Speed breakdown
    speed = metrics.speed
    print("-" * 65)
    print(f" Speed Breakdown (per image):")
    print(f"  - Preprocess:   {speed.get('preprocess', 0.0):.2f} ms")
    print(f"  - Inference:    {speed.get('inference', 0.0):.2f} ms")
    print(f"  - Postprocess:  {speed.get('postprocess', 0.0):.2f} ms (NMS-free in YOLO26!)")
    total_ms = speed.get('preprocess', 0.0) + speed.get('inference', 0.0) + speed.get('postprocess', 0.0)
    print(f"  - Total:        {total_ms:.2f} ms (~{1000/max(total_ms, 0.001):.1f} FPS)")
    print("=" * 65)
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate YOLOv26 on COCO Dataset")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="Model checkpoint (yolo26n.pt, yolo26s.pt, etc.)")
    parser.add_argument("--data", type=str, default="coco8.yaml", help="Dataset yaml config (e.g., coco8.yaml, coco128.yaml, coco_val.yaml)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for validation")
    parser.add_argument("--device", type=str, default=None, help="Device ('0' or 'cpu')")
    args = parser.parse_args()

    run_validation(args.model, args.data, args.batch, args.device)
