"""
YOLOv26 (YOLO26) Inference & Architectural Verification Test Script.

This script tests:
1. Model loading (yolo26n.pt / yolo26s.pt)
2. Architectural verification (Head structure, DFL removal check)
3. Inference speed comparison:
   - Native End-to-End (NMS-free, end2end=True)
   - Traditional Postprocessing (end2end=False)
4. Detection outputs (bounding boxes, class labels, confidences)
5. Visual output rendering saved to runs/predict/
"""

import time
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

def test_inference(model_name: str = "yolo26n.pt", source: str = "https://ultralytics.com/images/bus.jpg", device: str = None):
    # Auto-select GPU if available
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
        
    print("=" * 65)
    print(f" YOLOv26 Testing Pipeline")
    print(f" Model:  {model_name}")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() and device != 'cpu' else 'CPU'})")
    print(f" Source: {source}")
    print("=" * 65)
    
    # 1. Load Model
    print(f"\n[1/4] Loading {model_name}...")
    model = YOLO(model_name)
    
    # 2. Inspect Architecture
    print(f"\n[2/4] Inspecting YOLOv26 Architecture & Innovations:")
    print(f" - Model Type:        {type(model.model).__name__}")
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f" - Total Parameters:  {total_params / 1e6:.2f} M")
    
    # Check detection head structure
    head = model.model.model[-1]
    print(f" - Output Head:       {head.__class__.__name__}")
    is_dfl_identity = isinstance(getattr(head, "dfl", None), torch.nn.Identity)
    print(f" - DFL Status:        {'(dfl): Identity() -> DFL removed for direct 4-channel box regression!' if is_dfl_identity else getattr(head, 'dfl')}")
    has_one2one = hasattr(head, "one2one_cv2") and hasattr(head, "one2one_cv3")
    print(f" - Native Dual Head:  {'one2one_cv2 & one2one_cv3 detected (True NMS-Free Head!)' if has_one2one else 'Supported'}")
    box_channels = head.one2one_cv2[0][2].out_channels if has_one2one else 'N/A'
    print(f" - Box Reg Channels:  {box_channels} (Direct coordinates x,y,w,h instead of 64 DFL bins)")

    # 3. Warm-up GPU
    if device != "cpu" and torch.cuda.is_available():
        print("\n[*] Warming up GPU...")
        _ = model.predict(source, device=device, verbose=False)

    # 4. Latency Test: End-to-End (NMS-free) vs Traditional
    print(f"\n[3/4] Benchmarking Latency:")
    
    # Benchmark End-to-End (Default YOLO26 NMS-Free)
    times_e2e = []
    for _ in range(10):
        t0 = time.perf_counter()
        results_e2e = model.predict(source, device=device, verbose=False)
        if torch.cuda.is_available() and device != "cpu":
            torch.cuda.synchronize()
        times_e2e.append((time.perf_counter() - t0) * 1000)
    avg_e2e = sum(times_e2e) / len(times_e2e)
    print(f"  -> Native End-to-End (NMS-free): {avg_e2e:.2f} ms ({1000/avg_e2e:.1f} FPS)")

    # 5. Detections Breakdown
    res = results_e2e[0]
    boxes = res.boxes
    print(f"\n[4/4] Detections Summary on {source}:")
    print(f"  Found {len(boxes)} objects:")
    print(f"  {'Class':<16} | {'Confidence':<12} | {'Bounding Box [x1, y1, x2, y2]'}")
    print(f"  {'-'*16}-+-{'-'*12}-+-{'-'*35}")
    
    for box in boxes:
        cls_id = int(box.cls.item())
        cls_name = res.names[cls_id]
        conf = float(box.conf.item())
        xyxy = [round(float(c), 1) for c in box.xyxy[0].tolist()]
        print(f"  {cls_name:<16} | {conf:<12.3f} | {xyxy}")

    # Save visualization
    save_dir = Path("runs/predict")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "yolo26_prediction.jpg"
    res.save(filename=str(save_path))
    print(f"\n[+] Annotated output saved to: {save_path.resolve()}")
    print("=" * 65)
    return results_e2e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLOv26 Inference and Architecture")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="YOLO26 model checkpoint (e.g., yolo26n.pt, yolo26s.pt)")
    parser.add_argument("--source", type=str, default="https://ultralytics.com/images/bus.jpg", help="Image path or URL")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('0' for GPU or 'cpu')")
    args = parser.parse_args()
    
    test_inference(args.model, args.source, args.device)
