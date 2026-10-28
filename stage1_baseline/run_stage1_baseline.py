"""
Stage 1 Baseline Runner & Profiler for NVIDIA UnifiedPerception

This script runs the unmodified NVIDIA UnifiedPerception pipeline on a surveillance clip,
micro-benchmarks every pipeline stage (DINOv2 backbone vs Detection Head vs Action Head),
measures GPU memory, records ground truth predictions, and exports a comprehensive report.
"""

import os
import sys
import time
import json
from pathlib import Path
import torch

# Ensure UnifiedPerception is importable
repo_root = Path(__file__).resolve().parent.parent / "UnifiedPerception"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from unified_perception.backbone import extract_features
from unified_perception.heads import REGISTRY
from unified_perception.perception_types import BackboneOutput, HeadResult

def run_stage1_benchmark(
    video_path: str,
    output_dir: str,
    image_size: int = 640,
    stride: int = 1,
    num_frames: int = -1
):
    video_p = Path(video_path).resolve()
    out_p = Path(output_dir).resolve()
    out_p.mkdir(parents=True, exist_ok=True)
    
    if not video_p.exists():
        raise FileNotFoundError(f"Video not found: {video_p}")
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    print("=" * 70)
    print(" STAGE 1: UNIFIED PERCEPTION BASELINE BENCHMARK")
    print(f" Video Source:     {video_p.name} ({video_p})")
    print(f" Output Directory: {out_p}")
    print(f" Compute Device:   {device} ({gpu_name})")
    print(f" Image Resolution: {image_size}x{image_size}")
    print(f" Frame Stride:     {stride}")
    print("=" * 70)

    # 1. Warm-up GPU
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        dummy = torch.randn(1, 3, image_size, image_size, device=device)
        _ = dummy @ dummy.transpose(-1, -2)
        torch.cuda.synchronize()

    # 2. Benchmark Stage A: DINOv2 Feature Extraction
    print("\n[*] [Stage 1.A] Running DINOv2 Backbone...")
    t0_backbone = time.perf_counter()
    start_event = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    end_event = torch.cuda.Event(enable_timing=True) if device == "cuda" else None

    if start_event:
        start_event.record()

    feats: BackboneOutput = extract_features(
        mp4_path=str(video_p),
        num_frames=num_frames,
        model_name="vits14",
        device=device,
        image_size=image_size,
        hidden_layer_indices=None,
        keep_frames=True,
        stride=stride,
    )

    if end_event:
        end_event.record()
        torch.cuda.synchronize()
        backbone_gpu_ms = start_event.elapsed_time(end_event)
    else:
        backbone_gpu_ms = (time.perf_counter() - t0_backbone) * 1000

    backbone_wall_ms = (time.perf_counter() - t0_backbone) * 1000
    n_frames = feats.cls.shape[0]
    print(f"    -> Extracted {n_frames} frames.")
    print(f"    -> Backbone Total Time: {backbone_wall_ms:.2f} ms ({backbone_wall_ms / n_frames:.2f} ms/frame)")

    # 3. Benchmark Stage B: Detection Head (Adapter-v2 + YOLO26)
    print("\n[*] [Stage 1.B] Running Detection Head (Adapter-v2 + YOLO26s)...")
    detection_head = REGISTRY.get("detection")
    t0_det = time.perf_counter()
    if start_event:
        start_event.record()

    det_result: HeadResult = detection_head.predict(feats)

    if end_event:
        end_event.record()
        torch.cuda.synchronize()
        det_gpu_ms = start_event.elapsed_time(end_event)
    else:
        det_gpu_ms = (time.perf_counter() - t0_det) * 1000

    det_wall_ms = (time.perf_counter() - t0_det) * 1000
    print(f"    -> Detection Total Time: {det_wall_ms:.2f} ms ({det_wall_ms / n_frames:.2f} ms/frame)")

    # 4. Benchmark Stage C: Action Recognition Head (Linear Probe)
    print("\n[*] [Stage 1.C] Running Action Recognition Head (K400)...")
    action_head = REGISTRY.get("action")
    t0_act = time.perf_counter()
    if start_event:
        start_event.record()

    act_result: HeadResult = action_head.predict(feats)

    if end_event:
        end_event.record()
        torch.cuda.synchronize()
        act_gpu_ms = start_event.elapsed_time(end_event)
    else:
        act_gpu_ms = (time.perf_counter() - t0_act) * 1000

    act_wall_ms = (time.perf_counter() - t0_act) * 1000
    print(f"    -> Action Total Time: {act_wall_ms:.2f} ms")

    # Annotate videos
    print("\n[*] Generating video overlays...")
    try:
        detection_head.annotate(str(video_p), det_result, str(out_p / "detection_annotated.mp4"), image_size=image_size)
    except Exception as e:
        print(f"    -> Detection overlay notice: {e}")

    try:
        action_head.annotate(str(video_p), act_result, str(out_p / "action_annotated.mp4"))
    except Exception as e:
        print(f"    -> Action overlay notice: {e}")

    # 5. Measure GPU Memory Footprint
    max_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0

    # 6. Synthesize Results & Statistics
    total_pipeline_ms = backbone_wall_ms + det_wall_ms + act_wall_ms
    effective_fps = (n_frames / (total_pipeline_ms / 1000.0))

    # Parse Detections
    detection_counts = {}
    total_detections = 0
    if isinstance(det_result.primary, list):
        for frame_dets in det_result.primary:
            if isinstance(frame_dets, list):
                total_detections += len(frame_dets)
                for d in frame_dets:
                    cname = d.get("class_name", "unknown")
                    detection_counts[cname] = detection_counts.get(cname, 0) + 1

    # Save Combined Perception JSON (The Golden Baseline Record)
    golden_record = {
        "benchmark_meta": {
            "device": gpu_name,
            "video": video_p.name,
            "total_frames": n_frames,
            "image_size": image_size,
            "stride": stride,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "latency_profile_ms": {
            "backbone_total": round(backbone_wall_ms, 2),
            "backbone_per_frame": round(backbone_wall_ms / n_frames, 2),
            "detection_total": round(det_wall_ms, 2),
            "detection_per_frame": round(det_wall_ms / n_frames, 2),
            "action_total": round(act_wall_ms, 2),
            "total_pipeline": round(total_pipeline_ms, 2),
            "total_per_frame": round(total_pipeline_ms / n_frames, 2),
            "effective_fps": round(effective_fps, 2),
            "peak_vram_mb": round(max_memory_mb, 2),
        },
        "task_outputs": {
            "detection_summary": {
                "total_boxes": total_detections,
                "class_counts": detection_counts,
            },
            "action_summary": {
                "top1_action": act_result.primary,
                "top_predictions": act_result.details.get("topk", []) if act_result.details else [],
            },
        },
        "backbone_metadata": feats.metadata,
        "detection_full": det_result.to_json(),
        "action_full": act_result.to_json(),
    }

    golden_json_path = out_p / "baseline_golden_record.json"
    with open(golden_json_path, "w") as f:
        json.dump(golden_record, f, indent=2)

    # 7. Print Formatted Summary
    print("\n" + "=" * 70)
    print(" STAGE 1 BASELINE PROFILE RESULTS")
    print("=" * 70)
    print(f" Total Frames Processed:    {n_frames}")
    print(f" Total Pipeline Time:       {total_pipeline_ms:.2f} ms")
    print(f" Average Latency per Frame: {total_pipeline_ms / n_frames:.2f} ms")
    print(f" Baseline Throughput:       {effective_fps:.1f} FPS")
    print(f" Peak GPU VRAM:             {max_memory_mb:.1f} MB")
    print("-" * 70)
    print(" LATENCY BREAKDOWN BY COMPONENT:")
    pct_bb = (backbone_wall_ms / total_pipeline_ms) * 100
    pct_det = (det_wall_ms / total_pipeline_ms) * 100
    pct_act = (act_wall_ms / total_pipeline_ms) * 100
    print(f"  [1] DINOv2 Backbone:     {backbone_wall_ms / n_frames:6.2f} ms/frame  ({pct_bb:5.1f}% of pipeline)")
    print(f"  [2] Detection Head:      {det_wall_ms / n_frames:6.2f} ms/frame  ({pct_det:5.1f}% of pipeline)")
    print(f"  [3] Action Recog Head:   {act_wall_ms:6.2f} ms total   ({pct_act:5.1f}% of pipeline)")
    print("-" * 70)
    print(" TASK OUTPUTS (GROUND TRUTH):")
    print(f"  - Primary Action:         {act_result.primary}")
    print(f"  - Objects Detected:       {total_detections} boxes across {len(detection_counts)} classes")
    for cname, count in sorted(detection_counts.items(), key=lambda x: -x[1]):
        print(f"      • {cname:<16}: {count}")
    print("-" * 70)
    print(f"[+] Golden Baseline Record saved to: {golden_json_path}")
    print("=" * 70)
    return golden_record

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1 Baseline Runner")
    parser.add_argument("--video", type=str, default="stage1_baseline/sample_videos/test_clip_10s.mp4")
    parser.add_argument("--out", type=str, default="stage1_baseline/baseline_results")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--frames", type=int, default=-1)
    args = parser.parse_args()

    run_stage1_benchmark(args.video, args.out, stride=args.stride, num_frames=args.frames)
