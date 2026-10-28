"""
run_sparse_pipeline.py — End-to-End Evaluation of 5 FPS Sparse DINOv2 Sentry Perception.

Runs the complete perception pipeline (Gated DINOv2 Backbone + ViTYOLOAdapter_v2 + YOLO26s + ActionHead)
on test_clip_10s.mp4 at 5 FPS (stride=3, 50 frames).
Measures exact wall-clock latency on the NVIDIA RTX PRO 4000 Blackwell GPU and saves
sparse_results.json formatted identically to baseline_golden_record.json for automated benchmarking.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Add paths
sys.path.insert(0, "/home/cim/NVIDIA SDP/UnifiedPerception")
sys.path.insert(0, "/home/cim/NVIDIA SDP/stage1_baseline")

from temporal_token_cache import extract_features_sparse
from unified_perception.heads import REGISTRY
from unified_perception.perception_types import BackboneOutput, HeadResult

def run_sparse_benchmark(
    video_path: str,
    output_dir: str,
    stride: int = 3,
    image_size: int = 640,
    tau: float = 0.015,
    dilation: int = 2,
    keyframe_interval: int = 5,
    device: str = "cuda",
):
    video_p = Path(video_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("      NVIDIA SDP — 5 FPS SPARSE TOKEN CACHED PERCEPTION BENCHMARK")
    print(f"Device:            {torch.cuda.get_device_name(0)}")
    print(f"Video:             {video_p.name}")
    print(f"Stride:            {stride} (Effective 5 FPS)")
    print(f"Gating Tau:        {tau}")
    print(f"Patch Dilation:    {dilation} (5x5 window)")
    print(f"Keyframe Interval: {keyframe_interval} (Every {keyframe_interval/5.0:.1f}s)")
    print("=" * 80)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Sparse DINOv2 Feature Extraction with Token Caching
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[*] Step 1: Running Gated DINOv2 Backbone with Temporal Token Caching...")
    start_backbone = torch.cuda.Event(enable_timing=True)
    end_backbone = torch.cuda.Event(enable_timing=True)

    start_backbone.record()
    feats, frame_stats = extract_features_sparse(
        mp4_path=str(video_p),
        device=device,
        image_size=image_size,
        stride=stride,
        tau=tau,
        dilation=dilation,
        keyframe_interval=keyframe_interval,
    )
    end_backbone.record()
    torch.cuda.synchronize(device)
    backbone_time_ms = start_backbone.elapsed_time(end_backbone)

    T = feats.cls.shape[0]
    kf_count = sum(1 for s in frame_stats if s["is_keyframe"])
    sp_count = T - kf_count
    avg_active_tokens = sum(s["active_tokens"] for s in frame_stats) / float(T)
    avg_active_pct = (avg_active_tokens / 2026.0) * 100.0

    kf_latencies = [s["latency_ms"] for s in frame_stats if s["is_keyframe"]]
    sp_latencies = [s["latency_ms"] for s in frame_stats if not s["is_keyframe"]]
    avg_kf_ms = sum(kf_latencies) / len(kf_latencies) if kf_latencies else 0.0
    avg_sp_ms = sum(sp_latencies) / len(sp_latencies) if sp_latencies else 0.0

    print(f"    -> Extracted {T} frames in {backbone_time_ms:.2f} ms ({backbone_time_ms/T:.2f} ms/frame)")
    print(f"    -> Keyframes: {kf_count}/{T} (Avg latency: {avg_kf_ms:.2f} ms)")
    print(f"    -> Sparse Frames: {sp_count}/{T} (Avg latency: {avg_sp_ms:.2f} ms)")
    print(f"    -> Mean Active Tokens: {avg_active_tokens:.1f} / 2026 ({avg_active_pct:.2f}%)")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Downstream Heads Dispatch
    # ─────────────────────────────────────────────────────────────────────────
    head_results: dict[str, HeadResult] = {}
    head_timings_ms: dict[str, float] = {}

    for head_name in ["detection", "action"]:
        print(f"\n[*] Step 2: Running downstream head: '{head_name}'...")
        head_cls = REGISTRY[head_name]
        head_instance = head_cls

        start_head = torch.cuda.Event(enable_timing=True)
        end_head = torch.cuda.Event(enable_timing=True)

        start_head.record()
        result = head_instance.predict(feats)
        end_head.record()
        torch.cuda.synchronize(device)

        elapsed_ms = start_head.elapsed_time(end_head)
        head_timings_ms[head_name] = elapsed_ms
        head_results[head_name] = result

        if result.error:
            print(f"    [!] Error in {head_name}: {result.error}")
        else:
            if head_name == "detection":
                total_dets = sum(len(f) for f in result.primary if isinstance(f, list))
                print(f"    -> Success: {total_dets} detections in {elapsed_ms:.2f} ms ({elapsed_ms/T:.2f} ms/frame)")
            elif head_name == "action":
                print(f"    -> Success: Action='{result.primary}' in {elapsed_ms:.2f} ms")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Latency & Peak Memory Profiling
    # ─────────────────────────────────────────────────────────────────────────
    total_pipeline_ms = backbone_time_ms + sum(head_timings_ms.values())
    per_frame_pipeline_ms = total_pipeline_ms / float(T)
    effective_fps = 1000.0 / per_frame_pipeline_ms
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # Class distribution analysis
    detection_result = head_results.get("detection")
    class_counts = {}
    total_boxes = 0
    if detection_result and detection_result.primary:
        for f in detection_result.primary:
            if isinstance(f, list):
                for d in f:
                    cname = d.get("class_name", "unknown")
                    class_counts[cname] = class_counts.get(cname, 0) + 1
                    total_boxes += 1

    action_result = head_results.get("action")
    action_label = action_result.primary if action_result else "None"

    # Assemble comprehensive result structure
    sparse_record = {
        "benchmark_meta": {
            "device": torch.cuda.get_device_name(0),
            "video": video_p.name,
            "total_frames": T,
            "image_size": image_size,
            "stride": stride,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "gating_params": {
                "tau": tau,
                "dilation": dilation,
                "keyframe_interval": keyframe_interval,
            },
        },
        "token_savings_profile": {
            "total_tokens_per_frame": 2026,
            "mean_active_tokens": round(avg_active_tokens, 1),
            "mean_active_percentage": round(avg_active_pct, 2),
            "keyframes_count": kf_count,
            "sparse_frames_count": sp_count,
            "avg_keyframe_ms": round(avg_kf_ms, 2),
            "avg_sparse_frame_ms": round(avg_sp_ms, 2),
        },
        "latency_profile_ms": {
            "backbone_total": round(backbone_time_ms, 2),
            "backbone_per_frame": round(backbone_time_ms / T, 2),
            "detection_total": round(head_timings_ms.get("detection", 0.0), 2),
            "detection_per_frame": round(head_timings_ms.get("detection", 0.0) / T, 2),
            "action_total": round(head_timings_ms.get("action", 0.0), 2),
            "total_pipeline": round(total_pipeline_ms, 2),
            "total_per_frame": round(per_frame_pipeline_ms, 2),
            "effective_fps": round(effective_fps, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
        },
        "task_outputs": {
            "detection_summary": {
                "total_boxes": total_boxes,
                "class_counts": class_counts,
            },
            "action_summary": {
                "top1_action": action_label,
            },
        },
        "detection_full": detection_result.to_json() if detection_result else {},
        "frame_diagnostics": frame_stats,
    }

    record_path = out_dir / "sparse_results.json"
    with open(record_path, "w") as f:
        json.dump(sparse_record, f, indent=2)
    print(f"\n[+] Sparse Benchmark Record successfully written to: {record_path}")

    # Terminal summary table
    print("\n" + "=" * 80)
    print("                     SPARSE PIPELINE BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total Frames Processed:   {T}")
    print(f"Mean Active Tokens:       {avg_active_tokens:.1f} / 2026 ({avg_active_pct:.2f}%)")
    print(f"Token Reduction:          {100.0 - avg_active_pct:.2f}% fewer tokens processed")
    print("-" * 80)
    print(f"Backbone Latency:         {backbone_time_ms:.2f} ms ({backbone_time_ms/T:.2f} ms/frame)")
    print(f"Detection Latency:        {head_timings_ms.get('detection', 0.0):.2f} ms ({head_timings_ms.get('detection', 0.0)/T:.2f} ms/frame)")
    print(f"Action Latency:           {head_timings_ms.get('action', 0.0):.2f} ms")
    print(f"Total Pipeline Latency:   {total_pipeline_ms:.2f} ms ({per_frame_pipeline_ms:.2f} ms/frame)")
    print(f"Pipeline Throughput:      {effective_fps:.2f} FPS")
    print(f"Peak VRAM:                {peak_vram_mb:.2f} MB")
    print("-" * 80)
    print(f"Total Detections:         {total_boxes} boxes {class_counts}")
    print(f"Action Recognized:        '{action_label}'")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="/home/cim/NVIDIA SDP/stage1_baseline/sample_videos/test_clip_10s.mp4")
    parser.add_argument("--output-dir", default="/home/cim/NVIDIA SDP/stage1_baseline/sparse_5fps_results")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.015)
    parser.add_argument("--dilation", type=int, default=2)
    parser.add_argument("--keyframe-interval", type=int, default=5)
    args = parser.parse_args()

    run_sparse_benchmark(
        video_path=args.video,
        output_dir=args.output_dir,
        stride=args.stride,
        tau=args.tau,
        dilation=args.dilation,
        keyframe_interval=args.keyframe_interval,
    )
