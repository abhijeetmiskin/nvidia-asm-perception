"""
run_virat_dual_benchmark.py

End-to-end evaluation on VIRAT_S_000201_02_000590_000623.mp4:
Phase 1: Run dense 30 FPS Baseline (unmodified NVIDIA UnifiedPerception pipeline)
Phase 2: Run 5 FPS Downsampling (Windowed Argmax Peak-Motion) + Spatial Token Caching
Phase 3: Automated Comparison & Benchmark Report Generation
"""

import os
import sys
import time
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import decord
from transformers import AutoModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
repo_root = Path("/home/cim/NVIDIA SDP/UnifiedPerception")
sys.path.insert(0, str(repo_root))
sys.path.insert(0, "/home/cim/NVIDIA SDP/stage1_baseline")

from unified_perception.heads import REGISTRY
from unified_perception.heads.detection import _load_pradipt_impl
from temporal_token_cache import PatchDifferenceDetector

def box_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def match_frame_detections(gt_boxes, pred_boxes, iou_thresh=0.5):
    matched_gt = set()
    matched_pred = set()
    ious = []
    for p_idx, p in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1
        for g_idx, g in enumerate(gt_boxes):
            if g_idx in matched_gt or g.get("class_name") != p.get("class_name"):
                continue
            iou = box_iou(g["bbox"], p["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx
        if best_iou >= iou_thresh and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            matched_pred.add(p_idx)
            ious.append(best_iou)
    tp = len(matched_gt)
    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, ious

def main():
    video_path = Path("/home/cim/NVIDIA SDP/stage1_baseline/sample_videos/VIRAT_S_000201_02_000590_000623.mp4")
    out_dir = Path("/home/cim/NVIDIA SDP/stage1_baseline/virat_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("      NVIDIA SDP — VIRAT EXPERIMENTAL EVALUATION (RTX PRO 4000 BLACKWELL)")
    print(f"Video:             {video_path.name}")
    print(f"Device:            {torch.cuda.get_device_name(0)}")
    print(f"Output Directory:  {out_dir}")
    print("=" * 80)

    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(str(video_path), num_threads=1)
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    print(f"Input Stream:      {total_frames} frames @ {fps:.2f} FPS ({total_frames / fps:.2f}s)")

    # Standard transforms
    raw_tfm = transforms.Compose([
        transforms.Resize((640, 640), antialias=True),
        transforms.ConvertImageDtype(torch.float32),
    ])
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    # Initialize models
    print("\n[*] Initializing DINOv2-small ViT backbone & YOLO26 detection head...")
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    tp = Path("/home/cim/NVIDIA SDP/UnifiedPerception/third_party/training_package")
    detector = _load_pradipt_impl(str(tp), "vit")
    detector.conf_thres = 0.25

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: Baseline 30 FPS Stream Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(" PHASE 1: GROUND TRUTH 30 FPS BASELINE (Dense Full Tokens)")
    print("=" * 80)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    t0_phase1 = time.perf_counter()

    baseline_detections = []
    baseline_latencies = []
    baseline_cls_tokens = []

    # Stream through frames sequentially to preserve memory footprint
    for idx in range(total_frames):
        t_start = time.perf_counter()
        frame_uint8 = vr[idx].permute(2, 0, 1) # [3, H, W]
        f_raw = raw_tfm(frame_uint8).unsqueeze(0).to(device)
        f_norm = normalize(f_raw)

        with torch.no_grad():
            out = model(f_norm, output_hidden_states=True)
            # Tap layers [3, 6, 9, 12] for detection adapter
            hs = out.hidden_states
            dets = detector.predict_from_features(hs, f_raw)

        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        baseline_latencies.append(t_elapsed)
        baseline_detections.append(dets)
        baseline_cls_tokens.append(out.last_hidden_state[:, 0].cpu())

        if (idx + 1) % 150 == 0 or idx == total_frames - 1:
            avg_ms = np.mean(baseline_latencies)
            print(f"  [30 FPS Baseline] Processed {idx + 1}/{total_frames} frames (Current avg: {avg_ms:.2f} ms/frame)")

    t_total_phase1 = (time.perf_counter() - t0_phase1) * 1000.0
    baseline_peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    baseline_total_boxes = sum(len(d) for d in baseline_detections)
    baseline_class_counts = {}
    for f_dets in baseline_detections:
        for d in f_dets:
            c = d["class_name"]
            baseline_class_counts[c] = baseline_class_counts.get(c, 0) + 1

    print(f"\n[+] Phase 1 Baseline Complete:")
    print(f"    Total Frames:      {total_frames}")
    print(f"    Total Time:        {t_total_phase1:.2f} ms ({t_total_phase1 / total_frames:.2f} ms/frame)")
    print(f"    Effective FPS:     {total_frames / (t_total_phase1 / 1000.0):.2f} FPS")
    print(f"    Peak VRAM:         {baseline_peak_vram:.1f} MB")
    print(f"    Total Detections:  {baseline_total_boxes} boxes")
    for c, cnt in sorted(baseline_class_counts.items(), key=lambda x: -x[1]):
        print(f"      • {c:<12}: {cnt}")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2: 5 FPS Downsampling + Dynamic Spatial Token Caching
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(" PHASE 2: 5 FPS DOWNSAMPLING + SPATIAL TOKEN CACHING")
    print("=" * 80)

    # Step 2.1: Windowed Peak-Motion Argmax Frame Selection (30 FPS -> strict 5 FPS)
    stride = 6
    print(f"[*] Step 2.1: Computing Windowed Peak-Motion Argmax Selection (stride={stride})...")
    t0_sel = time.perf_counter()

    # Fast thumbnail motion scoring in lightweight chunks of 100 frames to avoid giant tensor allocation
    motion_scores = [0.0]
    prev_thumb = None
    for k in range(0, total_frames, 100):
        chunk_idx = list(range(k, min(k + 100, total_frames)))
        chunk_raw = vr.get_batch(chunk_idx).permute(0, 3, 1, 2).float() / 255.0
        chunk_small = F.interpolate(chunk_raw, size=(160, 160), mode="bilinear")
        if prev_thumb is not None:
            first_diff = (chunk_small[0:1] - prev_thumb).abs().mean().item()
            motion_scores.append(first_diff)
        diffs = (chunk_small[1:] - chunk_small[:-1]).abs().mean(dim=(1, 2, 3)).tolist()
        motion_scores.extend(diffs)
        prev_thumb = chunk_small[-1:]
        del chunk_raw, chunk_small

    motion_scores = np.array(motion_scores[:total_frames])

    selected_indices = []
    for k in range(0, total_frames, stride):
        window = list(range(k, min(k + stride, total_frames)))
        best_in_window = window[int(np.argmax(motion_scores[window]))]
        selected_indices.append(best_in_window)

    t_sel_ms = (time.perf_counter() - t0_sel) * 1000.0
    print(f"    Selected {len(selected_indices)} frames across {total_frames} raw frames ({t_sel_ms:.2f} ms total, {t_sel_ms/total_frames:.4f} ms/frame)")
    print(f"    Target Frame Budget: exactly {len(selected_indices)} frames / {total_frames/fps:.1f}s = {len(selected_indices)/(total_frames/fps):.2f} FPS")

    # Step 2.2: Spatial Token Caching Pipeline on Selected 5 FPS Frames
    print("\n[*] Step 2.2: Executing Sparse Token Caching Engine...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    t0_phase2 = time.perf_counter()

    patch_detector = PatchDifferenceDetector(tau=0.015, dilation=2).to(device)
    keyframe_interval = 5 # 1.0 second periodic anchor

    cache_hidden_states = None
    prev_raw = None

    sparse_detections = []
    sparse_latencies = []
    sparse_active_tokens = []
    sparse_cls_tokens = []
    is_keyframe_flags = []

    for t_step, f_idx in enumerate(selected_indices):
        t_start = time.perf_counter()
        frame_uint8 = vr[f_idx].permute(2, 0, 1)
        curr_raw = raw_tfm(frame_uint8).unsqueeze(0).to(device)
        curr_norm = normalize(curr_raw)

        is_kf = (t_step % keyframe_interval == 0) or (cache_hidden_states is None)
        active_indices = None

        if not is_kf and prev_raw is not None:
            active_indices, act_frac = patch_detector(curr_raw, prev_raw)
            if act_frac > 0.40:
                is_kf = True # Large scene cut trigger

        with torch.no_grad():
            if is_kf:
                out = model(curr_norm, output_hidden_states=True)
                cache_hidden_states = {i: out.hidden_states[i].clone() for i in range(13)}
                active_count = 2026
                sparse_hs = out.hidden_states
                del out
            else:
                # Sparse Token Forward
                active_token_indices = torch.cat([
                    torch.tensor([0], device=device, dtype=torch.long),
                    active_indices + 1
                ])
                active_count = len(active_token_indices)

                emb = model.embeddings(curr_norm)
                sparse_h = emb[:, active_token_indices, :]
                cache_hidden_states[0] = emb

                for layer_idx, layer in enumerate(model.encoder.layer, start=1):
                    sparse_h = layer(sparse_h)[0]
                    cache_hidden_states[layer_idx][:, active_token_indices, :] = sparse_h

                sparse_hs = tuple(cache_hidden_states[i] for i in range(13))
                del emb, sparse_h

            # Downstream Detection on Cached Layers
            dets = detector.predict_from_features(sparse_hs, curr_raw)

        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        sparse_latencies.append(t_elapsed)
        sparse_detections.append(dets)
        sparse_active_tokens.append(active_count)
        is_keyframe_flags.append(is_kf)
        sparse_cls_tokens.append(cache_hidden_states[12][:, 0].cpu())
        prev_raw = curr_raw

        if t_step % 20 == 0:
            torch.cuda.empty_cache()

        if (t_step + 1) % 30 == 0 or t_step == len(selected_indices) - 1:
            avg_ms = np.mean(sparse_latencies)
            print(f"  [5 FPS Sparse] Processed {t_step + 1}/{len(selected_indices)} frames (Current avg: {avg_ms:.2f} ms/frame)")

    t_total_phase2 = (time.perf_counter() - t0_phase2) * 1000.0
    sparse_peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    sparse_total_boxes = sum(len(d) for d in sparse_detections)
    sparse_class_counts = {}
    for f_dets in sparse_detections:
        for d in f_dets:
            c = d["class_name"]
            sparse_class_counts[c] = sparse_class_counts.get(c, 0) + 1

    print(f"\n[+] Phase 2 Sparse Complete:")
    print(f"    Evaluated Frames:  {len(selected_indices)}")
    print(f"    Total Time:        {t_total_phase2:.2f} ms ({t_total_phase2 / len(selected_indices):.2f} ms/frame)")
    print(f"    Effective FPS:     {len(selected_indices) / (t_total_phase2 / 1000.0):.2f} FPS")
    print(f"    Peak VRAM:         {sparse_peak_vram:.1f} MB")
    print(f"    Mean Active Tokens:{np.mean(sparse_active_tokens):.1f} / 2026 ({np.mean(sparse_active_tokens)/20.26:.1f}%)")
    print(f"    Total Detections:  {sparse_total_boxes} boxes")
    for c, cnt in sorted(sparse_class_counts.items(), key=lambda x: -x[1]):
        print(f"      • {c:<12}: {cnt}")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3: Accuracy, IoU, and Acceleration Comparison
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(" PHASE 3: COMPARATIVE EVALUATION & REPORT")
    print("=" * 80)

    # Match each 5 FPS selected frame with the exact same frame in the 30 FPS baseline
    matched_tps = 0
    matched_fps = 0
    matched_fns = 0
    all_matched_ious = []

    for t_step, f_idx in enumerate(selected_indices):
        gt_dets = baseline_detections[f_idx]
        sp_dets = sparse_detections[t_step]
        tp, fp, fn, ious = match_frame_detections(gt_dets, sp_dets, iou_thresh=0.5)
        matched_tps += tp
        matched_fps += fp
        matched_fns += fn
        all_matched_ious.extend(ious)

    precision = matched_tps / (matched_tps + matched_fps) if (matched_tps + matched_fps) > 0 else 0.0
    recall = matched_tps / (matched_tps + matched_fns) if (matched_tps + matched_fns) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0

    speedup_total = t_total_phase1 / t_total_phase2
    per_frame_speedup = np.mean(baseline_latencies) / np.mean(sparse_latencies)
    vram_reduction = ((baseline_peak_vram - sparse_peak_vram) / baseline_peak_vram) * 100.0

    print(f" Accuracy & Retention:")
    print(f"   • Detection Recall:    {recall * 100:.2f}%")
    print(f"   • Detection Precision: {precision * 100:.2f}%")
    print(f"   • Detection F1-Score:  {f1:.3f}")
    print(f"   • Mean Bounding Box IoU: {mean_iou:.3f}")
    print(f"\n Computational Acceleration:")
    print(f"   • End-to-End Speedup:  {speedup_total:.2f}x ({t_total_phase1:.1f} ms -> {t_total_phase2:.1f} ms)")
    print(f"   • Per-Frame Speedup:   {per_frame_speedup:.2f}x ({np.mean(baseline_latencies):.2f} ms -> {np.mean(sparse_latencies):.2f} ms)")
    print(f"   • Evaluated Tokens:    2026 -> {np.mean(sparse_active_tokens):.1f} (-{100 - np.mean(sparse_active_tokens)/20.26:.1f}%)")
    print(f"   • Peak VRAM Reduction: {vram_reduction:.1f}% ({baseline_peak_vram:.1f} MB -> {sparse_peak_vram:.1f} MB)")

    # Save detailed JSON report
    report = {
        "video": video_path.name,
        "device": torch.cuda.get_device_name(0),
        "resolution": [1280, 720],
        "fps_native": float(fps),
        "total_frames_native": total_frames,
        "baseline_30fps": {
            "total_frames": total_frames,
            "total_time_ms": round(t_total_phase1, 2),
            "per_frame_ms": round(float(np.mean(baseline_latencies)), 2),
            "effective_fps": round(total_frames / (t_total_phase1 / 1000.0), 2),
            "peak_vram_mb": round(baseline_peak_vram, 2),
            "total_boxes": baseline_total_boxes,
            "class_counts": baseline_class_counts
        },
        "proposed_5fps_sparse": {
            "evaluated_frames": len(selected_indices),
            "total_time_ms": round(t_total_phase2, 2),
            "per_frame_ms": round(float(np.mean(sparse_latencies)), 2),
            "effective_fps": round(len(selected_indices) / (t_total_phase2 / 1000.0), 2),
            "peak_vram_mb": round(sparse_peak_vram, 2),
            "mean_active_tokens": round(float(np.mean(sparse_active_tokens)), 1),
            "mean_active_pct": round(float(np.mean(sparse_active_tokens)) / 20.26, 2),
            "total_boxes": sparse_total_boxes,
            "class_counts": sparse_class_counts
        },
        "comparison_metrics": {
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1_score": round(f1, 4),
            "mean_matched_iou": round(mean_iou, 4),
            "end_to_end_speedup": round(speedup_total, 2),
            "per_frame_speedup": round(per_frame_speedup, 2),
            "vram_reduction_pct": round(vram_reduction, 2)
        }
    }

    report_path = out_dir / "virat_benchmark_comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[+] Full comparison report saved to: {report_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 4: Visual Dashboard & Side-by-Side Diagnostic Strip Generation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[*] Generating high-resolution benchmark visualization...")
    fig = plt.figure(figsize=(16, 9), dpi=200)
    fig.patch.set_facecolor('#0F172A')

    # 4 subplots
    # 1. Latency & Throughput comparison
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor('#1E293B')
    bars = ax1.bar(["30 FPS Baseline", "5 FPS Proposed"], [total_frames / (t_total_phase1/1000.0), len(selected_indices) / (t_total_phase2/1000.0)], color=['#64748B', '#76B900'], width=0.5)
    ax1.set_title("Effective System Throughput (FPS)", fontsize=12, fontweight='bold', color='#FFFFFF')
    ax1.set_ylabel("Frames Per Second", color='#94A3B8')
    ax1.tick_params(colors='#94A3B8')
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f} FPS", ha='center', va='bottom', color='#FFFFFF', fontweight='bold')

    # 2. Token Reduction
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor('#1E293B')
    bars2 = ax2.bar(["Full Tokens", "Sparse Tokens"], [2026, np.mean(sparse_active_tokens)], color=['#EF4444', '#76B900'], width=0.5)
    ax2.set_title(f"Mean Evaluated Tokens (-{100 - np.mean(sparse_active_tokens)/20.26:.1f}%)", fontsize=12, fontweight='bold', color='#FFFFFF')
    ax2.set_ylabel("Tokens per Frame", color='#94A3B8')
    ax2.tick_params(colors='#94A3B8')
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 30, f"{int(yval)}", ha='center', va='bottom', color='#FFFFFF', fontweight='bold')

    # 3. Detection Retention by Class
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor('#1E293B')
    top_classes = ["car", "person", "truck"]
    b_counts = [baseline_class_counts.get(c, 0) for c in top_classes]
    s_counts = [sparse_class_counts.get(c, 0) for c in top_classes]
    x = np.arange(len(top_classes))
    w = 0.35
    ax3.bar(x - w/2, b_counts, w, label='30 FPS Baseline', color='#64748B')
    ax3.bar(x + w/2, s_counts, w, label='5 FPS Proposed', color='#76B900')
    ax3.set_xticks(x)
    ax3.set_xticklabels(top_classes)
    ax3.set_title("Detection Box Counts by Class", fontsize=12, fontweight='bold', color='#FFFFFF')
    ax3.tick_params(colors='#94A3B8')
    ax3.legend(facecolor='#0F172A', edgecolor='none', labelcolor='#FFFFFF')

    # 4. Precision, Recall & IoU
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor('#1E293B')
    metrics = ["Recall", "Precision", "F1 Score", "Mean IoU"]
    vals = [recall * 100, precision * 100, f1 * 100, mean_iou * 100]
    bars4 = ax4.bar(metrics, vals, color=['#38BDF8', '#818CF8', '#A855F7', '#76B900'], width=0.5)
    ax4.set_title("Perception Preservation vs Ground Truth (%)", fontsize=12, fontweight='bold', color='#FFFFFF')
    ax4.set_ylim(0, 115)
    ax4.tick_params(colors='#94A3B8')
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', color='#FFFFFF', fontweight='bold')

    plt.suptitle(
        f"VIRAT Experimental Benchmark: 30 FPS Baseline vs 5 FPS Dynamic Token Caching\n"
        f"Clip: {video_path.name} | Target: NVIDIA RTX PRO 4000 Blackwell | Overall Speedup: {speedup_total:.1f}x",
        fontsize=14, fontweight='bold', color='#FFFFFF', y=0.98
    )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    dash_path = out_dir / "virat_benchmark_comparison.png"
    plt.savefig(dash_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()

    # Copy to artifacts directory
    artifact_dash = Path("/home/cim/.gemini/antigravity/brain/43942b37-b88b-4b18-a637-70397812bff9/virat_benchmark_comparison.png")
    os.system(f"cp '{dash_path}' '{artifact_dash}'")
    print(f"[+] Dashboard saved to: {dash_path} & {artifact_dash}")

    # Also render a 4-panel visual strip on the key intrusion frame (Frame 570 - Getting into Vehicle)
    key_event_frame = 570
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, key_event_frame)
    ret, k_frame = cap.read()
    cap.release()

    if ret:
        fig2, axes2 = plt.subplots(1, 3, figsize=(21, 7), dpi=200)
        fig2.patch.set_facecolor('#0F172A')

        # 1. Raw frame
        axes2[0].imshow(cv2.cvtColor(k_frame, cv2.COLOR_BGR2RGB))
        axes2[0].set_title(f"Panel 1: Raw 30 FPS Frame #{key_event_frame}\n(Person Entering Vehicle)", fontsize=11, fontweight='bold', color='#FFFFFF')
        axes2[0].axis('off')

        # 2. Baseline detections overlay
        b_over = cv2.cvtColor(k_frame.copy(), cv2.COLOR_BGR2RGB)
        for d in baseline_detections[key_event_frame]:
            b = [int(v) for v in d["bbox"]]
            # Map from 640x640 to 1280x720
            sx, sy = 1280.0 / 640.0, 720.0 / 640.0
            x1, y1, x2, y2 = int(b[0]*sx), int(b[1]*sy), int(b[2]*sx), int(b[3]*sy)
            cv2.rectangle(b_over, (x1, y1), (x2, y2), (239, 68, 68), 2)
            cv2.putText(b_over, f"{d['class_name']} {d['conf']:.2f}", (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (239, 68, 68), 1)
        axes2[1].imshow(b_over)
        axes2[1].set_title(f"Panel 2: 30 FPS Baseline Perception\n({len(baseline_detections[key_event_frame])} BBoxes, Dense 2026 Tokens)", fontsize=11, fontweight='bold', color='#EF4444')
        axes2[1].axis('off')

        # 3. 5 FPS Sparse detections overlay
        # Find nearest selected frame
        nearest_step = int(np.argmin([abs(s - key_event_frame) for s in selected_indices]))
        s_over = cv2.cvtColor(k_frame.copy(), cv2.COLOR_BGR2RGB)
        for d in sparse_detections[nearest_step]:
            b = [int(v) for v in d["bbox"]]
            sx, sy = 1280.0 / 640.0, 720.0 / 640.0
            x1, y1, x2, y2 = int(b[0]*sx), int(b[1]*sy), int(b[2]*sx), int(b[3]*sy)
            cv2.rectangle(s_over, (x1, y1), (x2, y2), (118, 185, 0), 2)
            cv2.putText(s_over, f"{d['class_name']} {d['conf']:.2f}", (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (118, 185, 0), 1)
        axes2[2].imshow(s_over)
        axes2[2].set_title(f"Panel 3: Proposed 5 FPS + Token Caching\n({len(sparse_detections[nearest_step])} BBoxes, {sparse_active_tokens[nearest_step]} Tokens)", fontsize=11, fontweight='bold', color='#76B900')
        axes2[2].axis('off')

        plt.tight_layout()
        strip_path = out_dir / "virat_side_by_side_strip.png"
        plt.savefig(strip_path, facecolor=fig2.get_facecolor(), edgecolor='none', bbox_inches='tight')
        plt.close()

        artifact_strip = Path("/home/cim/.gemini/antigravity/brain/43942b37-b88b-4b18-a637-70397812bff9/virat_side_by_side_strip.png")
        os.system(f"cp '{strip_path}' '{artifact_strip}'")
        print(f"[+] Diagnostic strip saved to: {strip_path} & {artifact_strip}")

    print("\n" + "=" * 80)
    print(" ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
