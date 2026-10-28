"""
render_comparison_frames.py — Generates side-by-side visual comparisons of baseline vs sparse detections.
"""
import json
from pathlib import Path
import cv2
import numpy as np
import decord
import torch

def draw_boxes(img, dets, color=(0, 255, 0), title=""):
    out = img.copy()
    for d in dets:
        box = [int(x) for x in d["bbox"]]
        cname = d.get("class_name", "obj")
        conf = d.get("conf", 0.0)
        cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), color, 2)
        label = f"{cname} {conf:.2f}"
        cv2.putText(out, label, (box[0], max(15, box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(out, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out

def main():
    golden_p = Path("/home/cim/NVIDIA SDP/stage1_baseline/baseline_5fps_results/baseline_golden_record.json")
    sparse_p = Path("/home/cim/NVIDIA SDP/stage1_baseline/sparse_5fps_results/sparse_results.json")
    video_p = "/home/cim/NVIDIA SDP/stage1_baseline/sample_videos/test_clip_10s.mp4"

    with open(golden_p) as f:
        golden = json.load(f)
    with open(sparse_p) as f:
        sparse = json.load(f)

    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_p, num_threads=1)

    gt_dets_all = golden["detection_full"]["primary"]
    sp_dets_all = sparse["detection_full"]["primary"]

    # Select 4 representative frames: Frame 1 (sparse), Frame 12 (sparse), Frame 24 (sparse), Frame 36 (sparse)
    sample_frame_indices = [1, 12, 24, 36]
    rows = []

    for f_idx in sample_frame_indices:
        raw_video_idx = f_idx * 3
        frame_rgb = vr[raw_video_idx].numpy()
        frame_resized = cv2.resize(frame_rgb, (640, 640))
        frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)

        gt_dets = gt_dets_all[f_idx]
        sp_dets = sp_dets_all[f_idx]

        vis_gt = draw_boxes(frame_bgr, gt_dets, color=(0, 200, 255), title=f"Baseline (Full ViT): Frame {f_idx} (stride*3={raw_video_idx})")
        vis_sp = draw_boxes(frame_bgr, sp_dets, color=(0, 255, 100), title=f"Sparse Cached (Gated): Frame {f_idx}")

        combined_row = np.hstack([vis_gt, vis_sp])
        rows.append(combined_row)

    final_mosaic = np.vstack(rows)
    out_p = Path("/home/cim/NVIDIA SDP/stage1_baseline/sparse_5fps_results/side_by_side_comparison.jpg")
    cv2.imwrite(str(out_p), final_mosaic)
    print(f"[+] Saved side-by-side comparison mosaic to: {out_p}")

if __name__ == "__main__":
    main()
