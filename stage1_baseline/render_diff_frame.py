"""
render_diff_frame.py — Generates side-by-side comparison on Frame 22 where
Baseline has 7 boxes and Sparse has 5 boxes (faint bicycles dropped).
"""
import json
from pathlib import Path
import cv2
import numpy as np
import decord

def draw_styled_detections(img, dets, title="", highlight_missing=False, missing_boxes=None):
    out = img.copy()
    h, w = out.shape[:2]

    # Semi-transparent top bar
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.8, out, 0.2, 0, out)
    cv2.putText(out, title, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    class_colors = {
        "person": (0, 220, 255),      # Yellow/Cyan
        "car": (255, 180, 0),         # Blue
        "motorcycle": (50, 255, 100), # Green
        "bicycle": (255, 100, 200),   # Magenta/Pink
    }

    for d in dets:
        box = [int(round(x)) for x in d["bbox"]]
        cname = d.get("class_name", "object")
        conf = d.get("conf", 0.0)
        box_col = class_colors.get(cname, (200, 200, 200))

        cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), box_col, 2)

        label = f"{cname} {conf:.2f}"
        font_scale = 0.52
        thickness = 1
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        y_text = max(box[1] - 6, lh + 6)
        cv2.rectangle(out, (box[0], y_text - lh - 4), (box[0] + lw + 6, y_text + 2), (15, 23, 42), -1)
        cv2.rectangle(out, (box[0], y_text - lh - 4), (box[0] + lw + 6, y_text + 2), box_col, 1)
        cv2.putText(out, label, (box[0] + 3, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    if highlight_missing and missing_boxes:
        for mbox in missing_boxes:
            box = [int(round(x)) for x in mbox["bbox"]]
            cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            lbl = f"DROPPED: {mbox['class_name']} ({mbox['conf']:.2f})"
            (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (box[0], box[1] - lh - 4), (box[0] + lw + 6, box[1] + 2), (0, 0, 180), -1)
            cv2.putText(out, lbl, (box[0] + 3, box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return out

def main():
    golden_p = Path("/home/cim/NVIDIA SDP/stage1_baseline/baseline_5fps_results/baseline_golden_record.json")
    sparse_p = Path("/home/cim/NVIDIA SDP/stage1_baseline/sparse_5fps_results/sparse_results.json")
    video_p = "/home/cim/NVIDIA SDP/stage1_baseline/sample_videos/test_clip_10s.mp4"
    out_dir = Path("/home/cim/NVIDIA SDP/stage1_baseline/sparse_5fps_results")

    with open(golden_p) as f:
        golden = json.load(f)
    with open(sparse_p) as f:
        sparse = json.load(f)

    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_p, num_threads=1)

    frame_idx = 22 # Frame 22: Baseline 7 boxes vs Sparse 5 boxes
    raw_video_idx = frame_idx * 3

    frame_rgb = vr[raw_video_idx].numpy()
    frame_resized = cv2.resize(frame_rgb, (800, 600))
    frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)

    sx = 800.0 / 640.0
    sy = 600.0 / 640.0

    def scale_dets(dets):
        scaled = []
        for d in dets:
            d_copy = dict(d)
            b = d["bbox"]
            d_copy["bbox"] = [b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy]
            scaled.append(d_copy)
        return scaled

    gt_dets = scale_dets(golden["detection_full"]["primary"][frame_idx])
    sp_dets = scale_dets(sparse["detection_full"]["primary"][frame_idx])

    def box_iou(b1, b2):
        xA = max(b1[0], b2[0])
        yA = max(b1[1], b2[1])
        xB = min(b1[2], b2[2])
        yB = min(b1[3], b2[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0

    missing_in_sparse = []
    for g in gt_dets:
        matched = False
        for s in sp_dets:
            if g["class_name"] == s["class_name"] and box_iou(g["bbox"], s["bbox"]) > 0.4:
                matched = True
                break
        if not matched:
            missing_in_sparse.append(g)

    # 1. Baseline: 7 boxes (including bicycles)
    vis_base = draw_styled_detections(
        frame_bgr, gt_dets,
        title=f"Baseline Detection: 7 Boxes (Full ViT — Frame {frame_idx})"
    )

    # 2. Sparse Clean: Exactly what sparse detected (5 boxes, bicycles absent)
    vis_sparse_clean = draw_styled_detections(
        frame_bgr, sp_dets,
        title=f"Sparse Detection: 5 Boxes (Token Cached — Frame {frame_idx})",
        highlight_missing=False
    )

    # 3. Sparse Highlighted: With red boxes showing where the 2 dropped bicycles were
    vis_sparse_highlighted = draw_styled_detections(
        frame_bgr, sp_dets,
        title=f"Sparse Detection: 5 Boxes (2 Faint Bicycles Dropped — Frame {frame_idx})",
        highlight_missing=True,
        missing_boxes=missing_in_sparse
    )

    p_base = out_dir / "baseline_detection_frame22_7boxes.png"
    p_sparse_clean = out_dir / "sparse_detection_frame22_5boxes.png"
    p_sparse_high = out_dir / "sparse_detection_frame22_highlighted.png"

    cv2.imwrite(str(p_base), vis_base)
    cv2.imwrite(str(p_sparse_clean), vis_sparse_clean)
    cv2.imwrite(str(p_sparse_high), vis_sparse_highlighted)

    print(f"[+] Saved Baseline (7 boxes):        {p_base}")
    print(f"[+] Saved Sparse Clean (5 boxes):    {p_sparse_clean}")
    print(f"[+] Saved Sparse Highlighted (diff): {p_sparse_high}")

if __name__ == "__main__":
    main()
