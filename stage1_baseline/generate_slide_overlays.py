"""
generate_slide_overlays.py — Generates individual high-resolution annotated frame overlays
for Baseline Detection and Sparse Detection cards in PowerPoint.
"""
import json
from pathlib import Path
import cv2
import numpy as np
import decord

def draw_styled_detections(img, dets, color=(0, 200, 255), label_bg=(0, 150, 200), title=""):
    out = img.copy()
    h, w = out.shape[:2]

    # Draw semi-transparent header bar
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    cv2.putText(out, title, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Class color map
    class_colors = {
        "person": (0, 220, 255),      # Cyan/Yellow
        "car": (255, 180, 0),         # Sky Blue
        "motorcycle": (50, 255, 100), # Neon Green
        "bicycle": (255, 100, 200),   # Pink/Magenta
    }

    for d in dets:
        box = [int(round(x)) for x in d["bbox"]]
        cname = d.get("class_name", "object")
        conf = d.get("conf", 0.0)

        box_col = class_colors.get(cname, color)

        # Draw bounding box with subtle thickness
        cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), box_col, 2)

        # Label background tag
        label = f"{cname} {conf:.2f}"
        font_scale = 0.55
        thickness = 1
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        y_text = max(box[1] - 6, lh + 6)
        cv2.rectangle(out, (box[0], y_text - lh - 4), (box[0] + lw + 6, y_text + 2), (15, 23, 42), -1)
        cv2.rectangle(out, (box[0], y_text - lh - 4), (box[0] + lw + 6, y_text + 2), box_col, 1)
        cv2.putText(out, label, (box[0] + 3, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

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

    gt_dets_all = golden["detection_full"]["primary"]
    sp_dets_all = sparse["detection_full"]["primary"]

    # Frame 25 (raw index 75): 7 rich detections (person, car, motorcycle, bicycle)
    chosen_frame_idx = 25
    raw_video_idx = chosen_frame_idx * 3

    frame_rgb = vr[raw_video_idx].numpy()
    frame_resized = cv2.resize(frame_rgb, (800, 600)) # Clean 4:3 high-res aspect ratio
    frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)

    # Scale bboxes from 640x640 imgsz to 800x600 display
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

    gt_dets = scale_dets(gt_dets_all[chosen_frame_idx])
    sp_dets = scale_dets(sp_dets_all[chosen_frame_idx])

    vis_baseline = draw_styled_detections(
        frame_bgr, gt_dets, 
        color=(0, 200, 255), 
        title=f"Baseline Detection (Full ViT Backbone — Frame {chosen_frame_idx})"
    )

    vis_sparse = draw_styled_detections(
        frame_bgr, sp_dets, 
        color=(50, 255, 100), 
        title=f"Sparse Detection (Token Cached Backbone — Frame {chosen_frame_idx})"
    )

    out_base = out_dir / "baseline_detection_overlay.png"
    out_sparse = out_dir / "sparse_detection_overlay.png"

    cv2.imwrite(str(out_base), vis_baseline)
    cv2.imwrite(str(out_sparse), vis_sparse)

    print(f"[+] Saved Baseline Detection Overlay: {out_base}")
    print(f"[+] Saved Sparse Detection Overlay:   {out_sparse}")

if __name__ == "__main__":
    main()
