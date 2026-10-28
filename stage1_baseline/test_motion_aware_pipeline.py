"""
test_motion_aware_pipeline.py — Runs the complete perception pipeline with Motion-Aware Temporal Selection.
"""
import sys
from pathlib import Path
import decord
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, "/home/cim/NVIDIA SDP/UnifiedPerception")
sys.path.insert(0, "/home/cim/NVIDIA SDP/stage1_baseline")

from temporal_token_cache import GatedDINOv2
from test_motion_aware_selection import compute_lightweight_motion_scores, select_motion_aware_frames
from unified_perception.heads.detection import _load_pradipt_impl
from unified_perception.heads.action import load_action_head
from torchvision import transforms

def main():
    video_p = "/home/cim/NVIDIA SDP/stage1_baseline/sample_videos/test_clip_10s.mp4"
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_p, num_threads=1)
    total = len(vr)

    print("[*] Computing motion scores on thumbnails (<0.05 ms / frame)...")
    motion_scores = compute_lightweight_motion_scores(vr)

    # Motion-aware 50 frames
    adaptive_indices = select_motion_aware_frames(motion_scores, stride=3)
    assert len(adaptive_indices) == 50, "Must be exactly 50 frames (5 FPS)!"

    # Load models
    tp = Path("/home/cim/NVIDIA SDP/UnifiedPerception/third_party/training_package")
    detector = _load_pradipt_impl(str(tp), "vit")
    detector.conf_thres = 0.25

    gated_backbone = GatedDINOv2(
        model_name="facebook/dinov2-small",
        device="cuda",
        image_size=640,
        tau=0.015,
        dilation=2,
        keyframe_interval=5,
    )

    raw_tfm = transforms.Compose([
        transforms.Resize((640, 640), antialias=True),
        transforms.ConvertImageDtype(torch.float32),
    ])
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    all_dets = []
    cls_list = []

    print("\n[*] Processing 50 Motion-Aware Selected Frames at 5 FPS...")
    for f_idx, raw_idx in enumerate(adaptive_indices):
        frame_uint8 = vr[raw_idx].permute(2, 0, 1)
        f_raw = raw_tfm(frame_uint8).unsqueeze(0).cuda()
        f_norm = normalize(f_raw)

        hs, cls, pt, stats = gated_backbone.forward_frame(
            curr_norm=f_norm, curr_raw=f_raw, frame_idx=f_idx
        )

        dets = detector.predict_from_features(tuple(hs.values()), f_raw)
        all_dets.append(dets)
        cls_list.append(cls)

    total_boxes = sum(len(d) for d in all_dets)
    classes = {}
    for d_list in all_dets:
        for d in d_list:
            c = d["class_name"]
            classes[c] = classes.get(c, 0) + 1

    # Action Head
    action_head = load_action_head(
        head_ckpt="/home/cim/NVIDIA SDP/UnifiedPerception/assets/k400_outputs/best.pth",
        classes_txt="/home/cim/NVIDIA SDP/UnifiedPerception/assets/kinetics400/annotations/classes.txt",
        train_dir="/home/cim/NVIDIA SDP/UnifiedPerception/assets/k400_outputs",
        device="cuda",
    )
    cls_tensor = torch.cat(cls_list, dim=0) # [50, 384]
    action_preds = action_head.classify(cls_tensor, topk=3)

    print("\n" + "=" * 70)
    print("        MOTION-AWARE 5 FPS PERCEPTION RESULTS")
    print("=" * 70)
    print(f"Total Frames Processed:    50 (Strict 5.0 FPS)")
    print(f"Total Detected Boxes:      {total_boxes} boxes {classes}")
    print(f"Action Top-1 Prediction:   '{action_preds[0][0]}' (conf={action_preds[0][1]:.3f})")
    print("=" * 70)

if __name__ == "__main__":
    main()
