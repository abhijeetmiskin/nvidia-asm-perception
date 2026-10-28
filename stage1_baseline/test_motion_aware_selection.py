"""
test_motion_aware_selection.py — Implements and validates Motion-Aware Temporal Frame Selection
under a strict 5 FPS budget (exactly 50 frames for a 10s video at 15 FPS).
"""
import sys
from pathlib import Path
import decord
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, "/home/cim/NVIDIA SDP/UnifiedPerception")
sys.path.insert(0, "/home/cim/NVIDIA SDP/stage1_baseline")

from unified_perception.heads.detection import _load_pradipt_impl
from transformers import AutoModel
from torchvision import transforms

def compute_lightweight_motion_scores(vr, device="cuda"):
    """Ultra-fast motion score computation on 160x160 thumbnails (< 0.05 ms / frame)."""
    total = len(vr)
    batch = vr.get_batch(range(total)).permute(0, 3, 1, 2).float() / 255.0 # [T, 3, H, W]
    batch_small = F.interpolate(batch, size=(160, 160), mode="bilinear").to(device)

    # Frame difference: |F_t - F_{t-1}|
    # For frame 0, motion is 0.0
    diffs = (batch_small[1:] - batch_small[:-1]).abs().mean(dim=(1, 2, 3)) # [T-1]
    motion_scores = torch.cat([torch.tensor([0.0], device=device), diffs]).cpu().numpy()
    return motion_scores

def select_motion_aware_frames(motion_scores, stride=3):
    """Selects the peak-motion frame within each 3-frame window, strictly enforcing 5 FPS."""
    total = len(motion_scores)
    selected_indices = []

    for k in range(0, total, stride):
        window = list(range(k, min(k + stride, total)))
        # Pick the frame in the window with the highest motion score
        best_in_window = window[int(np.argmax(motion_scores[window]))]
        selected_indices.append(best_in_window)

    return selected_indices

def main():
    video_p = "/home/cim/NVIDIA SDP/stage1_baseline/sample_videos/test_clip_10s.mp4"
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_p, num_threads=1)
    total = len(vr)

    print("=" * 75)
    print("      MOTION-AWARE TEMPORAL FRAME SELECTION (STRICT 5 FPS BUDGET)")
    print(f"Total video frames: {total} (10.0s @ 15 FPS)")
    print(f"Target Budget:       50 frames (Strict 5.0 FPS)")
    print("=" * 75)

    # 1. Compute motion scores across all 150 frames
    motion_scores = compute_lightweight_motion_scores(vr)

    # 2. Baseline Blind Stride-3
    blind_indices = list(range(0, total, 3))

    # 3. Motion-Aware Adaptive Selection
    adaptive_indices = select_motion_aware_frames(motion_scores, stride=3)

    print(f"\n[1] Blind Stride-3 Frame Count:     {len(blind_indices)} frames (5.0 FPS)")
    print(f"[2] Motion-Aware Frame Count:       {len(adaptive_indices)} frames (5.0 FPS)")
    assert len(blind_indices) == len(adaptive_indices) == 50, "Frame budget must be strictly 50!"

    # Comparison metrics
    diff_count = sum(1 for b, a in zip(blind_indices, adaptive_indices) if b != a)
    blind_motion_sum = sum(motion_scores[blind_indices])
    adaptive_motion_sum = sum(motion_scores[adaptive_indices])

    print(f"\n[+] Windows where adaptive selection picked a more active frame: {diff_count} / 50 ({diff_count/50*100:.1f}%)")
    print(f"[+] Total Motion Energy captured by Blind Stride:               {blind_motion_sum:.4f}")
    print(f"[+] Total Motion Energy captured by Motion-Aware Selection:     {adaptive_motion_sum:.4f} (+{(adaptive_motion_sum - blind_motion_sum)/blind_motion_sum*100:.1f}%)")

    print("\nSample Window Comparison (first 10 windows):")
    print(f"{'Window':<10} {'Frames in Window':<20} {'Blind Selected':<16} {'Motion-Aware Selected':<24} {'Gain / Status'}")
    print("-" * 75)
    for w_idx in range(10):
        k = w_idx * 3
        win = list(range(k, k + 3))
        b_sel = blind_indices[w_idx]
        a_sel = adaptive_indices[w_idx]
        win_scores = [round(float(motion_scores[f]), 5) for f in win]
        status = "Identical" if b_sel == a_sel else f"Captured Peak (Frame {a_sel})"
        print(f"W{w_idx:<9} {str(win):<20} Frame {b_sel:<10} Frame {a_sel} (score={motion_scores[a_sel]:.4f})  {status}")

if __name__ == "__main__":
    main()
