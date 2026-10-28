import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

video_path = Path("/home/cim/NVIDIA SDP/stage1_baseline/sample_videos/VIRAT_S_000201_02_000590_000623.mp4")
out_img_path = Path("/home/cim/NVIDIA SDP/stage1_baseline/virat_sample_preview_storyboard.png")
artifact_img_path = Path("/home/cim/.gemini/antigravity/brain/43942b37-b88b-4b18-a637-70397812bff9/virat_sample_preview_storyboard.png")

if not video_path.exists():
    print(f"Error: Video {video_path} does not exist yet.")
    exit(1)

cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration_sec = total_frames / fps if fps > 0 else 0

print(f"=== VIRAT VIDEO METADATA ===")
print(f"File: {video_path.name}")
print(f"Resolution: {width} x {height}")
print(f"FPS: {fps:.2f}")
print(f"Total Frames: {total_frames}")
print(f"Duration: {duration_sec:.2f} seconds")

sample_indices = [
    (0, "Frame 0: Stationary Parking Lot Overview"),
    (min(200, total_frames - 1), f"Frame 200: Pedestrian Trajectory & Scene"),
    (min(380, total_frames - 1), f"Frame 380: Approaching Vehicle"),
    (min(490, total_frames - 1), f"Frame 490: Event Start - Getting into Vehicle"),
    (min(570, total_frames - 1), f"Frame 570: Active Intrusion / Cabin Entry"),
    (min(650, total_frames - 1), f"Frame 650: Door Closing / Complete Entry")
]

fig, axes = plt.subplots(2, 3, figsize=(20, 11), dpi=200)
fig.patch.set_facecolor('#0F172A')

for ax_idx, (frame_idx, title) in enumerate(sample_indices):
    r, c = divmod(ax_idx, 3)
    ax = axes[r, c]
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(frame_rgb)
        ax.set_title(title, fontsize=12, fontweight='bold', color='#76B900', pad=10)
    else:
        ax.text(0.5, 0.5, f"Frame {frame_idx} not available", color='white', ha='center', va='center')
    
    ax.axis("off")

plt.suptitle(
    f"VIRAT Surveillance Video Preview: {video_path.name}\n"
    f"1080p @ 30.0 FPS | Total Frames: {total_frames} ({duration_sec:.1f}s) | Annotated Action: Getting into Vehicle (Frames 479-692)",
    fontsize=14, fontweight='bold', color='#FFFFFF', y=0.98
)

plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
plt.savefig(out_img_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
plt.savefig(artifact_img_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
plt.close()
cap.release()

print(f"[+] Preview storyboard saved successfully.")
