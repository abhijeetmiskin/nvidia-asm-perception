"""
Phase B.1: Spatial-Temporal Patch Redundancy Measurement Script for DINOv2 in ASM

Quantifies the exact percentage of redundant spatial patch tokens across consecutive
frames at 5 FPS in Automotive Surveillance Mode. Measures:
1. Per-patch pixel difference (MAE) on the 45x45 grid.
2. DINOv2 token cosine similarity across consecutive frames.
3. Spatial activity distribution heatmap (45x45).
4. Theoretical self-attention FLOPs reduction.
"""

import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Ensure UnifiedPerception is importable
repo_root = Path(__file__).resolve().parent.parent / "UnifiedPerception"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from unified_perception.backbone import extract_features
from unified_perception.perception_types import BackboneOutput

def measure_patch_redundancy(
    video_path: str = "stage1_baseline/sample_videos/test_clip_10s.mp4",
    output_dir: str = "stage1_baseline/phase_b1_redundancy_analysis",
    stride: int = 3, # 15 FPS / 3 = 5 FPS
    image_size: int = 640,
    patch_size: int = 14,
    grid_dim: int = 45 # 45x45 = 2025 patches
):
    video_p = Path(video_path).resolve()
    out_p = Path(output_dir).resolve()
    out_p.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print(" PHASE B.1: DINOv2 PATCH TOKEN REDUNDANCY ANALYSIS (5 FPS)")
    print(f" Video Source:     {video_p.name}")
    print(f" Target Frame Rate: 5 FPS (Stride = {stride})")
    print(f" Image Resolution: {image_size}x{image_size} (Grid: {grid_dim}x{grid_dim} = {grid_dim*grid_dim} patches)")
    print(f" Compute Device:   {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
    print(f" Output Directory: {out_p}")
    print("=" * 70)

    # 1. Extract features using DINOv2 at 5 FPS
    print("\n[*] Step 1: Extracting DINOv2 backbone features at 5 FPS...")
    t0 = time.perf_counter()
    feats: BackboneOutput = extract_features(
        mp4_path=str(video_p),
        num_frames=-1,
        model_name="vits14",
        device=device,
        image_size=image_size,
        hidden_layer_indices=None,
        keep_frames=True,
        stride=stride,
    )
    extract_time = time.perf_counter() - t0
    
    frames_raw = feats.frames # [T, 3, 640, 640] float in [0, 1]
    patch_tokens = feats.patch_tokens # [T, 2025, 384] float
    T = frames_raw.shape[0]
    print(f"    -> Extracted {T} frames in {extract_time:.2f}s.")
    print(f"    -> Patch tokens shape: {tuple(patch_tokens.shape)}")

    # 2. Compute Patch Pixel Difference across consecutive frames
    print("\n[*] Step 2: Calculating Patch-Level Pixel Delta (MAE) across consecutive frames...")
    # Reshape frames into 45x45 patches of size 14x14
    # 45 * 14 = 630 pixels. We slice [:, :, :630, :630] for clean patch alignment
    cropped_frames = frames_raw[:, :, :grid_dim * patch_size, :grid_dim * patch_size] # [T, 3, 630, 630]
    
    # Reshape into [T, 3, 45, 14, 45, 14] -> [T, 45, 45, 3, 14, 14]
    patches = cropped_frames.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5) # [T, 45, 45, 3, 14, 14]

    # Pixel MAE between consecutive frames: [T-1, 45, 45]
    frame_diffs = (patches[1:] - patches[:-1]).abs().mean(dim=(3, 4, 5)) # [T-1, 45, 45]
    patch_mae_np = frame_diffs.cpu().numpy() # [T-1, 45, 45]

    # 3. Compute DINOv2 Patch Token Cosine Similarity
    print("[*] Step 3: Calculating DINOv2 Semantic Feature Cosine Similarity...")
    # patch_tokens: [T, 2025, 384]
    tokens_curr = patch_tokens[:-1] # [T-1, 2025, 384]
    tokens_next = patch_tokens[1:]  # [T-1, 2025, 384]
    
    # Cosine similarity per patch token across consecutive frames
    cos_sim = F.cosine_similarity(tokens_curr, tokens_next, dim=-1) # [T-1, 2025]
    cos_sim_np = cos_sim.cpu().numpy()

    # 4. Statistical Analysis Across Multiple Activity Thresholds
    print("\n[*] Step 4: Quantifying Static vs. Dynamic Token Breakdown...")
    thresholds = [0.015, 0.025, 0.035, 0.05, 0.08]
    threshold_stats = {}
    
    for th in thresholds:
        active_mask = patch_mae_np > th # [T-1, 45, 45]
        active_counts = active_mask.sum(axis=(1, 2)) # [T-1]
        active_pcts = (active_counts / (grid_dim * grid_dim)) * 100
        static_pcts = 100.0 - active_pcts
        
        # Self-attention compute reduction factor: (N / N_active)^2
        # (Where full N = 2026, N_active = active_counts + 1 for CLS)
        n_active = active_counts + 1
        att_reduction = (2026.0 / n_active) ** 2
        
        threshold_stats[str(th)] = {
            "mean_active_tokens": float(np.mean(active_counts)),
            "mean_static_tokens": float(grid_dim * grid_dim - np.mean(active_counts)),
            "mean_static_percentage": float(np.mean(static_pcts)),
            "mean_active_percentage": float(np.mean(active_pcts)),
            "median_static_percentage": float(np.median(static_pcts)),
            "min_static_percentage": float(np.min(static_pcts)),
            "max_static_percentage": float(np.max(static_pcts)),
            "mean_attention_speedup_x": float(np.mean(att_reduction)),
        }

    # Reference threshold (tau = 0.025)
    ref_th = 0.025
    ref_mask = patch_mae_np > ref_th # [T-1, 45, 45]
    ref_active_pct = (ref_mask.sum(axis=(1, 2)) / (grid_dim * grid_dim)) * 100
    ref_static_pct = 100.0 - ref_active_pct
    
    # Cumulative spatial activity map: How often was each patch active across the whole video?
    cumulative_activity = ref_mask.mean(axis=0) * 100 # [45, 45] in percentage

    # Cosine Similarity distribution stats
    mean_cos_sim = float(np.mean(cos_sim_np))
    median_cos_sim = float(np.median(cos_sim_np))
    pct_tokens_high_sim = float(np.mean(cos_sim_np > 0.95) * 100)

    # 5. Generate Visual Diagnostic Plots
    print("\n[*] Step 5: Generating Spatial & Temporal Redundancy Visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    
    # Plot 1: 45x45 Spatial Activity Heatmap
    ax1 = axes[0, 0]
    im1 = ax1.imshow(cumulative_activity, cmap="hot", interpolation="nearest")
    ax1.set_title("DINOv2 45x45 Patch Activity Heatmap\n(% of Time Patch Has Movement Across 10s Clip)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Horizontal Patch Index (0 to 44)")
    ax1.set_ylabel("Vertical Patch Index (0 to 44)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Activity Frequency (%)")
    
    # Plot 2: Frame-by-Frame Static vs Active Tokens
    ax2 = axes[0, 1]
    time_axis = np.arange(1, T) * (stride / 15.0) # seconds
    ax2.plot(time_axis, ref_static_pct, label="Static Tokens (Can be Cached)", color="#2ecc71", linewidth=2.5)
    ax2.plot(time_axis, ref_active_pct, label="Active Tokens (Need DINOv2)", color="#e74c3c", linewidth=2.5)
    ax2.axhline(np.mean(ref_static_pct), color="#27ae60", linestyle="--", label=f"Mean Static: {np.mean(ref_static_pct):.1f}%")
    ax2.set_title("Temporal Redundancy Over Time at 5 FPS", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Video Timestamp (seconds)")
    ax2.set_ylabel("Percentage of 2,025 Tokens (%)")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="center right")

    # Plot 3: Token Cosine Similarity Distribution
    ax3 = axes[1, 0]
    ax3.hist(cos_sim_np.flatten(), bins=60, color="#3498db", edgecolor="black", alpha=0.8)
    ax3.axvline(mean_cos_sim, color="#e67e22", linestyle="--", linewidth=2, label=f"Mean Cos Sim: {mean_cos_sim:.3f}")
    ax3.axvline(0.95, color="#c0392b", linestyle=":", linewidth=2, label=f"> 0.95 Cos Sim: {pct_tokens_high_sim:.1f}% of tokens")
    ax3.set_title("Distribution of DINOv2 Feature Cosine Similarity\n(Consecutive 5 FPS Frames)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Cosine Similarity Between Consecutive Frames")
    ax3.set_ylabel("Number of Patch Token Pairs")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left")

    # Plot 4: Self-Attention Compute Reduction Across Thresholds
    ax4 = axes[1, 1]
    th_labels = [f"τ = {th}" for th in thresholds]
    speedups = [threshold_stats[str(th)]["mean_attention_speedup_x"] for th in thresholds]
    bars = ax4.bar(th_labels, speedups, color="#9b59b6", edgecolor="black", alpha=0.85)
    ax4.set_title("Theoretical Self-Attention Compute Reduction\n((N_full / N_active)^2)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Attention FLOPs Reduction Factor (X)")
    ax4.grid(True, alpha=0.3, axis="y")
    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f"{height:.1f}x",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plot_path = out_p / "redundancy_analysis_dashboard.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"    -> Saved visual dashboard to: {plot_path}")

    # 6. Save Structured JSON Metrics
    metrics_summary = {
        "analysis_meta": {
            "video": video_p.name,
            "total_frames_sampled": T,
            "stride": stride,
            "effective_fps": 5.0,
            "total_patches_per_frame": grid_dim * grid_dim,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "headline_findings": {
            "mean_static_tokens_percentage": round(float(np.mean(ref_static_pct)), 2),
            "mean_active_tokens_percentage": round(float(np.mean(ref_active_pct)), 2),
            "mean_active_tokens_count": round(float(np.mean(ref_mask.sum(axis=(1, 2)))), 1),
            "mean_static_tokens_count": round(float(grid_dim * grid_dim - np.mean(ref_mask.sum(axis=(1, 2)))), 1),
            "mean_token_cosine_similarity": round(mean_cos_sim, 4),
            "tokens_with_cosine_sim_over_0_95_pct": round(pct_tokens_high_sim, 2),
            "theoretical_attention_speedup_x": round(float(np.mean((2026.0 / (ref_mask.sum(axis=(1, 2)) + 1))**2)), 1),
        },
        "threshold_sensitivity": threshold_stats,
    }
    
    json_path = out_p / "redundancy_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"    -> Saved metrics JSON to: {json_path}")

    # 7. Print Console Summary
    print("\n" + "=" * 70)
    print(" PHASE B.1 REDUNDANCY ANALYSIS SUMMARY (AT 5 FPS)")
    print("=" * 70)
    print(f" Total Spatial Patches per Frame:  {grid_dim*grid_dim} (Grid: 45x45)")
    print(f" Mean Static (Unchanged) Patches:  {metrics_summary['headline_findings']['mean_static_tokens_percentage']}% (~{int(metrics_summary['headline_findings']['mean_static_tokens_count'])} patches)")
    print(f" Mean Active (Changing) Patches:   {metrics_summary['headline_findings']['mean_active_tokens_percentage']}% (~{int(metrics_summary['headline_findings']['mean_active_tokens_count'])} patches)")
    print(f" Mean DINOv2 Feature Cosine Sim:   {mean_cos_sim:.4f}")
    print(f" Tokens with Cosine Sim > 0.95:    {pct_tokens_high_sim:.1f}%")
    print("-" * 70)
    print(f" ★ THEORETICAL ATTENTION SPEEDUP:  {metrics_summary['headline_findings']['theoretical_attention_speedup_x']}x reduction in self-attention compute!")
    print("=" * 70)
    return metrics_summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase B.1 Patch Redundancy Measurement")
    parser.add_argument("--video", type=str, default="stage1_baseline/sample_videos/test_clip_10s.mp4")
    parser.add_argument("--out", type=str, default="stage1_baseline/phase_b1_redundancy_analysis")
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()

    measure_patch_redundancy(args.video, args.out, stride=args.stride)
