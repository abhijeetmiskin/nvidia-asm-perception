"""
generate_architecture_workflow.py

Generates a publication-grade, highly detailed end-to-end architecture and workflow
diagram for the NVIDIA SDP team, featuring:
1. Video Input & Multi-Camera Ingestion (15/30 FPS native or 5 FPS automotive)
2. Stage 1: Temporal Dimension Optimization (Motion-Aware Selection & Idle Sentry Gating)
3. Stage 2: Spatial Dimension Optimization (GPU Patch Difference & Dilation Mask)
4. Stage 3: DINOv2 Sparse ViT Execution & GPU Feature Cache
5. Stage 4: Downstream Multi-Task Perception Heads (Adapter-v2 + YOLO26s & K400 Action Head)
6. Hardware Telemetry & Benchmark Callouts (RTX PRO 4000 Blackwell)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle
from pathlib import Path

# Setup figure canvas: wide, ultra high-resolution
fig = plt.figure(figsize=(26, 17), dpi=300)
ax = fig.add_subplot(111)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Pure white background
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')

# ── Color Palette (NVIDIA Modern Enterprise Theme) ──
C_NV_GREEN = "#76B900"        # Official NVIDIA Green
C_NV_DARK_GREEN = "#117733"   # Deep accent green
C_SLATE_TITLE = "#0F172A"     # Deep navy slate
C_SLATE_TEXT = "#334155"      # Body text slate
C_SLATE_MUTED = "#64748B"     # Muted grey text
C_BORDER = "#CBD5E1"          # Crisp grey border
C_BG_CARD = "#F8FAFC"         # Light background fill
C_BG_ACCENT = "#F1F5F9"       # Section background fill
C_ALERT_ORANGE = "#D97706"    # Orange trigger badge
C_BLUE_TECH = "#2563EB"       # Blue for ViT attention
C_PURPLE = "#7C3AED"          # Purple for Action head
C_TEAL = "#0D9488"            # Teal for YOLO Detection

# Helper: Draw Container Card
def draw_card(x, y, w, h, bg_color=C_BG_CARD, border_color=C_BORDER, radius=1.0, lw=1.5, zorder=1):
    card = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.2,rounding_size={radius}",
        ec=border_color, fc=bg_color,
        lw=lw, zorder=zorder
    )
    ax.add_patch(card)
    return card

# Helper: Draw Connecting Arrow
def draw_arrow(x1, y1, x2, y2, color=C_NV_GREEN, lw=2.2, style="->", rad=0.0):
    connectionstyle = f"arc3,rad={rad}" if rad != 0.0 else "arc3,rad=0"
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=16,
            shrinkA=2, shrinkB=2,
            connectionstyle=connectionstyle
        ),
        zorder=6
    )

# ═══════════════════════════════════════════════════════════════════════════
# 1. TOP HEADER & TELEMETRY BANNER
# ═══════════════════════════════════════════════════════════════════════════
# Title box
ax.text(50, 97.2, "NVIDIA SENIOR DESIGN PROJECT (SDP) — AUTOMOTIVE SURVEILLANCE MODE (ASM)",
        ha="center", va="center", fontsize=15, fontweight="bold", color=C_NV_DARK_GREEN)
ax.text(50, 94.8, "End-to-End Perception Workflow: Temporal Motion Selection & Spatial DINOv2 Token Caching",
        ha="center", va="center", fontsize=21, fontweight="bold", color=C_SLATE_TITLE)
ax.text(50, 92.8, "Target Architecture: NVIDIA RTX PRO 4000 Blackwell (24GB VRAM)  |  Shared DINOv2 ViT-S/14  |  Zero Retraining Required",
        ha="center", va="center", fontsize=11, fontweight="medium", color=C_SLATE_MUTED)

# Benchmark Summary Badges (Top right & left)
draw_card(2.0, 91.5, 20.0, 4.0, bg_color="#F0FDF4", border_color=C_NV_GREEN, radius=0.6, lw=1.5)
ax.text(12.0, 94.2, "2.00x DINOv2 SPEEDUP", ha="center", va="center", fontsize=11, fontweight="bold", color=C_NV_DARK_GREEN)
ax.text(12.0, 92.6, "129.3 ms -> 64.7 ms (4.98 ms sparse)", ha="center", va="center", fontsize=9, color=C_SLATE_TEXT)

draw_card(78.0, 91.5, 20.0, 4.0, bg_color="#F0FDF4", border_color=C_NV_GREEN, radius=0.6, lw=1.5)
ax.text(88.0, 94.2, "-41.7% PEAK VRAM REDUCTION", ha="center", va="center", fontsize=11, fontweight="bold", color=C_NV_DARK_GREEN)
ax.text(88.0, 92.6, "4,157 MB -> 2,423 MB | +79.9% FPS", ha="center", va="center", fontsize=9, color=C_SLATE_TEXT)

# ═══════════════════════════════════════════════════════════════════════════
# 2. COLUMN 1: VIDEO INGESTION & TIER 1 TEMPORAL OPTIMIZATION (Left)
# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 Major Container
draw_card(2.0, 6.0, 29.0, 83.5, bg_color="#FFFFFF", border_color=C_BORDER, radius=1.0, lw=2.0)
draw_card(3.0, 84.5, 27.0, 4.0, bg_color="#F8FAFC", border_color=C_NV_GREEN, radius=0.6, lw=1.5)
ax.text(16.5, 86.8, "STAGE 1: TEMPORAL REDUCTION", ha="center", va="center", fontsize=13, fontweight="bold", color=C_NV_DARK_GREEN)
ax.text(16.5, 85.3, "Input Conditioning & Motion-Aware Selection", ha="center", va="center", fontsize=9.5, color=C_SLATE_MUTED)

# Box 1.1: Dual Video Ingestion Sources
draw_card(4.0, 71.0, 25.0, 11.5, bg_color=C_BG_CARD, border_color=C_BORDER, radius=0.8)
ax.text(16.5, 80.5, "1.1 Video Stream Sources", ha="center", va="center", fontsize=11.5, fontweight="bold", color=C_SLATE_TITLE)
ax.text(5.5, 77.8, "• Option A: 15 / 30 FPS Public Datasets", fontsize=9.5, fontweight="bold", color=C_BLUE_TECH)
ax.text(6.5, 76.2, "(MEVA, VIRAT, AVA, test_clip_10s.mp4)", fontsize=8.5, color=C_SLATE_MUTED)
ax.text(5.5, 74.0, "• Option B: Native 5 FPS Vehicle Surround", fontsize=9.5, fontweight="bold", color=C_NV_DARK_GREEN)
ax.text(6.5, 72.4, "(Front, Rear, Left, Right Fisheye Cameras)", fontsize=8.5, color=C_SLATE_MUTED)

# Box 1.2: Technique A - Motion-Aware Temporal Window Selection
draw_card(4.0, 40.5, 25.0, 28.0, bg_color="#F0FDF4", border_color=C_NV_GREEN, radius=0.8, lw=1.8)
ax.text(16.5, 66.2, "Technique A: Windowed Argmax Selection", ha="center", va="center", fontsize=11, fontweight="bold", color=C_NV_DARK_GREEN)
ax.text(16.5, 64.6, "[For 15/30 FPS Input -> Strictly Capped at 5.0 FPS]", ha="center", va="center", fontsize=8.5, color=C_SLATE_MUTED)

# Step A1: Thumbnail Downsampling
draw_card(5.5, 57.5, 22.0, 5.5, bg_color="#FFFFFF", border_color=C_BORDER, radius=0.5)
ax.text(6.5, 61.2, "Ultra-Fast Thumbnail Differencing (< 0.05 ms)", fontsize=9, fontweight="bold", color=C_SLATE_TITLE)
ax.text(6.5, 59.8, "• Resize frames to 160x160 thumbnails on GPU", fontsize=8.5, color=C_SLATE_TEXT)
ax.text(6.5, 58.5, "• Compute frame delta: M(f) = Mean(|F_t - F_{t-1}|)", fontsize=8.5, color=C_SLATE_MUTED)

# Step A2: Window Partitioning & Argmax
draw_card(5.5, 49.5, 22.0, 6.8, bg_color="#FFFFFF", border_color=C_BORDER, radius=0.5)
ax.text(6.5, 54.5, "3-Frame Temporal Windows (200 ms span)", fontsize=9, fontweight="bold", color=C_SLATE_TITLE)
ax.text(6.5, 53.0, "W_k = [f_{3k}, f_{3k+1}, f_{3k+2}]  for k in 0..49", fontsize=8.5, color=C_BLUE_TECH)
ax.text(6.5, 51.5, "Selection:  f*_k = argmax_{f in W_k} M(f)", fontsize=9, fontweight="bold", color=C_NV_DARK_GREEN)
ax.text(6.5, 50.3, "Guarantees 100% Peak Motion Capture (0% Blind Spots)", fontsize=8, color=C_SLATE_MUTED)

# Step A3: Mathematical 5.0 FPS Guarantee
draw_card(5.5, 42.0, 22.0, 6.2, bg_color="#ECFDF5", border_color=C_NV_GREEN, radius=0.5)
ax.text(6.5, 46.5, "Strict 5.0 FPS Enforcement Proof:", fontsize=9, fontweight="bold", color=C_NV_DARK_GREEN)
ax.text(6.5, 45.0, "Sum(1 frame per W_k) = 50 frames / 10s = 5.0 FPS", fontsize=8.5, fontweight="bold", color=C_SLATE_TITLE)
ax.text(6.5, 43.6, "+24.8% motion captured | Recovered 13 boxes (279 total)", fontsize=8, color=C_SLATE_TEXT)

# Box 1.3: Technique B - Intelligent Idle Sentry Gating
draw_card(4.0, 8.5, 25.0, 29.5, bg_color="#FFFBEB", border_color=C_ALERT_ORANGE, radius=0.8, lw=1.8)
ax.text(16.5, 36.0, "Technique B: Intelligent Idle Sentry Gating", ha="center", va="center", fontsize=11, fontweight="bold", color=C_ALERT_ORANGE)
ax.text(16.5, 34.4, "[Pradipt's Note: Skipping within 5 FPS Stream]", ha="center", va="center", fontsize=8.5, color=C_SLATE_MUTED)

draw_card(5.5, 27.2, 22.0, 6.0, bg_color="#FFFFFF", border_color=C_BORDER, radius=0.5)
ax.text(6.5, 31.4, "Lightweight Activity Detection (Δ < 0.05 ms)", fontsize=9, fontweight="bold", color=C_SLATE_TITLE)
ax.text(6.5, 29.9, "Computes global active patch fraction Δ", fontsize=8.5, color=C_SLATE_TEXT)
ax.text(6.5, 28.5, "Threshold Δ = 2.0% (Derived from sensor SNR floor)", fontsize=8.5, fontweight="bold", color=C_ALERT_ORANGE)

draw_card(5.5, 18.0, 22.0, 8.0, bg_color="#FEF3C7", border_color=C_ALERT_ORANGE, radius=0.5)
ax.text(6.5, 24.4, "Dual-Branch Decision Gate:", fontsize=9.5, fontweight="bold", color=C_SLATE_TITLE)
ax.text(6.5, 22.8, "• IF Δ < 2.0% (Empty Street / Zero Motion):", fontsize=8.5, fontweight="bold", color=C_ALERT_ORANGE)
ax.text(7.5, 21.4, "-> SLEEP: Bypass DINOv2 & YOLO completely", fontsize=8, color=C_SLATE_TEXT)
ax.text(7.5, 20.2, "-> Retain prior bboxes in memory (~99% power save)", fontsize=8, color=C_SLATE_MUTED)
ax.text(6.5, 18.8, "• IF Δ >= 2.0% (Pedestrian / Vehicle Entering):", fontsize=8.5, fontweight="bold", color=C_NV_DARK_GREEN)
ax.text(7.5, 17.6, "-> WAKEUP: Forward frame to Spatial Token Caching", fontsize=8, color=C_NV_DARK_GREEN)

draw_card(5.5, 10.0, 22.0, 6.8, bg_color="#FFFFFF", border_color=C_BORDER, radius=0.5)
ax.text(6.5, 15.2, "Physical SNR Justification for 2.0%:", fontsize=9, fontweight="bold", color=C_SLATE_TITLE)
ax.text(6.5, 13.8, "• 2.0% of 2,025 patches = ~40 patches", fontsize=8.5, color=C_SLATE_TEXT)
ax.text(6.5, 12.5, "• Sensor thermal noise = 0.3% - 0.8% (<16 patches)", fontsize=8, color=C_SLATE_MUTED)
ax.text(6.5, 11.2, "• Distant pedestrian + 5x5 dilation = 35-45 patches", fontsize=8, color=C_NV_DARK_GREEN)

# Internal arrows for Column 1
draw_arrow(16.5, 71.0, 16.5, 68.5, color=C_SLATE_TITLE, lw=2.0)
draw_arrow(16.5, 40.5, 16.5, 38.0, color=C_ALERT_ORANGE, lw=2.0)

# ═══════════════════════════════════════════════════════════════════════════
# 3. COLUMN 2: SPATIAL TOKEN REDUCTION & PATCH GATING ENGINE (Middle)
# ═══════════════════════════════════════════════════════════════════════════
draw_card(33.0, 6.0, 34.0, 83.5, bg_color="#FFFFFF", border_color=C_BORDER, radius=1.0, lw=2.0)
draw_card(34.0, 84.5, 32.0, 4.0, bg_color="#F8FAFC", border_color=C_NV_GREEN, radius=0.6, lw=1.5)
ax.text(50.0, 86.8, "STAGE 2 & 3: SPATIAL TOKEN CACHING & DINOv2", ha="center", va="center", fontsize=13, fontweight="bold", color=C_NV_DARK_GREEN)
ax.text(50.0, 85.3, "Patch Gating Engine, Sparse ViT Execution, & GPU Cache Refill", ha="center", va="center", fontsize=9.5, color=C_SLATE_MUTED)

# Box 2.1: Frame Standardization
draw_card(35.0, 71.0, 30.0, 11.5, bg_color=C_BG_CARD, border_color=C_BORDER, radius=0.8)
ax.text(50.0, 80.5, "2.1 Frame Standardization (640x640 Resolution)", ha="center", va="center", fontsize=11.5, fontweight="bold", color=C_SLATE_TITLE)
ax.text(36.5, 77.8, "• Input normalized to ImageNet [0, 1] tensor on CUDA:  F_t in R^{1 x 3 x 640 x 640}", fontsize=9, color=C_SLATE_TEXT)
ax.text(36.5, 76.0, "• Cropped to 630x630 active patch grid:  45 x 45 = 2,025 Spatial Patches (14x14 px)", fontsize=9, color=C_SLATE_TEXT)
ax.text(36.5, 74.2, "• Empirical Sentry Redundancy: 91.97% static background | 8.03% foreground active", fontsize=9, fontweight="bold", color=C_NV_DARK_GREEN)
ax.text(36.5, 72.4, "• Cosine similarity between consecutive frames = 0.9772 (Extreme temporal stability)", fontsize=8.5, color=C_SLATE_MUTED)

# Box 2.2: GPU Patch Difference Detector & Context Dilation
draw_card(35.0, 48.0, 30.0, 20.5, bg_color="#F0FDF4", border_color=C_NV_GREEN, radius=0.8, lw=1.8)
ax.text(50.0, 66.5, "2.2 GPU Patch Difference Detector (< 0.05 ms)", ha="center", va="center", fontsize=11, fontweight="bold", color=C_NV_DARK_GREEN)
ax.text(50.0, 65.0, "Ultra-fast PyTorch CUDA Kernel using 2D Average Pooling", ha="center", va="center", fontsize=8.5, color=C_SLATE_MUTED)

draw_card(36.5, 56.5, 27.0, 7.0, bg_color="#FFFFFF", border_color=C_BORDER, radius=0.5)
ax.text(37.5, 61.8, "Step 1: Patch MAE Computation via F.avg_pool2d", fontsize=9, fontweight="bold", color=C_SLATE_TITLE)
ax.text(37.5, 60.2, "Δ_pixel = Mean_c(|F_t - F_{t-1}|)   ->   diff in R^{1 x 1 x 630 x 630}", fontsize=8.5, color=C_BLUE_TECH)
ax.text(37.5, 58.6, "patch_diff = F.avg_pool2d(diff, k=14, stride=14) in R^{1 x 1 x 45 x 45}", fontsize=8.5, color=C_SLATE_TEXT)
ax.text(37.5, 57.2, "Binary threshold:  mask = (patch_diff > τ)  with τ = 0.015", fontsize=8.5, fontweight="bold", color=C_NV_DARK_GREEN)

draw_card(36.5, 49.0, 27.0, 6.5, bg_color="#FFFFFF", border_color=C_BORDER, radius=0.5)
ax.text(37.5, 53.8, "Step 2: 5x5 Morphological Dilation Padding", fontsize=9, fontweight="bold", color=C_SLATE_TITLE)
ax.text(37.5, 52.2, "mask_dilated = F.max_pool2d(mask, k=5, stride=1, pad=2)", fontsize=8.5, color=C_BLUE_TECH)
ax.text(37.5, 50.8, "Expands active boundaries by 2 patches in all directions", fontsize=8.5, color=C_SLATE_TEXT)
ax.text(37.5, 49.5, "Guarantees object edges and boundary context are 100% captured", fontsize=8, color=C_NV_DARK_GREEN)

# Box 2.3: Dual Routing (Keyframe vs Sparse Frame)
draw_card(35.0, 31.5, 30.0, 14.5, bg_color="#F8FAFC", border_color=C_BORDER, radius=0.8)
ax.text(50.0, 44.2, "2.3 Adaptive Routing & Keyframe Management", ha="center", va="center", fontsize=11, fontweight="bold", color=C_SLATE_TITLE)

# Left: Keyframe branch
draw_card(36.2, 33.0, 13.0, 9.5, bg_color="#FEF2F2", border_color="#EF4444", radius=0.5)
ax.text(42.7, 40.8, "KEYFRAME (20%)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#B91C1C")
ax.text(37.0, 39.0, "• Trigger: Period K=5 (1.0s)", fontsize=8, color=C_SLATE_TEXT)
ax.text(37.0, 37.8, "  OR Large cut (Δ > 40%)", fontsize=8, color=C_SLATE_TEXT)
ax.text(37.0, 36.4, "• Evaluates all 2,026 tokens", fontsize=8, color=C_SLATE_TEXT)
ax.text(37.0, 35.0, "• Fully refreshes GPU cache", fontsize=8, color=C_SLATE_TEXT)
ax.text(37.0, 33.8, "• Latency: 25.49 ms / frame", fontsize=8, fontweight="bold", color="#B91C1C")

# Right: Sparse frame branch
draw_card(50.8, 33.0, 13.0, 9.5, bg_color="#ECFDF5", border_color=C_NV_GREEN, radius=0.5)
ax.text(57.3, 40.8, "SPARSE FRAME (80%)", ha="center", va="center", fontsize=9.5, fontweight="bold", color=C_NV_DARK_GREEN)
ax.text(51.6, 39.0, "• Trigger: Normal sentry (Δ <= 40%)", fontsize=8, color=C_SLATE_TEXT)
ax.text(51.6, 37.8, "• Forwards active + CLS only", fontsize=8, color=C_SLATE_TEXT)
ax.text(51.6, 36.4, "• Sequence length: ~163 tokens", fontsize=8, color=C_SLATE_TEXT)
ax.text(51.6, 35.0, "• Attention FLOPs cut > 100x", fontsize=8, color=C_SLATE_TEXT)
ax.text(51.6, 33.8, "• Latency: 4.98 ms / frame!", fontsize=8, fontweight="bold", color=C_NV_DARK_GREEN)

# Box 2.4: DINOv2 Sparse ViT Forward & GPU Buffer Cache
draw_card(35.0, 8.5, 30.0, 21.0, bg_color="#EFF6FF", border_color=C_BLUE_TECH, radius=0.8, lw=1.8)
ax.text(50.0, 27.6, "Stage 3: DINOv2 ViT Sparse Forward & GPU Buffer", ha="center", va="center", fontsize=11, fontweight="bold", color=C_BLUE_TECH)

draw_card(36.5, 18.2, 27.0, 8.0, bg_color="#FFFFFF", border_color=C_BORDER, radius=0.5)
ax.text(37.5, 24.6, "Sparse ViT Encoder Slicing (12 Transformer Blocks):", fontsize=9, fontweight="bold", color=C_SLATE_TITLE)
ax.text(37.5, 23.2, "• ViT self-attention natively handles arbitrary sequence lengths N x D", fontsize=8.5, color=C_SLATE_TEXT)
ax.text(37.5, 21.8, "• Positional embeddings are injected at layer 0 (positions preserved)", fontsize=8.5, color=C_SLATE_TEXT)
ax.text(37.5, 20.4, "• Tokens forwarded:  T_sparse = [CLS, Patch_active_1, ..., Patch_active_M]", fontsize=8.5, color=C_BLUE_TECH)
ax.text(37.5, 19.0, "• Mean evaluated sequence: 876 tokens across stream (-56.7% tokens)", fontsize=8.5, fontweight="bold", color=C_NV_DARK_GREEN)

draw_card(36.5, 10.0, 27.0, 7.2, bg_color="#FFFFFF", border_color=C_BORDER, radius=0.5)
ax.text(37.5, 15.6, "In-Place GPU Feature Cache Refill [Layers 3, 6, 9, 12]:", fontsize=9, fontweight="bold", color=C_SLATE_TITLE)
ax.text(37.5, 14.2, "Cache Buffer:  C_layer in R^{1 x 2026 x 384} allocated once in GPU VRAM", fontsize=8.5, color=C_BLUE_TECH)
ax.text(37.5, 12.8, "In-Place Overwrite:  C_layer[:, active_indices, :] = sparse_output", fontsize=8.5, fontweight="bold", color=C_NV_DARK_GREEN)
ax.text(37.5, 11.4, "Result: Produces complete 45x45 feature maps without computing background!", fontsize=8, color=C_SLATE_TEXT)

# Connecting arrows inside Column 2
draw_arrow(50.0, 71.0, 50.0, 68.5, color=C_SLATE_TITLE, lw=2.0)
draw_arrow(50.0, 48.0, 50.0, 46.0, color=C_NV_GREEN, lw=2.0)
draw_arrow(50.0, 31.5, 50.0, 29.5, color=C_BLUE_TECH, lw=2.0)

# Connect Column 1 to Column 2 (Active frames flow into Column 2)
draw_arrow(29.0, 76.5, 33.0, 76.5, color=C_NV_GREEN, lw=2.5)
draw_arrow(29.0, 18.0, 33.0, 18.0, color=C_NV_DARK_GREEN, lw=2.5)

# ═══════════════════════════════════════════════════════════════════════════
# 4. COLUMN 3: DOWNSTREAM PERCEPTION HEADS & VALIDATION (Right)
# ═══════════════════════════════════════════════════════════════════════════
draw_card(69.0, 6.0, 29.0, 83.5, bg_color="#FFFFFF", border_color=C_BORDER, radius=1.0, lw=2.0)
draw_card(70.0, 84.5, 27.0, 4.0, bg_color="#F8FAFC", border_color=C_NV_GREEN, radius=0.6, lw=1.5)
ax.text(83.5, 86.8, "STAGE 4: PERCEPTION HEADS & OUTPUT", ha="center", va="center", fontsize=13, fontweight="bold", color=C_NV_DARK_GREEN)
ax.text(83.5, 85.3, "Downstream Task Execution with Zero Retraining", ha="center", va="center", fontsize=9.5, color=C_SLATE_MUTED)

# Box 4.1: Detection Head
draw_card(70.5, 55.0, 26.0, 27.5, bg_color="#F0FDFA", border_color=C_TEAL, radius=0.8, lw=1.8)
ax.text(83.5, 80.5, "4.1 Object Detection Head", ha="center", va="center", fontsize=11.5, fontweight="bold", color=C_TEAL)
ax.text(83.5, 79.0, "ViTYOLOAdapter_v2 + YOLO26s (NMS-Free)", ha="center", va="center", fontsize=8.5, color=C_SLATE_MUTED)

draw_card(71.8, 70.0, 23.4, 7.8, bg_color="#FFFFFF", border_color=C_BORDER, radius=0.5)
ax.text(72.8, 76.2, "Feature Pyramid Construction:", fontsize=9, fontweight="bold", color=C_SLATE_TITLE)
ax.text(72.8, 74.8, "• Consumes cached layers [3, 6, 9, 12] reshaped to 45x45", fontsize=8, color=C_SLATE_TEXT)
ax.text(72.8, 73.4, "• Multi-scale pyramid: P3/8, P4/16, P5/32 scales", fontsize=8, color=C_BLUE_TECH)
ax.text(72.8, 72.0, "• Detail Branch: CNN on raw frames preserves fine edges", fontsize=8, color=C_SLATE_TEXT)

draw_card(71.8, 60.5, 23.4, 8.5, bg_color="#FFFFFF", border_color=C_BORDER, radius=0.5)
ax.text(72.8, 67.4, "YOLO26s Detection & Accuracy Preservation:", fontsize=9, fontweight="bold", color=C_SLATE_TITLE)
ax.text(72.8, 66.0, "• End-to-end NMS-free inference via dual-head matching", fontsize=8, color=C_SLATE_TEXT)
ax.text(72.8, 64.6, "• 266 / 277 boxes detected (96.0% overall retention)", fontsize=8.5, fontweight="bold", color=C_NV_DARK_GREEN)
ax.text(72.8, 63.2, "• 0.968 Mean BBox IoU match against Golden Baseline", fontsize=8.5, fontweight="bold", color=C_NV_DARK_GREEN)
ax.text(72.8, 61.8, "• Cars: 95.9% | Persons: 97.7% | Motorcycles: 100%+", fontsize=8, color=C_SLATE_TEXT)

draw_card(71.8, 55.8, 23.4, 4.0, bg_color="#CCFBF1", border_color=C_TEAL, radius=0.5)
ax.text(72.8, 58.6, "Zero Retraining Contract:", fontsize=8.5, fontweight="bold", color=C_TEAL)
ax.text(72.8, 57.2, "Official adapter_best.pt & yolo26s.pt weights remain 100% untouched", fontsize=8, color=C_SLATE_TITLE)

# Box 4.2: Action Recognition Head
draw_card(70.5, 27.5, 26.0, 25.5, bg_color="#FAF5FF", border_color=C_PURPLE, radius=0.8, lw=1.8)
ax.text(83.5, 51.2, "4.2 Action Recognition Head", ha="center", va="center", fontsize=11.5, fontweight="bold", color=C_PURPLE)
ax.text(83.5, 49.7, "Whole-Scene Kinetics-400 Linear Probe", ha="center", va="center", fontsize=8.5, color=C_SLATE_MUTED)

draw_card(71.8, 41.5, 23.4, 7.2, bg_color="#FFFFFF", border_color=C_BORDER, radius=0.5)
ax.text(72.8, 47.0, "Temporal CLS Token Aggregation:", fontsize=9, fontweight="bold", color=C_SLATE_TITLE)
ax.text(72.8, 45.6, "• [CLS] token is actively forwarded on EVERY frame", fontsize=8, fontweight="bold", color=C_PURPLE)
ax.text(72.8, 44.2, "• Sequence of CLS tokens in R^{T x 384} across time", fontsize=8, color=C_SLATE_TEXT)
ax.text(72.8, 42.8, "• Preserves unbroken 8-frame sliding temporal window", fontsize=8, color=C_NV_DARK_GREEN)

draw_card(71.8, 33.0, 23.4, 7.5, bg_color="#FFFFFF", border_color=C_BORDER, radius=0.5)
ax.text(72.8, 38.8, "Linear Probe Classification (assets/k400_outputs):", fontsize=9, fontweight="bold", color=C_SLATE_TITLE)
ax.text(72.8, 37.4, "• Temporal mean pooling over 384-dim CLS sequence", fontsize=8, color=C_SLATE_TEXT)
ax.text(72.8, 36.0, "• Top-1 Action Baseline: 'playing badminton' (42.1%)", fontsize=8, color=C_SLATE_TEXT)
ax.text(72.8, 34.6, "• Top-1 Action Sparse: 'playing badminton' (41.8%)", fontsize=8, color=C_SLATE_TEXT)
ax.text(72.8, 33.4, "• Agreement: 100% exact match across all temporal windows", fontsize=8.5, fontweight="bold", color=C_NV_DARK_GREEN)

draw_card(71.8, 28.2, 23.4, 4.0, bg_color="#F3E8FF", border_color=C_PURPLE, radius=0.5)
ax.text(72.8, 31.0, "Zero Action Degradation:", fontsize=8.5, fontweight="bold", color=C_PURPLE)
ax.text(72.8, 29.6, "Active CLS forward guarantees temporal action awareness is unbroken", fontsize=8, color=C_SLATE_TITLE)

# Box 4.3: Unified Output Coordinator
draw_card(70.5, 8.5, 26.0, 17.5, bg_color=C_BG_CARD, border_color=C_BORDER, radius=0.8)
ax.text(83.5, 24.2, "4.3 Unified Output Coordinator", ha="center", va="center", fontsize=11.5, fontweight="bold", color=C_SLATE_TITLE)
ax.text(72.0, 21.8, "• perception.json: Bounding boxes + action predictions per frame", fontsize=8.5, color=C_SLATE_TEXT)
ax.text(72.0, 20.2, "• detection_annotated.mp4: Overlay with YOLO26 bounding boxes", fontsize=8.5, color=C_SLATE_TEXT)
ax.text(72.0, 18.6, "• action_annotated.mp4: Overlay with action recognition banners", fontsize=8.5, color=C_SLATE_TEXT)
ax.text(72.0, 17.0, "• annotated_grid.mp4: Synchronized 2x2 multi-camera stitched grid", fontsize=8.5, color=C_SLATE_TEXT)
ax.text(72.0, 15.0, "• Sentry Security Alerting: Dispatches instant alert on perimeter breach", fontsize=8.5, fontweight="bold", color=C_ALERT_ORANGE)
ax.text(72.0, 13.2, "• Hardware Telemetry: Active tokens, latency (ms), and VRAM logs", fontsize=8, color=C_SLATE_MUTED)
ax.text(72.0, 11.2, "• Effective Throughput: 10.38 FPS (+79.9% headroom on Blackwell)", fontsize=8.5, fontweight="bold", color=C_NV_DARK_GREEN)
ax.text(72.0, 9.6, "• VRAM Footprint: 2,423 MB (-41.7% memory savings on GPU)", fontsize=8.5, fontweight="bold", color=C_NV_DARK_GREEN)

# Connect Column 2 to Column 3 (Refilled cache feeds detection and action)
draw_arrow(65.0, 18.0, 69.0, 68.0, color=C_TEAL, lw=2.5, rad=-0.1)
draw_arrow(65.0, 14.0, 69.0, 40.0, color=C_PURPLE, lw=2.5, rad=0.1)

# Draw arrows from Heads to Output Coordinator
draw_arrow(83.5, 55.0, 83.5, 52.0, color=C_SLATE_TITLE, lw=1.8)
draw_arrow(83.5, 27.5, 83.5, 26.0, color=C_SLATE_TITLE, lw=1.8)

# ═══════════════════════════════════════════════════════════════════════════
# 5. BOTTOM HIGHLIGHT BANNER
# ═══════════════════════════════════════════════════════════════════════════
draw_card(2.0, 0.8, 96.0, 4.4, bg_color="#F8FAFC", border_color=C_NV_GREEN, radius=0.6, lw=1.5)
ax.text(4.0, 3.4, "KEY ARCHITECTURAL HIGHLIGHTS FOR NVIDIA SPONSORS & EVALUATORS:", fontsize=9.5, fontweight="bold", color=C_NV_DARK_GREEN)
ax.text(4.0, 2.0, "1. Temporal Optimization: Strictly enforces 5.0 FPS budget via Windowed Argmax (0% blind spots) and skips idle empty frames via SNR 2.0% gating (<0.05 ms).", fontsize=8.5, color=C_SLATE_TEXT)
ax.text(4.0, 1.0, "2. Spatial Token Reduction: Reduces evaluated DINOv2 tokens by 56.7%, dropping sparse ViT latency to 4.98 ms and peak VRAM by 41.7% on Blackwell GPU.", fontsize=8.5, color=C_SLATE_TEXT)
ax.text(54.0, 2.0, "3. Downstream Compatibility: In-place GPU cache re-filling guarantees 100% plug-and-play operation with Adapter-v2 and YOLO26s (zero weight modification).", fontsize=8.5, color=C_SLATE_TEXT)
ax.text(54.0, 1.0, "4. Perception Integrity: Preserves 96.0% object bounding boxes (0.968 IoU) and achieves 100% agreement on temporal action recognition.", fontsize=8.5, color=C_SLATE_TEXT)

# Save high-res PNG and PDF
out_dir = Path("/home/cim/NVIDIA SDP/stage1_baseline")
out_png = out_dir / "nvidia_sdp_comprehensive_workflow.png"
out_pdf = out_dir / "nvidia_sdp_comprehensive_workflow.pdf"

plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor='none')
plt.savefig(out_pdf, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print(f"[+] High-resolution Workflow PNG generated: {out_png}")
print(f"[+] Vector Workflow PDF generated: {out_pdf}")
