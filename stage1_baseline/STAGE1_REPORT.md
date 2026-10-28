# Stage 1 Baseline Profile Report: Automotive Surveillance Mode (ASM)

> **Project**: NVIDIA Senior Design Project (SDP) — Temporal Redundancy Reduction in ASM  
> **Host Target**: NVIDIA RTX PRO 4000 Blackwell (24GB VRAM, Compute Capability 12.0)  
> **Pipeline**: NVIDIA UnifiedPerception (DINOv2 ViT-S/14 + Adapter-v2 + YOLO26s + K400 Action Head)  
> **Date**: September 2026

---

## 1. Executive Summary

In Stage 1, we established the **unmodified Ground Truth Baseline** for the NVIDIA `UnifiedPerception` pipeline on realistic 1080p surveillance video. 

### Key Findings
1. **The Core Bottleneck Proven**: **DINOv2 accounts for 76.2% of the entire pipeline latency** (51.85 ms out of 68.02 ms per frame).
2. **Detection & Action Heads are Lightweight**: The downstream heads together account for only 23.8% of compute (Detection: 16.07 ms, Action: 0.1 ms).
3. **Throughput**: The baseline achieves **14.7 FPS** in full unoptimized mode.
4. **Golden Record Established**: Extracted **815 ground-truth bounding boxes** across 5 classes (`car`, `person`, `motorcycle`, `bicycle`, `skateboard`) and established the baseline action classification.

---

## 2. Experimental Environment & Hardware Setup

| Parameter | Value | Details |
| :--- | :--- | :--- |
| **GPU** | **NVIDIA RTX PRO 4000 Blackwell** | 24,467 MB VRAM, SM 12.0 (`sm_120`) |
| **CUDA Driver & Toolkit** | CUDA 13.2 / PyTorch 2.14.0+cu130 | Built with native Blackwell architecture support |
| **Backbone Model** | `facebook/dinov2-small` (`vits14`) | 384 feature dimension, $14 \times 14$ patch size |
| **Detection Head** | Adapter-v2 + YOLO26 (Small) | Consumes DINOv2 layers 3, 6, 9, 12 + CNN detail branch |
| **Action Head** | Linear Probe on Kinetics-400 | Temporal mean pooling over CLS tokens |
| **Input Video** | `test_clip_10s.mp4` | 1920×1080 @ 15.0 FPS, 150 frames, stationary surveillance view |
| **Input Dimensions** | $640 \times 640$ | $45 \times 45 = 2,025$ spatial patch tokens per frame |

---

## 3. Micro-Profiling Latency Breakdown

```
Total Per-Frame Latency: 68.02 ms (14.7 FPS)
┌─────────────────────────────────────────────────────────────┬──────────────┬────────┐
│ Pipeline Component                                          │ Latency (ms) │ Share  │
├─────────────────────────────────────────────────────────────┼──────────────┼────────┤
│ [1] DINOv2 ViT-S/14 Backbone Forward (2,025 patches)        │  51.85 ms    │ 76.2 % │
│ [2] Detection Head (Adapter-v2 Feature Pyramid + YOLO26s)   │  16.07 ms    │ 23.6 % │
│ [3] Action Recognition Head (Kinetics-400 Linear Probe)     │   0.10 ms    │  0.2 % │
└─────────────────────────────────────────────────────────────┴──────────────┴────────┘
```

```
Pipeline Latency Distribution:
  DINOv2 Backbone:  [█████████████████████████████████████] 76.2%
  Detection Head:   [████████████] 23.6%
  Action Head:      [▏] 0.2%
```

### Resource Utilization
* **Peak GPU VRAM Allocated**: **12,222.5 MB** (during batch DINOv2 multi-layer hidden states retention).
* **Average Video Decode**: ~1.4 ms per frame.

---

## 4. Ground Truth Perception Outputs (`baseline_golden_record.json`)

The baseline detected **815 object instances** across the 150-frame clip with the following distribution:

| Class Name | Detected Instances | Context in Surveillance Clip |
| :--- | :---: | :--- |
| **`car`** | **287** | Parked vehicles and moving street traffic |
| **`person`** | **263** | Pedestrians walking across the perimeter |
| **`motorcycle`** | **176** | Motorcyclists moving along roadway |
| **`bicycle`** | **84** | Cyclists passing vehicle field-of-view |
| **`skateboard`** | **5** | Transient action instances |
| **Total Boxes** | **815** | Saved in [`baseline_golden_record.json`](file:///home/cim/NVIDIA%20SDP/stage1_baseline/baseline_results/baseline_golden_record.json) |

### Action Recognition Output
* **Top-1 Action**: `playing badminton` (confidence: 42.1%)
* **Top-5 Predicted Actions**: Sport / outdoor human interaction classes from Kinetics-400.

---

## 5. Artifacts Generated in Stage 1

All baseline outputs are saved in [`stage1_baseline/baseline_results/`](file:///home/cim/NVIDIA%20SDP/stage1_baseline/baseline_results/):
1. **[`baseline_golden_record.json`](file:///home/cim/NVIDIA%20SDP/stage1_baseline/baseline_results/baseline_golden_record.json)**: The complete ground-truth JSON containing every bounding box coordinate, confidence score, and action vector.
2. **[`detection_annotated.mp4`](file:///home/cim/NVIDIA%20SDP/stage1_baseline/baseline_results/detection_annotated.mp4)**: 1080p surveillance video overlaid with YOLO26s bounding boxes and labels across all 150 frames.
3. **[`action_annotated.mp4`](file:///home/cim/NVIDIA%20SDP/stage1_baseline/baseline_results/action_annotated.mp4)**: Surveillance video overlaid with action recognition category banners.
4. **[`detection_sample_frame.jpg`](file:///home/cim/NVIDIA%20SDP/stage1_baseline/baseline_results/detection_sample_frame.jpg)**: High-resolution snapshot demonstrating detection quality on vehicles and pedestrians.

---

## 6. Strategic Takeaway for Stage 2 & Stage 3

The Stage 1 baseline numbers provide clear justification for the project direction:
1. **Optimizing the heads is unnecessary**: The detection head takes 16 ms and the action head takes 0.1 ms. Even if the heads were optimized to 0 ms, the pipeline would still be bottlenecked by DINOv2 (51.85 ms).
2. **DINOv2 is the sole high-value target**: Because DINOv2 consumes **76.2% of total execution time**, eliminating redundant patch computations will directly unlock **2× to 3× speedups**.
3. **Validation Ready**: We now possess the `baseline_golden_record.json` file. Any token-caching or patch-gating optimization in Stage 3 will be directly evaluated against this file using an automated script to verify $< 1\%$ mAP regression.
