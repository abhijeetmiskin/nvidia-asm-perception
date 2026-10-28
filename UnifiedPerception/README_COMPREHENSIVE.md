# UnifiedPerception — Automotive Surveillance Mode (ASM) Pipeline

> **NVIDIA Senior Design Project (SDP)**  
> Comprehensive Architecture, Pipeline Documentation, and Usage Guide

---

## 1. Project Overview & Context

**Automotive Surveillance Mode (ASM)** monitors a vehicle's perimeter when parked or stationary using onboard surround cameras. 

In standard surveillance mode, processing every video frame through full-scale neural networks causes significant battery drain and computational overhead. To address this, NVIDIA designed the **UnifiedPerception** architecture: a unified, self-contained multi-task vision pipeline where a **single DINOv2 Vision Transformer (ViT) forward pass feeds multiple downstream perception heads**.

```
                           Input Video Stream (.mp4)
                                      │
                                      ▼
         ┌────────────────────────────────────────────────────────┐
         │         Shared DINOv2 Backbone (ViT-S/14)              │
         │  • Single forward pass per frame / clip                │
         │  • Frozen self-supervised visual features              │
         └────────────────────────────┬───────────────────────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   │                                     │
                   ▼                                     ▼
┌─────────────────────────────────────┐ ┌─────────────────────────────────────┐
│       Object Detection Head         │ │       Action Recognition Head       │
│                                     │ │                                     │
│ • Adapter-v2 (Feature Pyramid)      │ │ • Whole-scene Linear Probe          │
│ • YOLO26 Detector (yolo26s.pt)      │ │ • Kinetics-400 Action Classes       │
│ • Low-level CNN Detail Branch       │ │ • Temporal CLS Token Aggregation    │
│                                     │ │                                     │
│ Output: Bounding Boxes + Classes    │ │ Output: Top-5 Actions (e.g. walk,   │
│         (person, vehicle, bag)      │ │         loiter, touch door handle)  │
└──────────────────┬──────────────────┘ └──────────────────┬──────────────────┘
                   │                                       │
                   └──────────────────┬────────────────────┘
                                      │
                                      ▼
         ┌────────────────────────────────────────────────────────┐
         │              Unified Orchestrator                      │
         │  • JSON metadata (perception.json)                     │
         │  • Annotated overlay videos (.mp4)                     │
         │  • Multi-camera 2×2 stitched grid view                 │
         └────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure & File Manifest

```
UnifiedPerception/
├── README.md                              # NVIDIA's original quick-start guide
├── README_COMPREHENSIVE.md                # This comprehensive architectural reference
├── ACTION_DETECTION_DATASETS_RANKED.md    # Survey & ranking of 26 surveillance action datasets
├── setup.sh                               # Environment creation and DINOv2 pre-download script
├── requirements.txt                       # Pinned Python dependencies (PyTorch, Ultralytics, Decord)
├── pyproject.toml                         # Package configuration for editable installation
│
├── assets/                                # Pre-trained Action Recognition Assets
│   ├── k400_outputs/
│   │   └── best.pth                       # Trained linear probe checkpoint for action classification
│   └── kinetics400/
│       └── annotations/
│           └── classes.txt                # 400 human action category names
│
├── third_party/                           # Pre-trained Object Detection Assets & Code
│   └── training_package/
│       ├── yolo26s.pt                     # Official YOLO26 (Small) weights
│       ├── runs/
│       │   └── vit_dinov2_v2_coco/
│       │       └── adapter_best.pt        # Trained Adapter-v2 weights (DINOv2 -> YOLO26 neck)
│       ├── sentry/                        # Adapter module architectures
│       │   ├── adapter/
│       │   │   ├── vit_yolo_adapter_v2.py # Cross-attention ViT-to-CNN feature pyramid adapter
│       │   │   ├── detail_branch.py       # CNN branch capturing raw pixel edge details
│       │   │   └── yolo_neck_runner.py    # YOLO26 neck and head execution
│       │   └── backbone/                  # Feature extractors (DINOv2, DINOv3, Radio)
│       └── scripts/
│           └── run_shared_backbone.py     # Standalone baseline runner
│
└── unified_perception/                    # Importable Python Package
    ├── __init__.py                        # Package export declarations
    ├── __main__.py                        # Entry point for `python -m unified_perception`
    ├── cli.py                             # Argument parsing and flag definitions
    ├── perception_types.py                # Dataclasses (BackboneOutput, HeadResult)
    ├── backbone.py                        # Frame sampling and DINOv2 feature extraction
    ├── orchestrator.py                    # Pipeline coordinator and video overlay generator
    └── heads/                             # Downstream Task Heads
        ├── __init__.py                    # Head registry (detection, action)
        ├── detection.py                   # In-process wrapper for Adapter-v2 + YOLO26
        └── action.py                      # Linear probe wrapper for Kinetics-400 actions
```

---

## 3. Deep Dive into the Three Core Subsystems

### A. The Shared Backbone (`unified_perception/backbone.py`)
- **Model**: `facebook/dinov2-small` (`vits14`, 384-dimensional embedding, 14×14 patch size).
- **Processing**:
  - Resizes input frames to $640 \times 640$ (producing a $45 \times 45 = 2,025$ patch grid).
  - Passes normalized frames through DINOv2 ViT with `output_hidden_states=True`.
- **Extracted Outputs (`BackboneOutput`)**:
  - `cls`: `[T, 384]` sequence of classification tokens across time.
  - `patch_tokens`: `[T, 2025, 384]` dense spatial features from the final transformer block.
  - `hidden_states`: Dictionary storing layer hidden states. Specifically retains layers **3, 6, 9, and 12** to construct a multi-resolution feature pyramid.
  - `frames`: Raw unnormalized $[0, 1]$ image tensors used by the detection head's detail branch.

### B. The Detection Head (`unified_perception/heads/detection.py`)
- **Architecture**: **Adapter-v2 + YOLO26 (Small)**.
- **Why an Adapter is Needed**: DINOv2 outputs flat 1D token sequences. Object detectors (like YOLO) require multi-scale 2D spatial feature pyramids ($P_3, P_4, P_5$ at strides 8, 16, and 32) to detect both small and large objects.
- **Components**:
  1. **Spatial Reshaper**: Maps 1D patch tokens from DINOv2 layers 3, 6, 9, 12 back into 2D feature maps.
  2. **Detail Branch (`detail_branch.py`)**: A lightweight convolutional net that runs on raw pixel frames to recover high-frequency boundary and edge details lost by patch tokenization.
  3. **YOLO26 Neck & Head (`yolo26s.pt`)**: Performs object classification and bounding box regression with native end-to-end NMS-free inference.

### C. The Action Head (`unified_perception/heads/action.py`)
- **Architecture**: Linear Probe trained on **Kinetics-400**.
- **Mechanism**:
  - Takes the sequence of DINOv2 `[CLS]` tokens `[T, 384]`.
  - Aggregates them across time (temporal mean pooling).
  - Feeds the 384-dim vector into a single linear classifier (`assets/k400_outputs/best.pth`).
  - Returns the Top-5 action predictions with confidence probabilities.
  - Supports an optional **Sliding Window Mode** (`SENTRY_ACTION_SLIDING=1`) to recognize localized temporal events (e.g. someone approaching, peering inside, pulling a handle).

---

## 4. Input Dimensions: 640 vs 644

The pipeline supports two square input resolutions:
* **`--image-size 640` (Default & Recommended)**: Matches the pre-trained detection Adapter-v2. Both detection and action heads run successfully.
* **`--image-size 644`**: Since $644 = 14 \times 46$, this provides an exact multiple of the ViT 14×14 patch size (no fractional patch interpolation), giving cleaner CLS tokens for action recognition. However, the detection adapter is trained specifically on 640, so detection will report an error at 644.

---

## 5. Quickstart & Usage Guide

### A. Environment Setup
The package is designed to run in a dedicated virtual environment inside its own directory:

```bash
cd "/home/cim/NVIDIA SDP/UnifiedPerception"

# Run setup (creates .venv, installs dependencies, and pre-downloads DINOv2 weights)
bash setup.sh

# Activate the virtual environment (required before running)
source .venv/bin/activate
```

---

### B. Running Inference on a Single Video Clip

```bash
python -m unified_perception \
    --source /path/to/surveillance_video.mp4 \
    --output-dir out/clip_run \
    --image-size 640
```

#### What You Get in `out/clip_run/`:
1. `perception.json`: Complete structured data containing bounding box coordinates, class labels, and action predictions for every frame.
2. `detection_annotated.mp4`: Source video overlaid with bounding boxes and detection labels.
3. `action_annotated.mp4`: Source video overlaid with the top-1 action banner.
4. `annotated_with_label.mp4`: Combined video showing both detection boxes and action labels simultaneously.

---

### C. Running on Multi-Camera Renders

For vehicles equipped with surround cameras (front, rear, left, right fisheye):

```bash
python -m unified_perception \
    --render-dir /path/to/render_folder \
    --output-dir out/multicam_run
```
*Processes each camera feed independently and automatically stitches all 4 views into a 2×2 synchronized video grid (`annotated_grid.mp4`).*

---

### D. Python API Example

You can embed the pipeline directly into custom scripts:

```python
from unified_perception import analyze

result = analyze(
    mp4="perimeter_check.mp4",
    output_dir="out/test_run",
    image_size=640,
    stride=2  # Process every 2nd frame
)

# Access detection results
detections = result["heads"]["detection"]["primary"]
print(f"Total frames processed: {len(detections)}")

# Access action results
action = result["heads"]["action"]["primary"]
print(f"Primary observed action: {action}")
```

---

## 6. Configuration Environment Variables

| Variable | Default Path / Value | Purpose |
| :--- | :--- | :--- |
| `SENTRY_TRAINING_PACKAGE` | `third_party/training_package` | Points to detection adapter code and weights |
| `SENTRY_ACTION_CKPT` | `assets/k400_outputs/best.pth` | Action recognition model checkpoint |
| `SENTRY_CLASSES_TXT` | `assets/kinetics400/annotations/classes.txt` | 400 action category names |
| `SENTRY_ACTION_SLIDING`| `0` | Set `1` to enable sliding-window temporal action detection |
| `REUSE_OUTPUT_DIR` | `0` | Set `1` to overwrite output directories without timestamp suffixes |
| `HF_HOME` | `.cache/hf` | Local HuggingFace cache directory |

---

## 7. The NVIDIA Project Challenge & Optimization Target

### The Problem
In [`unified_perception/backbone.py`](unified_perception/backbone.py#L189-L190), full DINOv2 self-attention is executed on every frame:
```python
with torch.no_grad():
    out = model(frames_norm, output_hidden_states=True)
```
For a 30 FPS video stream, this computes quadratic self-attention over **2,025 patch tokens 30 times per second**. In parked automotive surveillance, **85%–95% of these tokens represent completely static background** (pavement, parked cars, buildings, sky).

### What NVIDIA Expects from Us
1. **Analyze Temporal Redundancy**: Quantify how many patch tokens remain invariant across consecutive frames in automotive surveillance video.
2. **Reduce Repeated Computation**: Exploit temporal redundancy to bypass redundant DINOv2 calculations on unchanging regions.
3. **No Naive Frame Skipping**: The pipeline must preserve continuous temporal awareness for action recognition and smooth tracking.
4. **Preserve Downstream Performance**: Maintain accurate bounding boxes in the Detection Head and high Top-1/Top-5 accuracy in the Action Head.
5. **Key Candidate Technique**: **Temporal Token Caching & Patch Gating** — detecting changed spatial patches via low-cost frame differencing, routing only dynamic tokens through the deep ViT blocks, and retrieving static tokens from a temporal cache.
