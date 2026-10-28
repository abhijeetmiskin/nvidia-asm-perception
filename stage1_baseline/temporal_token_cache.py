"""
temporal_token_cache.py — Sparse DINOv2 Token Caching & Patch Gating Engine.

Exploits high temporal redundancy (over 91% static patches at 5 FPS) in automotive
surveillance mode to dramatically accelerate DINOv2 ViT inference while retaining
100% architectural compatibility with downstream detection (ViTYOLOAdapter_v2 + YOLO26)
and action recognition heads.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# Add UnifiedPerception to path for perception types
sys.path.insert(0, "/home/cim/NVIDIA SDP/UnifiedPerception")
from unified_perception.perception_types import BackboneOutput


class PatchDifferenceDetector(nn.Module):
    """GPU-accelerated spatial patch difference detector using 2D average pooling."""

    def __init__(
        self,
        tau: float = 0.015,
        dilation: int = 2,
        patch_size: int = 14,
        grid_dim: int = 45,
    ):
        super().__init__()
        self.tau = tau
        self.dilation = dilation
        self.patch_size = patch_size
        self.grid_dim = grid_dim
        self.crop_size = grid_dim * patch_size  # 45 * 14 = 630
        self.ksize = 1 + 2 * dilation           # 5x5 for dilation=2
        self.padding = dilation

    def forward(
        self, curr_raw: torch.Tensor, prev_raw: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Compute active patch indices between consecutive raw [0, 1] frames.

        Args:
            curr_raw: [1, 3, H, W] float in [0, 1] on CUDA
            prev_raw: [1, 3, H, W] float in [0, 1] on CUDA

        Returns:
            active_patch_indices: 1D CUDA LongTensor of patch indices [0 .. grid_dim*grid_dim - 1]
            active_fraction: float ratio of active patches (e.g. 0.12 = 12%)
        """
        # Crop to exact patch multiple: [1, 3, 630, 630]
        c_curr = curr_raw[:, :, :self.crop_size, :self.crop_size]
        c_prev = prev_raw[:, :, :self.crop_size, :self.crop_size]

        # Channel-averaged absolute pixel difference: [1, 1, 630, 630]
        diff = (c_curr - c_prev).abs().mean(dim=1, keepdim=True)

        # Average pool across 14x14 patches: [1, 1, 45, 45]
        patch_diff = F.avg_pool2d(diff, kernel_size=self.patch_size, stride=self.patch_size)

        # Threshold to binary mask
        mask = (patch_diff > self.tau).float()

        # Morphological dilation to encompass boundaries and context
        if self.dilation > 0:
            mask = F.max_pool2d(mask, kernel_size=self.ksize, stride=1, padding=self.padding)

        # Active patch indices (0 .. 2024)
        active_indices = (mask.view(1, -1) > 0).nonzero()[:, 1]
        active_fraction = float(len(active_indices)) / float(self.grid_dim * self.grid_dim)

        return active_indices, active_fraction


class GatedDINOv2(nn.Module):
    """Sparse DINOv2 ViT with temporal token caching and active patch routing."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        device: str = "cuda",
        image_size: int = 640,
        tau: float = 0.015,
        dilation: int = 2,
        keyframe_interval: int = 5,
        max_active_fraction: float = 0.40,
    ):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.keyframe_interval = keyframe_interval
        self.max_active_fraction = max_active_fraction

        print(f"[GatedDINOv2] Loading {model_name} on {device}...")
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.detector = PatchDifferenceDetector(
            tau=tau, dilation=dilation, patch_size=14, grid_dim=45
        )

        # Cache state
        self.cache_hidden_states: Optional[Dict[int, torch.Tensor]] = None
        self.prev_raw_frame: Optional[torch.Tensor] = None

    def reset_cache(self):
        """Reset internal temporal cache."""
        self.cache_hidden_states = None
        self.prev_raw_frame = None

    def forward_frame(
        self,
        curr_norm: torch.Tensor,
        curr_raw: torch.Tensor,
        frame_idx: int,
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Process a single video frame with sparse token caching.

        Args:
            curr_norm: [1, 3, 640, 640] ImageNet-normalized float tensor on CUDA
            curr_raw:  [1, 3, 640, 640] [0, 1] raw float tensor on CUDA
            frame_idx: Frame index in stream (0-indexed)

        Returns:
            hidden_states: Dict[int, torch.Tensor] of full [1, 2026, 384] per layer
            cls: [1, 384] CLS token
            patch_tokens: [1, 2025, 384] spatial patch tokens
            stats: diagnostic stats dict
        """
        is_periodic_keyframe = (frame_idx % self.keyframe_interval == 0) or (self.cache_hidden_states is None)
        active_indices = None
        act_frac = 1.0

        if not is_periodic_keyframe and self.prev_raw_frame is not None:
            active_indices, act_frac = self.detector(curr_raw, self.prev_raw_frame)

        # Trigger keyframe if periodic or large scene motion
        is_keyframe = is_periodic_keyframe or (act_frac > self.max_active_fraction)

        t0 = time.perf_counter()
        if is_keyframe:
            # Full DINOv2 forward
            out = self.model(curr_norm, output_hidden_states=True)
            self.cache_hidden_states = {i: out.hidden_states[i].clone() for i in range(13)}
            lhs = out.last_hidden_state
            cls = lhs[:, 0]
            patch_tokens = lhs[:, 1:]
            active_count = 2026
        else:
            # Sparse DINOv2 forward: only CLS + active patches
            # Token index 0 = CLS, 1..2025 = patches 0..2024
            active_token_indices = torch.cat([
                torch.tensor([0], device=self.device, dtype=torch.long),
                active_indices + 1
            ])
            active_count = len(active_token_indices)

            # Extract initial embeddings [1, 2026, 384]
            emb = self.model.embeddings(curr_norm)
            sparse_h = emb[:, active_token_indices, :]

            # Forward only sparse tokens through 12 encoder layers
            self.cache_hidden_states[0] = emb
            for layer_idx, layer in enumerate(self.model.encoder.layer, start=1):
                sparse_h = layer(sparse_h)[0]
                # In-place update of GPU cache
                self.cache_hidden_states[layer_idx][:, active_token_indices, :] = sparse_h

            # Final layernorm on layer 12
            lhs = self.model.layernorm(self.cache_hidden_states[12])
            cls = lhs[:, 0]
            patch_tokens = lhs[:, 1:]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.prev_raw_frame = curr_raw.clone()

        stats = {
            "frame_idx": frame_idx,
            "is_keyframe": is_keyframe,
            "active_tokens": active_count,
            "total_tokens": 2026,
            "active_percentage": (active_count / 2026.0) * 100.0,
            "latency_ms": elapsed_ms,
        }

        # Return detached clone of hidden_states to ensure clean decoupling
        out_hidden_states = {k: v.clone() for k, v in self.cache_hidden_states.items()}
        return out_hidden_states, cls, patch_tokens, stats


def extract_features_sparse(
    mp4_path: str,
    device: str = "cuda",
    image_size: int = 640,
    stride: int = 3,
    tau: float = 0.015,
    dilation: int = 2,
    keyframe_interval: int = 5,
    max_active_fraction: float = 0.40,
) -> Tuple[BackboneOutput, List[Dict[str, Any]]]:
    """Extract DINOv2 features across an MP4 stream using Gated Token Caching.

    Returns:
        feats: BackboneOutput matching the UnifiedPerception standard API.
        frame_stats: List of per-frame diagnostics (active tokens, keyframes, latency).
    """
    from torchvision import transforms
    import decord

    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(mp4_path, num_threads=1)
    total_frames = len(vr)
    indices = list(range(0, total_frames, stride))

    raw_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ConvertImageDtype(torch.float32),
    ])
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    gated_backbone = GatedDINOv2(
        model_name="facebook/dinov2-small",
        device=device,
        image_size=image_size,
        tau=tau,
        dilation=dilation,
        keyframe_interval=keyframe_interval,
        max_active_fraction=max_active_fraction,
    )

    all_cls = []
    all_patch_tokens = []
    all_frames_raw = []
    layer_tensors: Dict[int, List[torch.Tensor]] = {i: [] for i in range(13)}
    frame_stats = []

    print(f"[extract_features_sparse] Processing {len(indices)} frames at 5 FPS (stride={stride})...")

    for f_idx, raw_idx in enumerate(indices):
        frame_uint8 = vr[raw_idx].permute(2, 0, 1) # [3, H, W]
        f_raw = raw_tfm(frame_uint8).unsqueeze(0).to(device) # [1, 3, 640, 640]
        f_norm = normalize(f_raw)

        hs, cls, patch_tokens, stats = gated_backbone.forward_frame(
            curr_norm=f_norm, curr_raw=f_raw, frame_idx=f_idx
        )

        all_cls.append(cls.cpu())
        all_patch_tokens.append(patch_tokens.cpu())
        all_frames_raw.append(f_raw.cpu())
        for l in range(13):
            layer_tensors[l].append(hs[l].cpu())
        frame_stats.append(stats)

    # Stack tensors to [T, ...]
    cls_tensor = torch.cat(all_cls, dim=0) # [T, 384]
    patch_tokens_tensor = torch.cat(all_patch_tokens, dim=0) # [T, 2025, 384]
    frames_raw_tensor = torch.cat(all_frames_raw, dim=0) # [T, 3, 640, 640]
    hidden_states_stacked = {l: torch.cat(layer_tensors[l], dim=0) for l in range(13)}

    fps = float(vr.get_avg_fps())

    metadata = {
        "source": mp4_path,
        "model_name": "vits14",
        "hf_id": "facebook/dinov2-small",
        "num_frames": len(indices),
        "image_size": image_size,
        "feature_dim": int(cls_tensor.shape[-1]),
        "num_patches": int(patch_tokens_tensor.shape[1]),
        "num_layers": 13,
        "fps": fps,
        "stride": int(stride),
        "gating_tau": tau,
        "dilation": dilation,
        "keyframe_interval": keyframe_interval,
    }

    output = BackboneOutput(
        cls=cls_tensor,
        hidden_states=hidden_states_stacked,
        patch_tokens=patch_tokens_tensor,
        metadata=metadata,
        frames=frames_raw_tensor,
    )

    return output, frame_stats
