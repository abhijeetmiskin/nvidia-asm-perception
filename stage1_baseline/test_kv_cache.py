import sys
from pathlib import Path
import torch
from torchvision import transforms
import decord
from transformers import AutoModel

sys.path.insert(0, "/home/cim/NVIDIA SDP/UnifiedPerception")
from unified_perception.heads.detection import _load_pradipt_impl

def main():
    print("[test] Initializing detector and model...")
    tp = Path("/home/cim/NVIDIA SDP/UnifiedPerception/third_party/training_package")
    detector = _load_pradipt_impl(str(tp), "vit")
    detector.conf_thres = 0.25

    model = AutoModel.from_pretrained("facebook/dinov2-small").cuda().eval()

    video_path = "/home/cim/NVIDIA SDP/stage1_baseline/sample_videos/test_clip_10s.mp4"
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_path, num_threads=1)

    raw_tfm = transforms.Compose([
        transforms.Resize((640, 640), antialias=True),
        transforms.ConvertImageDtype(torch.float32),
    ])
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    f0_raw = raw_tfm(vr[0].permute(2, 0, 1)).unsqueeze(0).cuda()
    f1_raw = raw_tfm(vr[3].permute(2, 0, 1)).unsqueeze(0).cuda()
    f0_norm = normalize(f0_raw)
    f1_norm = normalize(f1_raw)

    print("\n--- Running Ground Truth Baseline on Frame 1 ---")
    with torch.no_grad():
        gt_out1 = model(f1_norm, output_hidden_states=True)
        gt_dets1 = detector.predict_from_features(gt_out1.hidden_states, f1_raw)

    print(f"Frame 1 Baseline Detections: {len(gt_dets1)} boxes")
    for d in gt_dets1:
        cname = d["class_name"]
        conf = d["conf"]
        box = [round(float(x), 1) for x in d["bbox"]]
        print(f"  [GT] {cname:12s} conf={conf:.3f} box={box}")

    print("\n--- Running KV Cached Attention (Active Q attending to All KV) ---")
    with torch.no_grad():
        f0_out = model(f0_norm, output_hidden_states=True)
        cache = {i: f0_out.hidden_states[i].clone() for i in range(13)}

        diff_raw = (f1_raw[:, :, :630, :630] - f0_raw[:, :, :630, :630]).abs().mean(dim=1, keepdim=True)
        patch_diff = torch.nn.functional.avg_pool2d(diff_raw, kernel_size=14, stride=14)

        tau = 0.025
        active_mask = (patch_diff > tau).float()
        active_mask = torch.nn.functional.max_pool2d(active_mask, kernel_size=3, stride=1, padding=1)
        active_indices = (active_mask.view(1, -1) > 0).nonzero()[:, 1]
        active_token_indices = torch.cat([torch.tensor([0], device="cuda"), active_indices + 1])

        print(f"Active tokens: {len(active_token_indices)} / 2026 ({len(active_token_indices)/20.25:.2f}%)")

        emb1 = model.embeddings(f1_norm)
        curr_full_h = cache[0].clone()
        curr_full_h[:, active_token_indices, :] = emb1[:, active_token_indices, :]

        sparse_hs = [None] * 13
        sparse_hs[0] = curr_full_h

        for layer_idx, layer in enumerate(model.encoder.layer, start=1):
            attn = layer.attention.attention
            normed_full = layer.norm1(curr_full_h)
            normed_active = normed_full[:, active_token_indices, :]

            # Q projection (active only)
            q_act = attn.transpose_for_scores(attn.query(normed_active)) # [B, heads, N_act, d]
            # K, V projection from full tokens
            k_full = attn.transpose_for_scores(attn.key(normed_full))    # [B, heads, N_all, d]
            v_full = attn.transpose_for_scores(attn.value(normed_full))  # [B, heads, N_all, d]

            # Scaled Dot-Product Attention: [B, heads, N_act, d]
            attn_out = torch.nn.functional.scaled_dot_product_attention(q_act, k_full, v_full)
            B, heads, N_act, d = attn_out.shape
            attn_out = attn_out.transpose(1, 2).reshape(B, N_act, heads * d)
            attn_out = layer.attention.output.dense(attn_out)

            if hasattr(layer, "layer_scale1") and layer.layer_scale1 is not None:
                attn_out = layer.layer_scale1(attn_out)
            active_h = curr_full_h[:, active_token_indices, :] + attn_out

            normed_act2 = layer.norm2(active_h)
            mlp_act = layer.mlp(normed_act2)
            if hasattr(layer, "layer_scale2") and layer.layer_scale2 is not None:
                mlp_act = layer.layer_scale2(mlp_act)
            active_h = active_h + mlp_act

            curr_full_h = curr_full_h.clone()
            curr_full_h[:, active_token_indices, :] = active_h
            sparse_hs[layer_idx] = curr_full_h

        sparse_dets = detector.predict_from_features(tuple(sparse_hs), f1_raw)
        print(f"\nKV-Cached Detections ({len(sparse_dets)} boxes):")
        for d in sparse_dets:
            cname = d["class_name"]
            conf = d["conf"]
            box = [round(float(x), 1) for x in d["bbox"]]
            print(f"  [KV] {cname:12s} conf={conf:.3f} box={box}")

if __name__ == "__main__":
    main()
