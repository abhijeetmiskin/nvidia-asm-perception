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

    # Frame 0 and Frame 1 (stride 3 -> raw index 0 and 3)
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

    print("\n--- Running Frame 0 Cache + Frame 1 Sparse Patch Gating ---")
    with torch.no_grad():
        f0_out = model(f0_norm, output_hidden_states=True)
        cache = {i: f0_out.hidden_states[i].clone() for i in range(13)}

        # Fast patch diff on raw [0, 1] frames on GPU
        # 45 patches * 14 = 630
        diff_raw = (f1_raw[:, :, :630, :630] - f0_raw[:, :, :630, :630]).abs().mean(dim=1, keepdim=True)
        patch_diff = torch.nn.functional.avg_pool2d(diff_raw, kernel_size=14, stride=14)  # [1, 1, 45, 45]

        for tau in [0.015, 0.025, 0.035]:
            active_mask = (patch_diff > tau).float()
            # 1-patch dilation
            active_mask = torch.nn.functional.max_pool2d(active_mask, kernel_size=3, stride=1, padding=1)

            active_indices = (active_mask.view(1, -1) > 0).nonzero()[:, 1]
            active_token_indices = torch.cat([torch.tensor([0], device="cuda"), active_indices + 1])

            pct_active = (len(active_token_indices) - 1) / 20.25
            print(f"\n[Tau={tau}] Active tokens: {len(active_token_indices)} / 2026 ({pct_active:.2f}%)")

            # Sparse transformer forward
            emb1 = model.embeddings(f1_norm)
            sparse_h = emb1[:, active_token_indices, :]

            sparse_hs = [None] * 13
            sparse_hs[0] = emb1

            for layer_idx, layer in enumerate(model.encoder.layer, start=1):
                sparse_h = layer(sparse_h)[0]
                updated = cache[layer_idx].clone()
                updated[:, active_token_indices, :] = sparse_h
                sparse_hs[layer_idx] = updated

            sparse_dets1 = detector.predict_from_features(tuple(sparse_hs), f1_raw)
            print(f"  Sparse Detections ({len(sparse_dets1)} boxes):")
            for d in sparse_dets1:
                cname = d["class_name"]
                conf = d["conf"]
                box = [round(float(x), 1) for x in d["bbox"]]
                print(f"    {cname:12s} conf={conf:.3f} box={box}")

if __name__ == "__main__":
    main()
