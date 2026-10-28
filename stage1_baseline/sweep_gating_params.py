import sys
from pathlib import Path
import torch
from torchvision import transforms
import decord
from transformers import AutoModel

sys.path.insert(0, "/home/cim/NVIDIA SDP/UnifiedPerception")
from unified_perception.heads.detection import _load_pradipt_impl

def box_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def match_detections(gt_dets, pred_dets, iou_thresh=0.5):
    # Match predictions to GT by class and IoU
    matched_gt = set()
    ious = []
    for p in pred_dets:
        best_iou = 0.0
        best_gt_idx = -1
        for i, g in enumerate(gt_dets):
            if i in matched_gt or g["class_name"] != p["class_name"]:
                continue
            iou = box_iou(g["bbox"], p["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        if best_iou >= iou_thresh and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            ious.append(best_iou)
    recall = len(matched_gt) / len(gt_dets) if gt_dets else 1.0
    precision = len(matched_gt) / len(pred_dets) if pred_dets else (1.0 if not gt_dets else 0.0)
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    return recall, precision, mean_iou

def main():
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

    # Evaluate across first 10 frames (stride 3 -> frames 0, 3, 6, 9, 12, 15, 18, 21, 24, 27)
    sampled_indices = [i * 3 for i in range(10)]
    frames_raw = [raw_tfm(vr[idx].permute(2, 0, 1)).unsqueeze(0).cuda() for idx in sampled_indices]
    frames_norm = [normalize(f) for f in frames_raw]

    print("Computing Ground Truth for first 10 frames...")
    gt_results = []
    with torch.no_grad():
        for i in range(len(sampled_indices)):
            out = model(frames_norm[i], output_hidden_states=True)
            dets = detector.predict_from_features(out.hidden_states, frames_raw[i])
            gt_results.append((out.hidden_states, dets))

    total_gt_boxes = sum(len(d) for _, d in gt_results)
    print(f"Total GT detections across 10 frames: {total_gt_boxes}")

    # Parameter grid search
    tau_list = [0.010, 0.015, 0.020, 0.025]
    dilation_list = [1, 2] # 1 -> 3x3 maxpool, 2 -> 5x5 maxpool

    print("\n" + "=" * 78)
    print(f"{'Tau':<8} {'Dilation':<10} {'Active%':<10} {'Active#':<10} {'Recall':<10} {'Precision':<10} {'Mean IoU':<10}")
    print("=" * 78)

    for tau in tau_list:
        for dil in dilation_list:
            ksize = 1 + 2 * dil
            padding = dil

            total_active_tokens = []
            recalls = []
            precisions = []
            ious = []

            # Initialize cache with Frame 0
            cache = {l: gt_results[0][0][l].clone() for l in range(13)}

            for t in range(1, len(sampled_indices)):
                prev_raw = frames_raw[t - 1]
                curr_raw = frames_raw[t]
                curr_norm = frames_norm[t]

                # Patch difference on raw image [0, 1]
                diff_raw = (curr_raw[:, :, :630, :630] - prev_raw[:, :, :630, :630]).abs().mean(dim=1, keepdim=True)
                patch_diff = torch.nn.functional.avg_pool2d(diff_raw, kernel_size=14, stride=14)

                active_mask = (patch_diff > tau).float()
                active_mask = torch.nn.functional.max_pool2d(active_mask, kernel_size=ksize, stride=1, padding=padding)

                active_indices = (active_mask.view(1, -1) > 0).nonzero()[:, 1]
                active_token_indices = torch.cat([torch.tensor([0], device="cuda"), active_indices + 1])

                n_act = len(active_token_indices)
                total_active_tokens.append(n_act)

                # Sparse forward
                emb = model.embeddings(curr_norm)
                sparse_h = emb[:, active_token_indices, :]

                sparse_hs = [None] * 13
                sparse_hs[0] = emb

                for layer_idx, layer in enumerate(model.encoder.layer, start=1):
                    sparse_h = layer(sparse_h)[0]
                    updated = cache[layer_idx].clone()
                    updated[:, active_token_indices, :] = sparse_h
                    cache[layer_idx] = updated
                    sparse_hs[layer_idx] = updated

                pred_dets = detector.predict_from_features(tuple(sparse_hs), curr_raw)
                gt_dets = gt_results[t][1]

                rec, prec, miou = match_detections(gt_dets, pred_dets)
                recalls.append(rec)
                precisions.append(prec)
                ious.append(miou)

            avg_act_pct = (sum(total_active_tokens) / len(total_active_tokens) - 1) / 20.25
            avg_act_cnt = sum(total_active_tokens) / len(total_active_tokens)
            avg_rec = sum(recalls) / len(recalls)
            avg_prec = sum(precisions) / len(precisions)
            avg_iou = sum(ious) / len(ious)

            print(f"{tau:<8.3f} {dil:<10d} {avg_act_pct:<10.2f}% {avg_act_cnt:<10.1f} {avg_rec*100:<9.1f}% {avg_prec*100:<9.1f}% {avg_iou:<10.3f}")

if __name__ == "__main__":
    main()
