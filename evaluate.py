import argparse
import copy
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.dataset import APMultiTaskDataset, set_seed
from models.model import ApexTransNet


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def connected_components(mask: np.ndarray) -> List[np.ndarray]:
    mask = mask.astype(bool)
    visited = np.zeros(mask.shape, dtype=bool)
    comps: List[np.ndarray] = []
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            coords = []
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny == cy and nx == cx:
                            continue
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            comp = np.zeros(mask.shape, dtype=np.uint8)
            yy, xx = zip(*coords)
            comp[np.array(yy), np.array(xx)] = 1
            comps.append(comp)
    return comps


def filter_components(mask: np.ndarray, min_area: int = 0, top_k: int = 0) -> np.ndarray:
    comps = connected_components(mask)
    kept = []
    for comp in comps:
        area = int(comp.sum())
        if area >= min_area:
            kept.append((area, comp))
    kept.sort(key=lambda x: x[0], reverse=True)
    if top_k and top_k > 0:
        kept = kept[:top_k]
    out = np.zeros_like(mask, dtype=np.uint8)
    for _, comp in kept:
        out = np.maximum(out, comp)
    return out


def morph_close(mask: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1:
        return mask.astype(np.uint8)
    pad = kernel // 2
    padded = np.pad(mask, ((pad, pad), (pad, pad)), mode="constant")
    dilated = np.zeros_like(mask)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            dilated[y, x] = padded[y:y + kernel, x:x + kernel].max()
    padded = np.pad(dilated, ((pad, pad), (pad, pad)), mode="constant", constant_values=1)
    eroded = np.zeros_like(mask)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            eroded[y, x] = padded[y:y + kernel, x:x + kernel].min()
    return eroded.astype(np.uint8)


def pixel_counts(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = float(np.logical_and(np.logical_not(pred), gt).sum())
    return tp, fp, fn


def lesion_counts(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int]:
    pred_comps = connected_components(pred)
    gt_comps = connected_components(gt)
    matched_gt = set()
    tp = 0
    for pc in pred_comps:
        best_idx = None
        best_iou = 0.0
        for i, gc in enumerate(gt_comps):
            inter = np.logical_and(pc, gc).sum()
            union = np.logical_or(pc, gc).sum()
            iou = float(inter / union) if union else 0.0
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_idx is not None and best_iou > 0.0 and best_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_idx)
    fp = len(pred_comps) - tp
    fn = len(gt_comps) - len(matched_gt)
    return tp, fp, fn


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


@torch.no_grad()
def collect_predictions(model, loader, device, threshold: float, min_area: int, morph_kernel: int, top_k: int, case_score_mode: str):
    rows = []
    model.eval()
    for batch in loader:
        image = batch["image"].to(device)
        out = model(image)
        probs = torch.sigmoid(out["seg_logits"]).cpu().numpy()[:, 0]
        cls_probs = torch.sigmoid(out["cls_logits"]).cpu().numpy().reshape(-1)
        gt_masks = batch["hard_mask"].numpy()[:, 0].astype(np.uint8)
        meta = batch["meta"]
        for i in range(image.size(0)):
            pred = (probs[i] >= threshold).astype(np.uint8)
            pred = morph_close(pred, morph_kernel)
            pred = filter_components(pred, min_area=min_area, top_k=top_k)
            if case_score_mode == "cls":
                case_score = float(cls_probs[i])
            elif case_score_mode == "max_prob":
                case_score = float(probs[i].max())
            else:
                comps = connected_components(pred)
                if comps:
                    comp_scores = [float(probs[i][comp.astype(bool)].mean() * np.sqrt(comp.sum())) for comp in comps]
                    case_score = max(comp_scores)
                else:
                    case_score = 0.0
            rows.append(
                {
                    "case_id": meta["case_id"][i],
                    "label_group": meta["label_group"][i],
                    "gt": gt_masks[i],
                    "pred": pred,
                    "case_score": case_score,
                    "cls_prob": float(cls_probs[i]),
                    "prob_max": float(probs[i].max()),
                    "pred_area": int(pred.sum()),
                    "gt_area": int(gt_masks[i].sum()),
                }
            )
    return rows


def summarize(rows: List[Dict], cls_threshold: float) -> Dict[str, float]:
    eps = 1e-6
    tp = fp = fn = 0.0
    ltp = lfp = lfn = 0
    case_tp = case_fp = case_tn = case_fn = 0
    for row in rows:
        a, b, c = pixel_counts(row["pred"], row["gt"])
        tp += a
        fp += b
        fn += c
        la, lb, lc = lesion_counts(row["pred"], row["gt"])
        ltp += la
        lfp += lb
        lfn += lc
        true_pos = row["label_group"] == "ap"
        pred_pos = row["case_score"] >= cls_threshold
        if true_pos and pred_pos:
            case_tp += 1
        elif (not true_pos) and pred_pos:
            case_fp += 1
        elif (not true_pos) and (not pred_pos):
            case_tn += 1
        else:
            case_fn += 1
    return {
        "pixel_dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "pixel_iou": (tp + eps) / (tp + fp + fn + eps),
        "pixel_precision": (tp + eps) / (tp + fp + eps),
        "pixel_recall": (tp + eps) / (tp + fn + eps),
        "lesion_precision": safe_div(ltp, ltp + lfp),
        "lesion_recall": safe_div(ltp, ltp + lfn),
        "case_accuracy": safe_div(case_tp + case_tn, case_tp + case_fp + case_tn + case_fn),
        "case_sensitivity": safe_div(case_tp, case_tp + case_fn),
        "case_specificity": safe_div(case_tn, case_tn + case_fp),
        "case_precision": safe_div(case_tp, case_tp + case_fp),
        "case_npv": safe_div(case_tn, case_tn + case_fn),
        "case_tp": case_tp,
        "case_fp": case_fp,
        "case_tn": case_tn,
        "case_fn": case_fn,
    }


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_stage2_server.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--cls-threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=0)
    parser.add_argument("--morph-kernel", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--case-score-mode", type=str, choices=["cls", "max_prob", "lesion_aware"], default="cls")
    parser.add_argument("--metrics-name", type=str, default="test_metrics.csv")
    parser.add_argument("--per-case-name", type=str, default=None)
    args = parser.parse_args()

    cfg = copy.deepcopy(load_config(args.config))
    if args.threshold is not None:
        cfg["threshold"] = args.threshold
    run_name = args.run_name or cfg.get("run_name", "apex_trans")
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = APMultiTaskDataset(
        cfg["metadata_csv"],
        args.split,
        tuple(cfg["image_size"]),
        cfg.get("target_mode", "soft_vote"),
        cfg.get("use_uncertainty_weight", True),
        cfg.get("downweight_two_annotator_cases", True),
        augment=False,
    )
    loader = DataLoader(ds, batch_size=cfg.get("batch_size", 4), shuffle=False, num_workers=cfg.get("num_workers", 0))
    model = ApexTransNet(
        in_channels=1,
        encoder_pretrained_path=None,
        transformer_layers=cfg.get("transformer_layers", 4),
        use_boundary_head=cfg.get("use_boundary_head", True),
        use_location_head=cfg.get("use_location_head", True),
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    rows = collect_predictions(
        model,
        loader,
        device,
        cfg["threshold"],
        args.min_area,
        args.morph_kernel,
        args.top_k,
        args.case_score_mode,
    )
    metrics = summarize(rows, args.cls_threshold)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_row = {
        "run_name": run_name,
        "split": args.split,
        "threshold": cfg["threshold"],
        "cls_threshold": args.cls_threshold,
        "min_area": args.min_area,
        "morph_kernel": args.morph_kernel,
        "top_k": args.top_k,
        "case_score_mode": args.case_score_mode,
        **metrics,
    }
    write_csv(output_dir / args.metrics_name, [metric_row])
    per_case_name = args.per_case_name or args.metrics_name.replace(".csv", "_per_case.csv")
    per_case_rows = [
        {
            "case_id": r["case_id"],
            "label_group": r["label_group"],
            "case_score": r["case_score"],
            "cls_prob": r["cls_prob"],
            "prob_max": r["prob_max"],
            "pred_area": r["pred_area"],
            "gt_area": r["gt_area"],
        }
        for r in rows
    ]
    write_csv(output_dir / per_case_name, per_case_rows)
    print(json.dumps(metric_row, ensure_ascii=False, indent=2))
    print(f"Saved metrics to {output_dir / args.metrics_name}")
    print(f"Saved per-case scores to {output_dir / per_case_name}")


if __name__ == "__main__":
    main()
