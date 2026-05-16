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


def pixel_counts(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = float(np.logical_and(np.logical_not(pred), gt).sum())
    tn = float(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())
    return tp, fp, fn, tn


def lesion_counts(pred: np.ndarray, gt: np.ndarray, iou_thr: float = 0.3) -> Tuple[int, int, int]:
    pred_cc = connected_components(pred)
    gt_cc = connected_components(gt)
    if not pred_cc and not gt_cc:
        return 0, 0, 0
    if not pred_cc:
        return 0, 0, len(gt_cc)
    if not gt_cc:
        return 0, len(pred_cc), 0

    matched = set()
    tp = 0
    for pred_comp in pred_cc:
        best_iou = 0.0
        best_idx = None
        for idx, gt_comp in enumerate(gt_cc):
            if idx in matched:
                continue
            inter = np.logical_and(pred_comp, gt_comp).sum()
            union = np.logical_or(pred_comp, gt_comp).sum()
            iou = inter / (union + 1e-6)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= iou_thr:
            matched.add(best_idx)
            tp += 1
    fp = len(pred_cc) - tp
    fn = len(gt_cc) - tp
    return tp, fp, fn


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


@torch.no_grad()
def run_eval(model, loader, device, threshold: float, min_area: int, morph_kernel: int, top_k: int, cls_threshold: float, iou_thr: float):
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
            cls_prob = float(cls_probs[i])
            cls_pred = 1 if cls_prob >= cls_threshold else 0
            cls_gt = int(batch["cls_label"][i].item())

            tp, fp, fn, tn = pixel_counts(pred, gt_masks[i])
            ltps, lfps, lfns = lesion_counts(pred, gt_masks[i], iou_thr=iou_thr)
            rows.append({
                "case_id": meta["case_id"][i],
                "label_group": meta["label_group"][i],
                "consensus_type": meta.get("consensus_type", [""] * image.size(0))[i] if isinstance(meta.get("consensus_type", None), list) else meta.get("consensus_type", ""),
                "dice": (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6),
                "iou": (tp + 1e-6) / (tp + fp + fn + 1e-6),
                "precision": (tp + 1e-6) / (tp + fp + 1e-6),
                "recall": (tp + 1e-6) / (tp + fn + 1e-6),
                "specificity": (tn + 1e-6) / (tn + fp + 1e-6),
                "lesion_precision": safe_div(ltps, ltps + lfps),
                "lesion_recall": safe_div(ltps, ltps + lfns),
                "cls_prob": cls_prob,
                "cls_pred": cls_pred,
                "cls_gt": cls_gt,
                "pixel_tp": int(tp),
                "pixel_fp": int(fp),
                "pixel_fn": int(fn),
                "pixel_tn": int(tn),
                "lesion_tp": int(ltps),
                "lesion_fp": int(lfps),
                "lesion_fn": int(lfns),
            })
    return rows


def summarize(rows: List[Dict]) -> Dict[str, float]:
    metrics = ["dice", "iou", "precision", "recall", "lesion_precision", "lesion_recall"]
    out = {m: float(np.mean([float(r[m]) for r in rows])) for m in metrics}
    cls_pred = np.array([int(r["cls_pred"]) for r in rows])
    cls_gt = np.array([int(r["cls_gt"]) for r in rows])
    out["case_accuracy"] = float((cls_pred == cls_gt).mean())
    out["case_sensitivity"] = float(((cls_pred == 1) & (cls_gt == 1)).sum() / max(1, (cls_gt == 1).sum()))
    out["case_specificity"] = float(((cls_pred == 0) & (cls_gt == 0)).sum() / max(1, (cls_gt == 0).sum()))
    out["case_precision"] = float(((cls_pred == 1) & (cls_gt == 1)).sum() / max(1, (cls_pred == 1).sum()))
    out["case_npv"] = float(((cls_pred == 0) & (cls_gt == 0)).sum() / max(1, (cls_pred == 0).sum()))
    out["case_tp"] = int(((cls_pred == 1) & (cls_gt == 1)).sum())
    out["case_fp"] = int(((cls_pred == 1) & (cls_gt == 0)).sum())
    out["case_tn"] = int(((cls_pred == 0) & (cls_gt == 0)).sum())
    out["case_fn"] = int(((cls_pred == 0) & (cls_gt == 1)).sum())
    return out


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
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
    parser.add_argument("--iou-thr", type=float, default=0.3)
    parser.add_argument("--metrics-name", type=str, default="test_metrics_case_mean.csv")
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

    rows = run_eval(
        model,
        loader,
        device,
        cfg["threshold"],
        args.min_area,
        args.morph_kernel,
        args.top_k,
        args.cls_threshold,
        args.iou_thr,
    )
    metrics = summarize(rows)
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
        "iou_thr": args.iou_thr,
        **metrics,
    }
    write_csv(output_dir / args.metrics_name, [metric_row])
    per_case_name = args.per_case_name or args.metrics_name.replace(".csv", "_per_case.csv")
    per_case_rows = [
        {
            "case_id": r["case_id"],
            "label_group": r["label_group"],
            "consensus_type": r["consensus_type"],
            "dice": r["dice"],
            "iou": r["iou"],
            "precision": r["precision"],
            "recall": r["recall"],
            "specificity": r["specificity"],
            "lesion_precision": r["lesion_precision"],
            "lesion_recall": r["lesion_recall"],
            "cls_prob": r["cls_prob"],
            "cls_pred": r["cls_pred"],
            "cls_gt": r["cls_gt"],
            "pixel_tp": r["pixel_tp"],
            "pixel_fp": r["pixel_fp"],
            "pixel_fn": r["pixel_fn"],
            "pixel_tn": r["pixel_tn"],
            "lesion_tp": r["lesion_tp"],
            "lesion_fp": r["lesion_fp"],
            "lesion_fn": r["lesion_fn"],
        }
        for r in rows
    ]
    write_csv(output_dir / per_case_name, per_case_rows)
    print(json.dumps(metric_row, ensure_ascii=False, indent=2))
    print(f"Saved metrics to {output_dir / args.metrics_name}")
    print(f"Saved per-case scores to {output_dir / per_case_name}")


if __name__ == "__main__":
    main()
