import argparse
import copy
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate import collect_predictions, summarize
from models.dataset import APMultiTaskDataset, set_seed
from models.model import ApexTransNet


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_floats(text: str):
    return [float(x) for x in text.split(",") if x.strip()]


def parse_ints(text: str):
    return [int(x) for x in text.split(",") if x.strip()]


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_stage2_server.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--thresholds", type=str, default="0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--min-areas", type=str, default="0,16,32,64,96,128,192")
    parser.add_argument("--morph-kernels", type=str, default="0,3")
    parser.add_argument("--top-ks", type=str, default="0,1,2,3")
    parser.add_argument("--case-score-mode", type=str, choices=["cls", "max_prob", "lesion_aware"], default="cls")
    parser.add_argument("--cls-threshold", type=float, default=0.5)
    parser.add_argument("--out", type=str, default="outputs/postprocess_search_val.csv")
    args = parser.parse_args()

    cfg = copy.deepcopy(load_config(args.config))
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

    results = []
    for thr in parse_floats(args.thresholds):
        for min_area in parse_ints(args.min_areas):
            for morph in parse_ints(args.morph_kernels):
                for top_k in parse_ints(args.top_ks):
                    rows = collect_predictions(
                        model,
                        loader,
                        device,
                        thr,
                        min_area=min_area,
                        morph_kernel=morph,
                        top_k=top_k,
                        case_score_mode=args.case_score_mode,
                    )
                    metrics = summarize(rows, args.cls_threshold)
                    score = metrics["pixel_dice"] + 0.2 * metrics["lesion_precision"] + 0.2 * metrics["lesion_recall"]
                    results.append(
                        {
                            "split": args.split,
                            "threshold": thr,
                            "min_area": min_area,
                            "morph_kernel": morph,
                            "top_k": top_k,
                            "selection_score": score,
                            **metrics,
                        }
                    )
    results.sort(key=lambda r: (r["selection_score"], r["pixel_dice"], r["lesion_precision"]), reverse=True)
    out = Path(args.out)
    write_rows(out, results)
    print("Best post-processing setting:")
    print(json.dumps(results[0], ensure_ascii=False, indent=2))
    print(f"Saved search table to {out}")


if __name__ == "__main__":
    main()
