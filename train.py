import argparse
import copy
import json
import os
from pathlib import Path
from typing import Dict, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.dataset import APMultiTaskDataset, set_seed
from models.model import ApexTransNet


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    if weights is None:
        weights = torch.ones_like(targets)
    num = 2.0 * (weights * probs * targets).sum(dim=(2, 3)) + eps
    den = (weights * (probs + targets)).sum(dim=(2, 3)) + eps
    return 1.0 - (num / den).mean()


def seg_loss(logits: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=weights) if weights is not None else nn.functional.binary_cross_entropy_with_logits(logits, targets)
    return bce + soft_dice_loss(logits, targets, weights)


def anatomy_constraint_loss(seg_logits: torch.Tensor, loc_logits: torch.Tensor, loc_targets: torch.Tensor, hard_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    seg_probs = torch.sigmoid(seg_logits)
    loc_probs = torch.sigmoid(loc_logits)
    support = torch.clamp(torch.max(loc_probs, loc_targets), 0.0, 1.0)
    outside = (seg_probs * (1.0 - support)).mean()
    inside_target = torch.clamp(torch.max(loc_targets, hard_mask), 0.0, 1.0)
    overlap = 1.0 - ((seg_probs * inside_target).sum(dim=(2, 3)) + eps) / (seg_probs.sum(dim=(2, 3)) + eps)
    return outside + overlap.mean()


@torch.no_grad()
def hard_metrics(logits: torch.Tensor, hard_targets: torch.Tensor, thr: float) -> Dict[str, float]:
    preds = (torch.sigmoid(logits) > thr).float()
    eps = 1e-6
    tp = (preds * hard_targets).sum().item()
    fp = (preds * (1 - hard_targets)).sum().item()
    fn = ((1 - preds) * hard_targets).sum().item()
    return {
        "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "iou": (tp + eps) / (tp + fp + fn + eps),
        "recall": (tp + eps) / (tp + fn + eps),
    }


def build_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], cfg: Dict, stage: str, device: torch.device) -> Dict[str, torch.Tensor]:
    target = batch["target"].to(device)
    hard_mask = batch["hard_mask"].to(device)
    loc_map = batch["loc_map"].to(device)
    boundary = batch["boundary"].to(device)
    cls_label = batch["cls_label"].to(device)
    weights = batch["weight_map"].to(device) if cfg["use_uncertainty_weight"] and stage == "stage2" else None
    cls_bce = nn.BCEWithLogitsLoss()
    loss_seg = seg_loss(out["seg_logits"], target, weights=weights)
    loss_loc = seg_loss(out["loc_logits"], loc_map) if stage == "stage2" and "loc_logits" in out else torch.tensor(0.0, device=device)
    loss_cls = cls_bce(out["cls_logits"], cls_label) if stage == "stage2" else torch.tensor(0.0, device=device)
    loss_boundary = seg_loss(out["boundary_logits"], boundary, weights=weights) if stage == "stage2" and "boundary_logits" in out else torch.tensor(0.0, device=device)
    loss_anatomy = anatomy_constraint_loss(out["seg_logits"], out["loc_logits"], loc_map, hard_mask) if stage == "stage2" and "loc_logits" in out else torch.tensor(0.0, device=device)
    loss = (
        loss_seg
        + cfg["loc_loss_weight"] * loss_loc
        + cfg["cls_loss_weight"] * loss_cls
        + cfg["boundary_loss_weight"] * loss_boundary
        + cfg["anatomy_loss_weight"] * loss_anatomy
    )
    return {"loss": loss, "seg": loss_seg, "loc": loss_loc, "cls": loss_cls, "boundary": loss_boundary, "anatomy": loss_anatomy}


def run_epoch(model, loader, optimizer, device, cfg, stage: str, is_train: bool):
    model.train(is_train)
    total = {"loss": 0.0, "seg": 0.0, "loc": 0.0, "cls": 0.0, "boundary": 0.0, "anatomy": 0.0, "dice": 0.0, "iou": 0.0}
    n = 0
    for batch in loader:
        image = batch["image"].to(device)
        hard_mask = batch["hard_mask"].to(device)
        out = model(image)
        losses = build_loss(out, batch, cfg, stage, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        m = hard_metrics(out["seg_logits"].detach(), hard_mask.detach(), cfg["threshold"])
        bs = image.size(0)
        for k in ["loss", "seg", "loc", "cls", "boundary", "anatomy"]:
            total[k] += float(losses[k].item()) * bs
        total["dice"] += m["dice"] * bs
        total["iou"] += m["iou"] * bs
        n += bs
    return {k: v / max(1, n) for k, v in total.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_stage1.json")
    parser.add_argument("--stage", type=str, choices=["stage1", "stage2"], default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Stage1 best checkpoint for stage2 fine-tuning.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--selection-split", type=str, choices=["val", "test"], default=None, help="Split used for saving best checkpoint. Use val for publication-grade training; test is only for internal model screening.")
    args = parser.parse_args()

    cfg = copy.deepcopy(load_config(args.config))
    for key, value in [("epochs", args.epochs), ("batch_size", args.batch_size), ("learning_rate", args.learning_rate), ("threshold", args.threshold)]:
        if value is not None:
            cfg[key] = value
    if args.selection_split is not None:
        cfg["selection_split"] = args.selection_split
    stage = args.stage or cfg.get("stage", "stage1")
    run_name = args.run_name or cfg["run_name"]
    cfg["run_name"] = run_name
    selection_split = cfg.get("selection_split", "val")
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = tuple(cfg["image_size"])
    train_ds = APMultiTaskDataset(cfg["metadata_csv"], "train", image_size, cfg["target_mode"], cfg["use_uncertainty_weight"], cfg["downweight_two_annotator_cases"], augment=True)
    val_ds = APMultiTaskDataset(cfg["metadata_csv"], "val", image_size, cfg["target_mode"], cfg["use_uncertainty_weight"], cfg["downweight_two_annotator_cases"], augment=False)
    test_ds = APMultiTaskDataset(cfg["metadata_csv"], "test", image_size, cfg["target_mode"], cfg["use_uncertainty_weight"], cfg["downweight_two_annotator_cases"], augment=False) if selection_split == "test" else None
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=True) if test_ds is not None else None
    model = ApexTransNet(
        in_channels=1,
        encoder_pretrained_path=cfg.get("encoder_pretrained_path"),
        transformer_layers=cfg.get("transformer_layers", 4),
        use_boundary_head=cfg.get("use_boundary_head", True),
        use_location_head=cfg.get("use_location_head", True),
    ).to(device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        print(f"Loaded resume checkpoint: {args.resume}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    output_dir = Path(cfg["output_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{run_name}_resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    best_dice = -1.0
    best_path = checkpoint_dir / f"best_{run_name}.pt"
    last_path = checkpoint_dir / f"last_{run_name}.pt"
    for epoch in range(1, cfg["epochs"] + 1):
        tr = run_epoch(model, train_loader, optimizer, device, cfg, stage, True)
        va = run_epoch(model, val_loader, None, device, cfg, stage, False)
        te = run_epoch(model, test_loader, None, device, cfg, stage, False) if test_loader is not None else None
        selected = te if selection_split == "test" else va
        scheduler.step()
        message = f"[{stage} epoch {epoch:03d}] train loss={tr['loss']:.4f} dice={tr['dice']:.4f} | val loss={va['loss']:.4f} dice={va['dice']:.4f} iou={va['iou']:.4f}"
        if te is not None:
            message += f" | test loss={te['loss']:.4f} dice={te['dice']:.4f} iou={te['iou']:.4f}"
        print(message)
        torch.save({"epoch": epoch, "model": model.state_dict(), "config": cfg, "best_dice": best_dice, "selection_split": selection_split}, last_path)
        if selected["dice"] > best_dice:
            best_dice = selected["dice"]
            torch.save({"epoch": epoch, "model": model.state_dict(), "config": cfg, "best_dice": best_dice, "selection_split": selection_split}, best_path)
            print(f"  -> saved {best_path.name} with {selection_split} dice {best_dice:.4f}")


if __name__ == "__main__":
    main()
