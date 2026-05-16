import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rows(csv_path: Path, split: str) -> List[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if row["split"] == split]


def read_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def read_binary(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0).astype(np.uint8)


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    w, h = size
    return np.asarray(Image.fromarray((image * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0


def resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    w, h = size
    return (np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.NEAREST), dtype=np.uint8) > 0).astype(np.uint8)


def resize_float_map(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    w, h = size
    return np.asarray(Image.fromarray((arr * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0


def max_filter(mask: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1:
        return mask
    pad = kernel // 2
    padded = np.pad(mask, ((pad, pad), (pad, pad)), mode="constant")
    out = np.zeros_like(mask)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            out[y, x] = padded[y:y + kernel, x:x + kernel].max()
    return out


def min_filter(mask: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1:
        return mask
    pad = kernel // 2
    padded = np.pad(mask, ((pad, pad), (pad, pad)), mode="constant", constant_values=1)
    out = np.zeros_like(mask)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            out[y, x] = padded[y:y + kernel, x:x + kernel].min()
    return out


def compute_boundary(mask: np.ndarray, width: int = 2) -> np.ndarray:
    return ((max_filter(mask, 2 * width + 1) - min_filter(mask, 2 * width + 1)) > 0).astype(np.uint8)


def random_aug(image: np.ndarray, hard_mask: np.ndarray, loc_map: np.ndarray, soft_target: np.ndarray, weight_map: np.ndarray):
    if random.random() < 0.5:
        image = np.fliplr(image).copy()
        hard_mask = np.fliplr(hard_mask).copy()
        loc_map = np.fliplr(loc_map).copy()
        soft_target = np.fliplr(soft_target).copy()
        weight_map = np.fliplr(weight_map).copy()
    if random.random() < 0.35:
        image = np.clip(image * random.uniform(0.92, 1.08) + random.uniform(-0.05, 0.05), 0.0, 1.0)
    if random.random() < 0.2:
        image = np.clip(image, 1e-6, 1.0) ** random.uniform(0.92, 1.08)
    return image, hard_mask, loc_map, soft_target, weight_map


class APMultiTaskDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        split: str,
        image_size: Tuple[int, int],
        target_mode: str = "soft_vote",
        use_uncertainty_weight: bool = True,
        downweight_two_annotator_cases: bool = True,
        augment: bool = False,
    ):
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.rows = load_rows(self.csv_path, split)
        self.model_root = Path(__file__).resolve().parents[1]
        self.processed_root = self.csv_path.parent.parent if self.csv_path.parent.name == "metadata" else self.csv_path.parent
        self.image_size = image_size
        self.target_mode = target_mode
        self.use_uncertainty_weight = use_uncertainty_weight
        self.downweight_two_annotator_cases = downweight_two_annotator_cases
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def resolve_path(self, raw_path: str, kind: str) -> Path:
        path = Path(raw_path)
        if path.exists():
            return path
        name = path.name
        candidates = {
            "image": [self.processed_root / "images" / name],
            "mask_smy": [self.processed_root / "masks_smy" / name],
            "mask_vote2": [self.processed_root / "masks_vote2" / name],
            "soft_target": [
                self.model_root / "cache" / "soft_targets_vote" / name,
                self.processed_root / "soft_targets_vote" / name,
            ],
            "uncertainty": [
                self.model_root / "cache" / "uncertainty_vote" / name,
                self.processed_root / "uncertainty_vote" / name,
            ],
            "tooth_root": [
                self.model_root / "cache" / "tooth_root_maps" / name,
                self.processed_root / "tooth_root_maps" / name,
            ],
        }.get(kind, [])
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return path

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        image = resize_image(read_gray(self.resolve_path(row["image_path_resolved"], "image")), self.image_size)
        hard_mask = resize_mask(read_binary(self.resolve_path(row["mask_vote2_path"], "mask_vote2")), self.image_size)
        soft_target = resize_float_map(read_gray(self.resolve_path(row["soft_target_path"], "soft_target")), self.image_size)
        loc_map = resize_mask(read_binary(self.resolve_path(row["tooth_root_map_path"], "tooth_root")), self.image_size)
        weight_map = resize_float_map(read_gray(self.resolve_path(row["uncertainty_path"], "uncertainty")), self.image_size)

        if self.target_mode == "smy":
            target = resize_mask(read_binary(self.resolve_path(row["mask_smy_path"], "mask_smy")), self.image_size).astype(np.float32)
        elif self.target_mode == "hard_vote2":
            target = hard_mask.astype(np.float32)
        else:
            target = soft_target.astype(np.float32)

        if not self.use_uncertainty_weight:
            weight_map = np.ones_like(weight_map, dtype=np.float32)
        sample_weight = float(row.get("sample_weight", 1.0))
        if not self.downweight_two_annotator_cases:
            sample_weight = 1.0
        weight_map = np.clip(weight_map * sample_weight, 0.0, 1.0)

        if self.augment:
            image, hard_mask, loc_map, target, weight_map = random_aug(image, hard_mask, loc_map, target, weight_map)

        boundary = compute_boundary(hard_mask)
        image = (image - float(image.mean())) / float(image.std() + 1e-6)
        cls_label = 1.0 if row["label_group"] == "ap" else 0.0
        return {
            "image": torch.from_numpy(image).unsqueeze(0).float(),
            "target": torch.from_numpy(target).unsqueeze(0).float(),
            "hard_mask": torch.from_numpy(hard_mask).unsqueeze(0).float(),
            "loc_map": torch.from_numpy(loc_map).unsqueeze(0).float(),
            "boundary": torch.from_numpy(boundary).unsqueeze(0).float(),
            "weight_map": torch.from_numpy(weight_map).unsqueeze(0).float(),
            "cls_label": torch.tensor([cls_label], dtype=torch.float32),
            "meta": {
                "case_id": row["case_id"],
                "label_group": row["label_group"],
                "consensus_type": row["consensus_type"],
            },
        }
