from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def morphology_close(mask: np.ndarray, kernel: int = 9) -> np.ndarray:
    img = Image.fromarray((mask * 255).astype(np.uint8))
    img = img.filter(ImageFilter.MaxFilter(kernel))
    img = img.filter(ImageFilter.MinFilter(kernel))
    return (np.asarray(img, dtype=np.uint8) > 0).astype(np.uint8)


def build_tooth_root_map(image: np.ndarray) -> np.ndarray:
    nonzero = image[image > 0.02]
    if nonzero.size == 0:
        return np.zeros_like(image, dtype=np.uint8)
    base_thr = max(0.18, float(np.percentile(nonzero, 56)))
    coarse = (image >= base_thr).astype(np.uint8)
    coarse = morphology_close(coarse, kernel=9)

    h, w = coarse.shape
    yy = np.linspace(0.0, 1.0, h).reshape(h, 1)
    root_bias = (yy >= 0.35).astype(np.uint8)
    coarse = coarse * root_bias
    coarse = morphology_close(coarse, kernel=11)
    return coarse.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = load_rows(Path(cfg["metadata_csv"]))
    out_dir = Path(cfg["tooth_root_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        img = np.asarray(Image.open(Path(row["image_path_resolved"])).convert("L"), dtype=np.float32) / 255.0
        m = build_tooth_root_map(img)
        Image.fromarray((m * 255).astype(np.uint8)).save(out_dir / f"{row['case_id']}.png")

    print(f"Generated tooth/root maps into {out_dir}")


if __name__ == "__main__":
    main()
