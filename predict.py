#!/usr/bin/env python3
"""Inference script for ApexTransNet.

Example:
    python predict.py \
        --input /path/to/radiograph_or_folder \
        --checkpoint checkpoints/best_apex_trans_stage2_512.pt \
        --output outputs/predictions \
        --threshold 0.55 \
        --save-prob \
        --save-overlay
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch

from models.model import ApexTransNet

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def list_images(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f'Unsupported image extension: {path.suffix}')
        return [path]
    if path.is_dir():
        images = sorted(p for p in path.rglob('*') if p.suffix.lower() in IMAGE_EXTS)
        if not images:
            raise ValueError(f'No images found under: {path}')
        return images
    raise FileNotFoundError(path)


def preprocess_image(path: Path, image_size: tuple[int, int]) -> tuple[torch.Tensor, Image.Image, np.ndarray]:
    image = Image.open(path).convert('L')
    original_array = np.asarray(image, dtype=np.uint8)
    resized = image.resize((image_size[1], image_size[0]), resample=Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    mean = float(array.mean())
    std = float(array.std())
    if std < 1e-6:
        std = 1.0
    array = (array - mean) / std
    tensor = torch.from_numpy(array[None, None, :, :]).float()
    return tensor, image, original_array


def load_model(checkpoint: Path, device: torch.device, transformer_layers: int) -> ApexTransNet:
    model = ApexTransNet(
        in_channels=1,
        encoder_pretrained_path=None,
        transformer_layers=transformer_layers,
        use_boundary_head=True,
        use_location_head=True,
    )
    ckpt = torch.load(checkpoint, map_location='cpu')
    state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
    state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = mask.astype(bool)
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            pixels = []
            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if not seen[ny, nx] and mask[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            comp = np.zeros_like(mask, dtype=bool)
            yy, xx = zip(*pixels)
            comp[np.array(yy), np.array(xx)] = True
            components.append(comp)
    return components


def filter_components(mask: np.ndarray, min_area: int = 0, top_k: int = 0) -> np.ndarray:
    if min_area <= 0 and top_k <= 0:
        return mask.astype(bool)
    components = connected_components(mask)
    if min_area > 0:
        components = [c for c in components if int(c.sum()) >= min_area]
    if top_k > 0:
        components = sorted(components, key=lambda c: int(c.sum()), reverse=True)[:top_k]
    output = np.zeros_like(mask, dtype=bool)
    for comp in components:
        output |= comp
    return output


def save_probability(prob: np.ndarray, original_size: tuple[int, int], path: Path) -> None:
    image = Image.fromarray((np.clip(prob, 0, 1) * 255).astype(np.uint8))
    image = image.resize(original_size, resample=Image.BILINEAR)
    image.save(path)


def save_mask(mask: np.ndarray, original_size: tuple[int, int], path: Path) -> np.ndarray:
    image = Image.fromarray((mask.astype(np.uint8) * 255))
    image = image.resize(original_size, resample=Image.NEAREST)
    image.save(path)
    return np.asarray(image) > 0


def save_overlay(original: np.ndarray, mask: np.ndarray, path: Path, alpha: float = 0.45) -> None:
    rgb = np.stack([original, original, original], axis=-1).astype(np.float32)
    color = np.array([238, 95, 145], dtype=np.float32)
    rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * color
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path)


def run_prediction(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device)
    model = load_model(Path(args.checkpoint), device, args.transformer_layers)
    image_size = (args.image_size, args.image_size)
    images = list_images(input_path)

    rows = []
    with torch.no_grad():
        for image_path in images:
            tensor, pil_image, original_array = preprocess_image(image_path, image_size)
            tensor = tensor.to(device)
            outputs = model(tensor)
            seg_prob = torch.sigmoid(outputs['seg_logits'])[0, 0].cpu().numpy()
            cls_prob = float(torch.sigmoid(outputs['cls_logits'])[0].cpu().item())

            mask = seg_prob >= args.threshold
            mask = filter_components(mask, min_area=args.min_area, top_k=args.top_k)

            stem = image_path.stem
            original_size = pil_image.size
            mask_path = output_dir / f'{stem}_mask.png'
            mask_original = save_mask(mask, original_size, mask_path)

            prob_path = ''
            if args.save_prob:
                prob_path = output_dir / f'{stem}_prob.png'
                save_probability(seg_prob, original_size, prob_path)

            overlay_path = ''
            if args.save_overlay:
                overlay_path = output_dir / f'{stem}_overlay.png'
                save_overlay(original_array, mask_original, overlay_path, alpha=args.overlay_alpha)

            rows.append({
                'image': str(image_path),
                'case_id': stem,
                'classification_probability': f'{cls_prob:.6f}',
                'classification_prediction': int(cls_prob >= args.cls_threshold),
                'mask_area_pixels': int(mask_original.sum()),
                'mask_area_fraction': f'{float(mask_original.mean()):.8f}',
                'segmentation_threshold': args.threshold,
                'classification_threshold': args.cls_threshold,
                'mask_path': str(mask_path),
                'probability_path': str(prob_path) if prob_path else '',
                'overlay_path': str(overlay_path) if overlay_path else '',
            })
            print(f'[{stem}] cls_prob={cls_prob:.4f}, mask_area={int(mask_original.sum())} px')

    csv_path = output_dir / 'predictions.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved prediction summary to {csv_path}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run ApexTransNet inference on panoramic radiographs.')
    parser.add_argument('--input', required=True, help='Path to a single image or a folder of images.')
    parser.add_argument('--checkpoint', default='checkpoints/best_apex_trans_stage2_512.pt', help='Path to model checkpoint.')
    parser.add_argument('--output', default='outputs/predictions', help='Directory for masks, overlays, and prediction CSV.')
    parser.add_argument('--image-size', type=int, default=512, help='Model input size. The released model uses 512.')
    parser.add_argument('--threshold', type=float, default=0.55, help='Segmentation probability threshold.')
    parser.add_argument('--cls-threshold', type=float, default=0.5, help='Case-level AP classification threshold.')
    parser.add_argument('--transformer-layers', type=int, default=4, help='Transformer bottleneck depth used by the checkpoint.')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='Inference device.')
    parser.add_argument('--min-area', type=int, default=0, help='Remove predicted connected components smaller than this area in resized mask space.')
    parser.add_argument('--top-k', type=int, default=0, help='Keep only the largest K predicted components in resized mask space. 0 keeps all.')
    parser.add_argument('--save-prob', action='store_true', help='Save probability heatmaps as 8-bit PNG files.')
    parser.add_argument('--save-overlay', action='store_true', help='Save colored mask overlays on the original image.')
    parser.add_argument('--overlay-alpha', type=float, default=0.45, help='Overlay opacity for predicted masks.')
    return parser.parse_args()


if __name__ == '__main__':
    run_prediction(parse_args())
