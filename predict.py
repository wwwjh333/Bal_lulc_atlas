from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from rasterio.windows import Window
from tqdm import tqdm

from label_config import NUM_CLASSES

BANDS = (1, 2, 3, 4)
CLASS_TO_COLOR = {
    0: (255, 0, 0),        # Building
    1: (133, 133, 133),    # Impervious Surface
    2: (34, 139, 34),      # Tree Canopy
    3: (128, 236, 104),    # Herbaceous Vegetation
    4: (0, 0, 255),        # Open Water
    5: (128, 0, 0),        # Bare Surface
}


def _get_sam_registry(net_type: str):
    if net_type != "sam":
        raise ValueError(f"Unknown net_type={net_type!r}; expected 'sam'.")
    from models.sam.build_sam import sam_model_registry
    return sam_model_registry


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    registry = _get_sam_registry(args.net_type)
    return registry[args.encoder_type](args).to(device)


def _load_checkpoint(path: str, map_location) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return {"state_dict": ckpt["model_state_dict"]}
    return {"state_dict": ckpt}


def make_grid(full_len: int, patch_size: int, stride: int) -> list[int]:
    if full_len <= patch_size:
        return [0]
    coords = list(range(0, full_len - patch_size + 1, stride))
    last = full_len - patch_size
    if coords[-1] != last:
        coords.append(last)
    return coords


def make_tent_weight(patch_size: int) -> np.ndarray:
    c = (patch_size - 1) / 2.0
    x = np.arange(patch_size, dtype=np.float32)
    w1 = 1.0 - np.abs(x - c) / (c + 1e-6)
    return np.clip(np.outer(w1, w1), 1e-3, 1.0).astype(np.float32)


def iter_tiff_files(input_path: str):
    exts = {".tif", ".tiff"}
    if os.path.isfile(input_path):
        if os.path.splitext(input_path)[1].lower() in exts:
            yield input_path
        return
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")
    for dirpath, _, filenames in os.walk(input_path):
        for fn in sorted(filenames):
            if os.path.splitext(fn)[1].lower() in exts:
                yield os.path.join(dirpath, fn)


def output_path_for(input_path: str, input_root: str, output_dir: str) -> str:
    input_root = os.path.abspath(input_root)
    rel = os.path.relpath(os.path.abspath(input_path), input_root)
    if rel.startswith(".."):
        rel = os.path.basename(input_path)
    out = Path(output_dir) / rel
    if out.suffix.lower() not in {".tif", ".tiff"}:
        out = out.with_suffix(".tif")
    return str(out)


@torch.no_grad()
def predict_tiff(
    model: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    image_path: str,
    output_path: str,
) -> None:
    patch_size = args.patch_size
    stride = args.stride
    num_classes = int(args.num_classes)
    weight = make_tent_weight(patch_size)

    with rasterio.open(image_path) as src:
        full_w, full_h = src.width, src.height
        transform, crs = src.transform, src.crs
        lefts = make_grid(full_w, patch_size, stride)
        tops = make_grid(full_h, patch_size, stride)

        acc = np.zeros((full_h, full_w, num_classes), dtype=np.float32)
        acc_w = np.zeros((full_h, full_w), dtype=np.float32)

        for top, left in tqdm(
            ((t, l) for t in tops for l in lefts),
            total=len(tops) * len(lefts),
            desc=os.path.basename(image_path),
            unit="patch",
        ):
            arr = src.read(BANDS, window=Window(left, top, patch_size, patch_size))
            arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            x = x.to(device=device, dtype=torch.float32) / 255.0
            if x.shape[-2:] != (args.image_size, args.image_size):
                x = F.interpolate(
                    x, size=(args.image_size, args.image_size),
                    mode="bilinear", align_corners=False,
                )

            logits = model(x)
            if logits.shape[-2:] != (patch_size, patch_size):
                logits = F.interpolate(
                    logits, size=(patch_size, patch_size),
                    mode="bilinear", align_corners=False,
                )

            logits_hwc = logits.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
            acc[top:top + patch_size, left:left + patch_size] += logits_hwc * weight[..., None]
            acc_w[top:top + patch_size, left:left + patch_size] += weight

    if np.any(acc_w <= 0):
        raise RuntimeError(f"Uncovered pixels in {image_path}")

    merged = np.argmax(acc / acc_w[..., None], axis=-1).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=merged.shape[0],
        width=merged.shape[1],
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(merged, 1)
        if num_classes == len(CLASS_TO_COLOR):
            dst.write_colormap(1, CLASS_TO_COLOR)

    print(f"Saved {output_path} ({merged.shape[0]}x{merged.shape[1]})")


def main(args: argparse.Namespace) -> None:
    device = torch.device(f"cuda:{args.gpu_device}" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"checkpoint not found: {args.weights}")

    ckpt = _load_checkpoint(args.weights, map_location=device)
    model = build_model(args, device)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    print(
        f"Loaded {args.weights} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    input_root = args.input if os.path.isdir(args.input) else os.path.dirname(args.input) or "."
    inputs = list(iter_tiff_files(args.input))
    if not inputs:
        raise RuntimeError(f"No .tif/.tiff files found under: {args.input}")

    for in_path in inputs:
        out_path = output_path_for(in_path, input_root, args.output_dir)
        predict_tiff(model, device, args, in_path, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Large GeoTIFF land-cover prediction")
    parser.add_argument("-net_type", type=str, default="sam", choices=["sam"])
    parser.add_argument("-model_type", type=str, default="seg", choices=["seg"])
    parser.add_argument(
        "-encoder_type", type=str, default="default",
        choices=["default", "vit_b", "vit_l", "vit_h"],
    )
    parser.add_argument(
        "-block_type", type=str, default="default",
        choices=["default", "adapter", "fwa", "lora", "adalora"],
    )
    parser.add_argument("-num_classes", type=int, default=NUM_CLASSES)
    parser.add_argument("-image_size", type=int, default=1024)
    parser.add_argument("-out_size", type=int, default=1024)
    parser.add_argument(
        "-sam_ckpt",
        type=str,
        default="pretrain_weights/sam/sam_vit_b_01ec64.pth",
    )
    parser.add_argument(
        "-weights",
        type=str,
        default="",
    )
    parser.add_argument("-input", type=str, default="data/bal_dc_benchmark/images")
    parser.add_argument("-output_dir", type=str, default="data/bal_dc_benchmark/predictions")
    parser.add_argument("-patch_size", type=int, default=1024)
    parser.add_argument("-stride", type=int, default=512)
    parser.add_argument("-gpu_device", type=int, default=0)

    args = parser.parse_args()
    main(args)
