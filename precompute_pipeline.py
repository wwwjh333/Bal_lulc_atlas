import os
import glob
import json
import math
import argparse
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window
from tqdm import tqdm


SOBEL_X = torch.tensor(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]],
    dtype=torch.float32,
).view(1, 1, 3, 3)

SOBEL_Y = torch.tensor(
    [[-1, -2, -1],
     [ 0,  0,  0],
     [ 1,  2,  1]],
    dtype=torch.float32,
).view(1, 1, 3, 3)


@torch.no_grad()
def compute_complexity_fast(
    img_t: torch.Tensor,
    downsample: int = 4,
    std_clip: float = 2.5,
) -> float:
    I = img_t.mean(dim=0, keepdim=True)
    if downsample > 1:
        I = F.avg_pool2d(I.unsqueeze(0), downsample, downsample).squeeze(0)

    I4 = I.unsqueeze(0)
    sx = SOBEL_X.to(I4.device, I4.dtype)
    sy = SOBEL_Y.to(I4.device, I4.dtype)
    gx = F.conv2d(I4, sx, padding=1)
    gy = F.conv2d(I4, sy, padding=1)
    g  = torch.sqrt(gx * gx + gy * gy + 1e-12).flatten()

    m = g.mean()
    s = g.std(unbiased=False) + 1e-12
    g = torch.clamp(g, 0.0, m + std_clip * s)

    return float(g.mean().item())


def normalize_c(c_raw: float, c10: float, c90: float) -> float:
    if c90 <= c10 + 1e-12:
        return float(1.0 - math.exp(-5.0 * max(0.0, c_raw)))
    return float(max(0.0, min(1.0, (c_raw - c10) / (c90 - c10))))


def normalize_c_map(C: np.ndarray, c10: float, c90: float) -> np.ndarray:
    x = np.asarray(C, dtype=np.float64)
    if c90 <= c10 + 1e-12:
        out = 1.0 - np.exp(-5.0 * np.maximum(0.0, x))
    else:
        out = (x - c10) / (c90 - c10)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def spectral_difference_map(
    img_t: torch.Tensor,
    model_patch_size: int,
    rgb_indices: Sequence[int] = (0, 1, 2),
    nir_index: int = 3,
) -> torch.Tensor:
    rgb  = torch.stack([img_t[i] for i in rgb_indices], dim=0).mean(dim=0)
    nir  = img_t[nir_index]
    diff = (nir - rgb).abs().unsqueeze(0).unsqueeze(0)
    out  = F.avg_pool2d(diff, kernel_size=model_patch_size, stride=model_patch_size)
    return out.squeeze(0).squeeze(0)


def spatial_complexity_map(
    img_t: torch.Tensor,
    model_patch_size: int,
) -> torch.Tensor:
    I  = img_t.mean(dim=0, keepdim=True).unsqueeze(0)
    sx = SOBEL_X.to(I.device, I.dtype)
    sy = SOBEL_Y.to(I.device, I.dtype)
    gx = F.conv2d(I, sx, padding=1)
    gy = F.conv2d(I, sy, padding=1)
    g  = torch.sqrt(gx * gx + gy * gy + 1e-12)
    out = F.avg_pool2d(g, kernel_size=model_patch_size, stride=model_patch_size)
    return out.squeeze(0).squeeze(0)


def collect_files(data_dir: str, recursive: bool) -> list:
    files = []
    for ext in ("*.tif", "*.tiff"):
        pattern = (
            os.path.join(data_dir, "**", ext)
            if recursive
            else os.path.join(data_dir, ext)
        )
        files.extend(glob.glob(pattern, recursive=recursive))
    return sorted(files)


def read_tiff(path: str, window_size: int) -> np.ndarray:
    with rasterio.open(path) as src:
        if window_size > 0:
            H, W = src.height, src.width
            ws   = window_size
            win  = Window(
                max(0, W // 2 - ws // 2),
                max(0, H // 2 - ws // 2),
                min(ws, W),
                min(ws, H),
            )
            return src.read(window=win)
        return src.read()


def step1_compute_stats(files: list, args: argparse.Namespace) -> tuple:
    print("\n[Step 1] Computing raw complexity scores ...")
    c_raw_list = []
    files_ok   = []

    for path in tqdm(files, desc="Step 1"):
        try:
            img   = read_tiff(path, args.window_size)
            img_t = torch.from_numpy(img).float() / 255.0
            c_raw = compute_complexity_fast(img_t, args.downsample, args.std_clip)
            c_raw_list.append(c_raw)
            files_ok.append(path)
        except Exception as e:
            if args.verbose:
                print(f"  [WARN] {path}: {e}")

    arr  = np.array(c_raw_list, dtype=np.float32)
    c10  = float(np.percentile(arr, 10))
    c90  = float(np.percentile(arr, 90))
    mean = float(np.mean(arr))
    std  = float(np.std(arr))

    print(f"  Files OK : {len(files_ok)} / {len(files)}")
    print(f"  c10={c10:.6f}  c90={c90:.6f}  mean={mean:.6f}  std={std:.6f}")

    stats = {
        "num_files": len(files_ok),
        "downsample": args.downsample,
        "std_clip":   args.std_clip,
        "c10": c10, "c90": c90, "mean": mean, "std": std,
    }
    stats_path = os.path.join(args.out_dir, "complexity_stats.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved -> {stats_path}")

    return c_raw_list, files_ok, c10, c90


def step2_normalize_and_save(
    files_ok: list,
    c_raw_list: list,
    c10: float,
    c90: float,
    args: argparse.Namespace,
) -> dict:
    print("\n[Step 2] Normalizing and saving complexity records ...")

    if args.save_maps:
        if not args.maps_out_dir:
            raise ValueError("--save_maps requires --maps_out_dir")
        os.makedirs(args.maps_out_dir, exist_ok=True)

    results = {}
    rgb_idx = tuple(args.rgb_indices)
    nir_i   = int(args.nir_index)

    for path, c_raw in tqdm(zip(files_ok, c_raw_list), total=len(files_ok), desc="Step 2"):
        try:
            key   = os.path.relpath(path, args.data_dir)
            c_hat = normalize_c(c_raw, c10, c90)
            rec   = {"c_raw": c_raw, "c_hat": c_hat}

            if args.save_maps:
                img   = read_tiff(path, window_size=0)
                img_t = torch.from_numpy(img).float() / 255.0

                D       = spectral_difference_map(img_t, args.model_patch_size, rgb_idx, nir_i)
                C       = spatial_complexity_map(img_t, args.model_patch_size)
                D_np    = D.cpu().numpy().astype(np.float32)
                C_np    = C.cpu().numpy().astype(np.float32)
                C_hat_np = normalize_c_map(C_np, c10, c90)

                maps_rel = os.path.splitext(key)[0] + ".npz"
                npz_path = os.path.join(args.maps_out_dir, maps_rel)
                os.makedirs(os.path.dirname(npz_path), exist_ok=True)
                np.savez_compressed(
                    npz_path,
                    D=D_np, C=C_np, C_hat=C_hat_np,
                    H_img=np.int32(img_t.shape[-2]),
                    W_img=np.int32(img_t.shape[-1]),
                    model_patch_size=np.int32(args.model_patch_size),
                )
                rec["maps_npz"] = maps_rel.replace("\\", "/")

            results[key] = rec

        except Exception as e:
            if args.verbose:
                print(f"  [WARN] {path}: {e}")

    complexity_path = os.path.join(args.out_dir, "complexity.json")
    with open(complexity_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved -> {complexity_path}")
    if args.save_maps:
        print(f"  Maps  -> {args.maps_out_dir}")

    return results


def step3_compute_tau_d(results: dict, args: argparse.Namespace) -> None:
    if not args.save_maps:
        print("\n[Step 3] Skipped (--save_maps not set; no D maps available).")
        return

    print("\n[Step 3] Computing tau_d statistics from D maps ...")

    all_d  = []
    failed = 0

    for key, rec in tqdm(results.items(), desc="Step 3"):
        maps_npz = rec.get("maps_npz")
        if not maps_npz:
            continue
        npz_path = os.path.normpath(
            os.path.join(args.maps_out_dir, maps_npz.replace("/", os.sep))
        )
        if not os.path.isfile(npz_path):
            failed += 1
            continue
        with np.load(npz_path, mmap_mode="r") as z:
            all_d.append(z["D"].flatten().copy())

    if not all_d:
        print("  [WARN] No D maps found — skipping tau_d computation.")
        return

    D    = np.concatenate(all_d)
    pcts = [10, 25, 50, 75, 90]
    pvals = np.percentile(D, pcts)

    print(f"\n  Loaded {len(all_d)} images ({D.size:,} tokens total), failed={failed}")
    print(f"  mean : {D.mean():.6f}")
    print(f"  std  : {D.std():.6f}")
    print(f"  min  : {D.min():.6f}")
    for p, v in zip(pcts, pvals):
        marker = "  <-- recommended tau_d" if p == 50 else ""
        print(f"  p{p:<3d} : {v:.6f}{marker}")
    print(f"  max  : {D.max():.6f}")

    tau_d = float(np.percentile(D, 50))
    print(f"\n  Suggested: args.tau_d = {tau_d:.6f}")

    tau_d_stats = {
        "num_images": len(all_d),
        "num_tokens": int(D.size),
        "mean": float(D.mean()), "std": float(D.std()),
        "min":  float(D.min()),  "max": float(D.max()),
        **{f"p{p}": float(v) for p, v in zip(pcts, pvals)},
        "recommended_tau_d": tau_d,
    }
    tau_path = os.path.join(args.out_dir, "tau_d_stats.json")
    with open(tau_path, "w") as f:
        json.dump(tau_d_stats, f, indent=2)
    print(f"  Saved -> {tau_path}")


def main(args: argparse.Namespace) -> None:
    files = collect_files(args.data_dir, args.recursive)
    if not files:
        print(f"No TIFF files found in: {args.data_dir}")
        return
    print(f"Found {len(files)} TIFF files in {args.data_dir}")

    c_raw_list, files_ok, c10, c90 = step1_compute_stats(files, args)
    results = step2_normalize_and_save(files_ok, c_raw_list, c10, c90, args)
    step3_compute_tau_d(results, args)

    print("\n[Done] All steps complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end complexity pre-computation pipeline for TIFF datasets."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/pretrain_bal_dc/images",
        help="Root directory containing TIFF files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/pretrain_bal_dc",
        help="Output directory for complexity_stats.json, complexity.json, tau_d_stats.json.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for TIFFs under data_dir.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=0,
        help="For Step 1: read only a center crop of this size (0 = full image).",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=4,
        help="Spatial downsampling factor before Sobel. (default: 4)",
    )
    parser.add_argument(
        "--std_clip",
        type=float,
        default=2.5,
        help="Gradient clipping in units of std above mean. (default: 2.5)",
    )
    parser.add_argument(
        "--save_maps",
        action="store_true",
        help="Save per-image D / C / C_hat token maps as *.npz (required for Step 3).",
    )
    parser.add_argument(
        "--maps_out_dir",
        type=str,
        default="data/pretrain_bal_dc/maps",
        help="Root directory for *.npz token maps; required when --save_maps is set.",
    )
    parser.add_argument(
        "--model_patch_size",
        type=int,
        default=16,
        help="ViT patch size used for token map pooling; must match training. (default: 16)",
    )
    parser.add_argument(
        "--rgb_indices",
        type=int,
        nargs=3,
        default=[0, 1, 2],
        help="Band indices for R, G, B channels. (default: 0 1 2)",
    )
    parser.add_argument(
        "--nir_index",
        type=int,
        default=3,
        help="Band index for NIR channel. (default: 3)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print warnings for failed files.",
    )

    args = parser.parse_args()
    main(args)
