import glob
import json
import math
import os
from typing import Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


def _normalize_c_map(C: np.ndarray, c10: float, c90: float) -> np.ndarray:
    x = np.asarray(C, dtype=np.float64)
    if c90 <= c10 + 1e-12:
        out = 1.0 - np.exp(-5.0 * np.maximum(0.0, x))
    else:
        out = (x - c10) / (c90 - c10)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


class _AdaptiveMultiScaleSimMIMMaskGenerator:
    def __init__(
        self,
        model_patch_size: int,
        mask_patch_sizes=(32, 64, 128),
        mask_ratios=(0.60, 0.50, 0.38),
        pmax_128=0.45,
        c0=0.33,
        T=0.06,
        q64=0.45,
    ):
        self.model_patch_size = int(model_patch_size)
        self.mask_patch_sizes = [int(x) for x in mask_patch_sizes]
        assert len(self.mask_patch_sizes) == 3
        assert len(mask_ratios) == 3
        self.mask_ratio_map = {int(s): float(r) for s, r in zip(self.mask_patch_sizes, mask_ratios)}
        for mps in self.mask_patch_sizes:
            assert mps % self.model_patch_size == 0, (
                f"mask_patch_size {mps} must be multiple of model_patch_size {self.model_patch_size}"
            )
        self.pmax_128 = float(pmax_128)
        self.c0 = float(c0)
        self.T = float(T)
        self.q64 = float(q64)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _probs_from_complexity(self, c_hat: float) -> np.ndarray:
        c_hat = float(max(0.0, min(1.0, c_hat)))
        z = (self.c0 - c_hat) / max(self.T, 1e-6)
        p128 = self.pmax_128 * self._sigmoid(z)
        rem = max(0.0, 1.0 - p128)
        p64 = rem * self.q64
        p32 = rem - p64
        s = p32 + p64 + p128
        return np.array([p32 / s, p64 / s, p128 / s], dtype=np.float64)

    def _gen_for_scale(self, H_img: int, W_img: int, mask_patch_size: int, mask_ratio: float) -> np.ndarray:
        H_tok = H_img // self.model_patch_size
        W_tok = W_img // self.model_patch_size
        if H_tok <= 0 or W_tok <= 0:
            raise ValueError(f"Invalid token grid from image size {H_img}x{W_img}")
        k = max(1, mask_patch_size // self.model_patch_size)
        tok_min = min(H_tok, W_tok)
        if tok_min >= 2:
            k = min(k, max(1, tok_min // 2))
        rand_h = math.ceil(H_tok / k)
        rand_w = math.ceil(W_tok / k)
        token_count = rand_h * rand_w
        mask_count = int(np.ceil(token_count * float(mask_ratio)))
        coarse = np.zeros(token_count, dtype=np.int64)
        coarse[np.random.permutation(token_count)[:mask_count]] = 1
        coarse = coarse.reshape(rand_h, rand_w)
        return coarse.repeat(k, axis=0).repeat(k, axis=1)[:H_tok, :W_tok]


class BalDcSimMIM(Dataset):
    def __init__(
        self,
        args,
        data_path: str,
        model_patch_size: int = 16,
        extensions=("*.tif", "*.tiff"),
        recursive: bool = True,
    ):
        self.data_path = data_path
        self.model_patch_size = int(model_patch_size)

        image_list = []
        for ext in extensions:
            pattern = os.path.join(data_path, "**", ext) if recursive else os.path.join(data_path, ext)
            image_list.extend(glob.glob(pattern, recursive=recursive))
        self.image_list = sorted(image_list)
        print(f"[BalDcSimMIM] Found {len(self.image_list)} tiff files.")

        self.super_tile_tokens = int(getattr(args, "super_tile_tokens", 32))
        if not hasattr(args, "tau_d") or args.tau_d is None:
            raise ValueError("BalDcSimMIM requires args.tau_d (precomputed spectral-split threshold).")
        self.tau_d = float(args.tau_d)

        self.c10 = float(getattr(args, "c10", 0.17213906))
        self.c90 = float(getattr(args, "c90", 0.41866542))

        mrs = getattr(args, "mask_ratios", None)
        if mrs is not None:
            mrs = tuple(float(x) for x in mrs)
            if len(mrs) != 3:
                raise ValueError("mask_ratios must be a sequence of 3 floats (for 32/64/128 px scales)")
        else:
            mrs = (0.60, 0.50, 0.38)

        self._mask_gen = _AdaptiveMultiScaleSimMIMMaskGenerator(
            model_patch_size=model_patch_size,
            mask_patch_sizes=(32, 64, 128),
            mask_ratios=mrs,
            pmax_128=0.45,
            c0=0.33,
            T=0.06,
            q64=0.45,
        )

        hm = getattr(args, "heterogeneity_maps_dir", None)
        self.heterogeneity_maps_dir = os.path.abspath(hm) if hm else None
        if not self.heterogeneity_maps_dir or not args.complexity_json:
            raise ValueError(
                "BalDcSimMIM requires args.heterogeneity_maps_dir and args.complexity_json."
            )
        with open(args.complexity_json, "r") as f:
            self.complexity_map = json.load(f)

    def __len__(self):
        return len(self.image_list)

    def _heterogeneity_masks(self, D: np.ndarray, C_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if D.shape != C_hat.shape:
            raise ValueError("D and C_hat must have the same shape")
        gen = self._mask_gen
        H_tok, W_tok = D.shape
        tau_d = self.tau_d

        T = max(1, int(self.super_tile_tokens))
        mask_rgb = np.zeros((H_tok, W_tok), dtype=np.float32)
        mask_nir = np.zeros((H_tok, W_tok), dtype=np.float32)

        for ti in range(0, H_tok, T):
            ti_end = min(ti + T, H_tok)
            for sj in range(0, W_tok, T):
                sj_end = min(sj + T, W_tok)
                h_px = (ti_end - ti) * gen.model_patch_size
                w_px = (sj_end - sj) * gen.model_patch_size

                c_hat_tile = float(np.mean(C_hat[ti:ti_end, sj:sj_end]))
                probs = gen._probs_from_complexity(c_hat_tile)
                mps = int(np.random.choice(gen.mask_patch_sizes, p=probs))
                mr = gen.mask_ratio_map[mps]

                if float(np.mean(D[ti:ti_end, sj:sj_end])) >= tau_d:
                    mask_rgb[ti:ti_end, sj:sj_end] = gen._gen_for_scale(h_px, w_px, mps, mr).astype(
                        np.float32, copy=False
                    )
                    mask_nir[ti:ti_end, sj:sj_end] = gen._gen_for_scale(h_px, w_px, mps, mr).astype(
                        np.float32, copy=False
                    )
                else:
                    sub = gen._gen_for_scale(h_px, w_px, mps, mr).astype(np.float32, copy=False)
                    mask_rgb[ti:ti_end, sj:sj_end] = sub
                    mask_nir[ti:ti_end, sj:sj_end] = sub

        return mask_rgb, mask_nir

    def _load_d_c_hat(self, img_path: str, H_img: int, W_img: int) -> Tuple[np.ndarray, np.ndarray]:
        key = os.path.relpath(img_path, self.data_path)
        rec = self.complexity_map.get(key)
        if not isinstance(rec, dict):
            raise KeyError(f"No JSON entry for image key: {key}")
        maps_npz = rec.get("maps_npz")
        if not maps_npz:
            raise KeyError(f"maps_npz missing for key: {key}")
        npz_path = os.path.normpath(
            os.path.join(self.heterogeneity_maps_dir, maps_npz.replace("/", os.sep))
        )
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"npz not found: {npz_path}")
        with np.load(npz_path, mmap_mode="r") as z:
            D = np.asarray(z["D"], dtype=np.float32)
            if "C_hat" in z.files:
                C_hat = np.asarray(z["C_hat"], dtype=np.float32)
            elif "C" in z.files:
                C_hat = _normalize_c_map(np.asarray(z["C"], dtype=np.float32), self.c10, self.c90)
            else:
                raise KeyError(f"{npz_path} must contain C_hat or C")
            h0, w0, mps0 = int(z["H_img"]), int(z["W_img"]), int(z["model_patch_size"])
        if h0 != H_img or w0 != W_img or mps0 != self.model_patch_size or D.shape != C_hat.shape:
            raise ValueError(
                f"npz incompatible with {key}: ({h0},{w0},mps={mps0}) vs ({H_img},{W_img},{self.model_patch_size})"
            )
        return D, C_hat

    def __getitem__(self, index):
        img_path = self.image_list[index]
        with rasterio.open(img_path) as src:
            bands = src.read()

        if bands.size == 0:
            raise ValueError(f"{img_path}: empty raster")

        img_t = torch.from_numpy(bands).float() / 255.0
        H_img, W_img = img_t.shape[-2], img_t.shape[-1]

        D, C_hat = self._load_d_c_hat(img_path, H_img, W_img)
        mr, mn = self._heterogeneity_masks(D, C_hat)
        mask_rgb = torch.from_numpy(mr).unsqueeze(-1)
        mask_nir = torch.from_numpy(mn).unsqueeze(-1)

        stem = os.path.basename(img_path).rsplit(".", 1)[0]
        return img_t, {"rgb": mask_rgb, "nir": mask_nir}, {"filename_or_obj": stem}
