import os
import glob
import argparse

import rasterio
from rasterio.windows import Window
from tqdm import tqdm


def collect_pairs(imgs_dir: str, anno_dir: str) -> list:
    pairs = []
    for img_path in sorted(glob.glob(os.path.join(imgs_dir, "*.tif"))):
        stem, _ = os.path.splitext(os.path.basename(img_path))
        anno_path = os.path.join(anno_dir, f"{stem}.tif")
        if not os.path.isfile(anno_path):
            print(f"[Skip] No matching annotation: {anno_path}")
            continue
        pairs.append((img_path, anno_path, stem))
    return pairs


def build_windows(height: int, width: int, patch_size: int, stride: int) -> list:
    if height < patch_size or width < patch_size:
        return []
    return [
        Window(col_off=c, row_off=r, width=patch_size, height=patch_size)
        for r in range(0, height - patch_size + 1, stride)
        for c in range(0, width - patch_size + 1, stride)
    ]


def main(args: argparse.Namespace) -> None:
    imgs_dir = os.path.join(args.data_dir, "images")
    anno_dir = os.path.join(args.data_dir, "anno")
    out_images = os.path.join(args.output_dir, "images")
    out_labels = os.path.join(args.output_dir, "label")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    pairs = collect_pairs(imgs_dir, anno_dir)
    if not pairs:
        print("No valid image/annotation pairs found.")
        return

    print(f"Images dir : {imgs_dir}")
    print(f"Anno dir   : {anno_dir}")
    print(f"Valid pairs: {len(pairs)} | patch_size={args.patch_size}, "
          f"stride={args.stride}, limit={args.limit or 'unlimited'}")

    saved = 0
    for img_path, anno_path, stem in pairs:
        if args.limit is not None and saved >= args.limit:
            break

        with rasterio.open(img_path) as src_img, rasterio.open(anno_path) as src_anno:
            hi, wi = src_img.height, src_img.width
            ha, wa = src_anno.height, src_anno.width
            if hi != ha or wi != wa:
                print(f"[Skip] Size mismatch for {stem}: img {wi}x{hi} vs anno {wa}x{ha}")
                continue

            meta_img = src_img.meta.copy()
            meta_anno = src_anno.meta.copy()
            windows = build_windows(hi, wi, args.patch_size, args.stride)
            if not windows:
                print(f"[Skip] Image smaller than patch size: {stem} ({wi}x{hi})")
                continue

            for idx, window in enumerate(tqdm(windows, desc=stem), start=1):
                if args.limit is not None and saved >= args.limit:
                    break

                img_patch = src_img.read(window=window)
                anno_patch = src_anno.read(window=window)
                out_name = f"{stem}_{idx:04d}.tif"
                out_img_fp = os.path.join(out_images, out_name)
                out_lbl_fp = os.path.join(out_labels, out_name)

                p_meta_img = meta_img.copy()
                p_meta_img.update({
                    "width": args.patch_size,
                    "height": args.patch_size,
                    "count": img_patch.shape[0],
                    "transform": src_img.window_transform(window),
                })
                p_meta_anno = meta_anno.copy()
                p_meta_anno.update({
                    "width": args.patch_size,
                    "height": args.patch_size,
                    "count": anno_patch.shape[0],
                    "transform": src_anno.window_transform(window),
                })

                with rasterio.open(out_img_fp, "w", **p_meta_img) as dst:
                    dst.write(img_patch)
                with rasterio.open(out_lbl_fp, "w", **p_meta_anno) as dst:
                    dst.write(anno_patch)
                saved += 1

    print(f"\nDone. Saved: {saved} patch pair(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop paired GeoTIFF images and annotations into fixed-size patches.",
    )
    parser.add_argument("--data_dir", type=str, default="data/bal_dc_benchmark")
    parser.add_argument("--output_dir", type=str, default="data/bal_dc_benchmark/training_data")
    parser.add_argument("--patch_size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(args)
