import os
import glob
import argparse

import rasterio
from rasterio.windows import Window
from tqdm import tqdm


def build_windows(height: int, width: int, patch_size: int, stride: int) -> list:
    return [
        Window(col_off=c, row_off=r, width=patch_size, height=patch_size)
        for r in range(0, height - patch_size + 1, stride)
        for c in range(0, width - patch_size + 1, stride)
    ]


def main(args: argparse.Namespace) -> None:
    images_out = os.path.join(args.output_dir, "images")
    os.makedirs(images_out, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(args.image_folder, "*.tif")))
    if not image_files:
        print(f"No .tif files found in: {args.image_folder}")
        return

    print(f"Found {len(image_files)} image(s). Patch size: {args.patch_size}, "
          f"Stride: {args.stride}, Limit: {args.limit or 'unlimited'}")

    saved = 0
    for img_path in image_files:
        if args.limit is not None and saved >= args.limit:
            break

        file_stem = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\nProcessing: {os.path.basename(img_path)}")

        with rasterio.open(img_path) as src:
            meta = src.meta.copy()
            windows = build_windows(src.height, src.width, args.patch_size, args.stride)

            for window in tqdm(windows, desc="Cropping"):
                if args.limit is not None and saved >= args.limit:
                    break

                patch = src.read(window=window)
                patch_id = f"{file_stem}_r{window.row_off}_c{window.col_off}"
                out_path = os.path.join(images_out, f"{patch_id}.tif")
                patch_meta = meta.copy()
                patch_meta.update({
                    "width": args.patch_size,
                    "height": args.patch_size,
                    "count": patch.shape[0],
                    "transform": src.window_transform(window),
                })

                with rasterio.open(out_path, "w", **patch_meta) as dst:
                    dst.write(patch)

                saved += 1

    print(f"\nDone. Saved: {saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop large GeoTIFF images into fixed-size patches.",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="data/pretrain_bal_dc/raw",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/pretrain_bal_dc",
    )
    parser.add_argument("--patch_size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None, help="Max patches to save (default: unlimited)")
    args = parser.parse_args()
    main(args)
