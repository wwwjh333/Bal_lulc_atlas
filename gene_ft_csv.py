import argparse
import csv
import re
from pathlib import Path

IMAGE_SUBDIR = "images"
LABEL_SUBDIR = "label"
TRAIN_CSV_NAME = "train.csv"
TEST_CSV_NAME = "test.csv"

DEFAULT_TRAIN_STEMS = [
    "m_3907644_se_18_030_20230525",
    "m_3907644_sw_18_030_20230525",
    "m_3907643_nw_18_030_20230525",
    "m_3907627_se_18_030_20230525",
    "m_3807601_se_18_030_20230901",
    "m_3807708_se_18_030_20230901",
]

DEFAULT_TEST_STEMS = [
    "m_3907652_nw_18_030_20230525",
    "m_3907643_sw_18_030_20230525",
    "m_3907636_ne_18_030_20230525",
    "m_3807601_sw_18_030_20230901",
]

PATCH_RE = re.compile(r"^(.+)_([0-9]+)\.(?:tif|tiff)$", re.IGNORECASE)
IMAGE_EXTS = {".tif", ".tiff", ".TIF", ".TIFF"}


def collect_pairs(image_dir: Path, label_dir: Path) -> list[str]:
    img_names = {
        p.name for p in image_dir.iterdir()
        if p.is_file() and p.suffix in IMAGE_EXTS
    }
    lbl_names = {
        p.name for p in label_dir.iterdir()
        if p.is_file() and p.suffix in IMAGE_EXTS
    }
    return sorted(img_names & lbl_names)


def write_csv(path: Path, basenames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="gbk") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "label_name"])
        for name in basenames:
            writer.writerow([f"{IMAGE_SUBDIR}/{name}", f"{LABEL_SUBDIR}/{name}"])


def split_by_stems(
    names: list[str],
    train_stems: list[str],
    test_stems: list[str],
) -> tuple[list[str], list[str], list[str]]:
    overlap = set(train_stems) & set(test_stems)
    if overlap:
        raise SystemExit(f"Stems in both train and test: {sorted(overlap)}")

    train_set = set(train_stems)
    test_set = set(test_stems)
    train_out, test_out, unassigned = [], [], []

    for name in names:
        match = PATCH_RE.match(name)
        if not match:
            unassigned.append(name)
            continue
        big_stem = match.group(1)
        if big_stem in train_set:
            train_out.append(name)
        elif big_stem in test_set:
            test_out.append(name)
        else:
            unassigned.append(name)

    return sorted(train_out), sorted(test_out), sorted(unassigned)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fine-tuning CSV splits from cropped image/label patches."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/bal_dc_benchmark/training_data"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--train-stems", nargs="+", default=DEFAULT_TRAIN_STEMS)
    parser.add_argument("--test-stems", nargs="+", default=DEFAULT_TEST_STEMS)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir
    image_dir = data_dir / IMAGE_SUBDIR
    label_dir = data_dir / LABEL_SUBDIR
    out_dir = args.out_dir or data_dir

    if not image_dir.is_dir():
        raise SystemExit(f"Missing directory: {image_dir}")
    if not label_dir.is_dir():
        raise SystemExit(f"Missing directory: {label_dir}")

    names = collect_pairs(image_dir, label_dir)
    if not names:
        raise SystemExit("No paired image/label files found.")

    train_names, test_names, unassigned = split_by_stems(
        names, args.train_stems, args.test_stems
    )

    if unassigned:
        msg = f"{len(unassigned)} patch(es) not assigned to any train/test stem."
        if args.strict:
            raise SystemExit(f"{msg} Examples: {unassigned[:10]}")
        print(f"[WARN] {msg}")

    train_path = out_dir / TRAIN_CSV_NAME
    test_path = out_dir / TEST_CSV_NAME
    write_csv(train_path, train_names)
    write_csv(test_path, test_names)
    print(f"Wrote {train_path} ({len(train_names)} rows)")
    print(f"Wrote {test_path} ({len(test_names)} rows)")


if __name__ == "__main__":
    main()
