from __future__ import annotations

import argparse
import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from datasets import get_dataloader
from label_config import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES


def _get_sam_registry(net_type: str):
    if net_type != "sam":
        raise ValueError(f"Unknown args.net_type={net_type!r}; expected 'sam'.")
    from models.sam.build_sam import sam_model_registry
    return sam_model_registry


Mapping_class_names = CLASS_NAMES


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=np.float64)
    m = den > 0
    out[m] = num[m] / den[m]
    return out


def per_class_metrics_from_confusion(cm: np.ndarray) -> dict[str, np.ndarray]:
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    iou = _safe_div(tp, tp + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1}


def overall_metrics_from_confusion(cm: np.ndarray) -> dict[str, float]:
    per = per_class_metrics_from_confusion(cm)
    miou = float(np.mean(per["iou"]))
    mf1 = float(np.mean(per["f1"]))
    oa = float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0.0
    return {"miou": miou, "mf1": mf1, "oa": oa}


def _accumulate_cm_torch(
    cm: torch.Tensor, pred: torch.Tensor, y: torch.Tensor, num_classes: int,
) -> None:
    c = int(num_classes)
    yt = y.reshape(-1).to(dtype=torch.int64)
    yp = pred.reshape(-1).to(dtype=torch.int64)
    m = (yt >= 0) & (yt < c) & (yp >= 0) & (yp < c)
    if not torch.any(m):
        return
    idx = yt[m] * c + yp[m]
    bins = torch.bincount(idx, minlength=c * c).reshape(c, c)
    cm += bins.to(cm.dtype)


SCENE_ORDER = ["Sparse / Peri-Urban", "Typical Urban", "Dense Urban Core"]

_PREFIX_TO_SCENE = {
    "m_3807601_sw_18_030_20230901": "Dense Urban Core",
    "m_3907652_nw_18_030_20230525": "Dense Urban Core",
    "m_3907643_sw_18_030_20230525": "Typical Urban",
    "m_3907636_ne_18_030_20230525": "Sparse / Peri-Urban",
}


def _extract_sample_name(meta_item: Any) -> str:
    if isinstance(meta_item, str):
        return meta_item
    if isinstance(meta_item, dict):
        v = meta_item.get("filename_or_obj", "")
        return str(v) if v is not None else ""
    return str(meta_item)


def _prefix_from_name(name: str) -> str:
    base = os.path.basename(name)
    stem, _ = os.path.splitext(base)
    if "_" not in stem:
        return stem
    return stem.rsplit("_", 1)[0]


def _scene_from_prefix(prefix: str) -> str:
    return _PREFIX_TO_SCENE.get(prefix, "Unknown")


@torch.no_grad()
def evaluate_with_scenes(
    model: nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    ignore_index: int = IGNORE_INDEX,
) -> dict[str, np.ndarray]:
    model.eval()
    cm_scene: dict[str, torch.Tensor] = {}

    for batch in tqdm(loader, desc="test", leave=False):
        imgs = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        y = batch["label"].to(device=device, non_blocking=True)

        logits = model(imgs)
        pred = logits.argmax(dim=1)

        meta = batch.get("image_meta_dict", None)
        if isinstance(meta, list):
            meta_list = meta
        elif isinstance(meta, dict):
            v = meta.get("filename_or_obj", None)
            if isinstance(v, list):
                meta_list = [{"filename_or_obj": item} for item in v]
            else:
                meta_list = [meta] * int(pred.shape[0])
        else:
            meta_list = [meta] * int(pred.shape[0])

        bs = int(pred.shape[0])
        for i in range(bs):
            name = _extract_sample_name(meta_list[i]) if i < len(meta_list) else ""
            scene = _scene_from_prefix(_prefix_from_name(name))
            if scene not in cm_scene:
                cm_scene[scene] = torch.zeros(
                    (num_classes, num_classes), dtype=torch.int64, device=device,
                )
            _accumulate_cm_torch(
                cm_scene[scene], pred[i], y[i], num_classes, ignore_index,
            )

    return {k: v.detach().cpu().numpy() for k, v in cm_scene.items()}


def _print_report(title: str, cm: np.ndarray) -> None:
    overall = overall_metrics_from_confusion(cm)
    per = per_class_metrics_from_confusion(cm)

    print(f"\n== {title} ==")
    print(
        f"Overall  mIoU={overall['miou']:.4f}  "
        f"mF1={overall['mf1']:.4f}  OA={overall['oa']:.4f}"
    )
    print("Per-class:")
    print("  cls  name              IoU     Prec    Recall  F1")
    for i in range(cm.shape[0]):
        name = Mapping_class_names.get(i, f"class_{i}")
        print(
            f"  {i:>3d}  {name:<16s}  "
            f"{per['iou'][i]:.4f}  {per['precision'][i]:.4f}  "
            f"{per['recall'][i]:.4f}  {per['f1'][i]:.4f}"
        )


def _print_macro_average(cm_scene: dict[str, np.ndarray]) -> None:
    scene_metrics = {
        s: overall_metrics_from_confusion(cm_scene[s])
        for s in SCENE_ORDER if s in cm_scene
    }

    if len(scene_metrics) < len(SCENE_ORDER):
        missing = [s for s in SCENE_ORDER if s not in scene_metrics]
        print(f"\n[WARNING] Missing scenes for macro average: {missing}")
        return

    macro_miou = float(np.mean([v["miou"] for v in scene_metrics.values()]))
    macro_mf1 = float(np.mean([v["mf1"] for v in scene_metrics.values()]))
    macro_oa = float(np.mean([v["oa"] for v in scene_metrics.values()]))
    print("\n== MACRO AVERAGE (equal scene weight) ==")
    print(
        f"Overall  mIoU={macro_miou:.4f}  "
        f"mF1={macro_mf1:.4f}  OA={macro_oa:.4f}"
    )


def build_model(args, device: torch.device) -> nn.Module:
    registry = _get_sam_registry(args.net_type)
    model = registry[args.encoder_type](args)
    return model.to(device)


def _load_checkpoint(path: str, map_location) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return {"state_dict": ckpt["model_state_dict"], "args": ckpt.get("args", {})}
    return {"state_dict": ckpt, "args": {}}


def main(args: argparse.Namespace) -> None:
    set_random_seed(args.seed, deterministic=False)
    device = torch.device(
        f"cuda:{args.gpu_device}" if torch.cuda.is_available() else "cpu",
    )

    print(f"=> loading checkpoint {args.weights}")
    assert os.path.exists(args.weights), f"checkpoint not found: {args.weights}"
    ckpt = _load_checkpoint(args.weights, map_location=f"cuda:{args.gpu_device}")
    start_epoch = ckpt.get("epoch", -1) if isinstance(ckpt, dict) else -1

    _, test_loader = get_dataloader(args)
    model = build_model(args, device)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print(
        f"=> loaded checkpoint (epoch={start_epoch}, "
        f"missing={len(missing)}, unexpected={len(unexpected)})"
    )

    cm_scene = evaluate_with_scenes(
        model,
        test_loader,
        device,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
    )

    printed: set[str] = set()
    for s in SCENE_ORDER:
        if s in cm_scene:
            _print_report(f"SCENE: {s}", cm_scene[s])
            printed.add(s)
    for s in sorted(set(cm_scene.keys()) - printed):
        _print_report(f"SCENE: {s}", cm_scene[s])

    _print_macro_average(cm_scene)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM bal_dc semantic segmentation testing",
    )
    parser.add_argument("-net_type", type=str, default="sam", choices=["sam"])
    parser.add_argument("-model_type", type=str, default="seg", choices=["seg"])
    parser.add_argument(
        "-encoder_type", type=str, default="default",
        choices=["default", "vit_b", "vit_l", "vit_h"],
    )
    parser.add_argument(
        "-block_type", type=str, default="adalora",
        choices=["default", "adapter", "fwa", "lora", "adalora"],
    )
    parser.add_argument("-num_classes", type=int, default=NUM_CLASSES)
    parser.add_argument(
        "-ignore_index",
        type=int,
        default=IGNORE_INDEX,
        help="void/nodata label excluded from metrics (valid classes: 0..num_classes-1)",
    )
    parser.add_argument("-image_size", type=int, default=1024)
    parser.add_argument("-out_size", type=int, default=1024)
    parser.add_argument(
        "-sam_ckpt",
        default="pretrain_weights/sam/sam_vit_b_01ec64.pth",
        type=str,
    )
    parser.add_argument("-encoder_pretrain_ckpt", default="", type=str)
    parser.add_argument(
        "-weights",
        type=str,
        default="logs/bal_dc_benchmark_urbanmim_wo_ha_adalora/Model/best_miou_checkpoint.pth",
    )
    parser.add_argument("-dataset", default="bal_dc", type=str)
    parser.add_argument("-data_path", type=str, default="data/bal_dc_benchmark/training_data")
    parser.add_argument("-train_ground_truth_csv", type=str, default="train.csv")
    parser.add_argument("-test_ground_truth_csv", type=str, default="test.csv")
    parser.add_argument("-b", type=int, default=2)
    parser.add_argument("-w", type=int, default=4)
    parser.add_argument("-gpu_device", type=int, default=0)
    parser.add_argument("-seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
