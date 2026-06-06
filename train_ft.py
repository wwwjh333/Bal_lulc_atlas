from __future__ import annotations

import argparse
import logging
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from datasets import get_dataloader
from label_config import IGNORE_INDEX, NUM_CLASSES

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


def _get_sam_registry(net_type: str):
    if net_type != "sam":
        raise ValueError(f"Unknown args.net_type={net_type!r}; expected 'sam'.")
    from models.sam.build_sam import sam_model_registry
    return sam_model_registry


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


def setup_run_dir(exp_dir: str) -> tuple[logging.Logger, str, str]:
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = os.path.join(exp_dir, timestamp)
    ckpt_dir = os.path.join(run_dir, "Model")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = logging.getLogger(f"train_ft.{os.path.basename(exp_dir.rstrip(os.sep))}.{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(os.path.join(run_dir, "log.txt"))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger, run_dir, ckpt_dir


def _accumulate_cm_torch(
    cm: torch.Tensor,
    pred: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    ignore_index: int = IGNORE_INDEX,
) -> None:
    c = int(num_classes)
    yt = y.reshape(-1).to(dtype=torch.int64)
    yp = pred.reshape(-1).to(dtype=torch.int64)
    m = (yt >= 0) & (yt < c) & (yt != ignore_index) & (yp >= 0) & (yp < c)
    if not torch.any(m):
        return
    idx = yt[m] * c + yp[m]
    bins = torch.bincount(idx, minlength=c * c).reshape(c, c)
    cm += bins.to(cm.dtype)


def metrics_from_confusion(cm: np.ndarray) -> dict[str, float]:
    cm = cm.astype(np.float64)
    n = cm.shape[0]
    eps = 1e-9
    ious, recs, precs = [], [], []
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else 0.0)
        recs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        precs.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    f1s = [
        2.0 * p * r / (p + r) if (p + r) > eps else 0.0
        for p, r in zip(precs, recs)
    ]
    miou = float(np.mean(ious))
    mf1 = float(np.mean(f1s))
    oa = float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0.0
    return {"miou": miou, "mf1": mf1, "oa": oa}


def _wandb_lr_fields(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    groups = optimizer.param_groups
    if len(groups) == 1:
        return {"lr": float(groups[0]["lr"])}
    return {"lr": float(groups[1]["lr"]), "lr_encoder_peft": float(groups[0]["lr"])}


def build_model(args, device: torch.device) -> nn.Module:
    registry = _get_sam_registry(args.net_type)
    model = registry[args.encoder_type](args)
    return model.to(device)


def _params_m(numel: int) -> str:
    return f"{numel / 1e6:.2f}M"


def configure_trainable_by_block_type(
    model: nn.Module,
    block_type: str,
    net_type: str,
    logger: logging.Logger,
) -> tuple[int, int]:
    requested = (block_type or "default").lower()
    effective = requested
    if net_type != "sam" and requested != "default":
        logger.warning(
            "PEFT block_type only applies to SAM ViT ImageEncoderViT; "
            f"net_type={net_type!r}, training encoder with default (full) fine-tuning."
        )
        effective = "default"

    enc = getattr(model, "image_encoder", None)

    if effective == "default":
        for name, p in model.named_parameters():
            p.requires_grad = not name.startswith("prompt_encoder.")
    else:
        for p in model.parameters():
            p.requires_grad = False

        if enc is not None:
            if effective in ("adapter", "fwa"):
                n_ad = 0
                for n, p in enc.named_parameters():
                    if "MLP_Adapter" in n or "Space_Adapter" in n:
                        p.requires_grad = True
                        n_ad += p.numel()
                if n_ad == 0:
                    logger.warning(
                        "No MLP_Adapter/Space_Adapter parameters in image_encoder; "
                        f"check -block_type is adapter or fwa (effective={effective!r})."
                    )
            elif effective in ("lora", "adalora"):
                from models.common import loralib as lora

                for p in enc.parameters():
                    p.requires_grad = True
                lora.mark_only_lora_as_trainable(enc, bias="none")
            else:
                raise ValueError(f"Unknown block_type={effective!r}")

        md = getattr(model, "mask_decoder", None)
        if md is not None:
            for p in md.parameters():
                p.requires_grad = True

    pe = getattr(model, "prompt_encoder", None)
    if pe is not None:
        for p in pe.parameters():
            p.requires_grad = False

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in model.parameters())
    extra = f", effective={effective}" if effective != requested else ""

    if enc is not None:
        enc_train = sum(p.numel() for p in enc.parameters() if p.requires_grad)
        enc_tot = sum(p.numel() for p in enc.parameters())
        enc_pct = 100.0 * enc_train / max(enc_tot, 1)
        logger.info(
            f"image_encoder trainable: {_params_m(enc_train)} / {_params_m(enc_tot)} "
            f"({enc_pct:.2f}% of encoder) [block_type={requested}{extra}]"
        )
    else:
        pct = 100.0 * n_train / max(n_tot, 1)
        logger.info(
            f"trainable (full model): {_params_m(n_train)} / {_params_m(n_tot)} "
            f"({pct:.2f}%) [block_type={requested}{extra}]"
        )
    return n_train, n_tot


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    cm_t = torch.zeros(
        (args.num_classes, args.num_classes),
        dtype=torch.int64,
        device=device,
    )
    lossfunc = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    loss_sum, n_batches = 0.0, 0
    for batch in tqdm(loader, desc="val", leave=False):
        imgs = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        y = batch["label"].to(device=device, non_blocking=True)
        logits = model(imgs)
        loss_sum += lossfunc(logits, y).item()
        n_batches += 1
        pred = logits.argmax(dim=1)
        _accumulate_cm_torch(cm_t, pred, y, args.num_classes, args.ignore_index)

    cm = cm_t.detach().cpu().numpy()
    metrics = metrics_from_confusion(cm)
    metrics["loss"] = loss_sum / max(n_batches, 1)
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> float:
    model.train()

    lossfunc = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    running, n_samples = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for batch in pbar:
        imgs = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        y = batch["label"].to(device=device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = lossfunc(logits, y)
        loss.backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        if trainable:
            nn.utils.clip_grad_norm_(trainable, max_norm=0.1)
        optimizer.step()

        bs = logits.size(0)
        running += loss.item() * bs
        n_samples += bs
        pbar.set_postfix(loss=f"{running / max(n_samples, 1):.4f}")

    return running / max(n_samples, 1)


def main(args: argparse.Namespace) -> None:
    set_random_seed(args.seed, deterministic=False)
    device = torch.device(f"cuda:{args.gpu_device}" if torch.cuda.is_available() else "cpu")

    logger, run_dir, ckpt_dir = setup_run_dir(args.exp_dir)
    logger.info(f"run_dir={run_dir}")
    logger.info(args)

    exp_name = os.path.basename(args.exp_dir.rstrip(os.sep))

    wandb_run = None
    if int(getattr(args, "use_wandb", 0)):
        if wandb is None:
            logger.warning("wandb not installed; skipping W&B logging.")
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=(args.wandb_name or exp_name),
                entity=(args.wandb_entity or None),
                mode=args.wandb_mode,
                config=vars(args),
            )

    train_loader, val_loader = get_dataloader(args)
    model = build_model(args, device)
    configure_trainable_by_block_type(model, args.block_type, args.net_type, logger)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    peft_block = (args.block_type or "default").lower() in ("adapter", "fwa", "lora", "adalora")
    enc_peft_lr = getattr(args, "encoder_peft_lr", None)
    if peft_block and enc_peft_lr is not None and args.net_type == "sam":
        enc_trainable = [
            p for n, p in model.named_parameters()
            if p.requires_grad and n.startswith("image_encoder.")
        ]
        other_trainable = [
            p for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith("image_encoder.")
        ]
        if not enc_trainable or not other_trainable:
            logger.warning(
                "encoder_peft_lr requires both image_encoder and other trainable parameters; "
                "one side is empty, falling back to single Adam group (all use -lr)."
            )
            optimizer = torch.optim.Adam(
                trainable_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
            )
        else:
            optimizer = torch.optim.Adam(
                [
                    {"params": enc_trainable, "lr": float(enc_peft_lr)},
                    {"params": other_trainable, "lr": float(args.lr)},
                ],
                betas=(0.9, 0.999),
                eps=1e-8,
            )
            logger.info(
                f"optimizer param groups: image_encoder PEFT lr={enc_peft_lr:g}, "
                f"other trainable lr={args.lr:g}"
            )
    else:
        optimizer = torch.optim.Adam(
            trainable_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_path = os.path.join(ckpt_dir, "best_miou_checkpoint.pth")
    best_miou = -1.0

    for epoch in range(args.epoch):
        t0 = time.time()
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args)
        logger.info(
            f"epoch {epoch:>3d}  train_loss={avg_loss:.4f}  "
            f"time={(time.time() - t0):.1f}s"
        )
        if wandb_run is not None:
            wandb.log(
                {
                    "train/loss": float(avg_loss),
                    **_wandb_lr_fields(optimizer),
                    "epoch": int(epoch),
                },
                step=int(epoch),
            )

        do_eval = (
            (epoch and epoch < 5)
            or (epoch and epoch % args.val_freq == 0)
            or (epoch == args.epoch - 1)
        )
        if do_eval:
            metrics = evaluate(model, val_loader, device, args)
            logger.info(
                f"epoch {epoch:>3d}  val_loss={metrics['loss']:.4f}  "
                f"mIoU={metrics['miou']:.4f}  mF1={metrics['mf1']:.4f}  "
                f"OA={metrics['oa']:.4f}"
            )
            if wandb_run is not None:
                wandb.log(
                    {
                        "val/loss": float(metrics["loss"]),
                        "val/miou": float(metrics["miou"]),
                        "val/mf1":  float(metrics["mf1"]),
                        "val/oa":   float(metrics["oa"]),
                        **_wandb_lr_fields(optimizer),
                        "epoch":    int(epoch),
                    },
                    step=int(epoch),
                )

            if metrics["miou"] > best_miou:
                best_miou = metrics["miou"]
                state = (
                    model.module.state_dict()
                    if hasattr(model, "module") else model.state_dict()
                )
                torch.save(
                    {
                        "epoch":            epoch + 1,
                        "model":            args.net_type,
                        "state_dict":       state,
                        "optimizer":        optimizer.state_dict(),
                        "metrics":          metrics,
                        "args":             vars(args),
                    },
                    best_path,
                )
                logger.info(f"  -> saved best mIoU={best_miou:.4f} -> {best_path}")

        scheduler.step()

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM bal_dc semantic segmentation fine-tuning",
    )

    parser.add_argument(
        "-net_type", type=str, default="sam", choices=["sam"],
        help="net family (SAM ViT)",
    )
    parser.add_argument(
        "-model_type", type=str, default="seg",
        choices=["seg", "simmim"],
        help="model variant within the chosen net (see build_sam.py)",
    )
    parser.add_argument(
        "-encoder_type", type=str, default="default",
        choices=["default", "vit_b", "vit_l", "vit_h"],
        help="ViT encoder size (default == vit_b)",
    )
    parser.add_argument(
        "-block_type", type=str, default="adalora",
        choices=["default", "adapter", "fwa", "lora", "adalora"],
        help=(
            "Transformer block in encoder: default / adapter / fwa (FreqWeaver+MLP adapter) / "
            "lora / adalora (see image_encoder.BLOCK_REGISTRY; SAM ViT only)"
        ),
    )
    parser.add_argument(
        "-encoder_peft_lr",
        type=float,
        default=None,
        help=(
            "Optional: when net_type=sam and block_type is adapter/fwa/lora/adalora, "
            "set a separate LR for trainable PEFT params in image_encoder; "
            "mask decoder and other trainable params still use -lr. "
            "Default None uses -lr for all trainable params."
        ),
    )
    parser.add_argument(
        "-num_classes", type=int, default=NUM_CLASSES,
        help="number of semantic classes (= mask_decoder output channels)",
    )
    parser.add_argument("-image_size", type=int, default=1024, help="model input size")
    parser.add_argument("-out_size", type=int, default=1024, help="upsampled logits size")
    parser.add_argument(
        "-sam_ckpt",
        default="pretrain_weights/sam/sam_vit_b_01ec64.pth",
        type=str,
        help="Official Meta SAM checkpoint; loaded when present (seg merge base; simmim loads matched submodules such as image_encoder).",
    )
    parser.add_argument(
        "-encoder_pretrain_ckpt",
        default="logs/pretrain/bal_dc_mim_pretrain_wo_ha/checkpoints/last.pth",
        type=str,
        help="seg only: if present, merge and overwrite image_encoder.* in the official weights.",
    )

    parser.add_argument("-dataset", default="bal_dc", type=str)
    parser.add_argument(
        "-data_path", type=str,
        default="data/bal_dc_benchmark/training_data",
        help="root directory of segmentation data",
    )
    parser.add_argument(
        "-train_ground_truth_csv", type=str, default="train.csv",
        help="CSV filename (relative to data_path) for the training split",
    )
    parser.add_argument(
        "-test_ground_truth_csv", type=str, default="test.csv",
        help="CSV filename (relative to data_path) for the test/val split",
    )
    parser.add_argument("-b", type=int, default=2, help="batch size")
    parser.add_argument("-w", type=int, default=4)

    parser.add_argument("-epoch", type=int, default=50)
    parser.add_argument(
        "-lr", type=float, default=1e-4,
        help="Initial LR; for default full-model training, or the non-encoder-PEFT group when using -encoder_peft_lr",
    )
    parser.add_argument("-val_freq", type=int, default=5, help="interval between validations (in epochs)")
    parser.add_argument(
        "-ignore_index",
        type=int,
        default=IGNORE_INDEX,
        help="void/nodata label excluded from loss and metrics (valid classes: 0..num_classes-1)",
    )

    parser.add_argument(
        "-exp_dir",
        default="logs/bal_dc_benchmark_urbanmim_wo_ha_adalora",
        type=str,
        help=(
            "experiment directory; each run is written to "
            "``<exp_dir>/<timestamp>/{Model,log.txt}`` so multiple runs "
            "of the same experiment stay grouped together"
        ),
    )
    parser.add_argument("--use_wandb", type=int, default=0, help="enable W&B logging (1/0)")
    parser.add_argument("--wandb_project", type=str, default="Bal-DC_LULC_Atlas", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default="", help="W&B entity (team/user)")
    parser.add_argument("--wandb_name", type=str, default="bal_dc_urbanmim_wo_ha_adalora", help="W&B run name")
    parser.add_argument("--wandb_mode", type=str, default="online", help="online/offline/disabled"    )

    parser.add_argument("-gpu_device", type=int, default=0)
    parser.add_argument("-seed", type=int, default=3407, help="random seed")

    args = parser.parse_args()
    main(args)
