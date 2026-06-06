import csv
import json
import logging
import os
import random
import time
from datetime import datetime

import matplotlib.cm as cm
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from datasets import get_dataloader
from torchvision.utils import make_grid, save_image
from transformers import get_cosine_schedule_with_warmup

try:
    import wandb
except Exception:
    wandb = None


def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def _get_sam_registry(net_type: str):
    if net_type != "sam":
        raise ValueError(f"Unknown net_type={net_type!r}; expected 'sam'.")
    from models.sam.build_sam import sam_model_registry
    return sam_model_registry


def build_model(args, device: torch.device) -> torch.nn.Module:
    registry = _get_sam_registry(args.net_type)
    return registry[args.encoder_type](args).to(device)


def _normalize_pretrain_args(args) -> None:
    if getattr(args, "tau_d", None) is None:
        args.tau_d = float(getattr(args, "tau_d_fixed", 0.3435))


def _setup_pretrain_paths(exp_name: str) -> dict[str, str]:
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = os.path.join("logs", "pretrain", exp_name, timestamp)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    return {"log_path": run_dir, "ckpt_path": ckpt_dir}


def _make_logger(log_path: str, exp_name: str) -> logging.Logger:
    logger = logging.getLogger(f"train_pt.{exp_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(os.path.join(log_path, "train.log"))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _save_pretrain_checkpoint(payload: dict, output_dir: str, filename: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    torch.save(payload, os.path.join(output_dir, filename))


@torch.no_grad()
def get_grad_norm(parameters, norm_type=2.0):
    params = [p for p in parameters if p.grad is not None]
    if len(params) == 0:
        return 0.0
    device = params[0].grad.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in params]),
        norm_type,
    )
    return float(total_norm)


class MetricLogger:
    def __init__(self, args, accelerator, path_helper=None, logger=None):
        self.args = args
        self.accelerator = accelerator
        self.logger = logger
        self.path_helper = path_helper
        self.csv_fp = None
        self.csv_writer = None
        self.wandb_run = None
        self.start_time = time.time()

        if not accelerator.is_main_process or path_helper is None:
            return

        self.metrics_dir = os.path.join(path_helper["log_path"], "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)

        csv_path = os.path.join(self.metrics_dir, "train_metrics.csv")
        self.csv_fp = open(csv_path, "a", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(
            self.csv_fp,
            fieldnames=[
                "step", "epoch", "iter", "loss", "loss_rgb", "loss_nir",
                "lr", "grad_norm", "grad_norm_ema", "mem_mb", "samples_per_sec",
                "elapsed_sec"
            ]
        )
        if self.csv_fp.tell() == 0:
            self.csv_writer.writeheader()
            self.csv_fp.flush()

        use_wb = getattr(args, "use_wandb", 0)
        if isinstance(use_wb, bool):
            use_wb = use_wb
        else:
            use_wb = int(use_wb) != 0
        if use_wb:
            if wandb is None:
                if logger is not None:
                    logger.warning("use_wandb=1 but wandb is not installed. Skip W&B logging.")
            else:
                wandb_dir = os.path.join(self.metrics_dir, "wandb")
                os.makedirs(wandb_dir, exist_ok=True)
                run_name = getattr(args, "wandb_name", "") or args.exp_name
                mode = getattr(args, "wandb_mode", "online")
                self.wandb_run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity if args.wandb_entity else None,
                    name=run_name,
                    dir=wandb_dir,
                    config=vars(args),
                    resume="allow",
                    mode=mode if mode in ("online", "offline", "disabled") else "online",
                )
                if logger is not None:
                    logger.info(
                        f"W&B enabled: project={args.wandb_project}, run={self.wandb_run.name}"
                    )

    def log_step(self, metrics, step):
        if not self.accelerator.is_main_process:
            return

        if self.csv_writer is not None:
            row = {k: metrics.get(k, None) for k in self.csv_writer.fieldnames}
            self.csv_writer.writerow(row)
            self.csv_fp.flush()

        if self.wandb_run is not None:
            wandb.log(metrics, step=step)

    def log_args(self):
        if not self.accelerator.is_main_process or self.path_helper is None:
            return
        args_path = os.path.join(self.metrics_dir, "args.json")
        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(vars(self.args), f, indent=2, ensure_ascii=False, default=str)

    def close(self):
        if not self.accelerator.is_main_process:
            return
        if self.csv_fp is not None:
            self.csv_fp.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()


def build_pix_mask(mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    if mask.dim() == 4:
        mask = mask[..., 0]
    mask = mask.float()
    pix_mask = (
        mask
        .repeat_interleave(patch_size, dim=1)
        .repeat_interleave(patch_size, dim=2)
        .unsqueeze(1)
    )
    return pix_mask


@torch.no_grad()
def save_recon_vis(
    img: torch.Tensor,
    mask_rgb: torch.Tensor,
    mask_nir: torch.Tensor,
    x_rec: torch.Tensor,
    patch_size: int,
    save_path: str,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img = img.float()
    x_rec = x_rec.float()

    pix_mask_rgb = build_pix_mask(mask_rgb, patch_size)
    pix_mask_nir = build_pix_mask(mask_nir, patch_size)
    masked_img_rgb = img * (1.0 - pix_mask_rgb)
    masked_img_nir = img * (1.0 - pix_mask_nir)
    err = (img - x_rec).abs()

    def norm01(x):
        b, c, h, w = x.shape
        x_ = x.view(b, c, -1)
        mn = x_.min(dim=-1, keepdim=True).values.view(b, c, 1, 1)
        mx = x_.max(dim=-1, keepdim=True).values.view(b, c, 1, 1)
        return (x - mn) / (mx - mn + 1e-6)

    gt_rgb, gt_nir = img[:, :3], img[:, 3:4]
    mk_rgb, mk_nir = masked_img_rgb[:, :3], masked_img_nir[:, 3:4]
    rc_rgb, rc_nir = x_rec[:, :3], x_rec[:, 3:4]
    er_rgb, er_nir = err[:, :3], err[:, 3:4]

    row_rgb = torch.cat([norm01(gt_rgb), norm01(mk_rgb), norm01(rc_rgb), norm01(er_rgb)], dim=3)
    row_nir_1c = torch.cat([norm01(gt_nir), norm01(mk_nir), norm01(rc_nir), norm01(er_nir)], dim=3)
    row_nir = row_nir_1c.repeat(1, 3, 1, 1)

    gt_ndvi = (gt_nir - gt_rgb[:, 0:1]) / (gt_nir + gt_rgb[:, 0:1] + 1e-6)
    rc_ndvi = (rc_nir - rc_rgb[:, 0:1]) / (rc_nir + rc_rgb[:, 0:1] + 1e-6)
    gt_ndvi = torch.clamp(gt_ndvi, -1.0, 1.0)
    rc_ndvi = torch.clamp(rc_ndvi, -1.0, 1.0)

    gt_ndvi_01 = (gt_ndvi + 1.0) * 0.5
    rc_ndvi_01 = (rc_ndvi + 1.0) * 0.5
    masked_ndvi_01 = gt_ndvi_01 * (1.0 - pix_mask_nir)

    gt_ndvi_rgb = apply_cmap01(gt_ndvi_01, cmap_name="viridis")
    masked_ndvi_rgb = apply_cmap01(masked_ndvi_01, cmap_name="viridis")
    rc_ndvi_rgb = apply_cmap01(rc_ndvi_01, cmap_name="viridis")
    ndvi_err_rgb = (gt_ndvi_rgb - rc_ndvi_rgb).abs()

    row_ndvi = torch.cat([gt_ndvi_rgb, masked_ndvi_rgb, rc_ndvi_rgb, ndvi_err_rgb], dim=3)
    full = torch.cat([row_rgb, row_nir, row_ndvi], dim=2)
    grid = make_grid(full, nrow=1)
    save_image(grid, save_path)



def main(args):
    set_seed(args.seed)
    _normalize_pretrain_args(args)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.accum_steps,
        mixed_precision=("fp16" if args.amp else "no"),
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device

    net = build_model(args, device)

    optimizer = optim.AdamW(
        net.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    num_warmup_steps = int(args.max_steps * args.warmup_ratio)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_steps,
    )

    if accelerator.is_main_process:
        args.path_helper = _setup_pretrain_paths(args.exp_name)
        logger = _make_logger(args.path_helper["log_path"], args.exp_name)
        logger.info(str(args))
        if getattr(args, "tau_d_mode", "fixed") != "fixed":
            logger.warning(
                "tau_d_mode=%r: BalDcSimMIM still uses scalar args.tau_d (filled from tau_d_fixed).",
                getattr(args, "tau_d_mode", None),
            )
    else:
        logger = None
        args.path_helper = None

    train_loader = get_dataloader(args)

    net, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        net, optimizer, train_loader, lr_scheduler
    )
    accelerator.wait_for_everyone()

    metric_logger = MetricLogger(args, accelerator, args.path_helper, logger)
    if accelerator.is_main_process:
        metric_logger.log_args()

    epoch0 = 0
    global_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        sd = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt.get("model")
        if sd is None:
            raise KeyError(f"resume checkpoint missing weights: keys={list(ckpt.keys())}")
        net.load_state_dict(sd, strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        epoch0 = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        if "lr_scheduler" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        else:
            lr_scheduler.last_epoch = global_step - 1
        if accelerator.is_main_process and logger is not None:
            logger.info(f"=> resumed: epoch={epoch0}, global_step={global_step}")

    net.train()
    optimizer.zero_grad(set_to_none=True)

    grad_ema = None
    pbar = None
    if accelerator.is_main_process:
        pbar = tqdm(total=args.max_steps, initial=global_step, ncols=160, desc="Pretrain", leave=True)

    last_log_time = time.time()

    try:
        for epoch in range(epoch0, args.max_epochs):
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            for it, batch in enumerate(train_loader):
                if global_step >= args.max_steps:
                    if accelerator.is_main_process and pbar is not None:
                        pbar.close()
                        if logger is not None:
                            logger.info(f"Done: global_step={global_step}")
                    return

                img, mask = batch[0], batch[1]
                img = img.to(device, non_blocking=True)
                mask_rgb = mask["rgb"].to(device, non_blocking=True)
                mask_nir = mask["nir"].to(device, non_blocking=True)

                with accelerator.accumulate(net):
                    with accelerator.autocast():
                        out = net(img, mask_rgb, mask_nir)
                        if isinstance(out, (tuple, list)):
                            loss = out[0]
                        else:
                            loss = out[0]
                        loss_rgb = None
                        loss_nir = None
                        if isinstance(out, (tuple, list)) and len(out) == 4:
                            loss_rgb = out[2]
                            loss_nir = out[3]

                    accelerator.backward(loss)
                    loss_val = float(loss.detach())
                    loss_rgb_val = float(loss_rgb.detach()) if loss_rgb is not None else None
                    loss_nir_val = float(loss_nir.detach()) if loss_nir is not None else None

                    if accelerator.sync_gradients:
                        if args.clip_grad and args.clip_grad > 0:
                            grad_norm = accelerator.clip_grad_norm_(net.parameters(), args.clip_grad)
                            grad_val = float(grad_norm)
                        else:
                            grad_val = get_grad_norm(net.parameters())

                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                        if np.isfinite(grad_val):
                            grad_ema = grad_val if grad_ema is None else (0.9 * grad_ema + 0.1 * grad_val)
                        global_step += 1
                        if accelerator.is_main_process and pbar is not None:
                            pbar.update(1)

                        lr = optimizer.param_groups[0]["lr"]

                        if torch.cuda.is_available():
                            mem = torch.cuda.max_memory_allocated(device=device) / (1024**2)
                            torch.cuda.reset_peak_memory_stats(device=device)
                        else:
                            mem = 0.0

                        now = time.time()
                        eff_batch = args.b * accelerator.num_processes * args.accum_steps
                        samples_per_sec = eff_batch / max(now - last_log_time, 1e-6)
                        last_log_time = now

                        metrics = {
                            "step": global_step,
                            "epoch": epoch,
                            "iter": it,
                            "loss": loss_val,
                            "loss_rgb": loss_rgb_val,
                            "loss_nir": loss_nir_val,
                            "lr": lr,
                            "grad_norm": grad_val,
                            "grad_norm_ema": 0.0 if grad_ema is None else grad_ema,
                            "mem_mb": mem,
                            "samples_per_sec": samples_per_sec,
                            "elapsed_sec": now - metric_logger.start_time,
                        }

                        if accelerator.is_main_process and pbar is not None:
                            postfix = {
                                "loss": f"{loss_val:.4f}",
                                "lr": f"{lr:.1e}",
                                "grad": f"{0.0 if grad_ema is None else grad_ema:.2f}",
                                "memMB": f"{mem:.0f}",
                                "epoch": epoch,
                            }
                            if loss_rgb_val is not None and loss_nir_val is not None:
                                postfix["loss_rgb"] = f"{loss_rgb_val:.4f}"
                                postfix["loss_nir"] = f"{loss_nir_val:.4f}"
                            pbar.set_postfix(postfix)

                        if global_step % args.log_steps == 0:
                            metric_logger.log_step(metrics, global_step)
                            if accelerator.is_main_process and logger is not None:
                                logger.info(
                                    "step=%d epoch=%d iter=%d loss=%.6f loss_rgb=%s loss_nir=%s lr=%.8e grad=%.4f memMB=%.1f sps=%.2f"
                                    % (
                                        global_step,
                                        epoch,
                                        it,
                                        loss_val,
                                        f"{loss_rgb_val:.6f}" if loss_rgb_val is not None else "None",
                                        f"{loss_nir_val:.6f}" if loss_nir_val is not None else "None",
                                        lr,
                                        grad_val,
                                        mem,
                                        samples_per_sec,
                                    )
                                )

                        if accelerator.is_main_process and args.vis_steps > 0 and (global_step % args.vis_steps == 0):
                            model_ = accelerator.unwrap_model(net)
                            model_.eval()
                            vis_b = min(args.vis_num, img.shape[0])

                            vis_img = img[:vis_b].detach()
                            vis_mask_rgb = mask_rgb[:vis_b].detach()
                            vis_mask_nir = mask_nir[:vis_b].detach()

                            with torch.no_grad():
                                with accelerator.autocast():
                                    vis_out = model_(vis_img, vis_mask_rgb, vis_mask_nir)
                                    vis_rec = vis_out[1] if isinstance(vis_out, (tuple, list)) else vis_out

                            model_.train()

                            vis_dir = os.path.join(args.path_helper["log_path"], "recon_vis")
                            save_path = os.path.join(vis_dir, f"step_{global_step:06d}.png")

                            save_recon_vis(
                                img=vis_img,
                                mask_rgb=vis_mask_rgb,
                                mask_nir=vis_mask_nir,
                                x_rec=vis_rec.detach(),
                                patch_size=model_.patch_size,
                                save_path=save_path,
                            )

                        if args.save_steps > 0 and (global_step % args.save_steps == 0):
                            if accelerator.is_main_process:
                                sd = accelerator.unwrap_model(net).state_dict()
                                payload = {
                                    "epoch": epoch,
                                    "global_step": global_step,
                                    "model": args.net_type,
                                    "state_dict": sd,
                                    "optimizer": optimizer.state_dict(),
                                    "lr_scheduler": lr_scheduler.state_dict(),
                                    "scaler": None,
                                    "path_helper": args.path_helper,
                                }
                                ckpt_name = f"checkpoint-{global_step:06d}.pth"
                                _save_pretrain_checkpoint(
                                    payload, args.path_helper["ckpt_path"], ckpt_name,
                                )
                                _save_pretrain_checkpoint(
                                    payload, args.path_helper["ckpt_path"], "last.pth",
                                )
                            accelerator.wait_for_everyone()
    finally:
        metric_logger.close()
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM SimMIM pretrain")

    parser.add_argument("-net_type", type=str, default="sam", choices=["sam"])
    parser.add_argument(
        "-model_type",
        type=str,
        default="simmim",
        choices=["simmim"],
    )
    parser.add_argument(
        "-encoder_type",
        type=str,
        default="default",
        choices=["default", "vit_b", "vit_l", "vit_h"],
    )
    parser.add_argument("-image_size", type=int, default=1024)
    parser.add_argument(
        "-sam_ckpt",
        type=str,
        default="pretrain_weights/sam/sam_vit_b_01ec64.pth",
    )

    parser.add_argument(
        "-dataset",
        type=str,
        default="bal_dc_simmim",
        choices=["bal_dc_simmim"],
    )
    parser.add_argument(
        "-data_path",
        type=str,
        default="data/pretrain_bal_dc/images",
    )
    parser.add_argument(
        "-complexity_json",
        type=str,
        default="data/pretrain_bal_dc/complexity.json",
    )
    parser.add_argument(
        "-heterogeneity_maps_dir",
        type=str,
        default="data/pretrain_bal_dc/maps",
    )
    parser.add_argument("--super_tile_tokens", type=int, default=32)
    parser.add_argument(
        "--tau_d_mode", type=str, default="fixed", choices=["per_image_median", "fixed"],
    )
    parser.add_argument("--tau_d_fixed", type=float, default=0.3435)
    parser.add_argument("--c10", type=float, default=0.17213906)
    parser.add_argument("--c90", type=float, default=0.41866542)
    parser.add_argument(
        "--mask_ratios", type=float, nargs=3, default=None, metavar=("R32", "R64", "R128"),
    )

    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--clip_grad", type=float, default=5.0)
    parser.add_argument("-weight_decay", type=float, default=1e-5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)

    parser.add_argument("-exp_name", type=str, default="bal_dc_mim_pretrain")
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--wandb_project", type=str, default="Bal-DC_LULC_Atlas")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_name", type=str, default="bal_dc_pretrain")
    parser.add_argument("--wandb_mode", type=str, default="online")

    parser.add_argument("-b", type=int, default=2)
    parser.add_argument("-w", type=int, default=4)
    parser.add_argument("-gpu_device", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("-seed", type=int, default=3407)
    parser.add_argument("--vis_steps", type=int, default=100)
    parser.add_argument("--vis_num", type=int, default=2)

    args = parser.parse_args()
    main(args)
