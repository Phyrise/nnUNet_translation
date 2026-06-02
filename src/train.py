"""Training entry point for sCT 2.5D U-Net (MONAI)."""
from __future__ import annotations
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from monai.inferers import sliding_window_inference

from .utils import set_seed, load_config, get_logger, denormalize_ct
from .data import build_datasets
from .model import build_unet
from .losses import build_loss


def render_progress_png(out_path, epochs, train_losses, val_losses, epoch_times, lrs):
    """Render an nnUNet-style 3-panel progress figure (loss / time / lr)."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    ax = axes[0]
    if train_losses:
        ax.plot(epochs, train_losses, color="tab:blue", label="loss_tr")
    if val_losses:
        ax.plot(epochs, val_losses, color="tab:red", label="loss_val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, epoch_times, color="tab:blue", label="epoch duration")
    ax.set_xlabel("epoch")
    ax.set_ylabel("time [s]")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, lrs, color="tab:blue", label="learning rate")
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    tmp_path = str(out_path) + ".tmp.png"
    fig.savefig(tmp_path, dpi=120)
    plt.close(fig)
    os.replace(tmp_path, out_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--resume", default=None, help="path to checkpoint to resume")
    # C3: --override key=value (dot-separated nested key, e.g. training.lr=5e-4)
    # sweep script 에서 lr/out_dir 등을 yaml 수정 없이 override 가능
    p.add_argument("--override", action="append", default=[],
                   help="override config key (dot-separated). Repeatable. Example: --override training.lr=5e-4")
    return p.parse_args()


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """C3: --override key=value 를 cfg 에 적용. value 는 yaml-parse + numeric fallback."""
    import yaml
    for kv in overrides:
        if "=" not in kv:
            raise ValueError(f"--override 형식 오류 (key=value 필요): {kv!r}")
        key, val_str = kv.split("=", 1)
        # yaml-parse value (숫자/bool/문자열 자동 추론)
        try:
            val = yaml.safe_load(val_str)
        except Exception:
            val = val_str
        # yaml 1.1 spec 우회 — "1e-5" 같은 scientific notation 이 string 으로 남으면 float() 재시도
        if isinstance(val, str):
            try:
                val = float(val_str)
            except ValueError:
                try:
                    val = int(val_str)
                except ValueError:
                    pass  # 진짜 string 인 경우 유지
        # dot-separated key 를 nested dict 로 적용
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            if k not in d:
                raise KeyError(f"--override key 경로 부재: {key} (단계 '{k}')")
            d = d[k]
        if keys[-1] not in d:
            raise KeyError(f"--override leaf key 부재: {key} (leaf '{keys[-1]}')")
        d[keys[-1]] = val
    return cfg


def evaluate(model, loader, device, ct_clip, patch_size, sw_batch_size, sw_overlap):
    model.eval()
    mae_hu_total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            pred = sliding_window_inference(
                inputs=x,
                roi_size=patch_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=sw_overlap,
                mode="gaussian",
            )
            pred_hu = (pred.clamp(-1, 1) + 1) * 0.5 * (ct_clip[1] - ct_clip[0]) + ct_clip[0]
            y_hu = (y + 1) * 0.5 * (ct_clip[1] - ct_clip[0]) + ct_clip[0]
            mae_hu_total += (pred_hu - y_hu).abs().mean().item() * x.size(0)
            n += x.size(0)
    return mae_hu_total / max(n, 1)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    # C3: CLI override 적용 (yaml load 직후)
    if args.override:
        cfg = _apply_overrides(cfg, args.override)
    set_seed(cfg["training"]["seed"])

    out_dir = cfg["training"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger("train", os.path.join(out_dir, "train.log"))
    writer = SummaryWriter(os.path.join(out_dir, "tb"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    train_ds, val_ds, train_vols, val_vols = build_datasets(cfg)
    logger.info(f"Train cases: {[v.case_id for v in train_vols]}")
    logger.info(f"Val cases:   {[v.case_id for v in val_vols]}")
    logger.info(f"Train slices: {len(train_ds)}  Val slices: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = build_unet(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params/1e6:.2f}M")

    loss_fn = build_loss(cfg).to(device)
    # C5: optimizer 선택 — Adam (기존 default) 또는 SGD(momentum, nesterov)
    opt_name = cfg["training"].get("optimizer", "adam").lower()
    if opt_name == "sgd":
        momentum = cfg["training"].get("momentum", 0.99)
        nesterov = cfg["training"].get("nesterov", True)
        optimizer = SGD(
            model.parameters(),
            lr=cfg["training"]["lr"],
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=cfg["training"]["weight_decay"],
        )
        logger.info(f"Optimizer: SGD(lr={cfg['training']['lr']}, momentum={momentum}, nesterov={nesterov}, wd={cfg['training']['weight_decay']})")
    elif opt_name == "adam":
        optimizer = Adam(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
        logger.info(f"Optimizer: Adam(lr={cfg['training']['lr']}, wd={cfg['training']['weight_decay']})")
    else:
        raise ValueError(f"Unknown optimizer: {opt_name!r} (expected 'sgd' or 'adam')")
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["num_epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["training"]["amp"] and device.type == "cuda")

    start_epoch = 0
    best_mae = float("inf")
    history = {"epoch": [], "train_loss": [], "val_loss": [], "epoch_time": [], "lr": []}
    history_path = os.path.join(out_dir, "history.json")
    progress_png = os.path.join(out_dir, "progress.png")
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["sched"])
        start_epoch = ckpt["epoch"] + 1
        best_mae = ckpt.get("best_mae", float("inf"))
        logger.info(f"Resumed from {args.resume} at epoch {start_epoch}")
        if os.path.isfile(history_path):
            try:
                history = json.load(open(history_path, encoding="utf-8"))
            except Exception:
                pass

    ct_clip = (cfg["normalization"]["ct_clip_min"], cfg["normalization"]["ct_clip_max"])
    log_every = cfg["training"]["log_every"]
    val_every = cfg["training"]["val_every"]
    save_every = cfg["training"].get("save_every", 0)  # 0 disables periodic snapshots
    val_patch = tuple(cfg["slicing"]["patch_size"])
    val_sw_bs = cfg["inference"]["sw_batch_size"]
    val_sw_overlap = cfg["inference"]["sw_overlap"]

    for epoch in range(start_epoch, cfg["training"]["num_epochs"]):
        model.train()
        t0 = time.time()
        running = 0.0
        epoch_loss_sum = 0.0
        epoch_loss_n = 0
        for it, batch in enumerate(train_loader):
            x = batch["input"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg["training"]["amp"] and device.type == "cuda"):
                pred = model(x)
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            # H-15: AMP-safe gradient clipping (nnU-Net 표준 clip_norm=12.0)
            # NaN/Inf 회복 안전망. 효과는 [E] (본 프로젝트 A 학습에서 NaN
            # 발생 보고 없으므로 측정 X), 비용은 2줄 / 무시할만한 overhead.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
            scaler.step(optimizer)
            scaler.update()
            l_item = loss.item()
            running += l_item
            epoch_loss_sum += l_item
            epoch_loss_n += 1
            if (it + 1) % log_every == 0:
                avg = running / log_every
                logger.info(f"epoch {epoch} it {it+1}/{len(train_loader)} loss {avg:.4f}")
                writer.add_scalar("train/loss", avg, epoch * len(train_loader) + it)
                running = 0.0
        scheduler.step()
        dt = time.time() - t0
        cur_lr = scheduler.get_last_lr()[0]
        train_loss_avg = epoch_loss_sum / max(epoch_loss_n, 1)
        logger.info(f"epoch {epoch} done in {dt:.1f}s lr={cur_lr:.2e} avg_train_loss={train_loss_avg:.4f}")

        val_mae_normalized = float("nan")
        if (epoch + 1) % val_every == 0:
            mae_hu = evaluate(model, val_loader, device, ct_clip, val_patch, val_sw_bs, val_sw_overlap)
            writer.add_scalar("val/mae_hu", mae_hu, epoch)
            logger.info(f"epoch {epoch} val MAE(HU): {mae_hu:.2f}")
            # also store val loss in normalized space (~ MAE / dynamic_range / 2) so progress.png matches train_loss scale
            val_mae_normalized = mae_hu / (ct_clip[1] - ct_clip[0])
            ckpt_obj = {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "sched": scheduler.state_dict(),
                "epoch": epoch,
                "best_mae": best_mae,
                "cfg": cfg,
            }
            torch.save(ckpt_obj, os.path.join(out_dir, "last.pth"))
            if mae_hu < best_mae:
                best_mae = mae_hu
                ckpt_obj["best_mae"] = best_mae
                torch.save(ckpt_obj, os.path.join(out_dir, "best.pth"))
                logger.info(f"new best MAE: {best_mae:.2f} → saved best.pth")
            if save_every > 0 and (epoch + 1) % save_every == 0:
                snap_path = os.path.join(out_dir, f"checkpoint_epoch_{epoch+1:04d}.pth")
                torch.save(ckpt_obj, snap_path)
                logger.info(f"saved periodic snapshot: {os.path.basename(snap_path)}")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss_avg)
        history["val_loss"].append(val_mae_normalized)
        history["epoch_time"].append(dt)
        history["lr"].append(cur_lr)
        try:
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f)
            render_progress_png(
                progress_png,
                history["epoch"],
                history["train_loss"],
                [v if v == v else None for v in history["val_loss"]],  # NaN -> None for matplotlib
                history["epoch_time"],
                history["lr"],
            )
        except Exception as e:
            logger.warning(f"progress render failed: {e!r}")

    writer.close()
    logger.info(f"Training done. Best val MAE(HU): {best_mae:.2f}")


if __name__ == "__main__":
    main()
