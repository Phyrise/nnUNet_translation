"""Batch inference: run infer over an entire directory of CBCT NIfTI files.

Loads the U-Net once and reuses it for all cases — avoids the per-case model
load that infer.py does.
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from .infer import build_25d_input
from .model import build_unet
from .utils import denormalize_ct, get_logger, load_config, normalize_cbct


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--input-dir", required=True, help="folder of CBCT *_0000.nii.gz")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--pattern", default="*_0000.nii.gz")
    p.add_argument("--limit", type=int, default=None, help="optional cap on cases")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger("batch_infer", os.path.join(args.output_dir, "batch_infer.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    model = build_unet(cfg).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    logger.info(f"Loaded {args.ckpt}")
    if isinstance(ckpt, dict):
        logger.info(
            f"  ckpt epoch={ckpt.get('epoch')} best_mae={ckpt.get('best_mae')}"
        )

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No files matched {args.pattern} in {args.input_dir}")
    logger.info(f"Cases to infer: {len(files)}")

    num_adj = cfg["slicing"]["num_adjacent"]
    patch = tuple(cfg["slicing"]["patch_size"])
    sw_overlap = cfg["inference"]["sw_overlap"]
    sw_bs = cfg["inference"]["sw_batch_size"]
    ct_clip = (cfg["normalization"]["ct_clip_min"], cfg["normalization"]["ct_clip_max"])
    air = cfg["normalization"]["cbct_air_threshold"]

    t_start = time.time()
    with torch.no_grad():
        for i, fpath in enumerate(files, 1):
            case_id = os.path.basename(fpath).replace("_0000.nii.gz", "")
            out_path = os.path.join(args.output_dir, f"{case_id}.nii.gz")
            t0 = time.time()
            nii = nib.load(fpath)
            vol = nii.get_fdata().astype(np.float32)
            vol_norm = normalize_cbct(vol, air)
            stack = build_25d_input(vol_norm, num_adj)  # (D, C, H, W)

            out_slices = np.empty(
                (stack.shape[0], stack.shape[2], stack.shape[3]), dtype=np.float32
            )
            for z in range(stack.shape[0]):
                x = torch.from_numpy(stack[z][None]).to(device)
                pred = sliding_window_inference(
                    inputs=x,
                    roi_size=patch,
                    sw_batch_size=sw_bs,
                    predictor=model,
                    overlap=sw_overlap,
                    mode="gaussian",
                )
                out_slices[z] = pred[0, 0].clamp(-1, 1).cpu().numpy()

            sct_norm = np.transpose(out_slices, (1, 2, 0))
            sct_hu = denormalize_ct(sct_norm, ct_clip[0], ct_clip[1]).astype(np.int16)
            out_nii = nib.Nifti1Image(sct_hu, affine=nii.affine, header=nii.header)
            nib.save(out_nii, out_path)
            dt = time.time() - t0
            logger.info(
                f"[{i}/{len(files)}] {case_id} shape={vol.shape} done in {dt:.1f}s"
            )
    total = time.time() - t_start
    logger.info(f"Total: {len(files)} cases in {total:.1f}s ({total/len(files):.1f}s/case)")


if __name__ == "__main__":
    main()
