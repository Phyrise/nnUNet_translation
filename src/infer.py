"""Inference: CBCT NIfTI → sCT NIfTI (slice-by-slice, 2.5D U-Net)."""
from __future__ import annotations
import argparse
import os

import numpy as np
import nibabel as nib
import torch
from monai.inferers import sliding_window_inference

from .utils import load_config, get_logger, normalize_cbct, denormalize_ct
from .model import build_unet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--ckpt", required=True, help="checkpoint .pth")
    p.add_argument("--input", required=True, help="CBCT NIfTI file")
    p.add_argument("--output", required=True, help="output sCT NIfTI path")
    return p.parse_args()


def build_25d_input(vol_norm: np.ndarray, num_adjacent: int) -> np.ndarray:
    """vol_norm: (H, W, D) → (D, num_adjacent, H, W)."""
    half = num_adjacent // 2
    D = vol_norm.shape[-1]
    out = np.empty((D, num_adjacent, vol_norm.shape[0], vol_norm.shape[1]), dtype=np.float32)
    for z in range(D):
        for ci, dz in enumerate(range(-half, half + 1)):
            zi = min(max(z + dz, 0), D - 1)
            out[z, ci] = vol_norm[..., zi]
    return out


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger("infer")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = build_unet(cfg).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    logger.info(f"Loaded {args.ckpt}")

    nii = nib.load(args.input)
    vol = nii.get_fdata().astype(np.float32)
    vol_norm = normalize_cbct(vol, cfg["normalization"]["cbct_air_threshold"])
    num_adj = cfg["slicing"]["num_adjacent"]
    stack = build_25d_input(vol_norm, num_adj)  # (D, C, H, W)

    patch = tuple(cfg["slicing"]["patch_size"])
    sw_overlap = cfg["inference"]["sw_overlap"]
    sw_bs = cfg["inference"]["sw_batch_size"]

    out_slices = np.empty((stack.shape[0], stack.shape[2], stack.shape[3]), dtype=np.float32)
    with torch.no_grad():
        for z in range(stack.shape[0]):
            x = torch.from_numpy(stack[z][None]).to(device)  # (1, C, H, W)
            pred = sliding_window_inference(
                inputs=x,
                roi_size=patch,
                sw_batch_size=sw_bs,
                predictor=model,
                overlap=sw_overlap,
                mode="gaussian",
            )
            out_slices[z] = pred[0, 0].clamp(-1, 1).cpu().numpy()

    sct_norm = np.transpose(out_slices, (1, 2, 0))  # (H, W, D)
    sct_hu = denormalize_ct(
        sct_norm,
        cfg["normalization"]["ct_clip_min"],
        cfg["normalization"]["ct_clip_max"],
    ).astype(np.int16)

    out_nii = nib.Nifti1Image(sct_hu, affine=nii.affine, header=nii.header)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    nib.save(out_nii, args.output)
    logger.info(f"Saved sCT → {args.output}")


if __name__ == "__main__":
    main()
