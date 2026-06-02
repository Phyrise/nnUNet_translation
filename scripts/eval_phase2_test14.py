"""sct_unet 시도 4 best.pth 로 phase2 test 14 case inference + phase2 metric 평가.

목적: amed plan unet0601.md § 6.10 (a)-(e) — fair 비교 set 측정.
- phase2 test 14 case = 2HNE004, 024, 030, 034, 038, 043, 054, 059, 060, 070, 090, 092, 097, 108
- sct_unet train 에 2HNE004 포함 → 13 case (004 제외) 가 양방 외부 = 완전 fair
- 14 case 전체도 측정 (004 caveat 명시)

입력: nnUNet_translation/data/HN/{case}/{cbct.mha, ct.mha, mask.mha}
출력:
- outputs/eval_phase2_test14/{case}_sct.nii.gz  (sct_unet 예측 sCT, HU int16)
- outputs/eval_phase2_test14/metrics_per_case.json
- outputs/eval_phase2_test14/summary.json

CPU only — CUDA_VISIBLE_DEVICES="" 환경변수로 강제 가능.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
from monai.inferers import sliding_window_inference

# sct_unet 모듈 path
SCT_UNET_ROOT = Path(r"C:/Users/USER/Desktop/dev/sct_unet")
sys.path.insert(0, str(SCT_UNET_ROOT))

from src.infer import build_25d_input
from src.model import build_unet
from src.utils import denormalize_ct, load_config, normalize_cbct

# phase2 평가 함수 — sct_unet 의 src 와 namespace 충돌 회피 위해 direct file import
import importlib.util
PHASE2_ROOT = Path(r"C:/Users/USER/Desktop/dev/phase2_unet_weighted")
_phase2_metrics_path = PHASE2_ROOT / "src" / "eval" / "metrics.py"
_spec = importlib.util.spec_from_file_location("phase2_metrics", _phase2_metrics_path)
_phase2_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_phase2_metrics)
evaluate_volume = _phase2_metrics.evaluate_volume

# 경로
HN_ROOT = Path(r"C:/Users/USER/Desktop/dev/nnUNet_translation/data/HN")
CFG_PATH = SCT_UNET_ROOT / "configs" / "full_v3_no_amp.yaml"
CKPT_PATH = SCT_UNET_ROOT / "outputs" / "full_v3_no_amp" / "best.pth"
OUT_DIR = SCT_UNET_ROOT / "outputs" / "eval_phase2_test14"

# phase2 test 14 case (split CSV 인용)
TEST_CASES = [
    "2HNE004", "2HNE024", "2HNE030", "2HNE034", "2HNE038",
    "2HNE043", "2HNE054", "2HNE059", "2HNE060", "2HNE070",
    "2HNE090", "2HNE092", "2HNE097", "2HNE108",
]
# sct_unet train 에 포함된 case (004) — fair 13 case 와 14 case caveat 표시
SCT_UNET_TRAIN_OVERLAP = {"2HNE004"}


def load_sct_unet_model(cfg, ckpt_path, device):
    model = build_unet(cfg).to(device).eval()
    # map_location 으로 강제 CPU 로 unpickle 한 후 .to(device) — CUDA env 없을 때도 안전
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    print(f"  ckpt epoch={ckpt.get('epoch') if isinstance(ckpt, dict) else 'N/A'}, "
          f"best_mae={ckpt.get('best_mae') if isinstance(ckpt, dict) else 'N/A'}", flush=True)
    return model


def infer_one(model, cbct_zyx, cfg, device):
    """phase2 의 .mha layout (z, y, x) cbct_HU → sct_HU (z, y, x).

    내부적으로 sct_unet 의 nibabel layout (x, y, z) 변환 후 inference,
    결과를 다시 (z, y, x) 로 변환.
    """
    num_adj = cfg["slicing"]["num_adjacent"]
    patch = tuple(cfg["slicing"]["patch_size"])
    sw_overlap = cfg["inference"]["sw_overlap"]
    sw_bs = cfg["inference"]["sw_batch_size"]
    ct_clip = (cfg["normalization"]["ct_clip_min"], cfg["normalization"]["ct_clip_max"])
    air = cfg["normalization"]["cbct_air_threshold"]

    # (z, y, x) → (x, y, z) for sct_unet nibabel layout
    cbct_xyz = np.transpose(cbct_zyx, (2, 1, 0)).astype(np.float32)
    cbct_norm = normalize_cbct(cbct_xyz, air)
    stack = build_25d_input(cbct_norm, num_adj)  # (D, C, H, W) where D = z

    out_slices = np.empty((stack.shape[0], stack.shape[2], stack.shape[3]), dtype=np.float32)
    with torch.no_grad():
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

    # out_slices: (D, H, W) = (z, x_or_y, x_or_y)
    # sct_unet 의 batch_infer 는 (out_slices[z] = pred[0,0]) 인데 pred 는 (1, 1, H, W) → (H, W).
    # stack 은 (D, C, H, W) where (H, W) = (W_orig, X_orig?). build_25d_input 확인:
    # build_25d_input(vol_norm, num_adj) 에서 vol_norm 은 (H, W, D) — nibabel x,y,z 의 (H=y, W=x, D=z)?
    # batch_infer.py L72: out_slices[z] = pred[0,0]; sct_norm = np.transpose(out_slices, (1, 2, 0)) — (H, W, D)
    # → out_slices.shape = (D, H, W), sct_norm.shape = (H, W, D)
    # nibabel 저장 시 affine 으로 spatial 보존. 본 함수에서는 (z, y, x) → np.transpose((2,1,0)) → (x, y, z).
    # build_25d_input 은 마지막 axis 가 z 라고 가정 → (x, y, z) input 의 D=z 슬라이스
    # out_slices = (D, H, W) = (z, y, x)
    # → 본 (z, y, x) 가 바로 phase2 layout 과 일치 (운 좋게)
    # 단 확인: sct_unet 의 nibabel input (H, W, D)=(x, y, z) 의 H=x, W=y. _get_stack 는 vol[..., i] 슬라이스 (axis=-1=z).
    # → 슬라이스 차원이 (x, y) → pred output 도 (x, y). out_slices = (z, x, y).
    # phase2 mask (z, y, x) 와 out_slices (z, x, y) 의 last 2 axes 가 다름.
    # 따라서 다시 transpose 필요: out_slices (z, x, y) → (z, y, x) via swapaxes(-1, -2).
    out_slices_zyx = np.swapaxes(out_slices, -1, -2)  # (z, x, y) → (z, y, x)
    sct_hu = denormalize_ct(out_slices_zyx, ct_clip[0], ct_clip[1])
    return sct_hu  # (z, y, x), HU


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "eval.log"
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    # CPU 강제 (사용자 결정 — GPU sleep + GPU 충돌 회피)
    # CUDA_VISIBLE_DEVICES env var 가 Windows subprocess 에 전파 안 되는 이슈 우회
    device = torch.device("cpu")
    log(f"=== sct_unet 시도 4 best.pth → phase2 test 14 case 평가 ===")
    log(f"  device: {device} (강제 CPU)")
    log(f"  cfg: {CFG_PATH}")
    log(f"  ckpt: {CKPT_PATH}")
    log(f"  out: {OUT_DIR}")
    log(f"  cases: {TEST_CASES}")
    log("")

    cfg = load_config(str(CFG_PATH))
    model = load_sct_unet_model(cfg, str(CKPT_PATH), device)

    results = []
    t_all = time.time()
    for i, cid in enumerate(TEST_CASES, 1):
        case_dir = HN_ROOT / cid
        cbct_path = case_dir / "cbct.mha"
        ct_path = case_dir / "ct.mha"
        mask_path = case_dir / "mask.mha"
        if not (cbct_path.exists() and ct_path.exists() and mask_path.exists()):
            log(f"[{i:2d}/14] SKIP {cid}: file missing in {case_dir}")
            continue
        t0 = time.time()

        # load (.mha → numpy z,y,x)
        cbct_sitk = sitk.ReadImage(str(cbct_path))
        cbct_zyx = sitk.GetArrayFromImage(cbct_sitk).astype(np.float32)
        ct_zyx = sitk.GetArrayFromImage(sitk.ReadImage(str(ct_path))).astype(np.float32)
        mask_zyx = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))).astype(bool)

        # inference
        sct_zyx = infer_one(model, cbct_zyx, cfg, device)  # (z, y, x), HU

        # shape sanity
        if sct_zyx.shape != ct_zyx.shape:
            log(f"[{i:2d}/14] SHAPE MISMATCH {cid}: sct={sct_zyx.shape} vs ct={ct_zyx.shape}")
            # 비교 불가, skip
            continue

        # save sCT (.nii.gz, identity affine for now — phase2 grading 에 affine 불필요)
        out_sct_path = OUT_DIR / f"{cid}_sct.nii.gz"
        sct_int16 = sct_zyx.astype(np.int16)
        # SimpleITK 로 저장하면 spatial 정보 보존 가능
        sct_img = sitk.GetImageFromArray(sct_int16)
        sct_img.CopyInformation(cbct_sitk)
        sitk.WriteImage(sct_img, str(out_sct_path.with_suffix("").with_suffix(".mha")))

        # phase2 metric
        # evaluate_volume(sct_hu, gt_hu, body_mask) — (z, y, x) layout
        metrics = evaluate_volume(sct_zyx.astype(np.int16), ct_zyx.astype(np.int16), mask_zyx)
        metrics["case"] = cid
        metrics["sct_unet_train_overlap"] = cid in SCT_UNET_TRAIN_OVERLAP
        results.append(metrics)

        dt = time.time() - t0
        log(f"[{i:2d}/14] {cid}: MAE_w={metrics['mae_whole']:.2f} "
            f"MAE_soft={metrics['mae_soft']:.2f} MAE_bone={metrics['mae_bone']:.2f} "
            f"SSIM_soft={metrics['ssim_soft']:.4f} SSIM_bone={metrics['ssim_bone']:.4f} "
            f"PSNR={metrics['psnr']:.2f}  ({dt:.1f}s)  "
            f"{'[OVERLAP]' if metrics['sct_unet_train_overlap'] else ''}")

    total_min = (time.time() - t_all) / 60
    log(f"\n=== DONE in {total_min:.1f} min ===")

    # save per-case JSON
    (OUT_DIR / "metrics_per_case.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    # summary 14 case + 13 case fair
    def agg(rows, keys):
        out = {}
        for k in keys:
            vals = np.array([r[k] for r in rows if r[k] is not None and np.isfinite(r[k])])
            if vals.size == 0:
                out[f"{k}_mean"] = None; out[f"{k}_std"] = None
            else:
                out[f"{k}_mean"] = float(vals.mean())
                out[f"{k}_std"] = float(vals.std())
        return out

    metric_keys = ["mae_whole", "mae_soft", "mae_bone", "ssim_soft", "ssim_bone", "psnr"]
    summary = {
        "n_cases_14": len(results),
        "summary_14_case_with_004_caveat": agg(results, metric_keys),
        "n_cases_13_fair": len([r for r in results if not r["sct_unet_train_overlap"]]),
        "summary_13_case_fair": agg([r for r in results if not r["sct_unet_train_overlap"]], metric_keys),
        "model": "sct_unet_v3_no_amp_best.pth (시도 4)",
        "ckpt": str(CKPT_PATH),
        "config": str(CFG_PATH),
        "device": str(device),
        "total_time_min": total_min,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    log("\n--- summary 14-case (incl. 2HNE004 overlap caveat) ---")
    s14 = summary["summary_14_case_with_004_caveat"]
    for k in metric_keys:
        log(f"  {k}_mean={s14[f'{k}_mean']:.4f} (std={s14[f'{k}_std']:.4f})")
    log("\n--- summary 13-case fair (2HNE004 제외) ---")
    s13 = summary["summary_13_case_fair"]
    for k in metric_keys:
        log(f"  {k}_mean={s13[f'{k}_mean']:.4f} (std={s13[f'{k}_std']:.4f})")


if __name__ == "__main__":
    main()
