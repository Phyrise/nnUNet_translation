"""Post-train pipeline for sCT U-Net (full_v2): wait until 'Training done.'
appears in outputs/full_v2/train.log, then run batch_infer on the hold-out
60 cases + val.py + append results section to unet procceding1.md.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r"C:/Users/USER/Desktop/dev/sct_unet")
PY = Path(r"C:/Users/USER/anaconda3/envs/nnunet/python.exe")
VAL_PY = Path(r"C:/Users/USER/Desktop/dev/nnUNet_translation/val.py")
CONFIG = ROOT / "configs" / "full_v2.yaml"

OUT_DIR = ROOT / "outputs" / "full_v2"
TRAIN_LOG = OUT_DIR / "train.log"
BEST_CKPT = OUT_DIR / "best.pth"

INPUT_DIR = ROOT / "outputs" / "inference input"
GT_DIR = ROOT / "outputs" / "new_gt"
# User-specified: input under outputs/inference input, gt under outputs/new_gt,
# output dumped under outputs/inference output (v1 .nii.gz files are overwritten;
# v1 metrics already preserved in unet procceding.md).
PRED_OUT = ROOT / "outputs" / "inference output"
METRICS_OUT = ROOT / "outputs" / "metrics_v2_holdout60"

LOG_DIR = ROOT / "scripts" / "watchdog_v2_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
POST_LOG = LOG_DIR / "post_train.log"

PROGRESS_MD = ROOT / "unet procceding1.md"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(POST_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def wait_for_training_done(poll_secs: int = 60) -> None:
    log(f"polling for 'Training done.' in {TRAIN_LOG} every {poll_secs}s")
    while True:
        if TRAIN_LOG.exists():
            try:
                txt = TRAIN_LOG.read_text(encoding="utf-8", errors="ignore")
                if "Training done." in txt:
                    log("training complete marker found")
                    return
            except OSError:
                pass
        time.sleep(poll_secs)


def run_predict() -> None:
    PRED_OUT.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PY), "-m", "src.batch_infer",
        "--config", str(CONFIG),
        "--ckpt", str(BEST_CKPT),
        "--input-dir", str(INPUT_DIR),
        "--output-dir", str(PRED_OUT),
    ]
    log("running batch_infer: " + " ".join(cmd))
    out_path = LOG_DIR / "predict_v2.log"
    with open(out_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
    log(f"batch_infer exit code = {proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError(f"batch_infer failed (rc={proc.returncode}); see {out_path}")


def run_val() -> None:
    METRICS_OUT.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PY), str(VAL_PY),
        "--pred-dir", str(PRED_OUT),
        "--gt-image-dir", str(GT_DIR),
        "--save-dir", str(METRICS_OUT),
    ]
    log("running val.py: " + " ".join(cmd))
    out_path = LOG_DIR / "val_v2.log"
    with open(out_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    log(f"val.py exit code = {proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError(f"val.py failed (rc={proc.returncode}); see {out_path}")


def append_results_to_markdown() -> None:
    metrics_path = METRICS_OUT / "metrics.json"
    if not metrics_path.exists():
        log(f"ERROR: {metrics_path} not found, skipping markdown update")
        return
    d = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = d["details"]

    def stats(key):
        vals = [r[key] for r in rows if r[key] is not None and np.isfinite(r[key])]
        if not vals:
            return None
        a = np.array(vals)
        return dict(n=len(vals), total=len(rows), mean=float(a.mean()),
                    std=float(a.std()), median=float(np.median(a)),
                    min=float(a.min()), max=float(a.max()))

    keys = ["MAE_HU", "PSNR_dB", "SSIM", "MS_SSIM", "DICE_bone", "HD95_bone_mm"]
    s = {k: stats(k) for k in keys}
    rows_sorted = sorted(rows, key=lambda r: r["MAE_HU"])

    def fmt_row(r):
        return (
            f"| {r['case']} | {r['MAE_HU']:.2f} | {r['PSNR_dB']:.2f} | "
            f"{r['SSIM']:.4f} | {r['MS_SSIM']:.4f} | {r['DICE_bone']:.4f} | "
            f"{r['HD95_bone_mm']:.2f} |"
        )

    lines = ["\n---\n",
             "# 자동 결과 — sCT U-Net full_v2 (post_train_sct)\n",
             f"기록 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
             "\n## Hold-out 60 케이스 (2HNE010 ~ 2HNE109) — best.pth 평가\n",
             "| Metric | Mean | Std | Median | Min | Max | n_finite |",
             "|--------|-----:|----:|-------:|----:|----:|---------:|"]
    for k in keys:
        v = s[k]
        if v is None:
            lines.append(f"| {k} | N/A | N/A | N/A | N/A | N/A | 0/{len(rows)} |")
        else:
            lines.append(
                f"| {k} | {v['mean']:.4f} | {v['std']:.4f} | {v['median']:.4f} | "
                f"{v['min']:.4f} | {v['max']:.4f} | {v['n']}/{v['total']} |"
            )

    lines.append("\n## v1 (full_0429, 148 epoch stop) 와 비교\n")
    lines.append("| Metric | v1 best.pth (epoch 145) hold-out 60 | **v2 best.pth (200 epoch 완주) hold-out 60** |")
    lines.append("|--------|------------------------------------:|---------------------------------------------:|")
    refs_v1 = {
        "MAE_HU": "127.99",
        "PSNR_dB": "24.67",
        "SSIM": "0.9025",
        "MS_SSIM": "0.9021",
        "DICE_bone": "0.6999",
        "HD95_bone_mm": "5.01",
    }
    fmt_map = {"MAE_HU": "{:.2f}", "PSNR_dB": "{:.2f}", "SSIM": "{:.4f}",
               "MS_SSIM": "{:.4f}", "DICE_bone": "{:.4f}", "HD95_bone_mm": "{:.2f}"}
    for k in keys:
        v = s[k]
        new_str = "N/A" if v is None else fmt_map[k].format(v["mean"])
        lines.append(f"| {k} | {refs_v1[k]} | **{new_str}** |")

    lines.append("\n## Best 5 (lowest MAE_HU)\n")
    lines.append("| Case | MAE_HU | PSNR_dB | SSIM | MS_SSIM | DICE_bone | HD95_bone_mm |")
    lines.append("|------|-------:|--------:|-----:|--------:|----------:|-------------:|")
    for r in rows_sorted[:5]:
        lines.append(fmt_row(r))

    lines.append("\n## Worst 5 (highest MAE_HU)\n")
    lines.append("| Case | MAE_HU | PSNR_dB | SSIM | MS_SSIM | DICE_bone | HD95_bone_mm |")
    lines.append("|------|-------:|--------:|-----:|--------:|----------:|-------------:|")
    for r in rows_sorted[-5:][::-1]:
        lines.append(fmt_row(r))

    lines.append("\n## 산출물\n")
    lines.append(f"- 학습 가중치: `{BEST_CKPT}`")
    lines.append(f"- 학습 history: `{OUT_DIR / 'history.json'}`")
    lines.append(f"- 학습 progress 그래프: `{OUT_DIR / 'progress.png'}`")
    lines.append(f"- 인퍼런스 출력 (60): `{PRED_OUT}`")
    lines.append(f"- 메트릭 JSON: `{metrics_path}`")
    lines.append(f"- post_train 로그: `{POST_LOG}`")
    lines.append(f"- watchdog 로그: `{LOG_DIR / 'watchdog.log'}`\n")

    with open(PROGRESS_MD, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"appended results section to {PROGRESS_MD}")


def main() -> int:
    log("post-train pipeline (full_v2) started")
    wait_for_training_done()
    run_predict()
    run_val()
    try:
        append_results_to_markdown()
    except Exception as e:
        log(f"ERROR while appending results to markdown: {e!r}")
    log("post-train pipeline complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
