#!/usr/bin/env bash
# Run sCT U-Net inference on the hold-out CBCT folder, then evaluate with val.py.
# Hard-codes the user-specified folders so the workflow is one command.

set -euo pipefail

ROOT="C:/Users/USER/Desktop/dev/sct_unet"
PYTHON="C:/Users/USER/anaconda3/envs/nnunet/python.exe"
VAL_PY="C:/Users/USER/Desktop/dev/nnUNet_translation/val.py"

CKPT="${ROOT}/outputs/full_0429/best.pth"
CONFIG="${ROOT}/configs/default.yaml"

INPUT_DIR="${ROOT}/outputs/inference input"
OUTPUT_DIR="${ROOT}/outputs/inference output"
GT_DIR="${ROOT}/outputs/new_gt"
METRICS_DIR="${ROOT}/outputs/metrics_holdout60"

mkdir -p "${OUTPUT_DIR}" "${METRICS_DIR}"

echo "========== batch inference =========="
cd "${ROOT}"
"${PYTHON}" -m src.batch_infer \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    | tee "${ROOT}/outputs/batch_infer_holdout60.log"

echo "========== val.py =========="
"${PYTHON}" "${VAL_PY}" \
    --pred-dir "${OUTPUT_DIR}" \
    --gt-image-dir "${GT_DIR}" \
    --save-dir "${METRICS_DIR}" \
    | tee "${ROOT}/outputs/val_holdout60.log"

echo "[DONE] metrics at ${METRICS_DIR}/metrics.json"
