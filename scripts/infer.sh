#!/usr/bin/env bash
# Usage: ./scripts/infer.sh path/to/CBCT.nii.gz path/to/sCT.nii.gz
set -euo pipefail
python -m src.infer --config configs/default.yaml --ckpt outputs/best.pth --input "$1" --output "$2"
