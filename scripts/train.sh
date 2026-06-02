#!/usr/bin/env bash
# Run from sct_unet/ root.
set -euo pipefail
python -m src.train --config configs/default.yaml "$@"
