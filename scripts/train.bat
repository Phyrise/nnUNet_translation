@echo off
REM Run from sct_unet/ root.
python -m src.train --config configs/default.yaml %*
