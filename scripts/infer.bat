@echo off
REM Usage: scripts\infer.bat path\to\CBCT.nii.gz path\to\sCT.nii.gz
python -m src.infer --config configs/default.yaml --ckpt outputs/best.pth --input %1 --output %2
