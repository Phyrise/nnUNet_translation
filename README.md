# nnU-Net translation

[nnU-Net](https://github.com/MIC-DKFZ/nnUNet) adapted for medical image-to-image translation (e.g. MR → synthetic CT):
the self-configuring nnU-Net pipeline, with a regression output and translation losses (L1, AFP).

## Install

```bash
# use a dedicated environment: this package replaces the original nnunetv2
git clone https://github.com/Phyrise/nnUNet_translation && cd nnUNet_translation
pip install -e .
export nnUNet_raw=/data/nnUNet_raw nnUNet_preprocessed=/data/nnUNet_preprocessed nnUNet_results=/data/nnUNet_results
```

## Usage

```bash
# 1. paired images -> ready-to-train dataset (uses IDs 50 and 51)
nnUNetv2_translation_prepare -d 50 -n PelvisMRCT -i "data/*/mr.nii.gz" -t "data/*/ct.nii.gz"

# 2. train fold 0
nnUNetv2_train 50 3d_fullres 0 -tr nnUNetTrainerMRCT -p nnResUNetPlans

# 3. predict (inputs named CASE_0000.nii.gz): synthetic CT in HU, same grid as the input
nnUNetv2_predict -d 50 -c 3d_fullres -p nnResUNetPlans -tr nnUNetTrainerMRCT -f 0 -i INPUT -o OUTPUT -step_size 0.3
```

- `-i` / `-t`: a folder (same file names in each) or a pattern where `*` is the case name. Inputs and target must be
  co-registered on the same voxel grid. Several input channels: repeat `-i`, e.g.
  `-i "data/*/mr.nii.gz" -i "data/*/mask.nii.gz" --input_norm zscore noNorm`.
- Target normalization: `--target_norm CT_clip` (default: clip to [-1024, 3071] HU, dataset z-score; predictions are
  written back in HU). For an MR target use `zscore` (predictions stay z-scored).
- Default architecture: residual U-Net with trilinear upsampling (`nnUNetPlannerResUNet`). Other planners via `-pl`.
- Inference: `-step_size 0.3` reduces patch artifacts; `--rec median` takes the voxel-wise median of overlapping
  patches instead of the mean (more RAM).

## Trainers

| Trainer | Loss | Notes |
|---|---|---|
| `nnUNetTrainerMRCT` | L1 | default |
| `nnUNetTrainerMRCT_mse` | L2 | |
| `nnUNetTrainerMRCT_AFP` | AFP + L1 | needs `export nnUNet_AFP_model=.../checkpoint_final.pth` |
| `nnUNetTrainerMRCT_AFP_ft` | AFP + 0.5 L1 | fine-tunes an L1 model: `-pretrained_weights .../checkpoint_final.pth`, 500 epochs, lr 1e-3 |

All trainers: 1000 epochs, no data augmentation, no mirroring; `checkpoint_best.pth` has the lowest validation loss
(`-chk checkpoint_best.pth` at inference). The AFP model is any nnU-Net v2 segmentation model of the target modality
(e.g. trained on [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) labels).
SynthRAD2025 recipe: `nnUNetTrainerMRCT`, then `nnUNetTrainerMRCT_AFP_ft` from its checkpoint.

Check an installation (a few minutes on CPU): `python -m nnunetv2.tests.translation_smoke_test`

## Citation

1. **nnU-Net translation** — Longuefosse A. et al. *Adapted nnU-Net: A Robust Baseline for Cross-Modality Synthesis
   and Medical Image Inpainting.* SASHIMI workshop, MICCAI 2024, LNCS 15187, pp. 24–33.
2. **AFP loss** — Longuefosse A. et al. *Anatomical feature-prioritized loss for enhanced MR to CT translation.*
   Physics in Medicine & Biology, 2025. [doi:10.1088/1361-6560/adea07](https://doi.org/10.1088/1361-6560/adea07)
3. **SynthRAD2025**
   - Longuefosse A., Sequeiro Gonzalez J., Baldacci F., Seita J. *From Intensity to Anatomy: Optimizing nnU-Net for
     MR-to-CT Translation on SynthRAD2025.* ISBI 2026.
   - Sequeiro Gonzalez J., Longuefosse A. et al. *Deep Learning-Based Cross-Anatomy CT Synthesis Using Adapted
     nnResU-Net with Anatomical Feature Prioritized Loss.* [arXiv:2509.22394](https://arxiv.org/abs/2509.22394), 2025.
   - Rogowski V. et al. *Generating synthetic computed tomography for radiotherapy: SynthRAD2025 challenge report.*
     Medical Image Analysis, 2026. [arXiv:2605.13555](https://arxiv.org/abs/2605.13555)
4. **nnU-Net** — Isensee F. et al. *nnU-Net: a self-configuring method for deep learning-based biomedical image
   segmentation.* Nature Methods 18, 203–211 (2021).

Contact: arthur.longuefosse [at] gmail.com
