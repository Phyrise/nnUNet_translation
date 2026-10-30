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

Check an installation (about 10 minutes on CPU): `python -m nnunetv2.tests.translation_smoke_test`

## Encoder AFP and nnFoundation initialization (ISBI 2027)

Options validated in our ISBI 2027 paper (3D pre- to post-contrast breast MRI synthesis, MAMA-MIA). The exact paper
results come from the code at the tag [`isbi2027-results`](https://github.com/Phyrise/nnUNet_translation/tree/isbi2027-results)
and its own training loop, to be released in the upcoming ISBI2027 repository. The trainers below expose the same
components in the nnU-Net pipeline.

| Trainer | Loss | Generator encoder | Learning rate |
|---|---|---|---|
| `nnUNetTrainerMRCT_AFPexpert` | L1 + λ AFP, segmentation expert | random | 1e-2, 5-epoch warm-up |
| `nnUNetTrainerMRCT_AFPnnF` | L1 + λ AFP, nnFoundation encoder | random | 1e-2, 5-epoch warm-up |
| `nnUNetTrainerMRCT_nnFinit` | L1 | nnFoundation | 1e-3, 15-epoch warm-up |
| `nnUNetTrainerMRCT_AFPexpert_nnFinit` | L1 + λ AFP, segmentation expert | nnFoundation | 1e-3, 15-epoch warm-up |

`nnUNetTrainerMRCT_AFPexpert_nnFinit` is the best configuration of the paper. The warm-up is linear, followed by the
usual poly decay; SGD and the 1000 epochs are those of the other trainers. Each trainer has a `_300epochs` variant with
the 300 epochs of the paper (see [Paper recipe](#paper-recipe-300-epochs)).

- **Encoder AFP.** D = Σ_ℓ w_ℓ · mean |Φ_ℓ(Ŷ) − Φ_ℓ(Y)| over the **encoder stages only** of a frozen teacher; the
  prediction and the target are z-scored separately before the teacher. Stage weights: `nnUNet_AFP_stage_weights`
  (comma-separated, shallowest stage first, default all 1). `nnUNetTrainerMRCT_AFP` is unchanged (encoder and decoder
  features, no z-score).
- **Teachers.** *Expert*: an nnU-Net v2 segmentation model of the target modality, `nnUNet_AFP_model` as for
  `nnUNetTrainerMRCT_AFP` (a network trained outside nnU-Net can be exported, see below). *nnF*: the self-supervised
  nnFoundationCNN ResEnc-L encoder, `nnUNet_nnF_model` = its nnssl `checkpoint_final.pth`
  ([MIC-DKFZ/nnFoundationCNN](https://huggingface.co/MIC-DKFZ/nnFoundationCNN), CC-BY-SA-4.0).
- **λ** (`nnUNet_AFP_lambda`, required): a number, or the json written by the calibration, which matches the gradient
  norms of both terms at initialization: λ = median ‖∇L1‖ / median ‖∇D‖ over 64 fixed training batches. As in the
  paper, calibrate once per teacher at the random initialization and reuse the value for the `_nnFinit` trainer.
- **nnFoundation initialization** (`nnUNet_nnF_model`): the encoder tensors are copied exactly into the generator, the
  decoder keeps its random initialization. The plans must give the generator the ResEnc-L encoder (6 stages, features
  32-64-128-256-320-320, blocks 1-3-4-6-6-6, 3×3×3 kernels, one input channel), as `nnResUNetPlans` does for 3D patches
  large enough for 6 stages; otherwise training stops with the list of differences. Strides may differ: the shapes
  still match, but the copied filters then see another geometry than in pre-training, and a warning is logged. This was
  the case in the paper (last stride (1, 2, 2) instead of (2, 2, 2)).

```bash
export nnUNet_AFP_model=/path/to/segmentation_model/fold_0/checkpoint_final.pth
export nnUNet_nnF_model=/path/to/nnFoundationCNN/checkpoint_final.pth
nnUNetv2_translation_calibrate_afp 50 3d_fullres 0 -tr nnUNetTrainerMRCT_AFPexpert -p nnResUNetPlans
export nnUNet_AFP_lambda=$nnUNet_results/Dataset050_PelvisMRCT/nnUNetTrainerMRCT_AFPexpert__nnResUNetPlans__3d_fullres/fold_0/afp_lambda.json
nnUNetv2_train 50 3d_fullres 0 -tr nnUNetTrainerMRCT_AFPexpert_nnFinit -p nnResUNetPlans
```

Inference needs none of these variables. Equivalence with the tag (loss, gradients, initialization, learning rate,
teacher export): `python -m nnunetv2.tests.afp_nnf_equivalence_test` (about a minute on CPU, needs the tag:
`git fetch --tags`).

**Teacher export.** A segmentation network trained outside nnU-Net (a state_dict, such as the ResEnc-L experts of the
paper) becomes an nnU-Net model folder usable as `nnUNet_AFP_model`:

```bash
nnUNetv2_translation_export_teacher -i expert/checkpoint_ep300.pth --preset resenc_l --num_classes 2 \
    --modality first_postcontrast --trusted -o teachers/expert_scratch
export nnUNet_AFP_model=$PWD/teachers/expert_scratch/fold_0/checkpoint_final.pth
```

`--preset resenc_l` is the ResEnc-L of nnFoundation; any dynamic_network_architectures network is accepted with
`--architecture arch.json` (the `architecture` entry of nnU-Net plans: `network_class_name`, `arch_kwargs`,
`_kw_requires_import`). The state_dict is read from `--key` (default `network_weights`, `''` if the file is the
state_dict itself). Every key and shape must match the architecture, otherwise the export stops and lists the missing,
unexpected and mismatched keys. The source path and its sha256 are written to `teacher_export.json` and into the
checkpoint. `--trusted` allows `torch.load(weights_only=False)`, which checkpoints holding optimizer or RNG states
need: use it only for files you trust.

### Paper recipe (300 epochs)

| Table 1 arm | Trainer | Variables | λ of the paper |
|---|---|---|---|
| L1 | `nnUNetTrainerMRCT_warmup_300epochs` | none | |
| L1 + AFP_seg | `nnUNetTrainerMRCT_AFPexpert_300epochs` | `nnUNet_AFP_model` = scratch expert (exported) | 0.1264 |
| L1 + AFP_nnF | `nnUNetTrainerMRCT_AFPnnF_300epochs` | `nnUNet_nnF_model` | 11.78 |
| M2 | `nnUNetTrainerMRCT_AFPexpert_nnFinit_300epochs` | `nnUNet_AFP_model` = scratch expert, `nnUNet_nnF_model` | 0.1264 |

`nnUNetTrainerMRCT_warmup_300epochs` is the L1 trainer with the common recipe of the paper: SGD 1e-2, 5-epoch linear
warm-up, poly decay to epoch 300. λ was calibrated at the random initialization of the paper generator, and the
AFP_seg value was reused for M2; for other data, plans or teachers, run the calibration. Also 300 epochs:
`nnUNetTrainerMRCT_nnFinit_300epochs` (L1, nnF initialization, 1e-3 with a 15-epoch warm-up), which matches no
Table 1 arm. The paper arms with the nnF initialization and L1 or L1 + AFP_nnF (`l1_nnf`, `ssl_nnf`) used the
1e-2 / 5-epoch recipe, which no trainer here provides.

What still differs from the paper runs:
- **Patches.** Patch and batch size come from the plans. The paper used 96×160×160 patches, batch 2, and its own
  sampler (one patch in three centred on a lesion voxel, paired random flips along the three axes); the trainers here
  have no augmentation.
- **No validity mask.** The paper computed the L1 only inside the image, and gave the teacher the target outside it;
  here both terms see the whole patch.
- **Teacher precision.** The paper teacher had float16 weights; here the teacher keeps float32 weights and runs under
  the trainer's autocast.

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
5. **nnFoundation** (nnFoundation teacher and initialization) — Ulrich Harsy C., Wald T., Gotkowski K. et al.
   *nnFoundation: 3D Foundation Models for Radiology.* [arXiv:2609.26924](https://arxiv.org/abs/2609.26924), 2026.

Contact: arthur.longuefosse [at] gmail.com
