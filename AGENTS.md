# AGENTS.md — nnU-Net translation

Fork of nnU-Net v2 (2.5) for image-to-image translation (MR → CT, MR → MR, ...): the network regresses the target
image instead of predicting labels. README.md is the user documentation; `documentation/` is the original nnU-Net
documentation (segmentation) and does not describe the translation workflow.

## Workflow

```bash
export nnUNet_raw=... nnUNet_preprocessed=... nnUNet_results=...
nnUNetv2_translation_prepare -d 50 -n NAME -i "data/*/mr.nii.gz" -t "data/*/ct.nii.gz"
nnUNetv2_train 50 3d_fullres 0 -tr nnUNetTrainerMRCT -p nnResUNetPlans
nnUNetv2_predict -d 50 -c 3d_fullres -p nnResUNetPlans -tr nnUNetTrainerMRCT -f 0 -i IN -o OUT -step_size 0.3
python -m nnunetv2.tests.translation_smoke_test   # end-to-end check on synthetic data, CPU, a few minutes
```

Flags: plans are selected with `-p` (not `-pl`) in train and predict; `-pl` selects the planner in prepare.
`nnUNetv2_translation_prepare -h` lists every option. Prediction inputs are named `CASE_0000.nii.gz`,
`CASE_0001.nii.gz` (one file per input channel), with the file ending of the dataset (`.nii.gz` by default).

## How it works

- `prepare` (`nnunetv2/translation/prepare.py`) creates two datasets: `Dataset{ID}_{name}` (inputs, the one to train
  and predict with) and `Dataset{ID+1}_{name}Target` (targets, only used for preprocessing). Labels are dummy all-ones
  masks. The target dataset reuses the input plans so both are resampled onto the same grid, then each preprocessed
  target is written as the `seg` array of the input case. Trainers read the target from `batch['target']`.
- The target normalization and its intensity statistics are stored under `translation` in the input `dataset.json`,
  which is copied to the model folder. Exported predictions are converted back to target units with
  `nnunetv2/translation/intensity.py` (HU for `CT_clip`, `CT`, `CT_noclip`; per-image `zscore` cannot be inverted).
  Models whose dataset.json has no `translation` entry (older datasets) are exported unchanged.
- Images are read in LPS orientation (`imageio/simpleitk_reader_writer.py`) and written back in their original
  orientation. Cropping to the nonzero region is disabled on purpose (`preprocessing/cropping/cropping.py`) so input
  and target keep the same shape: do not re-enable it.
- Networks have one output channel (`nnUNetTrainer.build_network_architecture`). This fork therefore cannot train
  segmentation models: train the AFP segmentation model with the original nnU-Net.

## Where things are

| Path | Content |
|---|---|
| `nnunetv2/translation/` | dataset preparation CLI, intensity conversion |
| `nnunetv2/training/nnUNetTrainer/variants/translation/nnUNetTrainerMRCT.py` | all translation trainers |
| `nnunetv2/training/loss/AFP.py` | AFP loss; rebuilds any nnU-Net v2 checkpoint given by `nnUNet_AFP_model` |
| `nnunetv2/training/dataloading/data_loader_{2d,3d}.py` | `*_MRCT` loaders (float target, zero padding) |
| `nnunetv2/experiment_planning/experiment_planners/residual_unets/` | ResUNet planners (`nnResUNetPlans`) |
| `nnunetv2/inference/predict_from_raw_data.py` | sliding window, `rec_mean` / `rec_median` |
| `nnunetv2/inference/export_prediction.py` | resampling to the original geometry, conversion to target units |
| `nnunetv2/tests/translation_smoke_test.py` | end-to-end test |

## Changing the code

- New loss: subclass `nnUNetTrainerMRCT` and override `_build_loss` (called with `(output, target)`); trainers are
  found by class name anywhere under `nnunetv2/training/nnUNetTrainer/`, so names must be unique.
- Keep the existing trainer names: trained models store the trainer name and need it at inference.
- Run the smoke test after any change to preparation, training, inference or export.
- Use a dedicated environment: the package is named `nnunetv2` and replaces the original nnU-Net.
