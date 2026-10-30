"""
Trainers for image-to-image translation (e.g. MR -> CT). The network regresses the target image, which is stored as
the 'seg' array of each preprocessed case (see nnunetv2/translation/prepare.py).

    nnUNetTrainerMRCT          L1 loss (recommended default)
    nnUNetTrainerMRCT_mae      same as nnUNetTrainerMRCT (kept for existing models)
    nnUNetTrainerMRCT_mse      L2 loss
    nnUNetTrainerMRCT_AFP      Anatomical Feature-Prioritized loss + L1 (needs nnUNet_AFP_model, see loss/AFP.py)
    nnUNetTrainerMRCT_AFP_ft   AFP fine-tuning from an L1 model (-pretrained_weights): 500 epochs, lr 1e-3

ISBI 2027 options (encoder-only AFP, nnFoundation initialization, see README.md):
    nnUNetTrainerMRCT_AFPexpert          L1 + lambda * encoder AFP, nnU-Net segmentation teacher (nnUNet_AFP_model)
    nnUNetTrainerMRCT_AFPnnF             L1 + lambda * encoder AFP, nnFoundationCNN teacher (nnUNet_nnF_model)
    nnUNetTrainerMRCT_nnFinit            L1, generator encoder initialized from nnFoundationCNN (nnUNet_nnF_model)
    nnUNetTrainerMRCT_AFPexpert_nnFinit  expert encoder AFP + nnFoundationCNN initialization
The AFP trainers need nnUNet_AFP_lambda (a number or the json of nnUNetv2_translation_calibrate_afp). Learning rate:
1e-2 with a 5-epoch linear warm-up, or 1e-3 with a 15-epoch warm-up for the *_nnFinit trainers, then poly decay.
"""
import os
from time import time
from typing import Union, Tuple, List

import numpy as np
import SimpleITK as sitk
import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.utilities.file_and_folder_operations import join, isfile, save_json, subfiles
from torch import autocast, nn

from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D_MRCT
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D_MRCT
from nnunetv2.training.loss.AFP import AFP
from nnunetv2.training.loss.afp_encoder import EncoderAFP, L1AndEncoderAFP, afp_lambda_from_env, \
    expert_encoder_afp, nnfoundation_encoder_afp, stage_weights_from_env
from nnunetv2.training.lr_scheduler.polylr import WarmupPolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.translation.intensity import get_translation_info
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.nnfoundation import check_resenc_l_plans, describe_stride_differences, \
    init_encoder_from_nnfoundation


class nnUNetTrainerMRCT(nnUNetTrainer):
    """L1 regression, no data augmentation, no mirroring, best checkpoint chosen on the validation loss."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False
        self.num_iterations_per_epoch = 250
        self.num_epochs = 1000

    def _build_loss(self):
        return nn.L1Loss()

    @staticmethod
    def get_training_transforms(patch_size: Union[np.ndarray, Tuple[int]],
                                rotation_for_DA: dict,
                                deep_supervision_scales: Union[List, Tuple, None],
                                mirror_axes: Tuple[int, ...],
                                do_dummy_2d_data_aug: bool,
                                order_resampling_data: int = 1,
                                order_resampling_seg: int = 0,
                                border_val_seg: int = -1,
                                use_mask_for_norm: List[bool] = None,
                                is_cascaded: bool = False,
                                foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                ignore_label: int = None) -> AbstractTransform:
        # no augmentation: spatial transforms would have to be applied identically to the target image
        return nnUNetTrainer.get_validation_transforms(deep_supervision_scales, is_cascaded, foreground_labels,
                                                       regions, ignore_label)

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        # we need to disable mirroring here so that no mirroring will be applied in inference!
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        patch_size = self.configuration_manager.patch_size
        loader = nnUNetDataLoader2D_MRCT if dim == 2 else nnUNetDataLoader3D_MRCT
        dl_tr = loader(dataset_tr, self.batch_size, patch_size, patch_size, self.label_manager,
                       oversample_foreground_percent=self.oversample_foreground_percent,
                       sampling_probabilities=None, pad_sides=None)
        dl_val = loader(dataset_val, self.batch_size, patch_size, patch_size, self.label_manager,
                        oversample_foreground_percent=self.oversample_foreground_percent,
                        sampling_probabilities=None, pad_sides=None)
        return dl_tr, dl_val

    def _to_device(self, batch: dict):
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        return data, target

    def train_step(self, batch: dict) -> dict:
        data, target = self._to_device(batch)
        self.optimizer.zero_grad(set_to_none=True)
        # autocast only on cuda (slow on cpu, unsupported on mps)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data, target = self._to_device(batch)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)
        return {'loss': l.detach().cpu().numpy(), 'tp_hard': 0, 'fp_hard': 0, 'fn_hard': 0}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        self.logger.log('val_losses', np.mean(outputs_collated['loss']), self.current_epoch)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)
        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        epoch_duration = self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - \
                         self.logger.my_fantastic_logging['epoch_start_timestamps'][-1]
        self.print_to_log_file(f"Epoch time: {np.round(epoch_duration, decimals=2)} s")

        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # best checkpoint = lowest validation loss
        if self._best_ema is None or self.logger.my_fantastic_logging['val_losses'][-1] < self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['val_losses'][-1]
            self.print_to_log_file(f"Yayy! New best validation loss: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)
        self.current_epoch += 1

    def compute_validation_metrics(self, validation_output_folder: str):
        """MAE between the exported validation predictions and the raw targets, in target units (e.g. HU)."""
        gt_folder = join(self.preprocessed_dataset_folder_base, 'gt_segmentations')
        file_ending = self.dataset_json['file_ending']
        if get_translation_info(self.dataset_json) is None:
            self.print_to_log_file('Validation complete. No MAE: this dataset was not built with '
                                   'nnUNetv2_translation_prepare, so predictions are not in target units.',
                                   also_print_to_console=True)
            return
        per_case = {}
        for f in subfiles(validation_output_folder, suffix=file_ending, join=False):
            if not isfile(join(gt_folder, f)):
                continue
            pred = sitk.DICOMOrient(sitk.ReadImage(join(validation_output_folder, f), sitk.sitkFloat32), 'LPS')
            gt = sitk.DICOMOrient(sitk.ReadImage(join(gt_folder, f), sitk.sitkFloat32), 'LPS')
            if pred.GetSize() != gt.GetSize():
                self.print_to_log_file(f'{f}: prediction size {pred.GetSize()} != target size {gt.GetSize()}, skipped')
                continue
            per_case[f[:-len(file_ending)]] = float(np.mean(np.abs(sitk.GetArrayViewFromImage(pred) -
                                                                   sitk.GetArrayViewFromImage(gt))))
        if len(per_case) == 0:
            self.print_to_log_file('Validation complete. No target found in gt_segmentations, MAE not computed.',
                                   also_print_to_console=True)
            return
        mean_mae = float(np.mean(list(per_case.values())))
        save_json({'mean': {'MAE': mean_mae}, 'per_case': {k: {'MAE': v} for k, v in per_case.items()}},
                  join(validation_output_folder, 'summary.json'))
        self.print_to_log_file("Validation complete", also_print_to_console=True)
        self.print_to_log_file(f"Mean validation MAE (whole image, target units): {np.round(mean_mae, 4)}",
                               also_print_to_console=True)


class nnUNetTrainerMRCT_mae(nnUNetTrainerMRCT):
    pass


class nnUNetTrainerMRCT_mse(nnUNetTrainerMRCT):
    def _build_loss(self):
        return nn.MSELoss()


class nnUNetTrainerMRCT_AFP(nnUNetTrainerMRCT):
    """
    Anatomical Feature-Prioritized loss: L1 between the features of a frozen segmentation network computed on the
    prediction and on the target, plus an L1 term on the images. The segmentation network is a trained nnU-Net
    model given by the nnUNet_AFP_model environment variable (path to its checkpoint_final.pth).
    """
    mae_weight = 1.0

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.batch_size = 1
        self.initial_lr = 1e-2

    def _build_loss(self):
        checkpoint = os.environ.get('nnUNet_AFP_model')
        if checkpoint is None:
            raise RuntimeError('AFP needs a trained nnU-Net segmentation model of the target modality. Set '
                               'nnUNet_AFP_model to its checkpoint, e.g. export nnUNet_AFP_model=/path/to/'
                               'nnUNet_results/DatasetXXX/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/'
                               'checkpoint_final.pth')
        return AFP(checkpoint, mae_weight=self.mae_weight, device=self.device)


class nnUNetTrainerMRCT_AFP_ft(nnUNetTrainerMRCT_AFP):
    """Fine-tuning of an L1 model with AFP: use -pretrained_weights PATH_TO_L1_CHECKPOINT."""
    mae_weight = 0.5

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500
        self.initial_lr = 1e-3


def _env_path(name: str, description: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f'This trainer needs {description}: export {name}=/path/to/checkpoint_final.pth')
    return value


class _WarmupPolyLRMixin:
    """SGD (momentum 0.99, Nesterov), linear warm-up for recipe_warmup_epochs epochs, then poly decay to num_epochs."""
    recipe_lr = 1e-2
    recipe_warmup_epochs = 5

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = self.recipe_lr
        self.warmup_epochs = self.recipe_warmup_epochs

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = WarmupPolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, self.warmup_epochs)
        return optimizer, lr_scheduler


class _EncoderAFPMixin(_WarmupPolyLRMixin):
    """L1 + lambda * encoder-only AFP (loss/afp_encoder.py) with a frozen teacher, afp_teacher = 'expert' or 'nnF'."""
    afp_teacher = 'expert'

    def build_afp_distance(self) -> EncoderAFP:
        weights = stage_weights_from_env()
        if self.afp_teacher == 'expert':
            checkpoint = _env_path('nnUNet_AFP_model', 'a trained nnU-Net segmentation model of the target modality')
            distance = expert_encoder_afp(checkpoint, weights, self.device)
        elif self.afp_teacher == 'nnF':
            checkpoint = _env_path('nnUNet_nnF_model', 'the nnFoundationCNN checkpoint '
                                                       '(https://huggingface.co/MIC-DKFZ/nnFoundationCNN)')
            distance = nnfoundation_encoder_afp(checkpoint, weights, self.device)
        else:
            raise ValueError(f'Unknown AFP teacher {self.afp_teacher!r}')
        self.print_to_log_file(f'Encoder AFP: {self.afp_teacher} teacher {checkpoint}, {distance.n_stages} encoder '
                               f'stages, stage weights {distance.stage_weights}')
        return distance

    def _build_loss(self):
        distance = self.build_afp_distance()
        afp_lambda, calibration = afp_lambda_from_env()
        self.print_to_log_file(f'AFP lambda: {afp_lambda} (nnUNet_AFP_lambda={os.environ["nnUNet_AFP_lambda"]})')
        if calibration is not None:
            teacher = calibration.get('teacher', {})
            if teacher.get('checkpoint') != distance.provenance['checkpoint'] or \
                    teacher.get('stage_weights') != list(distance.stage_weights):
                self.print_to_log_file(f'WARNING: lambda was calibrated with another teacher or other stage weights '
                                       f'({teacher.get("checkpoint")}, {teacher.get("stage_weights")})')
        return L1AndEncoderAFP(distance, afp_lambda)


class _NNFInitMixin(_WarmupPolyLRMixin):
    """Generator encoder initialized from nnFoundationCNN (decoder kept) and the nnF fine-tuning recipe."""
    recipe_lr = 1e-3
    recipe_warmup_epochs = 15

    def initialize(self):
        super().initialize()
        self.load_nnfoundation_encoder()

    def load_nnfoundation_encoder(self) -> dict:
        checkpoint = _env_path('nnUNet_nnF_model', 'the nnFoundationCNN checkpoint '
                                                   '(https://huggingface.co/MIC-DKFZ/nnFoundationCNN)')
        cm = self.configuration_manager
        stride_differences = check_resenc_l_plans(cm.network_arch_class_name, cm.network_arch_init_kwargs,
                                                  self.num_input_channels)
        network = self.network.module if self.is_ddp else self.network
        network = getattr(network, '_orig_mod', network)  # torch.compile
        info = init_encoder_from_nnfoundation(network, checkpoint)
        self.print_to_log_file(f'Generator encoder initialized from nnFoundation: {info["encoder_tensors"]} tensors '
                               f'copied exactly from {checkpoint} (sha256 {info["sha256"]}), decoder unchanged')
        if stride_differences:
            self.print_to_log_file('WARNING: ' + describe_stride_differences(stride_differences))
        return info


class nnUNetTrainerMRCT_AFPexpert(_EncoderAFPMixin, nnUNetTrainerMRCT):
    """L1 + lambda * AFP on the encoder stages of a frozen nnU-Net segmentation model of the target modality
    (nnUNet_AFP_model). lr 1e-2, 5-epoch warm-up."""
    afp_teacher = 'expert'


class nnUNetTrainerMRCT_AFPnnF(_EncoderAFPMixin, nnUNetTrainerMRCT):
    """L1 + lambda * AFP on the frozen nnFoundationCNN encoder (nnUNet_nnF_model). lr 1e-2, 5-epoch warm-up."""
    afp_teacher = 'nnF'


class nnUNetTrainerMRCT_nnFinit(_NNFInitMixin, nnUNetTrainerMRCT):
    """L1, generator encoder initialized from nnFoundationCNN (nnUNet_nnF_model). lr 1e-3, 15-epoch warm-up."""


class nnUNetTrainerMRCT_AFPexpert_nnFinit(_NNFInitMixin, _EncoderAFPMixin, nnUNetTrainerMRCT):
    """Expert encoder AFP + nnFoundationCNN initialization, lr 1e-3, 15-epoch warm-up (best configuration of the
    ISBI 2027 paper). Calibrate lambda with nnUNetTrainerMRCT_AFPexpert (random initialization), as in the paper."""
    afp_teacher = 'expert'


class nnUNetTrainerMRCT_smoketest(nnUNetTrainerMRCT):
    """Tiny run used by nnunetv2/tests/translation_smoke_test.py. Not for real training."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1
        self.num_iterations_per_epoch = 2
        self.num_val_iterations_per_epoch = 1


class nnUNetTrainerMRCT_AFP_smoketest(nnUNetTrainerMRCT_AFP):
    """Tiny AFP run used by nnunetv2/tests/translation_smoke_test.py. Not for real training."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1
        self.num_iterations_per_epoch = 2
        self.num_val_iterations_per_epoch = 1


class _SmokeTestMixin:
    """1 epoch of 2 iterations. Not for real training."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1
        self.num_iterations_per_epoch = 2
        self.num_val_iterations_per_epoch = 1


class nnUNetTrainerMRCT_AFPexpert_smoketest(_SmokeTestMixin, nnUNetTrainerMRCT_AFPexpert):
    """Used by nnunetv2/tests/translation_smoke_test.py."""


class nnUNetTrainerMRCT_AFPnnF_smoketest(_SmokeTestMixin, nnUNetTrainerMRCT_AFPnnF):
    """Used by nnunetv2/tests/translation_smoke_test.py."""


class nnUNetTrainerMRCT_AFPexpert_nnFinit_smoketest(_SmokeTestMixin, nnUNetTrainerMRCT_AFPexpert_nnFinit):
    """Used by nnunetv2/tests/translation_smoke_test.py."""
