from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch, pickle, os
from typing import Union, Tuple, List
import SimpleITK as sitk
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor

import numpy as np
from nnunetv2.training.loss.AFP_multi_mask3 import AFP_multi_mask3


from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D_MRCT
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D_MRCT_mask

from torch import distributed as dist
from nnunetv2.utilities.collate_outputs import collate_outputs

from time import time, sleep
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p

from torch import autocast

class nnUNetTrainerMRCT_AFP_multi_mask3_weights(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 20
        self.num_epochs = 1500 
        # self.batch_size = 1
        self.AFP_loss = AFP_multi_mask3(net1="Navi_1label", net2= "Imene8", net3="TotalSeg117", net1_weight=3.0, net2_weight= 1.0, net3_weight = 0.5, mae_weight=0.5) 
        self.initial_lr = 1e-3

        self.val_save_interval = 100 #every 10th epoch, save val patch
        self.val_patient = "reg_MOUT-Y_MR06"

    def _build_loss(self):
        loss = self.AFP_loss
        return loss

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
        return nnUNetTrainer.get_validation_transforms(deep_supervision_scales, is_cascaded, foreground_labels,
                                                       regions, ignore_label)
    @staticmethod
    def get_validation_transforms(
            deep_supervision_scales: Union[List, Tuple, None],
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> AbstractTransform:
        val_transforms = []
        val_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            val_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))

        val_transforms.append(RenameTransform('seg', 'target', True))

        if regions is not None:
            # the ignore label must also be converted
            val_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
                                                                        if ignore_label is not None else regions,
                                                                        'target', 'target'))

        if deep_supervision_scales is not None:
            val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                               output_key='target'))

        val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        val_transforms = Compose(val_transforms)
        return val_transforms

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        # we need to disable mirroring here so that no mirroring will be applied in inferene!
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        mask = batch['mask']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # print(self.network)
            # assert(0)

            # del data

            
            l = self.loss(output, target, mask)

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
    
    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        initial_patch_size=self.configuration_manager.patch_size
        dim=dim

        if dim == 2:
            dl_tr = nnUNetDataLoader2D_MRCT(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader2D_MRCT(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)
        else:
            dl_tr = nnUNetDataLoader3D_MRCT_mask(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader3D_MRCT_mask(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)
        return dl_tr, dl_val

    def init_fixed_val_patch(self):
        dl = self.dataloader_val.generator
        ds = dl._data  
        key = self.val_patient
        data, seg, props = ds.load_case(key)
        mask = ds.load_mask(key)

        os.makedirs(os.path.join(self.output_folder, "epochs/"), exist_ok=True)

        def center_crop(vol, patch): #manually defined for MOUT-Y
            return vol[:, 200:200+patch[0], 150:150+patch[1], 40:40+patch[2]] 
            
        data_patch = center_crop(data, self.configuration_manager.patch_size)
        seg_patch = center_crop(seg, self.configuration_manager.patch_size)
        mask_patch = center_crop(mask, self.configuration_manager.patch_size)
        
        # torch.save(data_patch, "val_data")
        # torch.save(seg_patch, "val_seg")
        # torch.save(mask_patch, "val_mask")
        # assert(0)

        self.fixed_val_input = torch.from_numpy(data_patch.copy()).float().unsqueeze(0).to(self.device)
        self.fixed_val_target = seg_patch[0]
        self.fixed_val_mask = mask_patch[0]

        sitk.WriteImage(sitk.GetImageFromArray(data_patch[0]), os.path.join(self.output_folder, "epochs/", f"input.nii.gz"))
        sitk.WriteImage(sitk.GetImageFromArray(self.fixed_val_target), os.path.join(self.output_folder, "epochs/", f"target.nii.gz"))
        sitk.WriteImage(sitk.GetImageFromArray(self.fixed_val_mask), os.path.join(self.output_folder, "epochs/", f"mask.nii.gz"))

        self.fixed_val_id = key
        print("validation patch:", key, data.shape, "->", data_patch.shape)

    def on_validation_epoch_start(self):
        self.network.eval()
        interval = self.val_save_interval
        should_save = interval and interval > 0 and (self.current_epoch % interval == 0)
        if should_save:
            with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                output = self.network(self.fixed_val_input).cpu().numpy().astype(np.float32)
                # l = self.AFP_loss(output, target)
        
            out_file = os.path.join(self.output_folder, "epochs/", f"{self.current_epoch}.nii.gz")
            img = sitk.GetImageFromArray(output)
            sitk.WriteImage(img, os.path.join(self.output_folder, "epochs/", f"out{self.current_epoch}.nii.gz"))
            print(f"saved {self.fixed_val_id} to {out_file}")

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        mask = batch['mask']
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.AFP_loss(output, target, mask)
        return {'loss': l.detach().cpu().numpy(), 'tp_hard': 0, 'fp_hard': 0, 'fn_hard': 0}


    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        
        loss_here = np.mean(outputs_collated['loss'])

        # self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        # self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

    def on_epoch_end(self):
        # Log the end time of the epoch
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        # Logging train and validation loss
        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        
        # Log the duration of the epoch
        epoch_duration = self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1]
        self.print_to_log_file(f"Epoch time: {np.round(epoch_duration, decimals=2)} s")

        # Checkpoint handling for best and periodic saves
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        best_metric = 'val_losses'  # Example metric, adjust based on actual usage
        if self._best_ema is None or self.logger.my_fantastic_logging[best_metric][-1] < self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging[best_metric][-1]
            self.print_to_log_file(f"Yayy! New best EMA MAE: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def run_training(self):
        self.on_train_start()
        self.init_fixed_val_patch() 

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()
