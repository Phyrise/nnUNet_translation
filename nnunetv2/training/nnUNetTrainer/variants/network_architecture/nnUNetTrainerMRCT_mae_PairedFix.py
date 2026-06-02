"""C1 근본 fix trainer — nnUNetTrainerMRCT_mae + D051 직접 페어링 dataloader.

기존 nnUNetTrainerMRCT_mae 의 dataloader (`nnUNetDataLoader3D_MRCT`) 는 D050 만 로드 →
automate_translation.py 의 shutil.copy(D051/*.npy → D050/*_seg.npy) workaround 의존.

본 trainer 는 `nnUNetDataLoader3D_MRCT_PairedFix` 사용 — D051 의 같은 case_id 의 CT 를
dataloader 단에서 직접 로드. workaround 의존성 제거 + 페어링 보장.

근거: amed plan unet0601.md § 6.6 (NaN 진단) + 메모리 [project_nnunet_dataloader_bug.md].

사용 예:
    nnUNetv2_train 50 3d_fullres 0 -tr nnUNetTrainerMRCT_mae_PairedFix
    # D050 학습 → 내부에서 D051 자동 페어링
"""
import os
from typing import Tuple

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerMRCT_mae import nnUNetTrainerMRCT_mae
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D_MRCT_PairedFix
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D_MRCT
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset


class nnUNetTrainerMRCT_mae_PairedFix(nnUNetTrainerMRCT_mae):
    """MRCT_mae + D050↔D051 dataloader 페어링 fix."""

    def _get_target_dataset_folder(self) -> str:
        """D050 의 preprocessed folder 에서 D051 의 preprocessed folder 자동 결정.

        예: preprocessed/Dataset050_TrainHN_Input/nnUNetPlans_3d_fullres
            → preprocessed/Dataset051_TrainHN_Target/nnUNetPlans_3d_fullres
        """
        src = self.preprocessed_dataset_folder
        # Dataset050_TrainHN_Input → Dataset051_TrainHN_Target
        # 1) Dataset NNN → NNN+1
        # 2) _Input suffix → _Target suffix
        parts = src.replace("\\", "/").split("/")
        # 마지막에서 두 번째가 Dataset 폴더명 (마지막은 nnUNetPlans_3d_fullres)
        ds_idx = None
        for k, p in enumerate(parts):
            if p.startswith("Dataset") and len(p) >= 10 and p[7:10].isdigit():
                ds_idx = k
                break
        if ds_idx is None:
            raise RuntimeError(f"[PairedFix] DatasetNNN 폴더명 인식 실패: {src}")
        old = parts[ds_idx]
        ds_num = int(old[7:10])
        new_num = ds_num + 1
        # _Input → _Target (suffix 가 _Input 인 경우만)
        if "_Input" in old:
            new_name = f"Dataset{new_num:03d}" + old[10:].replace("_Input", "_Target")
        else:
            # convention 위반 시 단순 ID+1
            new_name = f"Dataset{new_num:03d}" + old[10:]
        parts[ds_idx] = new_name
        target = "/".join(parts)
        if not os.path.isdir(target):
            raise RuntimeError(
                f"[PairedFix] D{new_num:03d} preprocessed folder 부재: {target}\n"
                f"  먼저 'nnUNetv2_plan_and_preprocess -d {new_num}' 실행 필요."
            )
        return target

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        initial_patch_size = self.configuration_manager.patch_size

        if dim == 2:
            # 2D 는 기존 path 유지 — 본 fix 범위 밖
            return super().get_plain_dataloaders(initial_patch_size, dim)

        # 3D: D051 (target) dataset 인스턴스
        target_folder = self._get_target_dataset_folder()
        self.print_to_log_file(f"[C1 PairedFix] target dataset folder = {target_folder}")
        # tr / val 각각 D051 의 같은 case_id 만 load
        target_tr = nnUNetDataset(target_folder, case_identifiers=list(dataset_tr.keys()))
        target_val = nnUNetDataset(target_folder, case_identifiers=list(dataset_val.keys()))

        dl_tr = nnUNetDataLoader3D_MRCT_PairedFix(
            dataset_tr, target_tr, self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None,
        )
        dl_val = nnUNetDataLoader3D_MRCT_PairedFix(
            dataset_val, target_val, self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None,
        )
        return dl_tr, dl_val
