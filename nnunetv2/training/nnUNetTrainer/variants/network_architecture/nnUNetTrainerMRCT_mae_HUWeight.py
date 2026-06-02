"""Tier 1 E1 trainer — nnUNetTrainerMRCT_mae_PairedFix + WeightedL1Loss (HU range weight).

조합:
  - dataloader: nnUNetDataLoader3D_MRCT_PairedFix (C1 fix 의 D050+D051 직접 페어링)
  - loss: WeightedL1Loss (HU range weight, soft=1.5/bone=0.5/air=0.5)
  - 나머지 (optimizer, scheduler, epoch): nnUNetTrainerMRCT_mae 동일

목적: phase2 P_hu_range [M] 결과의 nnU-Net 재현 + C1 fix 검증.

사용:
    nnUNetv2_train 50 3d_fullres 0 -tr nnUNetTrainerMRCT_mae_HUWeight
"""
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerMRCT_mae_PairedFix import nnUNetTrainerMRCT_mae_PairedFix
from nnunetv2.training.loss.weighted_mae import WeightedL1Loss


class nnUNetTrainerMRCT_mae_HUWeight(nnUNetTrainerMRCT_mae_PairedFix):
    """C1 PairedFix + Tier 1 E1 (HU range weighted L1)."""

    def _build_loss(self):
        # 본 trainer 의 핵심 변경 — myMAE 대신 WeightedL1Loss
        # ct_mean / ct_std 는 Dataset051 의 plans 인용 기본값. 다른 dataset 시 override 권장.
        loss = WeightedL1Loss(
            soft_weight=1.5,
            bone_weight=0.5,
            air_weight=0.5,
            other_weight=1.0,
            ct_mean=-775.2418,
            ct_std=449.8501,
            soft_range=(-100.0, 100.0),
            bone_threshold=300.0,
            air_threshold=-800.0,
        )
        return loss
