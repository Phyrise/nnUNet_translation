"""Tier 1 E1 — WeightedL1Loss (HU range weight).

기존 myMAE (nn.L1Loss) 의 변형 — HU range 별 가중치 적용:
  - soft tissue [-100, 100] HU: weight = 1.5  (clinical focus)
  - bone (>= 300 HU) / air (<= -800 HU): weight = 0.5  (학습 안정 + outlier 영향 감소)
  - 기타: weight = 1.0

목적:
  - phase2 P_hu_range 의 [M] 결과 (MAE_whole 123.19 → MAE_soft 63.61, 6 모델 중 우수)
    를 nnU-Net framework 측에서 재현
  - hold-out 14 case 의 MAE_soft 개선 (현재 sct_unet C4 학습 = 184.63)

근거: amed plan unet0601.md § 6.15 (Tier 1) + EXPERIMENT_REPORT.md § 14.4.2 (phase2 P_hu_range)

전제:
  - target 이 z-score 정규화된 CT (CTPaperNormalization). raw HU 가 아님.
  - 가중치 영역 결정 시 target 을 raw HU 로 역변환 필요 (CT_MEAN, CT_STD 사용).
"""
import torch
from torch import nn, Tensor


class WeightedL1Loss(nn.Module):
    """L1 loss + HU-range-based weighting.

    Args:
        soft_weight: 소프트 조직 [-100, 100] HU 가중치 (default 1.5)
        bone_weight: 뼈 (>= bone_threshold) 가중치 (default 0.5)
        air_weight: 공기 (<= air_threshold) 가중치 (default 0.5)
        other_weight: 그 외 (default 1.0)
        ct_mean: CTPaperNormalization 의 dataset mean HU (default Dataset051 = -775.24)
        ct_std: CTPaperNormalization 의 dataset std HU (default Dataset051 = 449.85)
        soft_range: 소프트 조직 HU 범위 (default (-100, 100))
        bone_threshold: 뼈 HU 임계값 (default 300)
        air_threshold: 공기 HU 임계값 (default -800)
    """

    def __init__(
        self,
        soft_weight: float = 1.5,
        bone_weight: float = 0.5,
        air_weight: float = 0.5,
        other_weight: float = 1.0,
        ct_mean: float = -775.2418,
        ct_std: float = 449.8501,
        soft_range: tuple = (-100.0, 100.0),
        bone_threshold: float = 300.0,
        air_threshold: float = -800.0,
    ):
        super().__init__()
        self.soft_weight = soft_weight
        self.bone_weight = bone_weight
        self.air_weight = air_weight
        self.other_weight = other_weight
        self.ct_mean = ct_mean
        self.ct_std = ct_std
        self.soft_lo, self.soft_hi = soft_range
        self.bone_threshold = bone_threshold
        self.air_threshold = air_threshold

    def _hu_from_zscore(self, target: Tensor) -> Tensor:
        """z-score 정규화 target 을 raw HU 로 역변환."""
        return target * self.ct_std + self.ct_mean

    def _build_weight_map(self, target_hu: Tensor) -> Tensor:
        """target_hu 의 HU range 별 가중치 map (target 과 동일 shape)."""
        # other_weight 로 초기화
        w = torch.full_like(target_hu, self.other_weight)
        # soft tissue mask
        soft_mask = (target_hu >= self.soft_lo) & (target_hu <= self.soft_hi)
        w[soft_mask] = self.soft_weight
        # bone mask (soft 외부)
        bone_mask = target_hu >= self.bone_threshold
        w[bone_mask] = self.bone_weight
        # air mask
        air_mask = target_hu <= self.air_threshold
        w[air_mask] = self.air_weight
        return w

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # 1) target 을 HU 공간으로 역변환 (weight map 계산용)
        target_hu = self._hu_from_zscore(target)
        # 2) HU range 별 weight map 생성
        w = self._build_weight_map(target_hu)
        # 3) weighted L1 — sum(w * |input - target|) / sum(w)
        diff = torch.abs(input - target)
        weighted = (w * diff).sum() / (w.sum().clamp_min(1e-8))
        return weighted
