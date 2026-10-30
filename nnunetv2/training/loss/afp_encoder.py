"""
Encoder-only AFP: the feature distance of the Anatomical Feature-Prioritized loss (Longuefosse et al., Physics in
Medicine & Biology 2025, https://doi.org/10.1088/1361-6560/adea07) restricted to the encoder of a frozen teacher, as
used in the ISBI 2027 paper (3D pre- to post-contrast breast MRI synthesis):

    D(Y_hat, Y) = sum_l w_l * mean | Phi_l(z(Y_hat)) - Phi_l(z(Y)) |

Phi_l is the output of encoder stage l, z a z-score computed separately for each image (over the spatial axes), and
both images are zero-padded (centered) so that every axis is a multiple of 2 ** (n_stages - 1). The target features
carry no gradient. Two teachers:
    expert  a trained nnU-Net v2 segmentation model of the target modality (loaded like loss/AFP.py)
    nnF     the self-supervised nnFoundationCNN ResEnc-L encoder (nnssl checkpoint)

With unit weights this is the feature term of the tag isbi2027-results (afp_features.AFPFeatureDistance with every
encoder stage, separate z-score, historical padding); nnunetv2/tests/afp_nnf_equivalence_test.py checks it.
loss/AFP.py (historical AFP: encoder and decoder features, no z-score) is unchanged.
"""
import json
import os
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class EncoderAFP(nn.Module):
    def __init__(self, teacher: nn.Module, stage_weights: Optional[Sequence[float]] = None):
        super().__init__()
        encoder = getattr(teacher, 'encoder', None)
        if encoder is None or not hasattr(encoder, 'stages') or not getattr(encoder, 'return_skips', False):
            raise ValueError('EncoderAFP: the teacher must be a dynamic_network_architectures U-Net whose encoder '
                             'returns the output of every stage (PlainConvUNet, ResidualEncoderUNet, ...)')
        self.teacher = teacher.requires_grad_(False).eval()
        self.n_stages = len(encoder.stages)
        weights = [1.0] * self.n_stages if stage_weights is None else [float(w) for w in stage_weights]
        if len(weights) != self.n_stages or min(weights) < 0 or max(weights) <= 0:
            raise ValueError(f'EncoderAFP: expected {self.n_stages} non-negative stage weights (not all zero), '
                             f'got {weights}')
        self.stage_weights = tuple(weights)
        self.pad_factor = 2 ** (self.n_stages - 1)
        self.provenance = {}  # filled by the builders below

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()  # frozen teacher: instance norm and dropout stay in inference mode
        return self

    @staticmethod
    def zscore(x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(2, x.ndim))
        return (x - x.mean(dims, keepdim=True)) / (x.std(dims, keepdim=True) + 1e-8)

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        pads = []
        for size in reversed(x.shape[2:]):  # F.pad expects the last axis first
            missing = (-size) % self.pad_factor
            pads.extend((missing // 2, missing - missing // 2))
        return F.pad(x, pads)

    def features(self, x: torch.Tensor):
        self.teacher.eval()
        # z-score in float32 (the network output may be float16 under autocast)
        return self.teacher.encoder(self.pad(self.zscore(x.float())))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape or prediction.shape[1] != 1:
            raise ValueError(f'EncoderAFP: expected matching single-channel prediction and target, got '
                             f'{tuple(prediction.shape)} and {tuple(target.shape)}')
        fx = self.features(prediction)
        with torch.no_grad():
            fy = self.features(target.detach())
        terms = [w * (a.float() - b.float()).abs().mean() for w, a, b in zip(self.stage_weights, fx, fy)]
        return torch.stack(terms).sum()


class L1AndEncoderAFP(nn.Module):
    """L1(Y_hat, Y) + afp_lambda * D(Y_hat, Y)."""

    def __init__(self, distance: EncoderAFP, afp_lambda: float):
        super().__init__()
        self.distance = distance
        self.afp_lambda = float(afp_lambda)
        self.l1 = nn.L1Loss()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1(output, target) + self.afp_lambda * self.distance(output, target)


def expert_encoder_afp(checkpoint_path: str, stage_weights=None, device=torch.device('cuda')) -> EncoderAFP:
    """Encoder AFP with a trained nnU-Net v2 segmentation model (checkpoint_final.pth or its fold folder)."""
    from nnunetv2.training.loss.AFP import load_segmentation_network
    distance = EncoderAFP(load_segmentation_network(checkpoint_path).to(device), stage_weights)
    distance.provenance = {'teacher': 'expert', 'checkpoint': os.path.abspath(checkpoint_path),
                           'encoder_stages': distance.n_stages, 'stage_weights': list(distance.stage_weights)}
    return distance


def nnfoundation_encoder_afp(checkpoint_path: str, stage_weights=None, device=torch.device('cuda')) -> EncoderAFP:
    """Encoder AFP with the frozen nnFoundationCNN encoder (nnssl checkpoint_final.pth)."""
    from nnunetv2.utilities.nnfoundation import load_nnfoundation_network
    distance = EncoderAFP(load_nnfoundation_network(checkpoint_path).to(device), stage_weights)
    distance.provenance = {'teacher': 'nnF', 'checkpoint': os.path.abspath(checkpoint_path),
                           'encoder_stages': distance.n_stages, 'stage_weights': list(distance.stage_weights)}
    return distance


def stage_weights_from_env() -> Optional[Tuple[float, ...]]:
    """nnUNet_AFP_stage_weights='1,1,1,1,1,1' (one weight per encoder stage, shallowest first); default: all 1."""
    value = os.environ.get('nnUNet_AFP_stage_weights')
    if value is None or value.strip() == '':
        return None
    return tuple(float(v) for v in value.split(','))


def afp_lambda_from_env() -> Tuple[float, Optional[dict]]:
    """
    nnUNet_AFP_lambda: a number, or the json written by nnUNetv2_translation_calibrate_afp. Returns (lambda, content
    of the json or None).
    """
    value = os.environ.get('nnUNet_AFP_lambda')
    if value is None or value.strip() == '':
        raise RuntimeError('This trainer needs the AFP weight lambda: export nnUNet_AFP_lambda=VALUE, or calibrate it '
                           'by gradient-norm matching with nnUNetv2_translation_calibrate_afp and export '
                           'nnUNet_AFP_lambda=/path/to/afp_lambda.json')
    try:
        return float(value), None
    except ValueError:
        pass
    if not os.path.isfile(value):
        raise RuntimeError(f'nnUNet_AFP_lambda={value} is neither a number nor an existing json file')
    with open(value) as f:
        calibration = json.load(f)
    return float(calibration['lambda']), calibration
