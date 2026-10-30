"""
nnFoundation helpers: checkpoint loading, the ResEnc-L network, and the initialization of a generator encoder.

nnFoundationCNN (Harsy et al., nnFoundation: 3D Foundation Models for Radiology, arXiv:2609.26924; weights at
https://huggingface.co/MIC-DKFZ/nnFoundationCNN, licence CC-BY-SA-4.0) is a ResEnc-L network pre-trained with nnssl
(masked auto-encoding) on CT, MR and PET volumes. Its checkpoint is a torch.save dictionary with
    'network_weights'        state_dict of a dynamic_network_architectures ResidualEncoderUNet (1-channel MAE head)
    'nnssl_adaptation_plan'  architecture 'ResEncL', 1 input channel, z-score input, no resampling
The state_dict stores aliases of the same tensors ('all_modules.*', 'decoder.encoder.*'): they are expected.
"""
import hashlib
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
from torch import nn

# nnssl architecture_registry.get_res_enc_l
RESENC_L = dict(n_stages=6, features_per_stage=[32, 64, 128, 256, 320, 320], kernel_sizes=[[3, 3, 3]] * 6,
                strides=[[1, 1, 1]] + [[2, 2, 2]] * 5, n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
                n_conv_per_stage_decoder=[1, 1, 1, 1, 1], conv_bias=True)


def build_resenc_l(input_channels: int = 1, num_classes: int = 1) -> nn.Module:
    """The nnFoundationCNN network (ResEnc-L, InstanceNorm with affine, LeakyReLU, conv bias)."""
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
    return ResidualEncoderUNet(input_channels=input_channels, conv_op=nn.Conv3d, num_classes=num_classes,
                               norm_op=nn.InstanceNorm3d, norm_op_kwargs={'eps': 1e-5, 'affine': True},
                               nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=False,
                               **RESENC_L)


def strip_uniform_prefixes(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Removes a DDP ('module.') or torch.compile ('_orig_mod.') prefix shared by every key."""
    state = dict(state)
    while state:
        prefix = next((p for p in ('module.', '_orig_mod.') if all(k.startswith(p) for k in state)), None)
        if prefix is None:
            break
        state = {k[len(prefix):]: v for k, v in state.items()}
    return state


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_nnfoundation_checkpoint(checkpoint_path: str) -> Tuple[Dict[str, torch.Tensor], dict]:
    """(state_dict, adaptation plan) of an nnssl ResEnc-L checkpoint. Loaded with weights_only=True (downloaded file)."""
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or 'network_weights' not in payload or 'nnssl_adaptation_plan' not in payload:
        raise ValueError(f'{checkpoint_path} is not an nnssl checkpoint (expected the keys network_weights and '
                         f'nnssl_adaptation_plan, e.g. checkpoint_final.pth of MIC-DKFZ/nnFoundationCNN)')
    plan = payload['nnssl_adaptation_plan']
    arch = (plan.get('architecture_plans') or {}).get('arch_class_name')
    channels = plan.get('pretrain_num_input_channels')
    if arch != 'ResEncL' or channels != 1:
        raise ValueError(f'{checkpoint_path}: only single-channel ResEncL nnssl checkpoints are supported '
                         f'(architecture {arch}, {channels} input channels)')
    return strip_uniform_prefixes(payload['network_weights']), plan


def load_nnfoundation_network(checkpoint_path: str) -> nn.Module:
    """The full nnFoundationCNN network with strictly loaded weights (every key present, every shape equal)."""
    state, _ = load_nnfoundation_checkpoint(checkpoint_path)
    network = build_resenc_l(input_channels=1, num_classes=1)
    network.load_state_dict(state, strict=True)
    return network


def _last_name(obj) -> str:
    name = obj if isinstance(obj, str) else getattr(obj, '__name__', str(obj))
    return name.rsplit('.', 1)[-1]


def _per_stage(value, n: int) -> list:
    return list(value) if isinstance(value, (list, tuple)) else [value] * n


def _as_3d(value) -> List[int]:
    return [int(v) for v in value] if isinstance(value, (list, tuple)) else [int(value)] * 3


def check_resenc_l_plans(network_class_name: str, arch_kwargs: dict, num_input_channels: int) \
        -> List[Tuple[int, List[int], List[int]]]:
    """
    Raises a ValueError listing every difference between the generator encoder described by the plans and the
    ResEnc-L encoder of nnFoundation (the encoder tensors would not match). Strides do not change the tensor shapes:
    they are returned as (stage, plans stride, nnFoundation stride) so that the caller can report them.
    """
    errors = []
    if _last_name(network_class_name) not in ('ResidualUNet', 'ResidualEncoderUNet'):
        errors.append(f'network class {network_class_name} (needs ResidualUNet or ResidualEncoderUNet, i.e. a '
                      f'residual encoder: use the nnResUNetPlans or nnUNetResEncUNetLPlans planners)')
    if num_input_channels != 1:
        errors.append(f'{num_input_channels} input channels (nnFoundation takes 1)')
    n = arch_kwargs.get('n_stages')
    if n != RESENC_L['n_stages']:
        errors.append(f'n_stages {n} (ResEnc-L: {RESENC_L["n_stages"]}); the number of stages follows the patch size '
                      f'in the plans, the patch must be divisible by 32 along every axis')
    else:
        checks = (('features_per_stage', [int(v) for v in _per_stage(arch_kwargs.get('features_per_stage'), n)],
                   RESENC_L['features_per_stage']),
                  ('kernel_sizes', [_as_3d(v) for v in _per_stage(arch_kwargs.get('kernel_sizes'), n)],
                   RESENC_L['kernel_sizes']),
                  ('n_blocks_per_stage', [int(v) for v in _per_stage(arch_kwargs.get('n_blocks_per_stage'), n)],
                   RESENC_L['n_blocks_per_stage']))
        for key, value, expected in checks:
            if value != expected:
                errors.append(f'{key} {value} (ResEnc-L: {expected})')
    if not arch_kwargs.get('conv_bias', False):
        errors.append('conv_bias False (ResEnc-L: True)')
    if _last_name(arch_kwargs.get('conv_op')) != 'Conv3d':
        errors.append(f'conv_op {arch_kwargs.get("conv_op")} (ResEnc-L: Conv3d)')
    norm_kwargs = arch_kwargs.get('norm_op_kwargs') or {}
    if _last_name(arch_kwargs.get('norm_op')) != 'InstanceNorm3d' or not norm_kwargs.get('affine', False):
        errors.append(f'norm_op {arch_kwargs.get("norm_op")} {norm_kwargs} (ResEnc-L: InstanceNorm3d, affine)')
    if arch_kwargs.get('block') is not None and _last_name(arch_kwargs['block']) != 'BasicBlockD':
        errors.append(f'block {arch_kwargs["block"]} (ResEnc-L: BasicBlockD)')
    if arch_kwargs.get('stem_channels') not in (None, RESENC_L['features_per_stage'][0]):
        errors.append(f'stem_channels {arch_kwargs["stem_channels"]} (ResEnc-L: {RESENC_L["features_per_stage"][0]})')
    if errors:
        raise ValueError('nnFoundation initialization: the generator encoder in the plans does not have the ResEnc-L '
                         'geometry of nnFoundationCNN:\n  - ' + '\n  - '.join(errors))
    strides = [_as_3d(s) for s in _per_stage(arch_kwargs.get('strides'), n)]
    return [(i, s, e) for i, (s, e) in enumerate(zip(strides, RESENC_L['strides'])) if s != e]


def describe_stride_differences(differences: Sequence[Tuple[int, List[int], List[int]]]) -> str:
    stages = ', '.join(f'stage {i}: {tuple(s)} instead of {tuple(e)}' for i, s, e in differences)
    return (f'nnFoundation initialization: the generator strides differ from nnFoundation ({stages}). The tensor '
            f'shapes are the same, so the weights load exactly, but the receptive field of the copied filters differs '
            f'from pre-training. In the ISBI 2027 paper the generator also had its last stride (1, 2, 2) instead of '
            f'(2, 2, 2).')


def init_encoder_from_nnfoundation(network: nn.Module, checkpoint_path: str) -> dict:
    """
    Copies the nnFoundation encoder tensors ('encoder.*') into network.encoder; every other tensor (decoder, output
    layer) is left unchanged. The generator encoder keys must be exactly those of the checkpoint encoder, with the same
    shapes, otherwise a ValueError lists the differences. Returns a summary (checkpoint, sha256, number of tensors,
    stride differences).
    """
    state, _ = load_nnfoundation_checkpoint(checkpoint_path)
    source = {k: v for k, v in state.items() if k.startswith('encoder.')}
    own = network.state_dict()
    own_keys = {k for k in own if k.startswith('encoder.')}
    missing, unexpected = sorted(own_keys - set(source)), sorted(set(source) - own_keys)
    shapes = sorted(f'{k}: {tuple(own[k].shape)} vs {tuple(source[k].shape)}'
                    for k in own_keys & set(source) if own[k].shape != source[k].shape)
    if not source or missing or unexpected or shapes:
        details = [f'{len(missing)} generator encoder keys missing from the checkpoint {missing[:3]}',
                   f'{len(unexpected)} checkpoint encoder keys absent from the generator {unexpected[:3]}',
                   f'{len(shapes)} shape mismatches {shapes[:3]}']
        raise ValueError('nnFoundation initialization: the generator encoder does not match the checkpoint '
                         f'encoder ({checkpoint_path}):\n  - ' + '\n  - '.join(details))
    result = network.load_state_dict(source, strict=False)
    if result.unexpected_keys or any(not k.startswith('decoder.') for k in result.missing_keys):
        raise RuntimeError(f'nnFoundation initialization: unexpected load result {result}')
    after = network.state_dict()
    if not all(torch.equal(after[k].cpu(), v.to(after[k].dtype)) for k, v in source.items()):
        raise RuntimeError('nnFoundation initialization: encoder tensors differ from the checkpoint after loading')
    nnf_strides = [list(s) for s in RESENC_L['strides']]
    own_strides = [_as_3d(s) for s in getattr(getattr(network, 'encoder', None), 'strides', nnf_strides)]
    return {'checkpoint': str(checkpoint_path), 'sha256': sha256_file(checkpoint_path),
            'encoder_tensors': len(source), 'exact': True,
            'stride_differences': [(i, s, e) for i, (s, e) in enumerate(zip(own_strides, nnf_strides)) if s != e]}
