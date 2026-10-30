"""Frozen nnUNet segmentation networks with named features for AFP.

Additive API: the existing AFP classes and historical checkpoints are untouched.
Uses the original segmentation forward, observed with temporary hooks; no copied
encoder/decoder forward. Supports PlainConvUNet and ResidualEncoderUNet.
"""
from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def canonical_modality(name: str) -> str:
    name = str(name).strip().lower().replace('-', '').replace('_', '')
    return {'t1': 't1n', 't1n': 't1n', 't1c': 't1c', 't1ce': 't1c',
            't1gd': 't1c', 't2': 't2w', 't2w': 't2w', 'flair': 't2f',
            't2flair': 't2f', 't2f': 't2f'}.get(name, name)


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 24), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_strict_weights(network: nn.Module, state: Mapping[str, Tensor]) -> None:
    """Strip only uniform DDP/compile prefixes, then require a complete match."""
    state = dict(state)
    if not state:
        raise ValueError('Empty expert state dictionary')
    while True:
        prefix = next((p for p in ('module.', '_orig_mod.') if all(k.startswith(p) for k in state)), None)
        if prefix is None:
            break
        state = {key[len(prefix):]: value for key, value in state.items()}
    network.load_state_dict(state, strict=True)


@dataclass
class ExpertOutput:
    features: Mapping[str, Tensor]
    logits: Tensor


class NNUNetFeatureExpert(nn.Module):
    """Frozen native network; features retain gradients with respect to its input."""
    def __init__(self, network: nn.Module, modalities: Sequence[str]):
        super().__init__()
        if not hasattr(network, 'encoder') or not hasattr(network, 'decoder'):
            raise ValueError('An nnUNet encoder/decoder architecture is required')
        self.network = network.requires_grad_(False).eval()
        self.modalities = tuple(canonical_modality(m) for m in modalities)
        if not self.modalities or len(set(self.modalities)) != len(self.modalities):
            raise ValueError('Expert modalities must be nonempty and unique')
        self.encoder_names = tuple(f'encoder.{i}' for i in range(len(network.encoder.stages)))
        self.decoder_names = tuple(f'decoder.{i}' for i in range(len(network.decoder.stages)))
        self.feature_names = self.encoder_names + self.decoder_names
        strides = torch.as_tensor(network.encoder.strides, dtype=torch.int64)
        if strides.ndim != 2 or strides.shape[1] != 3:
            raise ValueError('This AFP wrapper requires a 3D expert')
        self.required_divisibility = tuple(int(v) for v in strides.prod(0))
        self.provenance = {}

    @classmethod
    def from_checkpoint(cls, checkpoint, *, device='cpu', dtype=torch.float32,
                        configuration='3d_fullres', plans_path=None, dataset_json_path=None,
                        expected_modalities=None):
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

        checkpoint = Path(checkpoint)
        root = checkpoint.parent.parent
        plans_file = Path(plans_path) if plans_path else root / 'plans.json'
        dataset_file = Path(dataset_json_path) if dataset_json_path else root / 'dataset.json'
        plans = json.loads(plans_file.read_text())
        dataset = json.loads(dataset_file.read_text())
        manager = PlansManager(plans)
        conf = manager.get_configuration(configuration)
        name = conf.network_arch_class_name
        if name.rsplit('.', 1)[-1] not in ('PlainConvUNet', 'ResidualEncoderUNet'):
            raise ValueError(f'Unsupported segmentation expert architecture: {name}')
        modalities = tuple(canonical_modality(dataset['channel_names'][str(i)])
                           for i in range(len(dataset['channel_names'])))
        if expected_modalities is not None and modalities != tuple(canonical_modality(m) for m in expected_modalities):
            raise ValueError(f'Checkpoint modalities {modalities} differ from requested {expected_modalities}')
        channels = determine_num_input_channels(manager, conf, dataset)
        if channels != len(modalities):
            raise ValueError('Cascaded or additional non-modality inputs are not supported')
        kwargs = dict(conf.network_arch_init_kwargs)
        # Explicit in kwargs: older nnUNet helper versions do not consistently
        # forward their separate deep_supervision argument.
        kwargs['deep_supervision'] = False
        with torch.random.fork_rng(devices=[]):
            network = get_network_from_plans(name, kwargs, conf.network_arch_init_kwargs_req_import,
                                             channels, manager.get_label_manager(dataset).num_segmentation_heads,
                                             allow_init=False, deep_supervision=False)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        state = next((payload[key] for key in ('network_weights', 'state_dict', 'model_state_dict')
                      if key in payload), None)
        if state is None:
            raise ValueError('Checkpoint does not contain segmentation network weights')
        load_strict_weights(network, state)
        expert = cls(network.to(device=device, dtype=dtype), modalities)
        expert.provenance = {'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_sha256(checkpoint),
                             'plans': str(plans_file), 'dataset_json': str(dataset_file),
                             'architecture': name, 'modalities': list(modalities),
                             'strict_weights_loaded': True, 'missing_keys': 0, 'unexpected_keys': 0}
        return expert

    def train(self, mode=True):
        super().train(mode)
        self.network.eval()
        return self

    def select_features(self, selection='historical'):
        if isinstance(selection, str):
            if selection == 'historical':
                return self.encoder_names[:-2] + self.decoder_names
            if selection == 'all':
                return self.feature_names
            raise ValueError(f'Unknown feature selection {selection!r}')
        names = tuple(selection)
        if len(names) != len(set(names)) or any(name not in self.feature_names for name in names):
            raise ValueError(f'Invalid/duplicate feature names {names}')
        return names

    def forward(self, x: Tensor) -> ExpertOutput:
        if x.ndim != 5 or x.shape[1] != len(self.modalities):
            raise ValueError(f'Expected B×{len(self.modalities)}×D×H×W expert input')
        if any(size % divisor for size, divisor in zip(x.shape[2:], self.required_divisibility)):
            raise ValueError(f'Input must be padded to multiples of {self.required_divisibility}')
        param = next(self.network.parameters())
        if x.device != param.device:
            raise ValueError(f'Input on {x.device}, expert on {param.device}')
        captured = {}

        def capture(name):
            def hook(module, args, output):
                if not isinstance(output, Tensor):
                    raise TypeError(f'{name} did not return a tensor')
                captured[name] = output
            return hook

        self.network.eval()
        with ExitStack() as stack:
            for names, stages in ((self.encoder_names, self.network.encoder.stages),
                                  (self.decoder_names, self.network.decoder.stages)):
                for name, stage in zip(names, stages):
                    handle = stage.register_forward_hook(capture(name))
                    stack.callback(handle.remove)
            logits = self.network(x.to(dtype=param.dtype))
        if isinstance(logits, (list, tuple)):
            logits = logits[0]  # standard nnUNet: highest-resolution deep supervision head first
        if not isinstance(logits, Tensor) or set(captured) != set(self.feature_names):
            raise RuntimeError('Incomplete native expert forward')
        return ExpertOutput(OrderedDict((name, captured[name]) for name in self.feature_names), logits)


def compose_named_inputs(target: Tensor, context: Optional[Mapping[str, Tensor]],
                         target_modality: str, expert_modalities: Sequence[str]) -> Tensor:
    """Assemble by modality name, rejecting a real target in the generator context."""
    target_modality = canonical_modality(target_modality)
    named = {}
    for name, image in (context or {}).items():
        key = canonical_modality(name)
        if key in named or key == target_modality:
            raise ValueError('Duplicate context modality or target leakage into context')
        if image.shape != target.shape or image.device != target.device:
            raise ValueError('Context must match target shape and device')
        named[key] = image.detach()
    named[target_modality] = target
    order = tuple(canonical_modality(m) for m in expert_modalities)
    if set(named) != set(order):
        raise ValueError(f'Expected exactly {order}; received {tuple(named)}')
    return torch.cat([named[name] for name in order], dim=1)


class AFPFeatureDistance(nn.Module):
    """Feature term only; callers retain their existing L1/Dice/boundary terms.

    Default reproduces historical AFP: separate z-score, padding to 2**(stages-1),
    sum of nine stage-wise mean L1 terms for a six-stage expert. Explicit `all`
    exposes the six encoder + five decoder features; logits remain separate.
    """
    def __init__(self, expert: NNUNetFeatureExpert, *, features='historical', normalization='separate',
                 normalize_features=False, distance='l1', padding='historical', target_modality=None,
                 normalization_precision='historical'):
        super().__init__()
        if normalization not in ('none', 'separate', 'shared_target'):
            raise ValueError(normalization)
        if distance not in ('l1', 'channel_unit_l2') or padding not in ('historical', 'network'):
            raise ValueError('Invalid distance or padding')
        if normalization_precision not in ('historical', 'float32'):
            raise ValueError('Invalid normalization precision')
        self.expert = expert
        self.selected = expert.select_features(features)
        self.normalization, self.normalize_features = normalization, normalize_features
        self.distance, self.padding = distance, padding
        self.normalization_precision = normalization_precision
        if target_modality is None:
            if len(expert.modalities) != 1:
                raise ValueError('A multimodal AFP must explicitly identify the synthesised target')
            target_modality = expert.modalities[0]
        self.target_modality = canonical_modality(target_modality)
        if self.target_modality not in expert.modalities:
            raise ValueError('Synthesised modality is absent from expert inputs')

    def _pad(self, x):
        factors = ((2 ** (len(self.expert.encoder_names) - 1),) * 3 if self.padding == 'historical'
                   else self.expert.required_divisibility)
        pads = []
        for size, factor in reversed(tuple(zip(x.shape[2:], factors))):
            missing = (-size) % factor
            pads.extend((missing // 2, missing - missing // 2))
        return F.pad(x, pads)

    def forward(self, prediction: Tensor, target: Tensor, context=None) -> Tensor:
        if prediction.ndim != 5 or prediction.shape != target.shape or prediction.shape[1] != 1:
            raise ValueError('Expected matched B×1×D×H×W prediction and target')
        if not self.selected:
            return prediction.sum() * 0
        # Legacy AFP normalizes the prediction before concatenating its FP32
        # context. Under AMP the prediction may be FP16; concatenating first
        # would silently change its statistics/gradient to FP32.
        px, tx = prediction, target.detach()
        if self.normalization_precision == 'float32':
            px, tx = px.float(), tx.float()
        ctx = {name: image.detach().float() for name, image in (context or {}).items()}
        if self.normalization != 'none':
            dims = (2, 3, 4)
            target_stats = (tx.mean(dims, keepdim=True), tx.std(dims, keepdim=True))
            pred_stats = ((px.mean(dims, keepdim=True), px.std(dims, keepdim=True))
                          if self.normalization == 'separate' else target_stats)
            px = (px - pred_stats[0]) / (pred_stats[1] + 1e-8)
            tx = (tx - target_stats[0]) / (target_stats[1] + 1e-8)
            ctx = {name: (image - image.mean(dims, keepdim=True)) /
                         (image.std(dims, keepdim=True) + 1e-8) for name, image in ctx.items()}
        px = compose_named_inputs(px, ctx, self.target_modality, self.expert.modalities)
        tx = compose_named_inputs(tx, ctx, self.target_modality, self.expert.modalities)
        fx = self.expert(self._pad(px)).features
        with torch.no_grad():
            fy = self.expert(self._pad(tx)).features
        terms = []
        for name in self.selected:
            a, b = fx[name], fy[name]
            if self.normalize_features:
                a, b = F.instance_norm(a), F.instance_norm(b)
            a, b = a.float(), b.float()
            if self.distance == 'l1':
                terms.append((a - b).abs().mean())
            else:
                terms.append((F.normalize(a, dim=1) - F.normalize(b, dim=1)).square().sum(1).mean())
        return torch.stack(terms).sum()
