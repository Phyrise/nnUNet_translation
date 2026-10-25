"""
Anatomical Feature-Prioritized (AFP) loss (Longuefosse et al., Physics in Medicine & Biology 2025,
https://doi.org/10.1088/1361-6560/adea07).

The prediction and the target go through a frozen nnU-Net segmentation network trained on the target modality.
The loss is the L1 distance between their feature maps (encoder stages except the two deepest, every decoder stage),
plus mae_weight * L1 between the images. Any nnU-Net v2 checkpoint can be used (e.g. a model trained on
TotalSegmentator labels): the network is rebuilt from the plans stored in the checkpoint.
"""
import torch
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import isdir, isfile, join
from torch import nn

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def load_segmentation_network(checkpoint_path: str) -> nn.Module:
    """Rebuilds a trained nnU-Net v2 network from its checkpoint (checkpoint_final.pth or its fold folder)."""
    if isdir(checkpoint_path):
        checkpoint_path = join(checkpoint_path, 'checkpoint_final.pth')
    if not isfile(checkpoint_path):
        raise FileNotFoundError(f'AFP: segmentation checkpoint not found: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    init_args = checkpoint['init_args']
    configuration = PlansManager(init_args['plans']).get_configuration(init_args['configuration'])

    state_dict = {}
    for k, v in checkpoint['network_weights'].items():
        for prefix in ('module.', '_orig_mod.'):
            if k.startswith(prefix):
                k = k[len(prefix):]
        state_dict[k] = v
    num_input_channels = len(init_args['dataset_json']['channel_names'])
    if num_input_channels != 1:
        raise ValueError(f'AFP: the segmentation network must take one channel, it takes {num_input_channels}')
    seg_layer_keys = sorted(k for k in state_dict if k.startswith('decoder.seg_layers.') and k.endswith('.weight'))
    num_classes = state_dict[seg_layer_keys[0]].shape[0]

    network = get_network_from_plans(configuration.network_arch_class_name, configuration.network_arch_init_kwargs,
                                     configuration.network_arch_init_kwargs_req_import, num_input_channels,
                                     num_classes, allow_init=False, deep_supervision=False)
    network.load_state_dict(state_dict)
    return network


class AFP(nn.Module):
    def __init__(self, checkpoint_path: str, layers=None, mae_weight: float = 0.0, normalize_before_L1: bool = False,
                 device: torch.device = torch.device('cuda')):
        super().__init__()
        self.model = load_segmentation_network(checkpoint_path).to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # feature maps, in this order: encoder stages except the two deepest, decoder stages, final logits
        n_stages = len(self.model.encoder.stages)
        hooked = list(self.model.encoder.stages[:n_stages - 2]) + list(self.model.decoder.stages) + \
                 [self.model.decoder.seg_layers[-1]]
        self._features = []
        for module in hooked:
            module.register_forward_hook(lambda m, i, o: self._features.append(o))
        # default: every feature map except the logits
        self.layers = layers if layers is not None else list(range(len(hooked) - 1))
        # inputs are padded so that every axis is divisible by 2 ** (n_stages - 1)
        self.pad_factor = 2 ** (n_stages - 1)

        self.L1 = nn.L1Loss()
        self.mae_weight = mae_weight
        self.normalize_before_L1 = normalize_before_L1
        print(f'AFP: {checkpoint_path}, {len(hooked)} feature maps, layers {self.layers}, mae_weight {mae_weight}')

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()  # the segmentation network stays frozen (instance norm, dropout)
        return self

    def center_pad(self, x):
        pad = []
        for s in reversed(x.shape[2:]):  # F.pad expects the last axis first
            total = (s + self.pad_factor - 1) // self.pad_factor * self.pad_factor - s
            pad.extend([total // 2, total - total // 2])
        return F.pad(x, pad, mode='constant', value=0)

    def _feature_maps(self, x):
        self._features = []
        self.model(x)
        features, self._features = self._features, []
        return features

    def forward(self, x, y):
        x = self.center_pad(x)
        y = self.center_pad(y)
        emb_x = self._feature_maps(x)
        with torch.no_grad():
            emb_y = self._feature_maps(y)

        loss = 0
        for i in self.layers:
            fx, fy = emb_x[i], emb_y[i]
            if self.normalize_before_L1:
                fx, fy = F.instance_norm(fx), F.instance_norm(fy)
            loss = loss + self.L1(fx, fy)

        if self.mae_weight > 0:
            loss = loss + self.L1(x, y.float()) * self.mae_weight
        return loss
