"""
nnUNetv2_translation_export_teacher: turns a segmentation network trained outside nnU-Net (a state_dict, e.g. the
ResEnc-L lesion experts of the ISBI 2027 paper) into an nnU-Net v2 model folder that the AFP trainers can use as
teacher (nnUNet_AFP_model):

    OUTPUT/plans.json, OUTPUT/dataset.json, OUTPUT/teacher_export.json, OUTPUT/fold_0/checkpoint_final.pth

The architecture is a dynamic_network_architectures class with its kwargs, given as a json in the nnU-Net plans format
({"network_class_name": ..., "arch_kwargs": {...}, "_kw_requires_import": [...]}) or as the ResEnc-L preset of
nnFoundation. Every key and shape of the network must match the state_dict. The source path and its sha256 are
recorded in the checkpoint and in teacher_export.json.

    nnUNetv2_translation_export_teacher -i expert/checkpoint_ep300.pth --preset resenc_l --num_classes 2 \\
        --modality first_postcontrast --trusted -o teachers/expert_scratch
    export nnUNet_AFP_model=teachers/expert_scratch/fold_0/checkpoint_final.pth
"""
import argparse
import json
import os
import sys
import time
from copy import deepcopy
from typing import Optional, Sequence

import torch
from batchgenerators.utilities.file_and_folder_operations import isfile, join, maybe_mkdir_p, save_json

from nnunetv2.utilities.nnfoundation import RESENC_L, sha256_file, strip_uniform_prefixes

IMPORTS = ['conv_op', 'norm_op', 'dropout_op', 'nonlin']
PRESETS = {
    'resenc_l': {
        'network_class_name': 'dynamic_network_architectures.architectures.unet.ResidualEncoderUNet',
        'arch_kwargs': dict(deepcopy(RESENC_L), conv_op='torch.nn.modules.conv.Conv3d',
                            norm_op='torch.nn.modules.instancenorm.InstanceNorm3d',
                            norm_op_kwargs={'eps': 1e-05, 'affine': True}, dropout_op=None, dropout_op_kwargs=None,
                            nonlin='torch.nn.LeakyReLU', nonlin_kwargs={'inplace': True}),
        '_kw_requires_import': IMPORTS},
}


class ExportError(RuntimeError):
    pass


def load_state_dict(source: str, key: Optional[str], trusted: bool = False) -> dict:
    """state_dict stored under `key` in a torch.save file (key None or '': the file is the state_dict)."""
    if trusted:
        payload = torch.load(source, map_location='cpu', weights_only=False)
    else:
        try:
            payload = torch.load(source, map_location='cpu', weights_only=True)
        except Exception as e:
            raise ExportError(f'{source} cannot be loaded with weights_only=True ({type(e).__name__}): it contains '
                              f'pickled objects (e.g. optimizer or RNG states). If you trust this file (your own '
                              f'checkpoint), rerun with --trusted.') from None
    if key:
        if not isinstance(payload, dict) or key not in payload:
            keys = sorted(map(str, payload)) if isinstance(payload, dict) else []
            raise ExportError(f'{source}: no key {key!r} ({len(keys)} top-level keys: {keys[:8]}'
                              f'{" ..." if len(keys) > 8 else ""}; use --key, or --key \'\' if the file is the '
                              f'state_dict itself)')
        state = payload[key]
    else:
        state = payload
    if not isinstance(state, dict) or not all(torch.is_tensor(v) for v in state.values()):
        raise ExportError(f'{source}[{key!r}] is not a state_dict of tensors')
    return strip_uniform_prefixes(state)


def build_network(architecture: dict, input_channels: int, num_classes: int) -> torch.nn.Module:
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    for k in ('network_class_name', 'arch_kwargs', '_kw_requires_import'):
        if k not in architecture:
            raise ExportError(f'architecture json: missing {k!r} (expected the "architecture" entry of nnU-Net plans)')
    return get_network_from_plans(architecture['network_class_name'], deepcopy(architecture['arch_kwargs']),
                                  architecture['_kw_requires_import'], input_channels, num_classes,
                                  allow_init=False, deep_supervision=False)


def export_teacher(source: str, output_folder: str, architecture: dict, input_channels: int, num_classes: int,
                   key: Optional[str] = 'network_weights', modality: str = 'target',
                   label_names: Optional[Sequence[str]] = None, configuration: str = '3d_fullres',
                   trusted: bool = False, overwrite: bool = False) -> dict:
    checkpoint = join(output_folder, 'fold_0', 'checkpoint_final.pth')
    if isfile(checkpoint) and not overwrite:
        raise ExportError(f'{checkpoint} exists (use --overwrite)')
    label_names = list(label_names) if label_names else [f'class_{i}' for i in range(1, num_classes)]
    if len(label_names) != num_classes - 1:
        raise ExportError(f'{len(label_names)} label names for {num_classes} classes (background excluded)')

    state = load_state_dict(source, key, trusted)
    network = build_network(architecture, input_channels, num_classes)
    own = network.state_dict()
    missing, unexpected = sorted(set(own) - set(state)), sorted(set(state) - set(own))
    shapes = sorted(f'{k}: network {tuple(own[k].shape)}, file {tuple(state[k].shape)}'
                    for k in set(own) & set(state) if own[k].shape != state[k].shape)
    if missing or unexpected or shapes:
        raise ExportError(
            f'{source}[{key!r}] does not match {architecture["network_class_name"]} with {input_channels} input '
            f'channel(s) and {num_classes} classes:\n'
            f'  - {len(missing)} network keys missing from the file {missing[:3]}\n'
            f'  - {len(unexpected)} file keys absent from the network {unexpected[:3]}\n'
            f'  - {len(shapes)} shape mismatches {shapes[:3]}')
    network.load_state_dict(state, strict=True)
    if not all(torch.equal(v, state[k].to(v.dtype)) for k, v in network.state_dict().items()):
        raise ExportError('weights differ from the file after loading')

    provenance = {'source': os.path.abspath(source), 'source_sha256': sha256_file(source), 'key': key,
                  'architecture': architecture['network_class_name'], 'input_channels': input_channels,
                  'num_classes': num_classes, 'tensors': len(state), 'strict_load': True,
                  'created': time.strftime('%Y-%m-%d %H:%M:%S %Z')}
    plans = {'dataset_name': 'Dataset000_ExportedTeacher', 'plans_name': 'exportedTeacherPlans',
             'configurations': {configuration: {'data_identifier': f'exportedTeacherPlans_{configuration}',
                                                'architecture': architecture}},
             'teacher_export': provenance}
    channel_names = {'0': modality} if input_channels == 1 else {str(i): f'{modality}_{i}'
                                                                   for i in range(input_channels)}
    dataset_json = {'channel_names': channel_names,
                    'labels': dict({'background': 0}, **{n: i + 1 for i, n in enumerate(label_names)}),
                    'numTraining': 0, 'file_ending': '.nii.gz'}
    maybe_mkdir_p(join(output_folder, 'fold_0'))
    save_json(plans, join(output_folder, 'plans.json'), sort_keys=False)
    save_json(dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
    save_json(provenance, join(output_folder, 'teacher_export.json'), sort_keys=False)
    torch.save({'network_weights': network.state_dict(), 'trainer_name': 'exported_teacher',
                'init_args': {'plans': plans, 'configuration': configuration, 'fold': 0, 'dataset_json': dataset_json},
                'teacher_export': provenance}, checkpoint)
    return provenance


def main():
    parser = argparse.ArgumentParser(
        description='Exports a segmentation state_dict as an nnU-Net v2 model folder usable as AFP teacher '
                    '(nnUNet_AFP_model). Strict: every key and shape must match the architecture.')
    parser.add_argument('-i', '--input', required=True, help='torch.save file holding the state_dict')
    parser.add_argument('--key', default='network_weights',
                        help="key of the state_dict in the file (default: network_weights; '' = the whole file)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--preset', choices=sorted(PRESETS), help='resenc_l: the ResEnc-L of nnFoundation')
    group.add_argument('--architecture', help='json with network_class_name, arch_kwargs, _kw_requires_import '
                                              '(the "architecture" entry of nnU-Net plans)')
    parser.add_argument('--input_channels', type=int, default=1, help='default: 1 (AFP teachers take one channel)')
    parser.add_argument('--num_classes', type=int, required=True, help='output channels, background included')
    parser.add_argument('--modality', default='target', help='channel name written to dataset.json')
    parser.add_argument('--labels', nargs='+', default=None, help='names of the classes 1..n-1 (default class_i)')
    parser.add_argument('-c', '--configuration', default='3d_fullres', help='configuration name (default 3d_fullres)')
    parser.add_argument('--trusted', action='store_true',
                        help='allow torch.load with weights_only=False (only for checkpoints you trust)')
    parser.add_argument('-o', '--output', required=True, help='output model folder')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.preset:
        architecture = deepcopy(PRESETS[args.preset])
    else:
        with open(args.architecture) as f:
            architecture = json.load(f)
    try:
        provenance = export_teacher(args.input, args.output, architecture, args.input_channels, args.num_classes,
                                    args.key or None, args.modality, args.labels, args.configuration, args.trusted,
                                    args.overwrite)
    except ExportError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
    print(f'{provenance["tensors"]} tensors exported from {provenance["source"]} (sha256 {provenance["source_sha256"]})')
    print(f'export nnUNet_AFP_model={join(os.path.abspath(args.output), "fold_0", "checkpoint_final.pth")}')


if __name__ == '__main__':
    main()
