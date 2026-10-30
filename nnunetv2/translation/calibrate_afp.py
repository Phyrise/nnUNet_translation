"""
nnUNetv2_translation_calibrate_afp: sets the AFP weight lambda of an encoder-AFP trainer by gradient-norm matching
(ISBI 2027 paper):

    lambda = median_b ||grad_theta L1_b|| / median_b ||grad_theta D_b||

over a fixed set of training batches b, with the generator at its initialization (random, or with the nnFoundation
encoder for *_nnFinit trainers), before any update, so that both terms start with the same gradient magnitude. The two
gradients are computed in separate passes, with the precision of training (autocast on cuda, gradients of the losses
scaled by 1024 then unscaled). The json can be given to training as is:

    nnUNetv2_translation_calibrate_afp 50 3d_fullres 0 -tr nnUNetTrainerMRCT_AFPexpert -p nnResUNetPlans
    export nnUNet_AFP_lambda=$nnUNet_results/Dataset050_.../nnUNetTrainerMRCT_AFPexpert__nnResUNetPlans__3d_fullres/fold_0/afp_lambda.json
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import join, save_json
from torch import autocast

from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels


def parameter_gradient_norm(loss: torch.Tensor, network: torch.nn.Module, scale: float = 1024.) -> float:
    """||grad_theta loss|| over the trainable parameters; the loss is scaled before backward (float16 range)."""
    grads = torch.autograd.grad(loss * scale, [p for p in network.parameters() if p.requires_grad], allow_unused=True)
    norm = float(torch.sqrt(sum(g.float().square().sum() for g in grads if g is not None))) / scale
    del grads
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError(f'non-finite or zero calibration gradient norm ({norm})')
    return norm


def calibrate(dataset_name_or_id: str, configuration: str, fold: int, trainer_name: str, plans_identifier: str,
              n_batches: int = 64, seed: int = 0, output_file: str = None,
              device: torch.device = torch.device('cuda')) -> dict:
    from nnunetv2.run.run_training import get_trainer_from_args
    from nnunetv2.training.nnUNetTrainer.variants.translation.nnUNetTrainerMRCT import _EncoderAFPMixin, \
        _NNFInitMixin

    os.environ['nnUNet_n_proc_DA'] = '0'  # single-threaded loader: the batches only depend on the seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, trainer_name, plans_identifier,
                                    device=device)
    if not isinstance(trainer, _EncoderAFPMixin):
        raise RuntimeError(f'{trainer_name} has no encoder AFP term (use e.g. nnUNetTrainerMRCT_AFPexpert)')

    # generator at its initialization, as in trainer.initialize() (the loss is built without lambda)
    cm = trainer.configuration_manager
    trainer.num_input_channels = determine_num_input_channels(trainer.plans_manager, cm, trainer.dataset_json)
    trainer.network = trainer.build_network_architecture(
        cm.network_arch_class_name, cm.network_arch_init_kwargs, cm.network_arch_init_kwargs_req_import,
        trainer.num_input_channels, trainer.label_manager.num_segmentation_heads,
        trainer.enable_deep_supervision).to(trainer.device)
    init = trainer.load_nnfoundation_encoder() if isinstance(trainer, _NNFInitMixin) else 'random'
    network = trainer.network.train()
    distance = trainer.build_afp_distance().train()
    dataloader, _ = trainer.get_dataloaders()

    use_autocast = trainer.device.type == 'cuda'
    rows = []
    start = time.time()
    for b in range(n_batches):
        data, target = trainer._to_device(next(dataloader))
        with autocast(trainer.device.type, enabled=True) if use_autocast else dummy_context():
            l1 = F.l1_loss(network(data), target)
        l1_norm = parameter_gradient_norm(l1, network)
        del l1
        with autocast(trainer.device.type, enabled=True) if use_autocast else dummy_context():
            d = distance(network(data), target)
        afp_norm = parameter_gradient_norm(d, network)
        del d
        rows.append({'batch': b, 'L1_grad_norm': l1_norm, 'AFP_grad_norm': afp_norm})
        print(f'batch {b + 1}/{n_batches}: ||grad L1|| {l1_norm:.4g}  ||grad D|| {afp_norm:.4g}', flush=True)

    median_l1 = float(np.median([r['L1_grad_norm'] for r in rows]))
    median_afp = float(np.median([r['AFP_grad_norm'] for r in rows]))
    result = {'lambda': median_l1 / median_afp,
              'rule': 'median ||grad_theta L1|| / median ||grad_theta D|| over fixed training batches, generator at '
                      'its initialization',
              'median_L1_grad_norm': median_l1, 'median_AFP_grad_norm': median_afp,
              'trainer': trainer_name, 'dataset': trainer.plans_manager.dataset_name, 'plans': plans_identifier,
              'configuration': configuration, 'fold': fold, 'batches': n_batches, 'batch_size': trainer.batch_size,
              'seed': seed, 'device': str(trainer.device), 'autocast': use_autocast, 'loss_scale': 1024.,
              'teacher': distance.provenance, 'generator_initialization': init,
              'seconds': time.time() - start, 'created': time.strftime('%Y-%m-%d %H:%M:%S %Z'), 'rows': rows}
    output_file = output_file or join(trainer.output_folder, 'afp_lambda.json')
    save_json(result, output_file, sort_keys=False)
    trainer.print_to_log_file(f'AFP lambda calibration: lambda = {result["lambda"]:.6g} '
                              f'(median ||grad L1|| {median_l1:.4g} / median ||grad D|| {median_afp:.4g}, '
                              f'{n_batches} batches) -> {output_file}')
    print(f'export nnUNet_AFP_lambda={output_file}')
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Calibrates the AFP weight lambda of an encoder-AFP trainer by gradient-norm matching: '
                    'lambda = median ||grad L1|| / median ||grad D|| on fixed training batches at initialization. '
                    'The teacher is set as for training (nnUNet_AFP_model or nnUNet_nnF_model, '
                    'nnUNet_AFP_stage_weights).',
        epilog='Example: nnUNetv2_translation_calibrate_afp 50 3d_fullres 0 -tr nnUNetTrainerMRCT_AFPexpert '
               '-p nnResUNetPlans')
    parser.add_argument('dataset_name_or_id', type=str, help='Dataset name or ID to train with')
    parser.add_argument('configuration', type=str, help='Configuration, e.g. 3d_fullres')
    parser.add_argument('fold', type=int, help='Fold (its training cases provide the batches)')
    parser.add_argument('-tr', type=str, required=True, help='Encoder-AFP trainer, e.g. nnUNetTrainerMRCT_AFPexpert')
    parser.add_argument('-p', type=str, default='nnUNetPlans', help='Plans identifier (default: nnUNetPlans)')
    parser.add_argument('--batches', type=int, default=64, help='Number of training batches (default: 64)')
    parser.add_argument('--seed', type=int, default=0, help='Seed of the initialization and of the batches')
    parser.add_argument('-o', type=str, default=None,
                        help='Output json (default: afp_lambda.json in the output folder of the trainer)')
    parser.add_argument('-device', type=str, default='cuda', choices=['cuda', 'cpu', 'mps'],
                        help='Device (default: cuda)')
    args = parser.parse_args()
    if args.device == 'cpu':
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    calibrate(args.dataset_name_or_id, args.configuration, args.fold, args.tr, args.p, args.batches, args.seed,
              args.o, torch.device(args.device))


if __name__ == '__main__':
    main()
