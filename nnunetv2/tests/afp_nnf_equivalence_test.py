"""
Equivalence of the ISBI 2027 options with the code that produced the paper results, in strict float32 (TF32 off):

  (i)   encoder AFP (loss/afp_encoder.py) vs AFPFeatureDistance of the tag isbi2027-results: loss value and generator
        gradients, for nnU-Net expert teachers (PlainConvUNet, ResidualEncoderUNet) and the nnFoundation teacher, and
        per-stage weights
  (ii)  nnFoundation encoder initialization (utilities/nnfoundation.py) vs the paper's init_encoder: every tensor of
        the generator after initialization, plus the ResEnc-L geometry check
  (iii) learning-rate curves of the new trainers vs the paper's lr_at
  (iv)  gradient norm of the lambda calibration (translation/calibrate_afp.py) vs the paper's norm_grad

Teachers and checkpoints are random and created by the test (no real weights needed). About a minute on CPU:

    python -m nnunetv2.tests.afp_nnf_equivalence_test
    python -m nnunetv2.tests.afp_nnf_equivalence_test --reference path/to/afp_features.py   # outside a git checkout

The reference afp_features.py is read with `git show isbi2027-results:nnunetv2/training/loss/afp_features.py` unless
--reference is given. Prints every measured gap; exits with status 0 and 'EQUIVALENCE TESTS PASSED' when all gaps are
within tolerance.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import types
from copy import deepcopy

import numpy as np
import torch
from torch import nn

TAG = 'isbi2027-results'
TOLERANCE = 1e-6  # relative, float32
# stage weights are checked against a per-stage sum of the tag distance, whose gradients are accumulated in another
# order (one teacher pass per stage): float32 rounding of that order is a few 1e-6 (printed by the test)
WEIGHTED_TOLERANCE = 1e-5
IMPORTS = ['conv_op', 'norm_op', 'dropout_op', 'nonlin']
COMMON = dict(conv_op='torch.nn.modules.conv.Conv3d', conv_bias=True,
              norm_op='torch.nn.modules.instancenorm.InstanceNorm3d', norm_op_kwargs={'eps': 1e-05, 'affine': True},
              dropout_op=None, dropout_op_kwargs=None, nonlin='torch.nn.LeakyReLU', nonlin_kwargs={'inplace': True})
EXPERTS = {
    'PlainConvUNet expert': dict(
        network_class_name='dynamic_network_architectures.architectures.unet.PlainConvUNet', _kw_requires_import=IMPORTS,
        arch_kwargs=dict(n_stages=5, features_per_stage=[8, 16, 32, 64, 64],
                         kernel_sizes=[[1, 3, 3]] + [[3, 3, 3]] * 4,
                         strides=[[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                         n_conv_per_stage=[2] * 5, n_conv_per_stage_decoder=[2] * 4, **COMMON)),
    'ResidualEncoderUNet expert': dict(
        network_class_name='dynamic_network_architectures.architectures.unet.ResidualEncoderUNet',
        _kw_requires_import=IMPORTS,
        arch_kwargs=dict(n_stages=4, features_per_stage=[8, 16, 32, 64], kernel_sizes=[[3, 3, 3]] * 4,
                         strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [1, 2, 2]], n_blocks_per_stage=[1, 2, 2, 2],
                         n_conv_per_stage_decoder=[1] * 3, **COMMON)),
}
# generator of the paper (generator_architecture.json of the ISBI 2027 runs): ResEnc-L encoder, last stride (1, 2, 2)
PAPER_GENERATOR = dict(
    network_class_name='dynamic_network_architectures.architectures.unet.ResidualUNet', _kw_requires_import=IMPORTS,
    arch_kwargs=dict(n_stages=6, features_per_stage=[32, 64, 128, 256, 320, 320], kernel_sizes=[[3, 3, 3]] * 6,
                     strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
                     n_blocks_per_stage=[1, 3, 4, 6, 6, 6], n_conv_per_stage_decoder=[2] * 5, decoder_type='trilinear',
                     **COMMON))
FAILURES = []


def encoder_stages(n):
    return tuple(f'encoder.{i}' for i in range(n))


# ---------------------------------------------------------------------------------------------------------------------
# Reference code of the paper, copied verbatim (hard-coded paths and sha checks removed)
# ---------------------------------------------------------------------------------------------------------------------
def paper_build_resenc_l(input_channels=1, num_classes=1):  # men_clinical/foundation_afp.py
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
    RESENC_L = dict(n_stages=6, features_per_stage=[32, 64, 128, 256, 320, 320], kernel_sizes=[[3, 3, 3]] * 6,
                    strides=[[1, 1, 1]] + [[2, 2, 2]] * 5, n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
                    n_conv_per_stage_decoder=[1, 1, 1, 1, 1], conv_bias=True)
    return ResidualEncoderUNet(input_channels=input_channels, conv_op=nn.Conv3d, num_classes=num_classes,
                               norm_op=nn.InstanceNorm3d, norm_op_kwargs={'eps': 1e-5, 'affine': True},
                               nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=False,
                               **RESENC_L)


def paper_init_encoder(model, foundation):  # mama_3d common.init_encoder (r6 = r10 = r11)
    ck = torch.load(foundation, map_location='cpu', weights_only=True)
    weights = ck['network_weights']
    enc = {k: v for k, v in weights.items() if k.startswith('encoder.')}
    own = model.state_dict(); keys = {k for k in own if k.startswith('encoder.')}
    assert set(enc) == keys
    out = model.load_state_dict(enc, strict=False)
    assert not out.unexpected_keys and all(k.startswith('decoder.') for k in out.missing_keys)
    after = model.state_dict()
    assert all(torch.equal(after[k], v) for k, v in enc.items())
    return {'checkpoint': str(foundation), 'encoder_tensors': len(enc), 'exact': True}


def paper_lr_at(epoch, peak, warm, horizon):  # mama_3d common.lr_at
    if epoch < warm: return peak * (epoch + 1) / warm
    return peak * max(0., 1 - (epoch - warm) / (horizon - warm)) ** .9


def paper_norm_grad(loss, model, scale=1024.):  # mama_3d calibrate.norm_grad
    grads = torch.autograd.grad(loss * scale, [p for p in model.parameters() if p.requires_grad], allow_unused=True)
    norm = float(torch.sqrt(sum(g.float().square().sum() for g in grads if g is not None))) / scale
    del grads
    if not np.isfinite(norm) or norm <= 0: raise RuntimeError('nonfinite or zero calibration gradient')
    return norm


# ---------------------------------------------------------------------------------------------------------------------
def load_reference(path=None) -> types.ModuleType:
    """afp_features.py of the tag, executed as a module without writing bytecode."""
    if path is None:
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        result = subprocess.run(['git', '-C', repo, 'show', f'{TAG}:nnunetv2/training/loss/afp_features.py'],
                                capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f'Cannot read the reference from the tag {TAG} ({result.stderr.strip()}). Fetch the tag '
                     f'(git fetch --tags) or pass --reference path/to/afp_features.py')
        source, origin = result.stdout, f'{TAG}:nnunetv2/training/loss/afp_features.py'
    else:
        with open(path) as f:
            source, origin = f.read(), path
    module = types.ModuleType('isbi2027_afp_features')
    sys.modules[module.__name__] = module
    exec(compile(source, origin, 'exec'), module.__dict__)
    print(f'reference: {origin}')
    return module


def report(name, value, tolerance):
    ok = value <= tolerance
    print(f'    {name:<44s} {value:.3e}  ({"ok" if ok else "FAILED"}, tolerance {tolerance:.0e})')
    if not ok:
        FAILURES.append(name)


def compare(label, loss_new, grads_new, loss_ref, grads_ref, tolerance=TOLERANCE):
    """Prints the gaps; tolerance None = informative only."""
    loss_abs = abs(float(loss_new) - float(loss_ref))
    grad_abs = max(float((a.double() - b.double()).abs().max()) for a, b in zip(grads_new, grads_ref))
    grad_scale = max(float(b.double().abs().max()) for b in grads_ref)
    identical = float(loss_new) == float(loss_ref) and all(torch.equal(a, b) for a, b in zip(grads_new, grads_ref))
    print(f'  {label}: loss {float(loss_new):.6f} (reference {float(loss_ref):.6f}), '
          f'{"bitwise identical" if identical else "not bitwise identical"}')
    print(f'    loss max abs gap {loss_abs:.3e}, generator gradient max abs gap {grad_abs:.3e} '
          f'(max |grad| {grad_scale:.3e})')
    if tolerance is None:
        print(f'    loss relative gap {loss_abs / abs(float(loss_ref)):.3e}, gradient relative gap '
              f'{grad_abs / grad_scale:.3e} (informative)')
        return
    report('loss relative gap', loss_abs / abs(float(loss_ref)), tolerance)
    report('gradient relative gap (max abs gap / max |grad|)', grad_abs / grad_scale, tolerance)


def small_generator(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Conv3d(1, 4, 3, padding=1), nn.LeakyReLU(0.01), nn.Conv3d(4, 1, 3, padding=1))


def loss_and_grads(distance, generator, x, y):
    loss = distance(generator(x), y)
    return loss.detach(), [g.detach() for g in torch.autograd.grad(loss, list(generator.parameters()))]


def write_expert_checkpoint(folder, architecture, seed) -> str:
    """nnU-Net v2 layout: folder/{plans.json,dataset.json} and folder/fold_0/checkpoint_final.pth (random weights)."""
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    plans = {'dataset_name': 'Dataset999_Expert', 'plans_name': 'nnUNetPlans',
             'configurations': {'3d_fullres': {'data_identifier': 'nnUNetPlans_3d_fullres', 'batch_size': 2,
                                               'patch_size': [32, 64, 64], 'architecture': architecture}}}
    dataset_json = {'channel_names': {'0': 'CT'}, 'labels': {'background': 0, 'a': 1, 'b': 2}, 'numTraining': 1,
                    'file_ending': '.nii.gz'}
    torch.manual_seed(seed)
    network = get_network_from_plans(architecture['network_class_name'], deepcopy(architecture['arch_kwargs']),
                                     architecture['_kw_requires_import'], 1, 3, allow_init=True,
                                     deep_supervision=False)
    os.makedirs(os.path.join(folder, 'fold_0'), exist_ok=True)
    for name, content in (('plans.json', plans), ('dataset.json', dataset_json)):
        with open(os.path.join(folder, name), 'w') as f:
            json.dump(content, f)
    checkpoint = os.path.join(folder, 'fold_0', 'checkpoint_final.pth')
    torch.save({'network_weights': network.state_dict(), 'trainer_name': 'nnUNetTrainer',
                'init_args': {'plans': plans, 'configuration': '3d_fullres', 'fold': 0,
                              'dataset_json': dataset_json}}, checkpoint)
    return checkpoint


def write_nnf_checkpoint(path, seed) -> str:
    """Random ResEnc-L in the nnssl checkpoint format of nnFoundationCNN."""
    from nnunetv2.utilities.nnfoundation import build_resenc_l
    torch.manual_seed(seed)
    network = build_resenc_l()
    plan = {'architecture_plans': {'arch_class_name': 'ResEncL', 'arch_kwargs': None,
                                   'arch_kwargs_requiring_import': None},
            'pretrain_num_input_channels': 1, 'key_to_encoder': 'encoder.stages', 'key_to_stem': 'encoder.stem'}
    torch.save({'network_weights': network.state_dict(), 'nnssl_adaptation_plan': plan, 'citations': []}, path)
    return path


# ---------------------------------------------------------------------------------------------------------------------
def test_afp(ref, workdir, nnf_checkpoint, device):
    from nnunetv2.training.loss.afp_encoder import expert_encoder_afp, nnfoundation_encoder_afp
    print('\n(i) encoder AFP vs the tag AFPFeatureDistance (same teacher weights, random inputs)')
    for i, (label, architecture) in enumerate(EXPERTS.items()):
        checkpoint = write_expert_checkpoint(os.path.join(workdir, f'expert{i}'), architecture, seed=10 + i)
        new = expert_encoder_afp(checkpoint, device=device)
        expert = ref.NNUNetFeatureExpert.from_checkpoint(checkpoint, device=device, dtype=torch.float32)
        reference = ref.AFPFeatureDistance(expert, features=encoder_stages(new.n_stages))
        generator = small_generator(20 + i).to(device)
        torch.manual_seed(30 + i)
        x = torch.randn(2, 1, 20, 36, 44, device=device)
        y = 3 * torch.randn(2, 1, 20, 36, 44, device=device) + 5
        compare(f'{label} ({new.n_stages} encoder stages)', *loss_and_grads(new, generator, x, y),
                *loss_and_grads(reference, generator, x, y))
        if i == 0:
            # per-stage weights vs sum_l w_l * (tag distance restricted to stage l). That reference runs one teacher
            # pass per stage, so the generator gradients are accumulated in another order: the same reference summed
            # in reverse stage order gives the float32 rounding level of this comparison.
            weights = (0.5, 1.0, 2.0, 0.0, 1.5)
            weighted = expert_encoder_afp(checkpoint, stage_weights=weights, device=device)
            per_stage = list(zip(weights, [ref.AFPFeatureDistance(expert, features=(name,))
                                           for name in encoder_stages(len(weights))]))
            forward_sum = lambda p, t: sum(w * d(p, t) for w, d in per_stage)
            reverse_sum = lambda p, t: sum(w * d(p, t) for w, d in reversed(per_stage))
            reference_result = loss_and_grads(forward_sum, generator, x, y)
            compare('  rounding level: reference summed in reverse stage order',
                    *loss_and_grads(reverse_sum, generator, x, y), *reference_result, tolerance=None)
            compare(f'{label}, stage weights {weights}', *loss_and_grads(weighted, generator, x, y),
                    *reference_result, tolerance=WEIGHTED_TOLERANCE)

    new = nnfoundation_encoder_afp(nnf_checkpoint, device=device)
    network = paper_build_resenc_l()
    ref.load_strict_weights(network, torch.load(nnf_checkpoint, map_location='cpu', weights_only=True)['network_weights'])
    reference = ref.AFPFeatureDistance(ref.NNUNetFeatureExpert(network.to(device), ['t1c']),
                                       features=encoder_stages(6))
    generator = small_generator(40).to(device)
    torch.manual_seed(41)
    x = torch.randn(1, 1, 20, 40, 44, device=device)
    y = 2 * torch.randn(1, 1, 20, 40, 44, device=device) - 1
    compare('nnFoundation teacher (ResEnc-L, 6 encoder stages)', *loss_and_grads(new, generator, x, y),
            *loss_and_grads(reference, generator, x, y))


def test_init(nnf_checkpoint):
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.nnfoundation import check_resenc_l_plans, init_encoder_from_nnfoundation
    print('\n(ii) nnFoundation encoder initialization vs the paper init_encoder (generator of the paper)')
    torch.manual_seed(20261018)
    generator = get_network_from_plans(PAPER_GENERATOR['network_class_name'],
                                       deepcopy(PAPER_GENERATOR['arch_kwargs']),
                                       PAPER_GENERATOR['_kw_requires_import'], 1, 1, allow_init=True,
                                       deep_supervision=False)
    initial = {k: v.clone() for k, v in generator.state_dict().items()}
    paper, new = deepcopy(generator), deepcopy(generator)
    paper_init_encoder(paper, nnf_checkpoint)
    info = init_encoder_from_nnfoundation(new, nnf_checkpoint)
    a, b = paper.state_dict(), new.state_dict()
    source = torch.load(nnf_checkpoint, map_location='cpu', weights_only=True)['network_weights']
    assert a.keys() == b.keys()
    gap = max(float((a[k].double() - b[k].double()).abs().max()) for k in a)
    equal = sum(torch.equal(a[k], b[k]) for k in a)
    encoder = [k for k in b if k.startswith('encoder.')]
    others = [k for k in b if not k.startswith(('encoder.', 'decoder.encoder.'))]
    print(f'  {equal}/{len(a)} generator tensors identical to the paper init, max abs gap {gap:.3e}; '
          f'{info["encoder_tensors"]} encoder tensors copied')
    report('max abs gap, new init vs paper init', gap, 0.)
    report('encoder tensors different from the checkpoint', sum(not torch.equal(b[k], source[k]) for k in encoder), 0)
    report('decoder tensors changed by the init', sum(not torch.equal(b[k], initial[k]) for k in others), 0)
    print(f'  stride differences reported by the init: {info["stride_differences"]}')

    differences = check_resenc_l_plans(PAPER_GENERATOR['network_class_name'], PAPER_GENERATOR['arch_kwargs'], 1)
    report('paper plans: unexpected stride report', float(differences != [(5, [1, 2, 2], [2, 2, 2])]), 0.)
    wrong = {'n_blocks_per_stage': [1, 2, 2, 2, 2, 2], 'features_per_stage': [32, 64, 128, 256, 320, 512],
             'kernel_sizes': [[1, 3, 3]] + [[3, 3, 3]] * 5, 'n_stages': 5, 'conv_bias': False}
    for key, value in wrong.items():
        try:
            check_resenc_l_plans(PAPER_GENERATOR['network_class_name'], dict(PAPER_GENERATOR['arch_kwargs'],
                                                                             **{key: value}), 1)
            report(f'plans with wrong {key} accepted', 1., 0.)
        except ValueError:
            pass
    try:
        check_resenc_l_plans(PAPER_GENERATOR['network_class_name'], PAPER_GENERATOR['arch_kwargs'], 2)
        report('plans with 2 input channels accepted', 1., 0.)
    except ValueError as e:
        print(f'  wrong geometry rejected, e.g.: {str(e).splitlines()[-1].strip()}')


def test_lr():
    from nnunetv2.training.nnUNetTrainer.variants.translation import nnUNetTrainerMRCT as trainers
    print('\n(iii) learning rate per epoch vs the paper lr_at (horizon 300, as in the paper)')
    plans = {'dataset_name': 'Dataset998_LR', 'plans_name': 'nnResUNetPlans',
             'configurations': {'3d_fullres': {'data_identifier': 'x', 'batch_size': 2, 'patch_size': [32, 64, 64],
                                               'architecture': PAPER_GENERATOR}}}
    dataset_json = {'channel_names': {'0': 'MR'}, 'labels': {'background': 0, 'target': 1}, 'numTraining': 1,
                    'file_ending': '.nii.gz'}
    cases = ((trainers.nnUNetTrainerMRCT_nnFinit, 1e-3, 15), (trainers.nnUNetTrainerMRCT_AFPexpert_nnFinit, 1e-3, 15),
             (trainers.nnUNetTrainerMRCT_AFPexpert, 1e-2, 5), (trainers.nnUNetTrainerMRCT_AFPnnF, 1e-2, 5))
    for cls, peak, warm in cases:
        trainer = cls(plans, '3d_fullres', 0, dataset_json, device=torch.device('cpu'))
        trainer.network = nn.Linear(1, 1)  # only the optimizer and the scheduler are used
        trainer.num_epochs = 300
        optimizer, scheduler = trainer.configure_optimizers()
        group = optimizer.param_groups[0]
        assert group['momentum'] == 0.99 and group['nesterov'] and group['weight_decay'] == 3e-5
        lrs = []
        for epoch in range(300):  # nnUNetTrainer.on_train_epoch_start
            scheduler.step(epoch)
            lrs.append(optimizer.param_groups[0]['lr'])
        expected = [paper_lr_at(e, peak, warm, 300) for e in range(300)]
        gap20 = max(abs(a - b) for a, b in zip(lrs[:20], expected[:20]))
        gap300 = max(abs(a - b) for a, b in zip(lrs, expected))
        print(f'  {cls.__name__}: peak {trainer.initial_lr:g}, warm-up {trainer.warmup_epochs} epochs; '
              f'lr epochs 0-4 {[f"{v:.3g}" for v in lrs[:5]]}, epoch {warm} {lrs[warm]:.3g}, epoch 299 {lrs[-1]:.3g}')
        report(f'max abs gap epochs 0-19', gap20 / peak, 0.)
        report(f'max abs gap epochs 0-299', gap300 / peak, 0.)


def test_grad_norm(nnf_checkpoint, device):
    from nnunetv2.training.loss.afp_encoder import nnfoundation_encoder_afp
    from nnunetv2.translation.calibrate_afp import parameter_gradient_norm
    print('\n(iv) calibration gradient norm vs the paper norm_grad')
    distance = nnfoundation_encoder_afp(nnf_checkpoint, device=device)
    generator = small_generator(50).to(device)
    torch.manual_seed(51)
    x, y = torch.randn(1, 1, 20, 40, 44, device=device), torch.randn(1, 1, 20, 40, 44, device=device)
    for name, loss_fn in (('L1', nn.functional.l1_loss), ('D', distance)):
        new = parameter_gradient_norm(loss_fn(generator(x), y), generator)
        old = paper_norm_grad(loss_fn(generator(x), y), generator)
        print(f'  ||grad {name}|| = {new:.6g} (paper {old:.6g})')
        report(f'||grad {name}|| relative gap', abs(new - old) / old, TOLERANCE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', default=None, help='afp_features.py of the tag (default: read with git)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_default_dtype(torch.float32)
    device = torch.device(args.device)
    with tempfile.TemporaryDirectory(prefix='afp_nnf_equivalence_') as workdir:
        for k in ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'):  # trainers of (iii) write their log there
            os.environ[k] = os.path.join(workdir, k)
        ref = load_reference(args.reference)
        nnf_checkpoint = write_nnf_checkpoint(os.path.join(workdir, 'nnf_checkpoint_final.pth'), seed=1)
        test_afp(ref, workdir, nnf_checkpoint, device)
        test_init(nnf_checkpoint)
        test_lr()
        test_grad_norm(nnf_checkpoint, device)
    if FAILURES:
        print(f'\nEQUIVALENCE TESTS FAILED: {FAILURES}')
        sys.exit(1)
    print('\nEQUIVALENCE TESTS PASSED')


if __name__ == '__main__':
    main()
