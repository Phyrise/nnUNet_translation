"""
End-to-end check of the translation pipeline on tiny synthetic data (about 10 minutes on CPU):
prepare -> train (L1 and AFP, 1 short epoch) -> predict (mean and median), then checks outputs. Also runs the ISBI 2027
options with random teachers: lambda calibration, expert and nnFoundation encoder AFP, nnFoundation initialization.

    python -m nnunetv2.tests.translation_smoke_test            # CPU, temporary folder
    python -m nnunetv2.tests.translation_smoke_test --device cuda --output smoke_test_output --keep

Exits with status 0 and prints 'SMOKE TEST PASSED' when everything works.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from glob import glob

import numpy as np
import SimpleITK as sitk

DATASET_ID = 990
N_CASES = 6
SHAPE_ZYX = (24, 40, 48)
SPACING_XYZ = (1.0, 1.0, 1.5)
RAS_DIRECTION = (-1, 0, 0, 0, -1, 0, 0, 0, 1)  # not LPS on purpose: outputs must keep the input orientation


def synthetic_image(array: np.ndarray, origin) -> sitk.Image:
    image = sitk.GetImageFromArray(array.astype(np.float32))
    image.SetSpacing(SPACING_XYZ)
    image.SetOrigin(origin)
    image.SetDirection(RAS_DIRECTION)
    return image


def synthetic_mr(rng, shape_zyx) -> np.ndarray:
    noise = sitk.GetImageFromArray(rng.random(shape_zyx).astype(np.float32))
    mr = sitk.GetArrayFromImage(sitk.SmoothingRecursiveGaussian(noise, 3.0))
    return (mr - mr.min()) / (mr.max() - mr.min()) * 1000


def make_cases(folder: str):
    rng = np.random.default_rng(0)
    for i in range(N_CASES):
        mr = synthetic_mr(rng, SHAPE_ZYX)
        ct = -1000 + 2.5 * mr  # target in HU
        os.makedirs(os.path.join(folder, f'case{i}'), exist_ok=True)
        for name, array in (('mr', mr), ('ct', ct)):
            sitk.WriteImage(synthetic_image(array, (10.0 * i, -5.0, 3.0)),
                            os.path.join(folder, f'case{i}', f'{name}.mha'))


def run(cmd, env):
    print('\n$ ' + ' '.join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)


def entry(module_function: str, *args):
    """Calls a console entry point through the current interpreter (works without the scripts on PATH)."""
    module, function = module_function.split(':')
    return [sys.executable, '-c', f'from {module} import {function}; {function}()', *args]


def make_fake_segmentation_checkpoint(target_dataset_id: int, output_file: str):
    """Random-weight nnU-Net segmentation model of the target modality, enough to exercise the AFP loss."""
    import torch
    from nnunetv2.experiment_planning.plan_and_preprocess_api import plan_experiment_dataset
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
    from nnunetv2.paths import nnUNet_raw

    plans, _ = plan_experiment_dataset(target_dataset_id)  # default PlainConvUNet planner
    configuration = PlansManager(plans).get_configuration('3d_fullres')
    network = get_network_from_plans(configuration.network_arch_class_name, configuration.network_arch_init_kwargs,
                                     configuration.network_arch_init_kwargs_req_import, 1, 3, allow_init=True,
                                     deep_supervision=False)
    with open(os.path.join(nnUNet_raw, convert_id_to_dataset_name(target_dataset_id), 'dataset.json')) as f:
        dataset_json = json.load(f)
    dataset_json['labels'] = {'background': 0, 'a': 1, 'b': 2}
    torch.save({'network_weights': network.state_dict(), 'trainer_name': 'nnUNetTrainer',
                'init_args': {'plans': plans, 'configuration': '3d_fullres', 'fold': 0,
                              'dataset_json': dataset_json}}, output_file)


def make_fake_nnfoundation_checkpoint(output_file: str):
    """Random-weight ResEnc-L in the nnssl checkpoint format of nnFoundationCNN."""
    import torch
    from nnunetv2.utilities.nnfoundation import build_resenc_l
    plan = {'architecture_plans': {'arch_class_name': 'ResEncL'}, 'pretrain_num_input_channels': 1}
    torch.save({'network_weights': build_resenc_l().state_dict(), 'nnssl_adaptation_plan': plan}, output_file)


def make_resenc_l_plans(nnunet_preprocessed: str, plans_name: str):
    """Copy of nnResUNetPlans whose 3d_fullres generator has the ResEnc-L encoder of nnFoundation, with the last stride
    (1, 2, 2) of the ISBI 2027 generator."""
    folder = os.path.join(nnunet_preprocessed, f'Dataset{DATASET_ID:03d}_Smoke')
    with open(os.path.join(folder, 'nnResUNetPlans.json')) as f:
        plans = json.load(f)
    plans['plans_name'] = plans_name
    configuration = plans['configurations']['3d_fullres']
    configuration['patch_size'] = [32, 64, 64]
    configuration['architecture']['arch_kwargs'].update(
        n_stages=6, features_per_stage=[32, 64, 128, 256, 320, 320], kernel_sizes=[[3, 3, 3]] * 6,
        strides=[[1, 1, 1]] + [[2, 2, 2]] * 4 + [[1, 2, 2]], n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
        n_conv_per_stage_decoder=[2] * 5)
    with open(os.path.join(folder, f'{plans_name}.json'), 'w') as f:
        json.dump(plans, f, indent=2)


def check_training(root: str, trainer: str, plans: str) -> str:
    """Prints the validation MAE of a finished smoke training and returns its output folder."""
    folder = os.path.join(glob(os.path.join(root, 'nnUNet_results', '*'))[0], f'{trainer}__{plans}__3d_fullres',
                          'fold_0')
    summary = os.path.join(folder, 'validation', 'summary.json')
    assert os.path.isfile(summary), f'{trainer}: validation summary missing'
    with open(summary) as f:
        print(f"{trainer}: validation MAE = {json.load(f)['mean']['MAE']:.1f} HU")
    return folder


def training_log(folder: str) -> str:
    logs = sorted(glob(os.path.join(folder, 'training_log_*.txt')))
    assert logs, f'{folder}: no training log'
    return ''.join(open(log).read() for log in logs)


def check_isbi2027_options(root: str, env: dict, expert_checkpoint: str, device: str):
    """Encoder AFP (expert and nnFoundation teachers), lambda calibration, nnFoundation initialization and recipe."""
    import torch
    nnf_checkpoint = os.path.join(root, 'fake_nnfoundation_checkpoint.pth')
    make_fake_nnfoundation_checkpoint(nnf_checkpoint)
    env = dict(env, nnUNet_AFP_model=expert_checkpoint, nnUNet_nnF_model=nnf_checkpoint)

    trainer = 'nnUNetTrainerMRCT_AFPexpert_smoketest'
    run(entry('nnunetv2.translation.calibrate_afp:main', str(DATASET_ID), '3d_fullres', '0', '-tr', trainer,
              '-p', 'nnResUNetPlans', '--batches', '3', '-device', device), env)
    calibration = glob(os.path.join(root, 'nnUNet_results', '*', f'{trainer}__nnResUNetPlans__3d_fullres', 'fold_0',
                                    'afp_lambda.json'))
    assert len(calibration) == 1, 'calibration json missing'
    with open(calibration[0]) as f:
        afp_lambda = json.load(f)['lambda']
    assert 0 < afp_lambda < float('inf'), afp_lambda
    run(entry('nnunetv2.run.run_training:run_training_entry', str(DATASET_ID), '3d_fullres', '0', '-tr', trainer,
              '-p', 'nnResUNetPlans', '-device', device), dict(env, nnUNet_AFP_lambda=calibration[0]))
    assert f'AFP lambda: {afp_lambda}' in training_log(check_training(root, trainer, 'nnResUNetPlans'))

    trainer = 'nnUNetTrainerMRCT_AFPnnF_smoketest'
    run(entry('nnunetv2.run.run_training:run_training_entry', str(DATASET_ID), '3d_fullres', '0', '-tr', trainer,
              '-p', 'nnResUNetPlans', '-device', device), dict(env, nnUNet_AFP_lambda='0.5'))
    check_training(root, trainer, 'nnResUNetPlans')

    trainer, plans = 'nnUNetTrainerMRCT_AFPexpert_nnFinit_smoketest', 'nnResUNetPlans_nnF'
    make_resenc_l_plans(env['nnUNet_preprocessed'], plans)
    run(entry('nnunetv2.run.run_training:run_training_entry', str(DATASET_ID), '3d_fullres', '0', '-tr', trainer,
              '-p', plans, '-device', device), dict(env, nnUNet_AFP_lambda='0.5'))
    folder = check_training(root, trainer, plans)
    log = training_log(folder)
    assert 'Generator encoder initialized from nnFoundation: 448 tensors' in log, 'nnFoundation init not logged'
    assert 'stage 5: (1, 2, 2) instead of (2, 2, 2)' in log, 'stride caveat not logged'
    lr = torch.load(os.path.join(folder, 'checkpoint_final.pth'), map_location='cpu',
                    weights_only=False)['logging']['lrs'][0]
    assert abs(lr - 1e-3 / 15) < 1e-12, f'nnF recipe: epoch 0 lr {lr}, expected {1e-3 / 15}'

    # inference does not need the teacher or the nnFoundation checkpoint
    input_folder = os.path.join(root, 'predict_input_nnf')
    os.makedirs(input_folder, exist_ok=True)
    shutil.copy(os.path.join(root, 'nnUNet_raw', f'Dataset{DATASET_ID:03d}_Smoke', 'imagesTr', 'case0_0000.nii.gz'),
                input_folder)
    out = os.path.join(root, 'predict_nnf')
    plain_env = {k: v for k, v in env.items() if k not in ('nnUNet_AFP_model', 'nnUNet_nnF_model', 'nnUNet_AFP_lambda')}
    run(entry('nnunetv2.inference.predict_from_raw_data:predict_entry_point', '-i', input_folder, '-o', out,
              '-d', str(DATASET_ID), '-c', '3d_fullres', '-p', plans, '-tr', trainer, '-f', '0', '-device', device,
              '-npp', '1', '-nps', '1'), plain_env)
    check_prediction(os.path.join(out, 'case0.nii.gz'), os.path.join(input_folder, 'case0_0000.nii.gz'))


def check_prediction(prediction_file: str, reference_file: str):
    pred, ref = sitk.ReadImage(prediction_file), sitk.ReadImage(reference_file)
    assert pred.GetPixelID() == sitk.sitkFloat32, f'{prediction_file}: expected float32 output'
    assert pred.GetSize() == ref.GetSize(), f'{prediction_file}: size {pred.GetSize()} != {ref.GetSize()}'
    for name in ('GetSpacing', 'GetOrigin', 'GetDirection'):
        assert np.allclose(getattr(pred, name)(), getattr(ref, name)(), atol=1e-4), \
            f'{prediction_file}: {name} differs from the input'
    assert np.all(np.isfinite(sitk.GetArrayViewFromImage(pred))), f'{prediction_file}: non-finite values'


def check_target_units(nnunet_preprocessed: str, nnunet_raw: str):
    """The stored translation info must map the preprocessed targets back to HU."""
    from nnunetv2.translation.intensity import denormalize
    dataset = f'Dataset{DATASET_ID:03d}_Smoke'
    with open(os.path.join(nnunet_preprocessed, dataset, 'dataset.json')) as f:
        info = json.load(f)['translation']
    assert info['target_normalization'] == 'CTNormalization_clip', info
    target = np.load(os.path.join(nnunet_preprocessed, dataset, 'nnUNetPlans_3d_fullres', 'case0.npz'))['seg']
    raw = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(nnunet_raw, f'Dataset{DATASET_ID + 1:03d}_SmokeTarget',
                                                             'imagesTr', 'case0_0000.nii.gz')))
    hu = denormalize(target, info['target_normalization'], info['target_intensity_properties'])
    assert abs(hu.mean() - np.clip(raw, -1024, 3071).mean()) < 1, (hu.mean(), raw.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--output', default=None, help='Working folder (default: a temporary folder)')
    parser.add_argument('--keep', action='store_true', help='Keep the working folder')
    args = parser.parse_args()

    root = os.path.abspath(args.output) if args.output else tempfile.mkdtemp(prefix='nnunet_translation_smoke_')
    os.makedirs(root, exist_ok=True)
    env = dict(os.environ)
    for k in ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'):
        env[k] = os.environ[k] = os.path.join(root, k)
    env.update({'nnUNet_n_proc_DA': '1', 'nnUNet_def_n_proc': '2', 'OMP_NUM_THREADS': '4'})
    os.environ.update({'nnUNet_def_n_proc': '2'})
    try:
        make_cases(os.path.join(root, 'cases'))

        run(entry('nnunetv2.translation.prepare:main', '-d', str(DATASET_ID), '-n', 'Smoke',
                  '-i', os.path.join(root, 'cases', '*', 'mr.mha'), '-t', os.path.join(root, 'cases', '*', 'ct.mha'),
                  '-np', '2'), env)
        check_target_units(env['nnUNet_preprocessed'], env['nnUNet_raw'])

        ckpt = os.path.join(root, 'fake_segmentation_checkpoint.pth')
        make_fake_segmentation_checkpoint(DATASET_ID + 1, ckpt)
        for trainer in ('nnUNetTrainerMRCT_smoketest', 'nnUNetTrainerMRCT_AFP_smoketest'):
            run(entry('nnunetv2.run.run_training:run_training_entry', str(DATASET_ID), '3d_fullres', '0',
                      '-tr', trainer, '-p', 'nnResUNetPlans', '-device', args.device),
                dict(env, nnUNet_AFP_model=ckpt))
            check_training(root, trainer, 'nnResUNetPlans')
        check_isbi2027_options(root, env, ckpt, args.device)

        input_folder = os.path.join(root, 'predict_input')
        os.makedirs(input_folder, exist_ok=True)
        raw_images = os.path.join(root, 'nnUNet_raw', f'Dataset{DATASET_ID:03d}_Smoke', 'imagesTr')
        for case in ('case0', 'case5'):
            shutil.copy(os.path.join(raw_images, f'{case}_0000.nii.gz'), input_folder)
        # larger than the patch size, so that overlapping patches are reconstructed
        big = synthetic_mr(np.random.default_rng(1), tuple(int(s * 1.5) for s in SHAPE_ZYX))
        sitk.WriteImage(synthetic_image(big, (0.0, 0.0, 0.0)), os.path.join(input_folder, 'big_0000.nii.gz'))
        for rec in ('mean', 'median'):
            out = os.path.join(root, f'predict_{rec}')
            run(entry('nnunetv2.inference.predict_from_raw_data:predict_entry_point', '-i', input_folder, '-o', out,
                      '-d', str(DATASET_ID), '-c', '3d_fullres', '-p', 'nnResUNetPlans',
                      '-tr', 'nnUNetTrainerMRCT_smoketest', '-f', '0', '-device', args.device, '--rec', rec,
                      '-npp', '1', '-nps', '1'), env)
            for case in ('case0', 'case5', 'big'):
                check_prediction(os.path.join(out, f'{case}.nii.gz'), os.path.join(input_folder, f'{case}_0000.nii.gz'))
        mean, median = (sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root, f'predict_{r}', 'big.nii.gz')))
                        for r in ('mean', 'median'))
        assert not np.allclose(mean, median), 'mean and median reconstructions should differ where patches overlap'
        print(f'\nSMOKE TEST PASSED ({root})')
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == '__main__':
    main()
