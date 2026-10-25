"""
End-to-end check of the translation pipeline on tiny synthetic data (a few minutes on CPU):
prepare -> train (L1 and AFP, 1 short epoch) -> predict (mean and median), then checks outputs.

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
            summary = glob(os.path.join(root, 'nnUNet_results', '*', f'{trainer}__nnResUNetPlans__3d_fullres',
                                        'fold_0', 'validation', 'summary.json'))
            assert len(summary) == 1, f'{trainer}: validation summary missing'
            with open(summary[0]) as f:
                print(f"{trainer}: validation MAE = {json.load(f)['mean']['MAE']:.1f} HU")

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
