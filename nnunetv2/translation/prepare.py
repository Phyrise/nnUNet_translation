"""
nnUNetv2_translation_prepare: one command from paired images to a dataset ready for nnUNetv2_train.

It creates two nnU-Net datasets:
    Dataset{ID}_{name}          the inputs (one channel per --input). Train and predict with this ID.
    Dataset{ID+1}_{name}Target  the targets. Only used to preprocess the targets with the same plans.

Steps: pair the files by case name, check that inputs and target share the same grid, write the raw datasets
(with all-ones dummy labels), plan the input dataset, give the target dataset the same plans (with its own
intensity normalization), preprocess both, then store each preprocessed target as the 'seg' array of its input
case. The standard nnU-Net data loading then yields (input, target) pairs.
"""
import argparse
import fnmatch
import glob
import multiprocessing
import os
import re
import shutil
import sys
from copy import deepcopy
from typing import Dict, List

import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import join, load_json, save_json, maybe_mkdir_p, isdir, \
    isfile

import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiment_dataset, \
    preprocess
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
from nnunetv2.preprocessing.normalization.map_channel_name_to_normalization import \
    channel_name_to_normalization_mapping
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

SUPPORTED_ENDINGS = ('.nii.gz', '.nii', '.mha', '.nrrd')


class PrepareError(Exception):
    """Problem with the user's data or arguments (reported without traceback)."""

GEOMETRY_ATOL = 1e-3


def _strip_ending(filename: str) -> str:
    for e in SUPPORTED_ENDINGS:
        if filename.endswith(e):
            return filename[:-len(e)]
    return None


def collect_files(spec: str) -> Dict[str, str]:
    """
    spec is either a folder (case name = file name without extension) or a glob pattern with exactly one '*'
    (case name = the part matched by '*'), e.g. 'data/*/mr.mha'.
    """
    if isdir(spec):
        files = {}
        for f in sorted(os.listdir(spec)):
            case = _strip_ending(f)
            if case is not None and not f.startswith('.'):
                files[case] = join(spec, f)
        return files

    if spec.count('*') != 1:
        raise PrepareError(f"'{spec}' is neither a folder nor a glob pattern with exactly one '*'.")
    regex = re.compile(fnmatch.translate(spec).replace('.*', '(.+)', 1))
    files = {}
    for f in sorted(glob.glob(spec)):
        m = regex.match(f)
        case = m.group(1) if m else None
        if case is None or os.path.basename(f).startswith('.'):
            continue
        if '/' in case or os.sep in case:
            raise PrepareError(f"'*' in '{spec}' matches across folders ({f}). Use a more specific pattern.")
        files[case] = f
    return files


def pair_cases(inputs: List[Dict[str, str]], target: Dict[str, str]) -> List[str]:
    all_cases = set(target.keys()).union(*[set(i.keys()) for i in inputs])
    problems = []
    for name, files in [(f'--input #{n}', i) for n, i in enumerate(inputs)] + [('--target', target)]:
        missing = sorted(all_cases - set(files.keys()))
        if missing:
            problems.append(f'{name} has no file for {len(missing)} case(s): {missing[:10]}')
    if problems:
        raise PrepareError('Cases cannot be paired:\n  ' + '\n  '.join(problems))
    if len(all_cases) == 0:
        raise PrepareError('No image found. Supported endings: ' + ', '.join(SUPPORTED_ENDINGS))
    return sorted(all_cases)


def _read_geometry(filename: str) -> dict:
    reader = sitk.ImageFileReader()
    reader.SetFileName(filename)
    reader.ReadImageInformation()
    return {'size': reader.GetSize(), 'spacing': reader.GetSpacing(), 'origin': reader.GetOrigin(),
            'direction': reader.GetDirection(), 'channels': reader.GetNumberOfComponents()}


def _check_case_geometry(case: str, files: List[str]) -> List[str]:
    """Returns a list of problems (empty if the input files and the target (last file) share one grid)."""
    ref = _read_geometry(files[-1])
    problems = []
    if ref['channels'] != 1:
        problems.append(f'{case}: {files[-1]} has {ref["channels"]} components, expected 1')
    for f in files[:-1]:
        g = _read_geometry(f)
        if g['channels'] != 1:
            problems.append(f'{case}: {f} has {g["channels"]} components, expected 1')
        if g['size'] != ref['size']:
            problems.append(f'{case}: size {g["size"]} ({f}) != {ref["size"]} ({files[-1]})')
            continue
        for k in ('spacing', 'origin', 'direction'):
            if not np.allclose(g[k], ref[k], atol=GEOMETRY_ATOL):
                problems.append(f'{case}: {k} {np.round(g[k], 4).tolist()} ({f}) != '
                                f'{np.round(ref[k], 4).tolist()} ({files[-1]})')
    return problems


def _write_image(source: str, destination: str):
    if source.endswith('.nii.gz') and destination.endswith('.nii.gz'):
        shutil.copyfile(source, destination)
    else:
        sitk.WriteImage(sitk.ReadImage(source), destination, True)


def _write_case(case: str, input_files: List[str], target_file: str, input_folder: str, target_folder: str,
                file_ending: str):
    for c, f in enumerate(input_files):
        _write_image(f, join(input_folder, 'imagesTr', f'{case}_{c:04d}{file_ending}'))
    _write_image(target_file, join(target_folder, 'imagesTr', f'{case}_0000{file_ending}'))

    # dummy label: the whole image is foreground, so intensity statistics are computed on all voxels. The geometry
    # comes from this case (not from another file), otherwise nnU-Net rejects or misaligns the case.
    for folder, reference in ((input_folder, input_files[0]), (target_folder, target_file)):
        g = _read_geometry(reference)
        label = sitk.Image(g['size'], sitk.sitkUInt8) + 1
        label.SetSpacing(g['spacing'])
        label.SetOrigin(g['origin'])
        label.SetDirection(g['direction'])
        sitk.WriteImage(label, join(folder, 'labelsTr', f'{case}{file_ending}'), True)


def _store_target_as_seg(case: str, input_folder: str, target_folder: str):
    """The preprocessed target becomes the 'seg' array of the input case (this is what the trainers learn)."""
    inp = np.load(join(input_folder, case + '.npz'))
    tgt = np.load(join(target_folder, case + '.npz'))
    data, target = inp['data'], tgt['data']
    if data.shape[1:] != target.shape[1:]:
        raise PrepareError(f'{case}: preprocessed input {data.shape} and target {target.shape} differ in shape.')
    np.savez_compressed(join(input_folder, case + '.npz'), data=data, seg=target.astype(np.float32))
    # stale unpacked files would otherwise be used instead of the new npz
    for suffix in ('.npy', '_seg.npy'):
        if isfile(join(input_folder, case + suffix)):
            os.remove(join(input_folder, case + suffix))


def make_target_plans(input_plans: dict, target_dataset_name: str, target_fingerprint: dict,
                      target_dataset_json: dict) -> dict:
    """Same plans as the input (spacing, transpose, resampling) so that both are preprocessed onto one grid."""
    plans = deepcopy(input_plans)
    plans['dataset_name'] = target_dataset_name
    plans['foreground_intensity_properties_per_channel'] = target_fingerprint[
        'foreground_intensity_properties_per_channel']
    normalization = [channel_name_to_normalization_mapping[target_dataset_json['channel_names']['0']].__name__]
    for c in plans['configurations'].values():
        if 'normalization_schemes' in c:
            c['normalization_schemes'] = normalization
            c['use_mask_for_norm'] = [False]
    return plans


def _check_free_dataset_ids(dataset_id: int, names: List[str], overwrite: bool):
    for did, name in zip((dataset_id, dataset_id + 1), names):
        for base in (nnUNet_raw, nnUNet_preprocessed, nnUNet_results):
            if base is None or not isdir(base):
                continue
            for existing in os.listdir(base):
                if not existing.startswith(f'Dataset{did:03d}_'):
                    continue
                if existing != name:
                    raise PrepareError(f'Dataset ID {did} is already used by {join(base, existing)}. '
                                       f'Choose another -d (this command uses -d and -d + 1).')
                if base != nnUNet_results and not overwrite:
                    raise PrepareError(f'{join(base, existing)} already exists. Use --overwrite to rebuild it.')


def _run_parallel(fn, args_list, num_processes: int):
    with multiprocessing.get_context('spawn').Pool(num_processes) as p:
        return p.starmap(fn, args_list)


def prepare(dataset_id: int, name: str, inputs: List[str], target: str, input_norm: List[str], target_norm: str,
            planner_name: str = 'nnUNetPlannerResUNet', configurations: List[str] = ('3d_fullres',),
            num_processes: int = default_num_processes, file_ending: str = '.nii.gz', overwrite: bool = False):
    if nnUNet_raw is None or nnUNet_preprocessed is None:
        raise PrepareError('Set the nnUNet_raw, nnUNet_preprocessed and nnUNet_results environment variables first.')
    if len(input_norm) != len(inputs):
        raise PrepareError(f'--input_norm needs one value per --input ({len(inputs)}), got {input_norm}')
    for n in list(input_norm) + [target_norm]:
        if n not in channel_name_to_normalization_mapping:
            raise PrepareError(f"Unknown normalization '{n}'. Choose from {list(channel_name_to_normalization_mapping)}")

    input_name = f'Dataset{dataset_id:03d}_{name}'
    target_name = f'Dataset{dataset_id + 1:03d}_{name}Target'
    _check_free_dataset_ids(dataset_id, [input_name, target_name], overwrite)

    # 1. pair and check
    input_files = [collect_files(i) for i in inputs]
    target_files = collect_files(target)
    cases = pair_cases(input_files, target_files)
    print(f'{len(cases)} paired cases, {len(inputs)} input channel(s): {cases[:5]}{" ..." if len(cases) > 5 else ""}')
    if len(cases) < 5:
        print('WARNING: nnU-Net uses a 5-fold split, at least 5 cases are needed for training.')
    per_case_files = {c: [i[c] for i in input_files] + [target_files[c]] for c in cases}
    problems = sum(_run_parallel(_check_case_geometry, [(c, per_case_files[c]) for c in cases], num_processes), [])
    if problems:
        raise PrepareError(f'{len(problems)} geometry problem(s). Inputs and target must be co-registered and '
                           f'resampled onto the same grid:\n  ' + '\n  '.join(problems[:20]))

    # 2. raw datasets
    input_raw, target_raw = join(nnUNet_raw, input_name), join(nnUNet_raw, target_name)
    for folder in (input_raw, target_raw, join(nnUNet_preprocessed, input_name),
                   join(nnUNet_preprocessed, target_name)):
        if isdir(folder):
            shutil.rmtree(folder)
    for folder in (input_raw, target_raw):
        maybe_mkdir_p(join(folder, 'imagesTr'))
        maybe_mkdir_p(join(folder, 'labelsTr'))
    _run_parallel(_write_case, [(c, per_case_files[c][:-1], per_case_files[c][-1], input_raw, target_raw,
                                 file_ending) for c in cases], num_processes)
    labels = {'background': 0, 'image': 1}
    save_json({'channel_names': {str(c): n for c, n in enumerate(input_norm)}, 'labels': labels,
               'numTraining': len(cases), 'file_ending': file_ending}, join(input_raw, 'dataset.json'),
              sort_keys=False)
    save_json({'channel_names': {'0': target_norm}, 'labels': labels, 'numTraining': len(cases),
               'file_ending': file_ending}, join(target_raw, 'dataset.json'), sort_keys=False)

    # 3. plans: the input dataset is planned, the target dataset reuses its plans
    extract_fingerprints([dataset_id, dataset_id + 1], num_processes=num_processes, check_dataset_integrity=True,
                         clean=True, verbose=False)
    planner = recursive_find_python_class(join(nnunetv2.__path__[0], 'experiment_planning'), planner_name,
                                          current_module='nnunetv2.experiment_planning')
    if planner is None:
        raise PrepareError(f'Planner {planner_name} not found')
    input_plans, plans_identifier = plan_experiment_dataset(dataset_id, planner)
    target_fingerprint = load_json(join(nnUNet_preprocessed, target_name, 'dataset_fingerprint.json'))
    target_plans = make_target_plans(input_plans, target_name, target_fingerprint, load_json(join(target_raw,
                                                                                                  'dataset.json')))
    save_json(target_plans, join(nnUNet_preprocessed, target_name, plans_identifier + '.json'), sort_keys=False)

    # the target normalization travels with the model (dataset.json is copied to nnUNet_results) so that
    # predictions can be written in target units (e.g. HU)
    input_json = load_json(join(input_raw, 'dataset.json'))
    input_json['translation'] = {
        'target_dataset': target_name,
        'target_normalization': channel_name_to_normalization_mapping[target_norm].__name__,
        'target_intensity_properties': target_fingerprint['foreground_intensity_properties_per_channel']['0'],
    }
    save_json(input_json, join(input_raw, 'dataset.json'), sort_keys=False)
    for raw, dataset_name in ((input_raw, input_name), (target_raw, target_name)):
        shutil.copy(join(raw, 'dataset.json'), join(nnUNet_preprocessed, dataset_name, 'dataset.json'))

    # 4. preprocess both, then pair
    preprocess([dataset_id, dataset_id + 1], plans_identifier, configurations, [num_processes])
    plans_manager = PlansManager(input_plans)
    for c in configurations:
        data_identifier = plans_manager.get_configuration(c).data_identifier
        _run_parallel(_store_target_as_seg, [(case, join(nnUNet_preprocessed, input_name, data_identifier),
                                              join(nnUNet_preprocessed, target_name, data_identifier))
                                             for case in cases], num_processes)

    # final validation compares predictions with the raw targets
    gt_folder = join(nnUNet_preprocessed, input_name, 'gt_segmentations')
    for case in cases:
        shutil.copyfile(join(target_raw, 'imagesTr', f'{case}_0000{file_ending}'), join(gt_folder, case + file_ending))

    print(f'\nDone. {input_name} is ready ({len(cases)} cases, plans: {plans_identifier}).\n'
          f'Train:   nnUNetv2_train {dataset_id} {configurations[0]} 0 -tr nnUNetTrainerMRCT -p {plans_identifier}\n'
          f'Predict: nnUNetv2_predict -d {dataset_id} -c {configurations[0]} -p {plans_identifier} '
          f'-tr nnUNetTrainerMRCT -f 0 -i INPUT_FOLDER -o OUTPUT_FOLDER\n'
          f'(inference inputs are named CASE_0000{file_ending}, CASE_0001{file_ending}, ... as in '
          f'{join(input_raw, "imagesTr")})')
    return input_name, target_name, plans_identifier


def main():
    parser = argparse.ArgumentParser(
        description='Builds a ready-to-train translation dataset from paired images (inputs -> target).',
        epilog="Example: nnUNetv2_translation_prepare -d 50 -n PelvisMRCT -i 'data/*/mr.nii.gz' "
               "-t 'data/*/ct.nii.gz'")
    parser.add_argument('-d', '--dataset_id', type=int, required=True,
                        help='Dataset ID for the inputs. ID + 1 is also used (for the targets).')
    parser.add_argument('-n', '--name', type=str, required=True, help='Dataset name, e.g. PelvisMRCT')
    parser.add_argument('-i', '--input', action='append', required=True,
                        help="Input images: a folder, or a glob pattern with one '*' standing for the case name "
                             "(e.g. 'data/*/mr.mha'). Repeat -i for several input channels.")
    parser.add_argument('-t', '--target', type=str, required=True, help='Target images, same format as -i.')
    parser.add_argument('--input_norm', nargs='+', default=None,
                        help='Normalization of each input channel (default: zscore, per image). Use noNorm for '
                             'masks.')
    parser.add_argument('--target_norm', type=str, default='CT_clip',
                        help='Normalization of the target. CT_clip (default): clip to [-1024, 3071] HU then '
                             'dataset-level z-score, predictions are written back in HU. For MR targets use '
                             'zscore (predictions stay z-scored).')
    parser.add_argument('-pl', type=str, default='nnUNetPlannerResUNet',
                        help='Experiment planner (default: nnUNetPlannerResUNet, residual U-Net, plans '
                             'nnResUNetPlans).')
    parser.add_argument('-c', nargs='+', default=['3d_fullres'], help='Configurations (default: 3d_fullres)')
    parser.add_argument('-np', type=int, default=default_num_processes, help='Number of processes')
    parser.add_argument('--file_ending', type=str, default='.nii.gz', choices=['.nii.gz', '.mha', '.nrrd'],
                        help='File format of the raw datasets (default: .nii.gz)')
    parser.add_argument('--overwrite', action='store_true', help='Rebuild the datasets if they already exist')
    args = parser.parse_args()

    input_norm = args.input_norm if args.input_norm is not None else ['zscore'] * len(args.input)
    try:
        prepare(args.dataset_id, args.name, args.input, args.target, input_norm, args.target_norm, args.pl, args.c,
                args.np, args.file_ending, args.overwrite)
    except PrepareError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
