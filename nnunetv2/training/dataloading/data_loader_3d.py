import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
import time

class nnUNetDataLoader3D(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(i)
            case_properties.append(properties)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)
        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}


class nnUNetDataLoader3D_MRCT(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.float32)
        case_properties = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(i)
            case_properties.append(properties)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=0)
        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}


class nnUNetDataLoader3D_MRCT_PairedFix(nnUNetDataLoaderBase):
    """C1 근본 fix — D050 (CBCT) + D051 (CT target) 을 dataloader 단에서 직접 페어링.

    기존 nnUNetDataLoader3D_MRCT 의 버그:
      - D050 만 self._data 로 로드 (CBCT data + dummy seg)
      - automate_translation.py 의 shutil.copy(D051/*.npy → D050/*_seg.npy) workaround 가
        D050 의 seg 자리에 CT 를 강제 주입
      - 새 plan_and_preprocess 가 돌면 *_seg.npy 가 dummy 로 복구 → CT pairing 끊김 → trivial collapse

    본 fix:
      - 별도 target_data (D051 의 nnUNetDataset) 를 받아서 같은 case_id 의 CT 를 직접 load
      - seg 자리에 D051 CT 를 사용 (workaround 의존성 제거)
      - case_id 일치 검증 (__init__ 에서 missing case assert)

    근거: amed plan unet0601.md § 6.6 (NaN 진단) + 메모리 [project_nnunet_dataloader_bug.md].
    """

    def __init__(self,
                 data,
                 target_data,
                 batch_size,
                 patch_size,
                 final_patch_size,
                 label_manager,
                 oversample_foreground_percent=0.0,
                 sampling_probabilities=None,
                 pad_sides=None,
                 probabilistic_oversampling=False):
        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager,
                         oversample_foreground_percent=oversample_foreground_percent,
                         sampling_probabilities=sampling_probabilities,
                         pad_sides=pad_sides,
                         probabilistic_oversampling=probabilistic_oversampling)
        # case_id 일치 검증 — D050 의 모든 case 가 D051 에도 있어야 함
        data_keys = set(self.indices)
        target_keys = set(target_data.keys())
        missing = data_keys - target_keys
        if missing:
            raise RuntimeError(
                f"[C1 PairedFix] D051 (target_data) 에 부재한 case: {sorted(missing)[:5]}"
                f" (... {len(missing)} 개 missing)"
            )
        self._target_data = target_data

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.float32)
        case_properties = []

        for j, i in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)

            # D050: CBCT input + dummy seg
            data, _, properties = self._data.load_case(i)
            # C1 fix: D051 의 같은 case_id 의 CT data 를 seg 로 사용
            target_data_arr, _, _ = self._target_data.load_case(i)
            # target_data_arr shape: (C, D, H, W). seg 도 (C, D, H, W) 형식 기대.
            seg = target_data_arr.astype(np.float32)

            case_properties.append(properties)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]
            this_slice_seg = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice_seg]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=0)
        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}


class nnUNetDataLoader3D_MRCT_mask(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.float32)
        mask_all = np.zeros(self.seg_shape, dtype=np.int16)

        case_properties = []

        for j, i in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(i)
            mask = self._data.load_mask(i)

            case_properties.append(properties)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]
            seg = seg[this_slice]
            mask = mask[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=0)
            mask_all[j] = np.pad(mask, ((0, 0), *padding), 'constant', constant_values=-1)

        return {'data': data_all, 'seg': seg_all, 'mask' : mask_all, 'properties': case_properties, 'keys': selected_keys}