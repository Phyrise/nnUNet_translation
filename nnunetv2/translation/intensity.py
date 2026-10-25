"""
Conversion of network outputs back to the intensity units of the target modality (e.g. HU for CT).

The target images are normalized during preprocessing (see nnunetv2/preprocessing/normalization). The network
therefore predicts normalized intensities. `nnUNetv2_translation_prepare` stores the target normalization in the
`translation` entry of the input dataset.json, which travels with the trained model. Predictions are converted back
with `denormalize_prediction` when they are exported.
"""
import warnings
from typing import Optional

import numpy as np

from nnunetv2.preprocessing.normalization.default_normalization_schemes import CTtanh

# normalization schemes that use dataset-level statistics and can be inverted exactly
_DATASET_ZSCORE_SCHEMES = ('CTNormalization', 'CTNormalization_clip', 'CTNormalization_noclip')
_IDENTITY_SCHEMES = ('NoNormalization',)


def get_translation_info(dataset_json: Optional[dict]) -> Optional[dict]:
    if dataset_json is None:
        return None
    return dataset_json.get('translation')


def is_invertible(normalization_scheme: str) -> bool:
    return normalization_scheme in _DATASET_ZSCORE_SCHEMES + _IDENTITY_SCHEMES + ('CTtanh',)


def denormalize(image: np.ndarray, normalization_scheme: str, intensity_properties: dict) -> np.ndarray:
    """
    Inverts the target normalization. Clipping cannot be inverted: values stay within the clipped range.
    """
    image = image.astype(np.float32, copy=False)
    if normalization_scheme in _DATASET_ZSCORE_SCHEMES:
        return image * max(intensity_properties['std'], 1e-8) + intensity_properties['mean']
    if normalization_scheme in _IDENTITY_SCHEMES:
        return image
    if normalization_scheme == 'CTtanh':
        return (image + 1.0) / 2.0 * (CTtanh._max - CTtanh._min) + CTtanh._min
    raise ValueError(f'Normalization {normalization_scheme} uses per-image statistics and cannot be inverted.')


def denormalize_prediction(image: np.ndarray, dataset_json: Optional[dict]) -> np.ndarray:
    """
    Converts an exported prediction to target units if the model was prepared with nnUNetv2_translation_prepare.
    Models without translation info (older datasets) are returned unchanged, as before.
    """
    info = get_translation_info(dataset_json)
    if info is None:
        return image
    scheme = info['target_normalization']
    if not is_invertible(scheme):
        warnings.warn(f'Target normalization {scheme} uses per-image statistics: predictions are saved in '
                      f'normalized units.')
        return image
    return denormalize(image, scheme, info['target_intensity_properties'])
