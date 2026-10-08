"""The augmentation block both stages share, and what it builds."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
import albumentations as A

from darts_model.data.config import (
    MAX_DARTS,
    AugmentationConfig,
    build_augmentation_config,
    build_data_config,
    validate_dart_count_weights,
)
from darts_model.data.gpu_augment import gpu_augment_from_config
from darts_model.data.pretrain_dataset import photometric_pipeline


def test_unknown_augmentation_key_raises() -> None:
    with pytest.raises(ValueError, match="hue_shift_limt"):
        build_augmentation_config({"hue_shift_limt": 30})


@pytest.mark.parametrize("key", ["gpu_noise", "gaussian_noise"])
def test_removed_noise_keys_name_their_replacement(key: str) -> None:
    with pytest.raises(ValueError, match="sensor_noise"):
        build_augmentation_config({key: True})


def test_absent_block_gives_defaults() -> None:
    assert build_augmentation_config(None) == AugmentationConfig()
    assert build_data_config({}).augmentation == AugmentationConfig()


def test_hue_saturation_value_limits_are_passed_explicitly() -> None:
    aug = AugmentationConfig(hue_shift_limit=11, sat_shift_limit=12,
                             val_shift_limit=13)
    hsv = next(t for t in photometric_pipeline(aug).transforms
               if isinstance(t, A.HueSaturationValue))
    assert tuple(hsv.hue_shift_limit) == (-11, 11)
    assert tuple(hsv.sat_shift_limit) == (-12, 12)
    assert tuple(hsv.val_shift_limit) == (-13, 13)


def test_hue_shift_is_in_opencv_half_degrees() -> None:
    """hue_shift_limit=30 is +-60 degrees: a shift of +30 turns pure red
    (0 degrees) into yellow (60 degrees), not orange (30)."""
    img = np.zeros((4, 4, 3), np.uint8)
    img[..., 0] = 255
    t = A.HueSaturationValue(hue_shift_limit=(30, 30), sat_shift_limit=(0, 0),
                             val_shift_limit=(0, 0), p=1.0)
    out = t(image=img)["image"]
    hue_deg = 2 * int(cv2.cvtColor(out, cv2.COLOR_RGB2HSV)[0, 0, 0])
    assert hue_deg == 60
    np.testing.assert_array_equal(out[0, 0], [255, 255, 0])


def test_gpu_noise_follows_sensor_noise() -> None:
    aug = AugmentationConfig(shot_noise_p=0.1, gauss_noise_p=0.2, iso_noise_p=0.3)
    g = gpu_augment_from_config(aug)
    assert g.enabled
    assert (g.shot_noise_p, g.gauss_noise_p, g.iso_noise_p) == (0.1, 0.2, 0.3)
    assert not gpu_augment_from_config(AugmentationConfig(sensor_noise=False)).enabled
    assert not gpu_augment_from_config(None).enabled


def test_dart_count_weights_cannot_exceed_max_darts() -> None:
    assert validate_dart_count_weights([0.05, 0.25, 0.3, 0.4]) == (0.05, 0.25, 0.3, 0.4)
    assert validate_dart_count_weights(()) == ()
    with pytest.raises(ValueError, match="MAX_DARTS"):
        validate_dart_count_weights([1.0] * (MAX_DARTS + 2))
    with pytest.raises(ValueError):
        validate_dart_count_weights([0.5, -0.1])
    with pytest.raises(ValueError, match="MAX_DARTS"):
        build_data_config({"data": {"dart_count_weights": [1, 1, 1, 1, 1]}})
