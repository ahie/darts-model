"""The ``data`` section of a detector config, and the augmentation block both
training stages share."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

#: Darts in a turn. Box targets are padded to this many and masked, so a frame
#: with more would carry dense instances the box targets cannot represent.
MAX_DARTS: int = 3

#: Keys earlier configs used, with what replaced them. Named in the error so an
#: old config fails with the fix rather than a bare "unexpected keyword".
_REMOVED_AUGMENTATION_KEYS = {
    "gaussian_noise": "sensor_noise",
    "gpu_noise": "sensor_noise",
}
_REMOVED_DATA_KEYS = {
    "gpu_index": "render_gpus",
}


@dataclass(frozen=True)
class AugmentationConfig:
    """Photometric augmentation, identical in both training stages.

    There is no geometric augmentation: the renderer already samples pose, and
    every target would need transforming.

    The CPU part runs in the dataloader workers
    (:func:`darts_model.data.pretrain_dataset.photometric_pipeline`); sensor
    noise runs on device
    (:func:`darts_model.data.gpu_augment.gpu_augment_from_config`).
    """

    #: ``RandomBrightnessContrast`` limits, applied with p=0.5.
    brightness_limit: float = 0.4
    contrast_limit: float = 0.4
    #: ``HueSaturationValue`` limits, applied together with p=0.3. Each is a
    #: symmetric range: the shift is drawn uniformly from [-limit, +limit].
    #:
    #: Units are OpenCV's 8-bit HSV, because albumentations shifts the uint8
    #: HSV image directly. Hue is in half-degrees on a 0-179 wheel, so 30 is
    #: +-60 degrees of hue rotation, a third of the colour wheel either way.
    #: Saturation and value are additive offsets on 0-255.
    hue_shift_limit: int = 30
    sat_shift_limit: int = 30
    val_shift_limit: int = 20
    #: A fixed 3px box blur with p=0.3; see ``photometric_pipeline``.
    gaussian_blur: bool = True
    #: CoarseDropout. Off by default; see ``photometric_pipeline``.
    random_erasing: bool = False
    #: Sensor noise (shot, Gaussian, ISO), applied per sample on device by
    #: :class:`darts_model.data.gpu_augment.GPUAugment` while training. On the
    #: CPU these three transforms are ~98% of the augmentation budget and
    #: starve the GPU, so there is no CPU path. Normalisation happens there
    #: regardless of this flag.
    sensor_noise: bool = True
    #: Per-sample probability of each noise transform.
    shot_noise_p: float = 0.2
    gauss_noise_p: float = 0.8
    iso_noise_p: float = 0.6


def build_augmentation_config(raw: dict | None) -> AugmentationConfig:
    """Validate an ``augmentation`` block. Absent or empty gives the defaults.

    Unknown keys raise: a misspelt key would otherwise fall back to its
    default without a word.
    """
    raw = dict(raw or {})
    removed = sorted(k for k in raw if k in _REMOVED_AUGMENTATION_KEYS)
    if removed:
        raise ValueError(
            "augmentation: " + ", ".join(
                f"{k} is replaced by {_REMOVED_AUGMENTATION_KEYS[k]}"
                for k in removed))
    valid = {f.name for f in dataclasses.fields(AugmentationConfig)}
    unknown = sorted(set(raw) - valid)
    if unknown:
        raise ValueError(
            "unknown augmentation keys: " + ", ".join(unknown)
            + ". Valid keys: " + ", ".join(sorted(valid)))
    return AugmentationConfig(**raw)


def validate_dart_count_weights(weights) -> tuple[float, ...]:
    """Check ``P(0, 1, ..., n darts)`` weights and return them as a tuple.

    Empty is allowed and means the renderer's uniform default.
    """
    w = tuple(float(x) for x in weights)
    if len(w) > MAX_DARTS + 1:
        raise ValueError(
            f"dart_count_weights has {len(w)} entries, i.e. up to {len(w) - 1} "
            f"darts per frame; targets hold at most MAX_DARTS={MAX_DARTS}, so "
            f"at most {MAX_DARTS + 1} entries (P(0..{MAX_DARTS} darts)).")
    if any(x < 0 for x in w) or (w and sum(w) <= 0):
        raise ValueError(
            f"dart_count_weights must be non-negative with a positive sum, "
            f"got {list(w)}")
    return w


@dataclass(frozen=True)
class DataConfig:
    """Renderer and loader settings for the detector."""

    image_size: int = 1024
    #: Numeral GLBs and the decal atlas. Empty uses the renderer's built-in
    #: default, the ``renderer/assets`` directory of the source tree.
    asset_dir: str = ""
    #: Background photographs, composited behind the board and used as its
    #: environment lighting. See scripts/download_places365.py. Empty renders
    #: without backgrounds; a non-empty path must hold images.
    bg_image_dir: str = ""
    #: CUDA devices the dataloader workers render on. Empty renders on the
    #: training GPU; listing others moves rendering off it. See
    #: :func:`darts_model.renderer.render_gpu_uuid`.
    render_gpus: tuple[int, ...] = ()
    #: Frames per epoch across all GPUs; see
    #: :func:`darts_model.renderer.frames_for_worker`.
    epoch_length: int = 10000
    val_epoch_length: int = 1000
    #: Probability of 0, 1, 2 and 3 darts in a frame. Empty is uniform.
    dart_count_weights: tuple[float, ...] = ()
    #: Share of multi-dart turns thrown at an existing dart rather than
    #: independently.
    grouping_prob: float = 0.55
    #: Within grouping turns, the share using the tight scatter (0.02-0.08 BU)
    #: rather than the loose one (0.08-0.18).
    tight_prob: float = 0.55
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


def check_removed_data_keys(data: dict) -> None:
    """Fail on a ``data`` key that has been replaced, naming the replacement."""
    for old, new in _REMOVED_DATA_KEYS.items():
        if old in data:
            raise ValueError(f"data.{old} is replaced by data.{new}")


def build_data_config(raw: dict) -> DataConfig:
    """Build a DataConfig from a parsed YAML config.

    Unknown keys raise, as they do for the model sections: a misspelt key
    would otherwise fall back to its default without a word.
    """
    data = dict(raw.get("data") or {})
    check_removed_data_keys(data)
    if "render_gpus" in data:
        data["render_gpus"] = tuple(int(g) for g in data["render_gpus"] or ())
    aug = build_augmentation_config(data.pop("augmentation", None))
    if "dart_count_weights" in data:
        data["dart_count_weights"] = validate_dart_count_weights(
            data["dart_count_weights"])
    return DataConfig(augmentation=aug, **data)
