"""Iterable dataset yielding rendered frames with their dense targets.

Applies the detector's photometric augmentation. Brightness, contrast, hue,
noise and blur are per-pixel intensity operations that leave every target
untouched, and a backbone pretrained on clean renders and then frozen inside a
pipeline applying +-0.4 brightness and contrast would see inputs unlike
anything it was trained on. A detector trained without augmentation falls from
20 detections to 4 under a mild global intensity shift, where augmented models
lose none.

Geometric augmentation is left out: it would have to be mirrored onto five
target maps, with the offset field's vectors rotated rather than merely
resampled, and the detector uses none either.

CoarseDropout is off by default. It blanks rectangles of the image while the
targets still describe what was there, which asks a dense head to predict class
and offset for pixels it cannot see.

Every rendered frame is yielded, including those with no dart: the configured
share of empty boards still supervises the class, uv and height maps.

Rendering is synchronous: a worker's renderer idles while it does the CPU work
for the previous frame. At 1024px with the segmentation pass that is 15.79ms of
render and 6.74ms of CPU in sequence. :class:`DenseDetectDataset` overlaps the
two on a background thread.
"""
from __future__ import annotations

import numpy as np
import torch

import albumentations as A

from darts_model.data.config import AugmentationConfig, validate_dart_count_weights
from darts_model.data.targets import build_dense_targets
from darts_model.renderer import RenderedDataset


def photometric_pipeline(aug: AugmentationConfig) -> A.Compose:
    """The CPU half of ``aug``; both training stages build it here.

    Image-only by construction, so no target needs a corresponding transform.

    The three noise transforms are deliberately absent: they run on the GPU in
    GPUAugment. Profiled at 1024px they are 47 of 48ms of CPU augmentation --
    ShotNoise alone 106ms per call -- and running them here drops GPU
    utilisation from 91-98% to an oscillating 22-89%. What remains costs 1.1ms.

    Every HueSaturationValue limit is passed explicitly: albumentations
    otherwise applies its own saturation and value defaults, and a config that
    names only the hue would not describe what runs. See
    :class:`AugmentationConfig` for the units.

    Blur is a fixed 3px box blur. albumentations has no smaller kernel (it
    clamps a requested (1, 3) to (3, 3)), and on a 2-3px tip a constant 3px
    blur is not gentle, which is worth remembering when reading tip precision.
    """
    out = [
        A.RandomBrightnessContrast(
            brightness_limit=aug.brightness_limit,
            contrast_limit=aug.contrast_limit, p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=aug.hue_shift_limit,
            sat_shift_limit=aug.sat_shift_limit,
            val_shift_limit=aug.val_shift_limit, p=0.3),
    ]
    if aug.gaussian_blur:
        out.append(A.Blur(blur_limit=(3, 3), p=0.3))
    # Off by default, deliberately. The holes are 5-15% of the image, which at
    # 1024 is 51-153px against a dart of roughly 100-200 -- a single hole can
    # bury a whole dart while the tip target still supervises those cells,
    # which is label noise aimed straight at the metric this model exists to
    # minimise. Not recommended.
    if aug.random_erasing:
        out.append(A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(0.05, 0.15),
            hole_width_range=(0.05, 0.15),
            fill="random", p=0.5))
    return A.Compose(out)

# Normalisation happens in GPUAugment, not here, exactly as in the detector's
# pipeline: a backbone pretrained on a different input distribution than it is
# fine-tuned on would look like a backbone that does not help.


class DensePretrainDataset(RenderedDataset):
    """Renders frames and derives dense targets, one worker per renderer.

    ``seed`` is the experiment seed; each worker's renderer seed is derived
    from it, ``split``, the global rank and the worker id
    (:func:`darts_model.renderer.renderer_seed`). ``epoch_length`` counts
    frames across all GPUs (:func:`darts_model.renderer.frames_for_worker`).
    """

    def __init__(self, asset_dir: str = "", image_size: int = 1024,
                 epoch_length: int = 2000, seed: int = 0, split: str = "train",
                 bg_image_dir: str = "", render_gpus: tuple[int, ...] = (),
                 dart_count_weights: tuple[float, ...] = (0.05, 0.25, 0.30, 0.40),
                 skill_placement: bool = True, out_stride: int = 4,
                 augmentation: AugmentationConfig | None = None) -> None:
        super().__init__(epoch_length=epoch_length, seed=seed, split=split,
                         render_gpus=render_gpus, bg_image_dir=bg_image_dir)
        self.asset_dir = asset_dir
        self.image_size = image_size
        #: Targets are reduced to the head's grid here rather than on device.
        #: At 1024 the float maps are ~20MB per sample and the head consumes a
        #: stride-4 grid, so reducing before transfer moves 16x less.
        self.out_stride = out_stride
        self.dart_count_weights = list(validate_dart_count_weights(dart_count_weights))
        self.skill_placement = skill_placement
        self._aug = (photometric_pipeline(augmentation)
                     if augmentation is not None else None)

    def _ensure_renderer(self, worker_id: int, rank: int):
        options = dict(
            width=self.image_size, height=self.image_size,
            skill_placement=self.skill_placement,
            dart_count_weights=self.dart_count_weights,
        )
        if self.asset_dir:
            options["asset_dir"] = self.asset_dir
        return self._make_renderer(worker_id, rank, **options)

    def __iter__(self):
        wid, rank, n_frames = self._worker_plan()
        renderer = self._ensure_renderer(wid, rank)
        for _ in range(n_frames):
            image_np, annotation = renderer.render_frame()
            yield self._sample(image_np, annotation)

    def _sample(self, image_np: np.ndarray, annotation: dict) -> dict:
        targets = build_dense_targets(annotation, self.image_size,
                                      out_stride=self.out_stride)
        if targets is None:
            raise RuntimeError(
                "renderer returned no seg_ids/depth, which the dense "
                "targets are built from. Rebuild the renderer from this "
                "repository."
            )
        rgb = image_np[..., :3]
        if self._aug is not None:
            rgb = self._aug(image=np.ascontiguousarray(rgb))["image"]
        # uint8 out. GPUAugment divides by 255, adds sensor noise and
        # normalises, all on device -- so the expensive part never touches
        # a dataloader worker and the bytes crossing the bus are a quarter
        # the size of float32.
        img = np.ascontiguousarray(rgb.transpose(2, 0, 1))
        # Targets already arrive at the head's grid: build_dense_targets
        # reduces the ids first and evaluates the continuous fields only at
        # the winning subpixels, which is identical to reducing afterwards
        # and 12x cheaper.
        sample = {"image": torch.from_numpy(img),
                  "class": torch.from_numpy(targets["seg_class"]),
                  "instance": torch.from_numpy(targets["seg_instance"])}
        for k in ("offset", "uv", "height", "dart_mask", "board_mask",
                  "drawn_mask"):
            sample[k] = torch.from_numpy(np.ascontiguousarray(targets[k]))
        return sample
