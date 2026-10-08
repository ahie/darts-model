"""The training input pipeline and the exported graph must agree, exactly.

The backbone is pretrained through ``GPUAugment``; if the detector's
normalisation silently stops running, it fine-tunes on raw 0-255 and nothing
raises -- metrics are merely worse.

Nothing here checks that the constants are *correct* -- only that there is one
definition of them and that both paths reach it, which is the property a test
can hold.
"""
from __future__ import annotations


import pytest
import torch


from darts_model.data.config import AugmentationConfig
from darts_model.data.gpu_augment import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    gpu_augment_from_config,
)
from darts_model.model.detector import DenseDartConfig, DenseDartLitModule


def _tiny() -> DenseDartConfig:
    return DenseDartConfig(
        backbone_widths=(16, 24, 32, 48), backbone_depths=(1, 1, 1, 1),
        head_width=16, head_depth=1, backbone_weights="",
        predict_ends=True, predict_keypoints=True, kp_head_width=16,
        kp_head_depth=1)


def test_detector_normalises_its_input() -> None:
    """The hook must apply ImageNet normalisation, not pass 0-255 through.

    A `uint8` batch of a constant value has a closed-form answer, so this
    cannot pass by comparing the pipeline against itself.
    """
    m = DenseDartLitModule(_tiny())
    img = torch.full((1, 3, 8, 8), 128, dtype=torch.uint8)
    out = m.on_after_batch_transfer({"image": img}, 0)["image"]

    for c in range(3):
        want = (128 / 255.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        assert torch.allclose(out[0, c], torch.full((8, 8), want), atol=1e-6), (
            f"channel {c}: got {out[0, c, 0, 0]:.4f}, want {want:.4f}")
    # The bug being guarded against, stated directly: raw 0-255 reaching the
    # backbone.
    assert out.abs().max() < 10.0, "input looks unnormalised"


def test_noise_is_off_outside_training() -> None:
    """Validation and export must see normalisation ALONE.

    `GPUAugment` gates noise on a training flag; if that gating broke, val
    metrics would carry augmentation noise and read as a worse model.
    """
    m = DenseDartLitModule(_tiny(), gpu_augment=gpu_augment_from_config(
        AugmentationConfig(sensor_noise=True)))
    img = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
    a = m.on_after_batch_transfer({"image": img.clone()}, 0)["image"]
    b = m.on_after_batch_transfer({"image": img.clone()}, 0)["image"]
    # No trainer attached => not training => deterministic.
    assert torch.allclose(a, b), "noise applied outside the training loop"


def test_export_wrapper_matches_the_training_hook() -> None:
    """The exported graph and the training pipeline, on the same input.

    Both are fed 0-255. What the wrapper's forward hands the network is
    captured and compared with the training hook's output, so this fails if
    the forward stops normalising, normalises twice, or either set of
    constants changes.
    """
    from darts_model.export.coreml import ExportWrapper

    lit = DenseDartLitModule(_tiny())
    wrapper = ExportWrapper(lit.model).eval()
    seen = []
    handle = lit.model.register_forward_pre_hook(
        lambda _m, args: seen.append(args[0].detach().clone()))

    img = torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.uint8)
    try:
        with torch.no_grad():
            wrapper(img.float())
    finally:
        handle.remove()
    trained = lit.on_after_batch_transfer({"image": img.clone()}, 0)["image"]

    assert len(seen) == 1
    assert torch.allclose(trained, seen[0], atol=1e-5), (
        "export prologue and training hook disagree: max delta "
        f"{(trained - seen[0]).abs().max():.3e}")


def test_float_input_is_refused() -> None:
    """Scaling and normalisation happen in one place. A float batch cannot say
    whether that has already happened, so it is an error, not a guess."""
    m = DenseDartLitModule(_tiny())
    for img in (torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8) * 255.0):
        with pytest.raises(TypeError, match="uint8"):
            m.on_after_batch_transfer({"image": img}, 0)
