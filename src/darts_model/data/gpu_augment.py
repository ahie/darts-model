"""Batched sensor-noise augmentation and normalisation on the GPU.

The three noise transforms dominate CPU augmentation: measured per image at
512x512, ShotNoise 9.1 ms, ISONoise 8.5 ms, GaussNoise 6.9 ms, against 3.2 ms
to render the frame in the first place.  Run per image in a dataloader worker,
that cost starves the GPU.

The distributions below mirror ``albumentations`` element for element, with
parameters drawn per sample rather than per batch.  Two differences from
running them on the CPU are deliberate:

* **Ordering.**  The noise runs after the CPU pipeline's ``Blur`` and
  ``CoarseDropout``.  That is the more physical order: real noise is added by
  the sensor, after the optics.
* **Normalisation.**  It happens here too, so samples cross from the
  dataloader as uint8, a quarter the size of float32.

Reference implementations: ``albumentations.augmentations.pixel.functional``
``shot_noise`` / ``iso_noise``, and ``GaussNoise`` (``per_channel=True``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from darts_model.data.config import AugmentationConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Per-sample strength ranges, drawn uniformly. Fixed rather than configured:
#: the configs set only whether noise runs and how often each kind does.
SHOT_SCALE_RANGE = (0.05, 0.2)
GAUSS_STD_RANGE = (0.02, 0.08)
ISO_COLOR_SHIFT_RANGE = (0.01, 0.04)
ISO_INTENSITY_RANGE = (0.1, 0.4)

# albumentations works in linear light for shot noise, undoing sRGB's transfer
# curve with a plain 2.2 power rather than the piecewise sRGB function.
_GAMMA = 2.2


def _rand(shape: tuple[int, ...], lo: float, hi: float, *, device, dtype) -> Tensor:
    """Uniform sample in ``[lo, hi)`` — one value per batch element."""
    return torch.rand(shape, device=device, dtype=dtype) * (hi - lo) + lo


def _poisson(rate: Tensor) -> Tensor:
    """``torch.poisson`` with a CPU detour on backends that lack the kernel.

    MPS has no ``aten::poisson``.  The round trip is slow and defeats the point
    of this module; it exists only so a CUDA config can be smoke-tested on a
    Mac, not for training.
    """
    if rate.device.type == "mps":
        return torch.poisson(rate.cpu()).to(rate.device)
    return torch.poisson(rate)


def rgb_to_hls(rgb: Tensor) -> Tensor:
    """RGB -> HLS for float images in [0, 1], matching OpenCV's float32 layout.

    Hue is returned in degrees on [0, 360); lightness and saturation on [0, 1].
    """
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = rgb.amax(dim=1)
    minc = rgb.amin(dim=1)
    total = maxc + minc
    diff = maxc - minc

    lightness = total * 0.5

    # Saturation has a different denominator either side of L = 0.5; guard the
    # achromatic case (diff == 0) before dividing so no NaN reaches the output.
    safe_diff = torch.where(diff > 0, diff, torch.ones_like(diff))
    sat = torch.where(
        lightness < 0.5,
        diff / torch.where(total > 0, total, torch.ones_like(total)),
        diff / torch.where(total < 2.0, 2.0 - total, torch.ones_like(total)),
    )
    sat = torch.where(diff > 0, sat, torch.zeros_like(sat))

    rc = (maxc - r) / safe_diff
    gc = (maxc - g) / safe_diff
    bc = (maxc - b) / safe_diff

    hue = torch.where(
        maxc == r,
        bc - gc,
        torch.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc),
    )
    hue = torch.where(diff > 0, (hue / 6.0) % 1.0, torch.zeros_like(hue))

    return torch.stack((hue * 360.0, lightness, sat), dim=1)


def hls_to_rgb(hls: Tensor) -> Tensor:
    """Inverse of :func:`rgb_to_hls`."""
    hue = (hls[:, 0] / 360.0) % 1.0
    lightness = hls[:, 1]
    sat = hls[:, 2]

    m2 = torch.where(
        lightness < 0.5,
        lightness * (1.0 + sat),
        lightness + sat - lightness * sat,
    )
    m1 = 2.0 * lightness - m2

    def channel(offset: float) -> Tensor:
        h = (hue + offset) % 1.0
        return torch.where(
            h < 1.0 / 6.0,
            m1 + (m2 - m1) * h * 6.0,
            torch.where(
                h < 0.5,
                m2,
                torch.where(
                    h < 2.0 / 3.0,
                    m1 + (m2 - m1) * (2.0 / 3.0 - h) * 6.0,
                    m1,
                ),
            ),
        )

    rgb = torch.stack((channel(1.0 / 3.0), channel(0.0), channel(-1.0 / 3.0)), dim=1)
    # Fully desaturated pixels collapse to L on every channel.
    return torch.where(sat.unsqueeze(1) > 0, rgb, lightness.unsqueeze(1).expand_as(rgb))


def shot_noise(x: Tensor, scale: Tensor) -> Tensor:
    """Poisson shot noise applied in linear light.

    *scale* is one value per batch element, broadcast over pixels.
    """
    s = scale.view(-1, 1, 1, 1)
    linear = x.clamp_min(0).pow(_GAMMA)
    # The 1e-6 term keeps the Poisson rate away from exactly zero, as upstream.
    noisy = _poisson((linear + s * 1e-6) / s) * s
    return noisy.clamp_(0.0, 1.0).pow_(1.0 / _GAMMA)


def gauss_noise(x: Tensor, std: Tensor, mean: Tensor) -> Tensor:
    """Additive Gaussian noise, sampled independently per channel."""
    noise = torch.randn_like(x) * std.view(-1, 1, 1, 1) + mean.view(-1, 1, 1, 1)
    return (x + noise).clamp_(0.0, 1.0)


def iso_noise(x: Tensor, color_shift: Tensor, intensity: Tensor) -> Tensor:
    """Camera sensor noise: Poisson on luminance, Gaussian on hue.

    The Poisson rate is the image's own lightness standard deviation scaled by
    *intensity*, so flat frames get less noise than busy ones — that coupling is
    what makes this differ from plain Gaussian noise, and it is computed per
    image here just as ``cv2.meanStdDev`` does upstream.
    """
    hls = rgb_to_hls(x)
    hue, lightness, sat = hls[:, 0], hls[:, 1], hls[:, 2]

    ci = color_shift.view(-1, 1, 1) * intensity.view(-1, 1, 1)
    inten = intensity.view(-1, 1, 1)

    # Per-image std of the lightness channel (population std, as OpenCV uses).
    std_l = lightness.flatten(1).std(dim=1, unbiased=False).view(-1, 1, 1)

    luminance_noise = _poisson((std_l * inten).expand_as(lightness))
    color_noise = torch.randn_like(hue) * ci

    hue = hue + color_noise
    lightness = lightness + luminance_noise * inten * (1.0 - lightness)

    out = hls_to_rgb(torch.stack((hue, lightness, sat), dim=1))
    return out.clamp_(0.0, 1.0)


class GPUAugment(nn.Module):
    """Sensor noise plus normalisation, applied to a whole batch on-device.

    Consumes the uint8 ``(B, C, H, W)`` batches the datasets produce and
    returns ImageNet-normalised float tensors.

    Noise is applied only when *training* is true; validation gets the
    normalisation path alone.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        shot_noise_p: float = 0.2,
        gauss_noise_p: float = 0.8,
        iso_noise_p: float = 0.6,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.shot_noise_p = shot_noise_p
        self.gauss_noise_p = gauss_noise_p
        self.iso_noise_p = iso_noise_p
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def _mask(self, n: int, p: float, device) -> Tensor:
        """Per-sample apply/skip decision, mirroring albumentations' per-image p."""
        return (torch.rand(n, device=device) < p).view(-1, 1, 1, 1)

    def forward(self, images: Tensor, *, training: bool) -> Tensor:
        # uint8 only. This is the single place input is scaled and normalised,
        # and a float tensor carries no record of whether that has happened:
        # 0-255 floats, 0-1 floats and already-normalised floats would all pass
        # through with no error and a wrong result.
        if images.dtype != torch.uint8:
            raise TypeError(
                f"GPUAugment takes raw uint8 images (0-255), got {images.dtype}. "
                "Scaling and normalisation happen here and nowhere else.")
        x = images.float().div_(255.0)

        if training and self.enabled:
            n, device, dtype = x.shape[0], x.device, x.dtype

            if self.shot_noise_p > 0.0:
                scale = _rand((n,), *SHOT_SCALE_RANGE, device=device, dtype=dtype)
                x = torch.where(
                    self._mask(n, self.shot_noise_p, device), shot_noise(x, scale), x
                )

            if self.gauss_noise_p > 0.0:
                std = _rand((n,), *GAUSS_STD_RANGE, device=device, dtype=dtype)
                mean = torch.zeros(n, device=device, dtype=dtype)
                x = torch.where(
                    self._mask(n, self.gauss_noise_p, device),
                    gauss_noise(x, std, mean),
                    x,
                )

            if self.iso_noise_p > 0.0:
                shift = _rand(
                    (n,), *ISO_COLOR_SHIFT_RANGE, device=device, dtype=dtype
                )
                inten = _rand(
                    (n,), *ISO_INTENSITY_RANGE, device=device, dtype=dtype
                )
                x = torch.where(
                    self._mask(n, self.iso_noise_p, device),
                    iso_noise(x, shift, inten),
                    x,
                )

        return (x - self.mean) / self.std


def gpu_augment_from_config(aug: AugmentationConfig | None) -> GPUAugment:
    """The on-device half of ``aug``; both training stages build it here.

    ``None`` gives normalisation alone, with no noise even while training.
    """
    if aug is None:
        return GPUAugment(enabled=False)
    return GPUAugment(enabled=aug.sensor_noise,
                      shot_noise_p=aug.shot_noise_p,
                      gauss_noise_p=aug.gauss_noise_p,
                      iso_noise_p=aug.iso_noise_p)
