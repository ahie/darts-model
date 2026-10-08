"""Backbone built for a task whose evidence is a few pixels wide.

Measured on this project rather than assumed:

  - Shift response at initialisation is essentially stage-independent for a
    plain ConvNeXt pyramid (mean-centred spatial cosine 0.94/0.88/0.80/0.67 at
    every depth under 1/2/4/8px shifts). The architecture is not what destroys
    fine detail; an invariance-seeking pretraining *objective* is: the same
    network pretrained that way sits at 0.97 in its deepest stage where random
    init sits at 0.66.
  - So the architecture's job is narrow -- do not discard detail structurally,
    and leave a fine level intact for dense heads to attach to. The defence
    against invariance is the training target, not the layer design.

Three choices follow, two of them against the usual advice:

  - Overlapping stride-2 stem, never a 4x4 patchify. A patch embedding throws
    away sub-patch phase in layer one, and a dart tip is smaller than one patch.
    Measured on a toy pyramid over rendered frames: at the finest stage under
    an 8px shift, overlap holds 0.445 mean-centred spatial cosine against
    patchify's 0.598, so it keeps materially more shift response. The gap is
    concentrated in the first stage and is nearly gone by the second (0.810 vs
    0.816) -- phase discarded in layer one is not recovered later, it is simply
    absent from everything downstream.
  - No anti-aliased downsampling. Blur-pool raised stride-8 spatial cosine from
    0.810 to 0.935 and saturated the next stage to 1.0000 -- it buys shift
    *equivariance* by removing high frequencies, which is exactly where a 2px
    tip lives. Standard advice, wrong for this task.
  - Width goes where the tip evidence is. The stride-4 stage is 96 channels
    wide, because that is the level a 2-3px tip is resolved at; a narrow
    stride-4 stage measurably limits how well the detector localises from it. The domain is one object class on a planar,
    near-colourless board, so deeper stages stay modest.

The last stage halves rather than dilating (``dilate_last=False``), giving four
distinct scales at strides 4/8/16/32 for the detector's neck to fuse. Dilation
keeps the coarsest grid at 64x64, which only helps a head that reads that grid
directly; the neck fuses down to stride 8, and with dilation it would receive
two levels at the same resolution instead of a genuine coarse one.

At the defaults: 3.47M params, 24.67 GMAC at a 1024px input.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FineBackboneConfig:
    in_channels: int = 3
    stem_width: int = 32
    #: One entry per stage, running at strides 4/8/16/32 (16/16 when
    #: ``dilate_last``). The defaults are the released model's; see the module
    #: docstring for why stride 4 is wide.
    widths: tuple[int, ...] = (96, 128, 160, 256)
    depths: tuple[int, ...] = (2, 2, 6, 2)
    #: Dilate the last stage instead of halving, keeping the coarsest grid at
    #: stride 16. Off for the released model; see the module docstring.
    dilate_last: bool = False
    drop_path: float = 0.1
    layer_scale_init: float = 1e-6


class LayerNorm2d(nn.Module):
    """Channels-first LayerNorm, as in ConvNeXt."""

    def __init__(self, ch: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ch))
        self.bias = nn.Parameter(torch.zeros(ch))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class Block(nn.Module):
    """ConvNeXt block. Dilation widens the receptive field in place."""

    def __init__(self, ch: int, drop_path: float = 0.0,
                 dilation: int = 1, layer_scale_init: float = 1e-6) -> None:
        super().__init__()
        pad = 3 * dilation
        self.dw = nn.Conv2d(ch, ch, 7, padding=pad, dilation=dilation, groups=ch)
        self.norm = LayerNorm2d(ch)
        self.pw1 = nn.Conv2d(ch, 4 * ch, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(4 * ch, ch, 1)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(ch)) \
            if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))
        if self.gamma is not None:
            y = self.gamma[:, None, None] * y
        return x + self.drop_path(y)


class FineBackbone(nn.Module):
    """Feature pyramid whose finest level stays at stride 4.

    Returns ``(last_stage, stage_outputs)``: the coarsest map, and one map per
    stage with the coarsest last.
    """

    def __init__(self, config: FineBackboneConfig | None = None) -> None:
        super().__init__()
        cfg = config or FineBackboneConfig()
        self.config = cfg

        # Stem to stride 4, as two overlapping stride-2 convolutions.
        #
        # Never a 4x4/4 patchify -- that discards sub-patch phase in layer one,
        # and a dart tip is smaller than one patch. But the naive alternative,
        # one stride-2 conv followed by a full-width 3x3 at half resolution,
        # puts 2.4 of 12.3 GMAC into a single layer at 1024px input. Halving twice
        # with a narrow first step keeps every output overlapping its
        # neighbours while doing the wide convolution on a quarter of the grid.
        half = max(cfg.stem_width // 2, 8)
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, half, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(half, cfg.stem_width, 3, stride=2, padding=1),
            LayerNorm2d(cfg.stem_width),
        )

        total = sum(cfg.depths)
        rates = [cfg.drop_path * i / max(total - 1, 1) for i in range(total)]

        self._frozen = False
        self._requested_training = True
        self.downs = nn.ModuleList()
        self.stages = nn.ModuleList()
        prev = cfg.stem_width
        idx = 0
        self._reductions: list[int] = []
        stride = 4                      # the stem already halved twice
        for i, (w, d) in enumerate(zip(cfg.widths, cfg.depths)):
            last = i == len(cfg.widths) - 1
            dilate = cfg.dilate_last and last
            if i == 0:
                # Stage 0 runs at the stem's resolution: the finest level is
                # kept, not reconstructed from a coarser one later.
                self.downs.append(nn.Sequential(LayerNorm2d(prev),
                                                nn.Conv2d(prev, w, 1)))
            elif dilate:
                # Keep resolution, widen the receptive field instead.
                self.downs.append(nn.Sequential(LayerNorm2d(prev),
                                                nn.Conv2d(prev, w, 1)))
            else:
                # 3x3 stride 2, overlapping. A 2x2/2 would be a patchify at
                # every stage boundary, discarding phase four times over.
                self.downs.append(nn.Sequential(
                    LayerNorm2d(prev), nn.Conv2d(prev, w, 3, stride=2, padding=1)))
                stride *= 2
            self._reductions.append(stride)
            self.stages.append(nn.Sequential(*[
                Block(w, rates[idx + j], 2 if dilate else 1, cfg.layer_scale_init)
                for j in range(d)
            ]))
            idx += d
            prev = w

    # --- freeze interface ---
    #
    # Used by DenseDartConfig.backbone_freeze_epochs, which holds the backbone
    # fixed for the first epochs so gradients from an untrained head cannot
    # disturb it.
    #
    # Frozen means fixed in the forward too, not only in the backward: a frozen
    # backbone runs in eval mode, so stochastic depth does not keep dropping
    # blocks out of the features the head is being fitted against. train() is
    # overridden so a parent's .train() cannot switch it back; the mode the
    # parent asked for is remembered and restored by unfreeze().

    def train(self, mode: bool = True) -> "FineBackbone":
        self._requested_training = mode
        return super().train(mode and not self._frozen)

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        self._frozen = True
        super().train(False)

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True
        self._frozen = False
        super().train(self._requested_training)

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def stage_channels(self) -> list[int]:
        return list(self.config.widths)

    @property
    def stage_reductions(self) -> list[int]:
        return list(self._reductions)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        outs = []
        for down, stage in zip(self.downs, self.stages):
            x = stage(down(x))
            outs.append(x)
        return outs[-1], outs
