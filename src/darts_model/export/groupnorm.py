"""Spelling GroupNorm so a GPU delegate will take it.

`nn.GroupNorm` lowers, through `torch.export`, to a `GATHER_ND` over its
per-channel affine vector -- a `[C]` constant indexed by `[1, C, 1]`. No
WebGPU delegate implements `GATHER_ND` in any form, and because the refusal
lands inside every norm rather than at the end of the graph, the partitioner
cuts the backbone apart around them: measured, the eight norms in the neck and
heads stranded 228 of 548 operations on the CPU, and the same weights that run
in 120ms under CoreML ran at 2.7fps in a browser.

Nothing about GroupNorm's arithmetic requires a gather. The gather is how the
converter chose to spell "reshape a length-C vector so it broadcasts over
NCHW". This spells it differently: statistics by plain reductions, and the
affine as a broadcast multiply-add against constants that are ALREADY the right
shape, so there is no reshape left to lower.

The weights are untouched -- same parameters, same function, different graph.
Which is exactly why `rewrite_group_norms` refuses to return without checking:
a rewrite that is subtly not the trained model would pass every smoke test and
score darts slightly wrong forever.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn

from darts_model.export.common import ExportError

#: How far the rewritten module may differ from the original, per element.
#: float32 noise through a 48-conv backbone, and nothing more.
TOLERANCE = 2e-4


class GroupNormExplicit(nn.Module):
    """`nn.GroupNorm`, spelled without an indexed affine.

    `weight` and `bias` are stored pre-shaped as (1, C, 1, 1) rather than (C,):
    a constant that is already broadcast-shaped gives the converter nothing to
    reshape, and the reshape is what became the gather.
    """

    def __init__(self, source: nn.GroupNorm) -> None:
        super().__init__()
        if source.num_channels % source.num_groups:
            raise ExportError(
                f"{source.num_channels} channels is not divisible by "
                f"{source.num_groups} groups")
        self.groups = int(source.num_groups)
        self.eps = float(source.eps)
        self.affine = bool(source.affine)
        if source.affine:
            self.register_buffer(
                "w", source.weight.detach().reshape(1, -1, 1, 1).clone())
            self.register_buffer(
                "b", source.bias.detach().reshape(1, -1, 1, 1).clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        # The group folded into the BATCH dimension, not added as a fifth.
        #
        # The obvious grouped view is (n, g, c/g, h, w), and it is what the
        # first attempt used. The WebGPU delegate refuses it outright --
        # "Tensor dimensions must be less than 5" -- and because that refusal
        # also lands inside every norm, delegation collapsed from 320 GPU ops
        # to 27 and the model ran at CPU speed: worse than the problem being
        # fixed. (n*g, c/g, h, w) carries the same elements with the same group
        # boundaries, in four dimensions.
        grouped = x.reshape(n * self.groups, c // self.groups, h, w)
        mean = grouped.mean(dim=(1, 2, 3), keepdim=True)
        # Biased variance, which is what F.group_norm uses. The unbiased one
        # differs by m/(m-1) over the group -- small, and wrong, and exactly
        # the sort of thing that passes a smoke test.
        var = grouped.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        normed = ((grouped - mean) / torch.sqrt(var + self.eps)).reshape(n, c, h, w)
        return normed * self.w + self.b if self.affine else normed


def _swap(module: nn.Module) -> int:
    swapped = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GroupNorm):
            setattr(module, name, GroupNormExplicit(child))
            swapped += 1
        else:
            swapped += _swap(child)
    return swapped


def _flatten(outputs: Any) -> list[torch.Tensor]:
    if isinstance(outputs, dict):
        return [outputs[k] for k in sorted(outputs)]
    if isinstance(outputs, (list, tuple, Sequence)):
        return list(outputs)
    return [outputs]


def rewrite_group_norms(
    module: nn.Module,
    probe: Callable[[], Any],
    *,
    tolerance: float = TOLERANCE,
    log: Callable[[str], None] = print,
) -> int:
    """Replace every GroupNorm in `module`, in place, and prove it equivalent.

    `probe` is called before and after the swap and must return the same
    tensors both times -- a dict, a sequence, or one tensor. Give it something
    INDEX-FREE: dense planes, not a readout.

    The reason is worth stating, because the obvious choice is wrong. A readout
    ending in `topk` selects K of N cells by score, and the float32 noise this
    rewrite moves scores by is enough to reorder near-ties. Comparing gathered
    values then reports differences of 0.19 -- not because the arithmetic
    changed, but because they are different cells. (Measured: 1.9e-5 on the
    planes, 0.19 on the readout of those same planes, on random input.)

    Feed the probe non-constant input, too. GroupNorm divides by the
    activation's own standard deviation, which on a constant frame is zero for
    every group, so agreement there proves nothing.

    Returns how many were replaced. Raises `ExportError` if the rewritten
    module is not the same function, rather than letting a model that is
    subtly not the trained one reach a device.
    """
    with torch.no_grad():
        before = [t.detach().clone() for t in _flatten(probe())]

    swapped = _swap(module)
    if not swapped:
        log("  no GroupNorm found; nothing rewritten")
        return 0

    with torch.no_grad():
        after = _flatten(probe())
    if len(after) != len(before):
        raise ExportError(
            f"the probe returned {len(before)} tensors before the rewrite and "
            f"{len(after)} after; it must return the same ones")

    worst = 0.0
    for a, b in zip(before, after):
        if a.shape != b.shape:
            raise ExportError(
                f"the probe returned {tuple(a.shape)} before the rewrite and "
                f"{tuple(b.shape)} after")
        worst = max(worst, float((a - b).abs().max()))
    if worst > tolerance:
        raise ExportError(
            f"the rewritten GroupNorm differs from the original by "
            f"{worst:.3e}, past the {tolerance:.0e} tolerance -- this is not "
            f"the same function, so the export would ship a model that is not "
            f"the one that was trained")
    log(f"  rewrote {swapped} GroupNorm module(s); worst drift {worst:.2e}")
    return swapped
