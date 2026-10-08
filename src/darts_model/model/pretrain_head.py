"""Dense pretraining heads: scaffolding to force spatial precision into a backbone.

The heads here are thrown away; only the backbone transfers. Their job is to
make invariance impossible to learn, by demanding per-pixel answers that a
smooth representation cannot give.

Four targets, all free from the renderer:

  class      8-way, including two structures that are thin on purpose -- the
             spider at 0.2-0.6mm and the dart point at ~27px per dart.
  offset     per-pixel vector to *its own* dart's tip. This is the target the
             design rests on: it supervises ~1000px per dart instead of the
             ~27 a tip class gives, it is continuous so precision is not capped
             by the grid, and because it is keyed by dart instance it separates
             overlapping darts -- the failure a location prior entrenches
             rather than fixes.
  height     millimetres above the board plane. Zero across the board, positive
             on darts, so the supervision sits exactly where depth is not
             already implied by the homography.
  uv         board-plane coordinates, teaching the mapping the board head
             exists to produce.

Predictions are made at stride 4 rather than full resolution. That is coarse
for a 1px wire, and would be fatal if the class map carried tip precision --
but it does not. Sub-cell accuracy comes from the offset regression, and the
class map only has to tell thin structure apart, which at stride 4 is still a
one-cell-wide discrimination.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from darts_model.data.targets import FLIGHT, METAL, NUM_CLASSES, TIP, WIRE

#: Per class, in SegClass order. Measured with tools/measure_class_freq.py
#: over 1000 rendered frames at 1024px (configs/pretrain.yaml), **after**
#: reduction to the stride-4 grid -- the distribution the loss actually sees,
#: not the one at output resolution. The two differ: priority reduction
#: inflates thin classes by construction, the wire about threefold and the tip
#: about twofold, so weights from full-resolution figures would under-weight
#: the tip and over-weight the wire.
#:
#: Used only to derive loss weights, so drift here is a weighting error rather
#: than a correctness bug -- but it is the kind that shows up as a model that
#: quietly ignores the classes it was built to learn. Re-measure after any
#: change to what the renderer draws or how the camera frames it.
CLASS_FREQ = (0.54188, 0.38327, 0.05461, 0.01301, 0.00039, 0.00211, 0.00474)


@dataclass
class DensePretrainConfig:
    #: Width of the fused trunk. This, not backbone width, limits the fine
    #: detail reaching the offset head: widening the backbone 4x at s/4
    #: (24 -> 96) improves every dense metric at matched step except
    #: ``offsetrel_tip`` (+0.2%), the one target needing sub-cell precision.
    trunk_width: int = 64
    trunk_blocks: int = 2
    #: Square-root inverse frequency rather than plain inverse. On the stride-4
    #: distribution plain inverse would weight the tip 823x over background and
    #: destabilise training; the root lands it at 28.7x, and the Tversky term
    #: below carries the rest without needing an extreme cross-entropy weight.
    weight_power: float = 0.5
    weight_clip: tuple[float, float] = (0.1, 20.0)
    #: Applied to thin classes only. Scale-invariant to class size, so it
    #: supervises the tip without the cross-entropy weights having to be
    #: extreme. alpha < beta penalises false negatives harder -- the tip is
    #: ~0.036% of cells, so missing it costs almost nothing in accuracy terms
    #: and everything in usefulness.
    thin_classes: tuple[int, ...] = (TIP, WIRE, METAL)
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    w_class: float = 1.0
    w_tversky: float = 1.0
    #: Highest, because the offset field is the target the design rests on.
    w_offset: float = 2.0
    #: Per-part weights for the offset loss, averaged within each part before
    #: combining. Order is (tip, barrel, flight).
    #:
    #: Unweighted L1 over all dart pixels puts 91.4% of the objective on flights
    #: and 0.3% on tips, because flights outnumber tip cells 26:1 *and* their
    #: offsets are 12x longer -- about 300x combined. That matters for what this
    #: loss is for: a flight's target ("the tip is 150px that way") is a claim
    #: about dart pose and is satisfiable with smooth low-frequency features,
    #: which is precisely the representation the pretraining exists to prevent.
    #: Only near-tip targets demand resolving position to a few pixels.
    #:
    #: Per-distance reliability weighting is not enough: w = d0/(d0+|t|) at
    #: d0=10px still leaves flights at 81.6% and near-tip pixels at 2.4%,
    #: because it can recover a factor of ~12 against a ~300x imbalance.
    #: Normalising within groups is what controls it.
    #:
    #: (2, 1, 0.5) puts the tip at ~57% of the objective, against 0.3% for a
    #: plain mean over dart pixels. Flights are kept rather than dropped -- an
    #: occluded tip has no near pixels at all, and long-range votes are the only
    #: thing that can localise it. One measured frame had 221 dart pixels and
    #: zero tip pixels.
    offset_part_weights: tuple[float, float, float] = (2.0, 1.0, 0.5)
    #: Typical per-component target magnitude for (tip, barrel, flight), in the
    #: same normalised units as the offset target. Measured over rendered frames
    #: at 1024px: 7.4px, 31.1px and 80.1px respectively.
    #:
    #: Each part's L1 is divided by its own scale, which turns the term into
    #: *relative* error and gives it a fixed, readable zero point: a part scores
    #: exactly 1.0 when it does no better than predicting zero, and 0.5 at half
    #: the trivial error.
    #:
    #: Without this, L1 values an absolute improvement identically at every
    #: scale -- taking a tip cell from 7px to 2px earns what taking a flight
    #: cell from 70px to 65px earns. Those are worth very different amounts
    #: here: absolute L1 yields a model that beats the trivial baseline on
    #: flights (63px against 80px) while sitting four times worse than it on
    #: tips (32px against 7.4px).
    offset_part_scales: tuple[float, float, float] = (0.00718, 0.03034, 0.07819)
    #: Multiplied back into the offset term after the per-part scaling.
    #:
    #: Dividing by a part's scale normalises the loss *value* to O(1) but
    #: multiplies its *gradient* by 1/scale -- about 139x for the tip. Left
    #: uncorrected, grad/offset runs ~650x grad/class and the term starves
    #: every other task (tip IoU 0.387 -> 0.017 against a control).
    #:
    #: Multiplying by a reference scale restores the gradient magnitude of the
    #: absolute formulation, while keeping the relative-error incentive that
    #: favours the tip. The logged offsetrel_* stay in relative units, so the
    #: readable "1.0 means no better than zero" survives -- the metric and the
    #: objective simply do not have to share a scale.
    offset_loss_scale: float = 0.00718
    w_height: float = 0.5
    w_uv: float = 0.5


def class_weights(cfg: DensePretrainConfig) -> torch.Tensor:
    freq = torch.tensor(CLASS_FREQ, dtype=torch.float32)
    med = freq.median()
    w = (med / freq.clamp_min(1e-8)) ** cfg.weight_power
    return w.clamp(*cfg.weight_clip)


class _ConvBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm = nn.GroupNorm(8, ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.norm(self.conv(x)))


class DensePretrainHead(nn.Module):
    """FPN trunk fusing every backbone stage onto the finest grid, plus one 1x1
    per target.

    Fusion runs *upward* -- coarse stages are upsampled to the fine grid.
    Fusing the other way, resampling fine stages down onto coarse ones,
    decimates the only stage that still carries fine detail.
    """

    def __init__(self, stage_channels: list[int],
                 config: DensePretrainConfig | None = None) -> None:
        super().__init__()
        cfg = config or DensePretrainConfig()
        self.config = cfg
        w = cfg.trunk_width

        self.lateral = nn.ModuleList([nn.Conv2d(c, w, 1) for c in stage_channels])
        self.fuse = nn.Sequential(*[_ConvBlock(w) for _ in range(cfg.trunk_blocks)])
        self.head_class = nn.Conv2d(w, NUM_CLASSES, 1)
        self.head_offset = nn.Conv2d(w, 2, 1)
        self.head_height = nn.Conv2d(w, 1, 1)
        self.head_uv = nn.Conv2d(w, 2, 1)

    def forward(self, stages: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        target_hw = stages[0].shape[-2:]
        # Summed per-level projections: already a general linear map of the
        # concatenated levels (sum_i L_i x_i == [L_1|...|L_n] [x_1;...;x_n]),
        # so a concat-and-1x1 would add parameters, not capacity.
        fused = None
        for lat, s in zip(self.lateral, stages):
            x = lat(s)
            if x.shape[-2:] != target_hw:
                x = F.interpolate(x, size=target_hw, mode="bilinear",
                                  align_corners=False)
            fused = x if fused is None else fused + x
        fused = self.fuse(fused)
        return {
            "class_logits": self.head_class(fused),
            "offset": self.head_offset(fused),
            "height": self.head_height(fused),
            "uv": self.head_uv(fused),
        }


def tversky_loss(probs: torch.Tensor, onehot: torch.Tensor,
                 alpha: float, beta: float, eps: float = 1e-6) -> torch.Tensor:
    dims = (0, 2, 3)
    tp = (probs * onehot).sum(dims)
    fp = (probs * (1 - onehot)).sum(dims)
    fn = ((1 - probs) * onehot).sum(dims)
    return (1.0 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)).mean()


class DensePretrainLoss(nn.Module):
    """Weighted CE + Tversky on thin classes + masked L1 on the regressions.

    Every regression target is normalised to O(1) before weighting, so the task
    weights in the config mean what they look like rather than silently encoding
    unit choices.
    """

    def __init__(self, config: DensePretrainConfig | None = None) -> None:
        super().__init__()
        self.config = config or DensePretrainConfig()
        self.register_buffer("cls_w", class_weights(self.config))

    def forward(self, pred: dict[str, torch.Tensor],
                target: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        cfg = self.config
        logits = pred["class_logits"]
        cls_t = target["class"].long()

        ce = F.cross_entropy(logits, cls_t, weight=self.cls_w.to(logits.dtype))

        probs = logits.softmax(1)
        onehot = F.one_hot(cls_t, NUM_CLASSES).permute(0, 3, 1, 2).to(probs.dtype)
        idx = list(cfg.thin_classes)
        tv = tversky_loss(probs[:, idx], onehot[:, idx],
                          cfg.tversky_alpha, cfg.tversky_beta)

        def masked_l1(p: torch.Tensor, t: torch.Tensor, m: torch.Tensor,
                      w: torch.Tensor | None = None) -> torch.Tensor:
            m = m.unsqueeze(1).to(p.dtype)
            if w is not None:
                m = m * w.unsqueeze(1).to(p.dtype)
            n = m.sum().clamp_min(1e-6) * p.shape[1]
            return ((p - t).abs() * m).sum() / n

        # Averaged within each dart part, then combined. Averaging over all
        # dart pixels at once would hand the objective to the flights by sheer
        # count and offset magnitude.
        off = pred["offset"].new_zeros(())
        wsum = 0.0
        for pw, sc, c in zip(cfg.offset_part_weights, cfg.offset_part_scales,
                             (TIP, METAL, FLIGHT)):
            sel = target["class"] == c
            if sel.any():
                # Divided by the part's own target scale: the term becomes
                # relative error, so 1.0 means "no better than predicting zero"
                # for every part alike.
                off = off + pw * masked_l1(pred["offset"], target["offset"],
                                           sel) / max(sc, 1e-8)
                wsum += pw
        off = off / max(wsum, 1e-6)
        # Relative internally, absolute in magnitude. See offset_loss_scale.
        off_term = off * cfg.offset_loss_scale
        hgt = masked_l1(pred["height"], target["height"], target["drawn_mask"])
        uv = masked_l1(pred["uv"], target["uv"], target["board_mask"])

        total = (cfg.w_class * ce + cfg.w_tversky * tv + cfg.w_offset * off_term
                 + cfg.w_height * hgt + cfg.w_uv * uv)
        # Per-part offset error, because the aggregate is dominated by flights
        # and would hide a model that votes well from 150px away while being
        # imprecise exactly where precision is available.
        stats = {"ce": ce.detach(), "tversky": tv.detach(),
                 "offset": off.detach(), "height": hgt.detach(), "uv": uv.detach()}
        with torch.no_grad():
            # Per-component mean absolute error, matching what masked_l1 and
            # the reference baselines use, so the two are comparable.
            err = (pred["offset"] - target["offset"]).abs().mean(dim=1)
            for name, c, sc in (("tip", TIP, cfg.offset_part_scales[0]),
                                ("barrel", METAL, cfg.offset_part_scales[1]),
                                ("flight", FLIGHT, cfg.offset_part_scales[2])):
                sel = target["class"] == c
                e = err[sel].mean() if sel.any() else err.new_zeros(())
                stats[f"offset_{name}"] = e
                # Relative to the trivial baseline, so <1 is genuine progress
                # and the three parts are directly comparable.
                stats[f"offsetrel_{name}"] = e / max(sc, 1e-8)
        return total, stats

