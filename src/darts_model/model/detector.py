"""Dense dart detection: every cell on a dart predicts that dart's whole box.

No queries, no matcher, no assignment.  Set prediction brings failure modes
that have nothing to do with detection -- unstable assignment, query identity
drifting as the backbone moves, duplicate suppression left to be learned -- and
a dense head has none of them.

The readout is Hough voting.  Each foreground cell decodes the oriented box of
the dart it sits on, and detections are the peaks those votes accumulate into.
That makes duplicate suppression arithmetic rather than something a network has
to discover, and it uses every cell: ~83 votes per dart at stride 8 against the
single responsible cell a peak-based design would trust.
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities import rank_zero_info

from darts_model.model.backbone import FineBackbone, FineBackboneConfig

if TYPE_CHECKING:
    from darts_model.data.gpu_augment import GPUAugment


@dataclass
class DenseDartConfig:
    # backbone -- must match the pretraining config so its weights load
    backbone_widths: tuple[int, ...] = (96, 128, 160, 256)
    backbone_depths: tuple[int, ...] = (2, 2, 6, 2)
    backbone_dilate_last: bool = False
    backbone_drop_path: float = 0.1
    backbone_weights: str = ""
    #: Start from EVERY weight of a previous run, not just its backbone.
    #:
    #: `backbone_weights` loads `model.backbone.` only, which is right when the
    #: heads must be retrained -- a new task, or a changed output. It is wrong
    #: for continuing a converged run: it throws away the heads that did most
    #: of the converging.
    #:
    #: This loads the whole model and NOTHING else. No optimizer moments, no
    #: scheduler position, no epoch counter -- which is the entire point. A
    #: `--resume` restores all of that and simply finishes the old cosine,
    #: whereas a fine-tune needs the old weights under a NEW schedule.
    init_weights: str = ""
    #: Epochs the backbone is held fixed (``requires_grad`` off, eval mode)
    #: while the randomly initialised heads converge against its features.
    #:
    #: 0, the default, trains everything from the first step under the LR
    #: warmup alone. Which cell supervises which dart is fixed by the
    #: silhouette, so there is no assignment for a moving backbone to
    #: re-shuffle, as there is in a query-based detector.
    #:
    #: A positive value -- the shipped configs/detector.yaml uses 50 -- protects
    #: the pretrained features from the large, uninformed gradients of a fresh
    #: head. At the unfreeze the backbone's LR re-warms over
    #: ``backbone_rewarmup_epochs``: by then the global warmup is long over, and
    #: fresh Adam moments at near-peak LR would otherwise hit the pretrained
    #: weights with exactly the large first updates the freeze kept away.
    backbone_freeze_epochs: int = 0
    #: Epochs, per optimizer step, over which the backbone's LR multiplier
    #: ramps 0 -> 1 after the unfreeze, on top of the global schedule. Unused
    #: when ``backbone_freeze_epochs`` is 0.
    backbone_rewarmup_epochs: float = 5.0

    #: Output stride, set by measurement over 439 darts: at 8 the minimum
    #: silhouette cell count per dart is 17 and only 0.5% fall under 25, so the
    #: vote budget is comfortable and the peak threshold is uncritical.  At 16 a
    #: third of darts drop under 25 votes; at 4 the head costs 4x for votes that
    #: are not needed.
    out_stride: int = 8
    #: Head width.  This is where the compute goes: two 3x3 convs at 256
    #: channels over 16384 cells is ~4.8 GMAC against the backbone's ~12.8,
    #: while 128 is ~1.2.  The head needs DEPTH -- a single 1x1 has no receptive
    #: field of its own -- but not width.
    head_width: int = 128
    head_depth: int = 2
    #: Add PANet's bottom-up path after the top-down one. Top-down alone carries
    #: semantics from the coarse levels down; the bottom-up return leg carries
    #: fine localisation back up, which is what a tip needs. Standard since
    #: YOLOv4 and one extra conv stack. Requires every level above the head's
    #: to halve resolution, so not with ``backbone_dilate_last``.
    use_pan: bool = True

    # loss
    #: Foreground is ~1.6% of cells, so the positive term needs help. Focal
    #: rather than a hand-derived pos_weight, which is a balance constant that
    #: silently breaks the head when it is derived wrongly.
    focal_alpha: float = 2.0
    focal_gamma: float = 4.0
    fg_weight: float = 1.0
    #: Masked L1 per box component. Centre, direction and extent are weighted
    #: separately because they are in different units.
    centre_weight: float = 5.0
    dir_weight: float = 0.75
    size_weight: float = 0.4

    #: Predict the two ends -- the landing point and the flight tip -- as their
    #: own per-cell offsets, instead of reading them off the box.
    #:
    #: The landing point is what scoring is computed from, and it is NOT
    #: recoverable from the box in general. For a dart pointing near the camera
    #: the entry projects into the INTERIOR of the silhouette, because the
    #: dart's own barrel and flight cover where it went in: measured over 175
    #: darts, the entry sits a median 1.4px from the silhouette's end but 8.9px
    #: (max 27) for the shortest-projecting decile, and no end of any box can
    #: represent that. Predicting it directly removes the dependence on the box
    #: convention entirely, and the flight tip comes free since the renderer
    #: reports it.
    #:
    #: Voted through the same accumulator as the box; four more channels.
    #: On in the released model, and required by the exporters.
    predict_ends: bool = True
    tip_weight: float = 5.0
    flight_weight: float = 2.5

    # instance embedding
    #: Channels of a per-cell embedding that says WHICH dart the cell is on,
    #: trained with a pull/push loss (De Brabandere et al., 2017): a dart's
    #: cells are pulled to their mean, and the means of different darts in a
    #: frame are pushed apart. 0 disables the head.
    #:
    #: Every other dart output is a per-cell regression toward the cell's own
    #: dart, so identity is only ever supervised through the centre offset,
    #: where a cell that describes the WRONG dart pays no more than one that
    #: is imprecise by the same distance. In tight groups that is measured:
    #: 16.5% of cells on a dart within 60px of another predict a landing point
    #: nearer the neighbour's than their own, against 0.5% for isolated
    #: darts, and the cells' box and landing outputs agree with each other
    #: while doing it -- the cell has the whole wrong dart, not a bad tip.
    embed_dim: int = 0
    embed_weight: float = 1.0
    #: Pull hinge: a cell within this distance of its dart's mean is not
    #: pulled further, so the loss does not compete with the regressions once
    #: a dart is coherent.
    embed_pull_margin: float = 0.5
    #: Push hinge: the means of two darts in one frame are pushed until they
    #: are twice this far apart. Twice the pull margin is the usual minimum;
    #: three times leaves the readout's claim threshold room between them.
    embed_push_margin: float = 1.5
    #: Pull toward the origin on the dart means, so the embedding space cannot
    #: drift without bound. Small: it only has to remove that freedom.
    embed_reg_weight: float = 0.001
    #: Cells whose winning dart holds less than this share of the cell's dart
    #: pixels are left out of the embedding loss. At a crossing a cell can be
    #: split nearly evenly between two darts, and pulling it toward whichever
    #: won the majority teaches noise. Has no effect on batches without
    #: ``instance_purity``.
    embed_min_purity: float = 0.7
    #: Epochs, per optimizer step, over which ``embed_weight`` ramps 0 -> 1.
    #: A fresh head puts every dart's mean at one point, so the push term
    #: starts near its ceiling of (2 * push_margin)^2 -- 9, against a
    #: converged model's other terms summing to well under 1 -- and under
    #: gradient clipping it would take over every shared layer's update.
    #: Adam normalises per parameter, so the head itself learns at full speed
    #: through the ramp while the shared layers keep following the other
    #: terms until the head has organised itself.
    embed_warmup_epochs: float = 5.0

    # per-dart readout (model/instance.py)
    #: Read each dart's landing point from all of its cells at once: the
    #: embedding groups them, an attention head reads them together. Requires
    #: ``embed_dim`` and ``predict_ends``. Becomes the readout that
    #: validation ranks on and the exported graph emits; the Hough readouts
    #: are logged alongside it.
    instance_head: bool = False
    #: Per-cell feature channels handed to the head, a 1x1 projection of the
    #: dart trunk.
    token_dim: int = 64
    instance_dim: int = 64
    instance_heads: int = 4
    #: Cross-attention layers from the dart's query to its cells, after one
    #: self-attention layer among the cells.
    instance_layers: int = 2
    #: Cells read per dart: the most-member ones among the kept candidates. A
    #: dart covers ~80-250 cells at stride 8.
    instance_tokens: int = 256
    #: Dart slots picked in-graph at inference. One more than a turn holds,
    #: so a spurious seed cannot take a real dart's slot.
    instance_slots: int = 4
    #: Width of the soft membership, in embedding units: the pull margin,
    #: inside which a dart's own cells are held. Measured from ground-truth
    #: seeds on a part-trained embedding, the zero-correction readout was
    #: 12.3px at 0.5, 15.9 at 0.75 and 19.2 at 1.0, the wider ones admitting
    #: more of a neighbour's cells.
    membership_sigma: float = 0.5
    #: Seeds and membership measure identity as the embedding plus the
    #: cell's predicted box centre in units of this many pixels. A dart's own
    #: cells vote centres within a few pixels of each other, a stray up to
    #: ~15 (0.14 units squared); a dart 120px away is 9 units squared off and
    #: takes no part whatever its embedding; a tight neighbour 30px away is
    #: left to the embedding.
    membership_centre_px: float = 40.0
    #: A seed's vote density counts kept cells whose predicted centre lies
    #: within this many pixels of the seed cell's: the Hough readout's 3x3
    #: neighbourhood of 4px bins.
    seed_radius_px: float = 6.0
    #: Loss on the per-dart landing point, a Huber in pixels at the input
    #: resolution: quadratic below ``instance_huber_px``, so the gradient keeps
    #: pointing in proportion at the sub-pixel scale where plain L1 gives
    #: only a sign. The weight puts a ~3px error at about the size of the
    #: converged dense tip term.
    instance_weight: float = 0.01
    instance_flight_weight: float = 0.005
    instance_huber_px: float = 2.0
    #: Training seeds are a dart's mean embedding over a random fraction of its
    #: cells in this range, as an inference seed is the mean of a peak's core.
    instance_seed_subset: tuple[float, float] = (0.3, 0.7)

    # query readout (model/queries.py)
    #: The alternative to ``instance_head``: plain learned queries
    #: cross-attend to the kept cells under a foreground bias, and each points
    #: out one dart -- confidence, landing and flight points, box -- matched
    #: to the darts by the Hungarian algorithm in training. No embedding and
    #: no seeds. Shares ``token_dim``, ``instance_dim``, ``instance_heads``,
    #: ``instance_layers`` (decoder layers), ``instance_weight``,
    #: ``instance_flight_weight`` and ``instance_huber_px`` with it.
    query_head: bool = False
    #: One query per dart a turn can hold.
    query_count: int = 3
    #: Cells the queries read: the top ones by foreground. Three darts of up
    #: to ~250 cells each at stride 8 fit with room.
    query_tokens: int = 512
    #: A query is a dart when its confidence clears this.
    query_conf_threshold: float = 0.5
    #: Matching cost per pixel of landing distance, and per unit confidence.
    #: Confidence in the cost is what kept the assignment stable when
    #: queries were last used here.
    query_cost_px: float = 0.1
    query_cost_conf: float = 1.0
    query_conf_weight: float = 0.1
    query_centre_weight: float = 0.005
    query_dir_weight: float = 0.05
    query_size_weight: float = 0.05
    #: Epochs at the start of training in which the per-dart readout reads
    #: the dense field detached: it learns against the converged model
    #: without moving it. A fresh query head's losses start ~20x the rest of
    #: the model's together, and its first gradients are uninformed; a fresh
    #: embedding at that scale tripled isolated darts' error in three epochs.
    readout_warmup_epochs: int = 0
    #: Learning rate of the per-dart readout's own weights, as a multiple of
    #: ``lr``. A fine-tune's lr suits the converged weights, not a head that
    #: starts from nothing.
    readout_lr_factor: float = 1.0
    #: What a per-dart readout's tokens are built from. "trunk": the dart
    #: trunk's output, the features the dense heads regress from. "neck+trunk":
    #: the PAN neck's output alongside it -- general features, before the
    #: trunk specialises them for per-cell regression, which is where two
    #: touching darts that the cells confuse may still differ.
    token_inputs: str = "trunk"
    #: Append the 2x2 stride-4 backbone vectors under each stride-8 cell to
    #: its token, projected to ``fine_dim`` each: the detail at which a tip,
    #: or a shaft passing behind another, is actually visible. Stride 8 only.
    fine_tokens: bool = False
    fine_dim: int = 32
    #: Let the query readout's attention see cells up to this many cells
    #: outside the silhouette (the foreground score max-pooled over that
    #: radius), where the dart meets the board. Coordinates still come only
    #: from silhouette cells: a background cell's own dense estimates are
    #: untrained.
    context_radius: int = 0

    # board keypoints
    #: Predict the 40 double-bed corners -- where each of the twenty radial
    #: wires crosses the two circles bounding the double ring.
    #:
    #: These rather than the twenty segment-centre keypoints the renderer also
    #: emits: a segment centre is a point on a ring with nothing in the image
    #: marking it, so its pixel position can only be inferred from the
    #: geometry around it. A wire meeting a ring is a corner you can see.
    #: Forty of them, well spread and individually identifiable, is also a far
    #: better-conditioned set for recovering board pose than twenty.
    #: On in the released model, and required by the exporters.
    predict_keypoints: bool = True
    num_keypoints: int = 40
    kp_head_width: int = 64
    kp_head_depth: int = 2
    kp_weight: float = 1.0
    kp_offset_weight: float = 1.0
    #: Gaussian radius of the keypoint target, in cells.
    kp_sigma: float = 2.0

    # tip heatmap (model/tips.py)
    #: A class-agnostic heatmap of landing points plus a sub-cell offset, built
    #: like the corner head, that the readout's landing points snap to.
    predict_tip_heatmap: bool = False
    tip_head_width: int = 64
    tip_head_depth: int = 2
    #: Grid the tip heatmap runs at: the detector's own stride, or 4 with a
    #: stride-8 detector. At 4 the head reads the backbone's stride-4 stage --
    #: 96 channels wide because that is where a 2-3px tip is resolved -- fused
    #: with the neck upsampled 2x. The stride-8 neck never reads that stage;
    #: there tip peaks levelled off near 2px, about the regression's own
    #: precision, where the corners reach 0.28px.
    tip_stride: int = 8
    #: Gaussian radius of the tip target, in cells of ``tip_stride``. 8px,
    #: half the corners': tips in a tight group are 10-20px apart, and wider
    #: bumps would put the predicted maximum between two tips -- a pull
    #: toward the neighbour, the error the snap is meant to remove.
    tip_sigma: float = 1.0
    tip_heat_weight: float = 1.0
    tip_offset_weight: float = 1.0
    #: Train the sub-cell offset on every cell within this many cells of a
    #: tip's own cell, each pointing at the tip, so a peak one cell off still
    #: lands on it; 0 trains the tip's own cell only, as for the corners.
    tip_offset_radius: int = 0
    #: How the offset loss weighs the cells it is trained on: "uniform", or
    #: "gaussian" by each cell's heatmap target, so a tip's own cell keeps
    #: most of the loss when its neighbours are trained too.
    tip_offset_weighting: str = "uniform"
    #: Let the query readout choose, per query, among its own landing
    #: estimate and the tip heatmap's peaks (queries.PeakPointer), learned
    #: from the landing loss and a cross-entropy on which peak detects the
    #: true tip, instead of a hand-set snap. Requires query_head and
    #: predict_tip_heatmap, and replaces tip_snap.
    tip_pointer: bool = False
    tip_pointer_weight: float = 0.05

    # stage-3 readout (model/token_readout.py)
    #: Learned queries over stride-4 tokens that carry the tip heatmap's dense
    #: supervision: the Hough readout and the tip assignment, in-graph and
    #: learned. Requires predict_tip_heatmap; shares query_count,
    #: query_conf_threshold and the matching costs with the query readout.
    token_readout: bool = False
    readout_tokens: int = 1024
    #: Half-width, in tip cells, of the window token self-attention runs in:
    #: 8 cells is 32px at stride 4, the scale at which touching darts must be
    #: told apart.
    readout_window: int = 8
    readout_enc_layers: int = 2
    readout_dec_layers: int = 3
    readout_dim: int = 96
    readout_heads: int = 4
    readout_conf_weight: float = 1.0
    readout_tip_weight: float = 0.1
    readout_estimate_weight: float = 0.05
    readout_flight_weight: float = 0.05
    readout_dart_attn_weight: float = 0.5
    readout_tip_attn_weight: float = 0.5
    #: Train only the readout: every other weight is frozen and the dense
    #: model runs without gradients. Every fine-tune of the converged dense
    #: model so far disturbed it while it restarted; frozen, it cannot move,
    #: and it costs a fraction of the memory.
    freeze_dense: bool = False
    #: Snap the shipped readout's landing points to tip peaks. Validation logs
    #: the other choice alongside either way.
    tip_snap: bool = False
    #: A peak a landing point may snap to: at least this confident, within
    #: this many pixels of the estimate, and within this many of the dart's
    #: axis through it.
    tip_snap_score: float = 0.3
    tip_snap_px: float = 8.0
    tip_snap_axis_px: float = 4.0
    #: How landing points are given peaks when snapping: "nearest", each dart
    #: its nearest acceptable peak (``tip_snap_px``, ``tip_snap_axis_px``); or
    #: "hungarian", one-to-one over the frame within ``tip_assign_gate_px``
    #: (tips.assign_tips). Only "nearest" runs inside an exported graph;
    #: "hungarian" is the app's to apply to the exported peaks.
    tip_assign: str = "nearest"
    tip_assign_gate_px: float = 40.0
    #: Peaks considered per frame.
    tip_peak_count: int = 16

    # readout (inference only, no gradient)
    #: Accumulator bin size in pixels.  Deliberately NOT the feature stride:
    #: predicted centres are continuous, and binning at stride 8 would merge the
    #: closest decile of dart pairs, measured at 2.5 feature cells apart.
    vote_bin_px: float = 4.0
    #: A cell votes only if its foreground score clears this.
    fg_threshold: float = 0.5
    #: A peak must collect at least this many votes.  Measured floor is 17 votes
    #: for the least-covered dart at stride 8, so this has real headroom.
    peak_min_votes: int = 8
    #: Group cells by the embedding instead of by the centre bin they voted
    #: into. Peaks are still found by centre votes; each peak's seed is the
    #: mean embedding of its centre-claimed core, and every voting cell then
    #: joins the seed nearest in embedding space, if within
    #: ``embed_push_margin``. Requires ``embed_dim``. Validation logs the other
    #: grouping alongside, so the two can be compared on one run.
    claim_by_embedding: bool = False

    # validation
    #: Detections are matched one-to-one to ground-truth darts on the landing
    #: point, and a pair further apart than this, in pixels at the input
    #: resolution, is not a match. 40 px at 1024 is about 1.5 treble-bed
    #: widths: a detection that far off would score a different bed, so it is
    #: not a localisation of that dart. It is also the error charged for a
    #: missed dart in ``val/landing_px_error_penalised``.
    match_gate_px: float = 40.0

    # optimiser
    lr: float = 5.0e-4
    backbone_lr_factor: float = 0.1
    weight_decay: float = 0.05
    #: Fractional epochs, and applied per OPTIMIZER STEP rather than per epoch:
    #: a per-epoch warmup is a staircase, not a ramp. YOLO warms over ~3 epochs
    #: but interpolates every iteration. A frozen backbone gets its own
    #: re-warmup at the unfreeze; see ``backbone_rewarmup_epochs``.
    warmup_epochs: float = 3.0
    #: Floor of the cosine decay, as a fraction of `lr` (YOLO's `lrf`).
    final_lr_factor: float = 0.01
    #: AdamW beta1 is warmed 0.8 -> 0.9 alongside the LR, the Adam analogue of
    #: YOLO's momentum warmup. A low beta1 early means the optimiser leans on
    #: the current gradient rather than an average dominated by the random
    #: initialisation it is trying to leave.
    warmup_beta1: float = 0.8
    #: Optimizer steps per epoch. Set by the trainer, not the YAML.
    #:
    #: It has to be passed in because it cannot be discovered: the dataset is
    #: an IterableDataset with no `__len__`, so Lightning's
    #: `estimated_stepping_batches` returns 1. Taken at face value that
    #: collapses warmup and the whole cosine into a single step.
    steps_per_epoch: int = 0
    max_epochs: int = 500
    gradient_clip_val: float = 0.5

    def __post_init__(self) -> None:
        self.instance_seed_subset = tuple(self.instance_seed_subset)
        if self.out_stride not in (4, 8, 16):
            raise ValueError(
                f"out_stride {self.out_stride} is not one of 4, 8, 16 -- the "
                "backbone emits strides 4/8/16/32 and the head fuses onto one")
        if self.use_pan and self.backbone_dilate_last:
            raise ValueError(
                "use_pan requires every backbone stage to halve resolution, "
                "but backbone_dilate_last keeps the last stage at stride 16, "
                "so the bottom-up path's stride-2 step cannot meet it. Set "
                "one of them to false.")
        if self.token_readout and (self.instance_head or self.query_head):
            raise ValueError("token_readout, query_head and instance_head are "
                             "alternative readouts; set one")
        if self.token_readout and not (self.predict_tip_heatmap
                                       and self.predict_ends):
            raise ValueError("token_readout requires predict_tip_heatmap and "
                             "predict_ends")
        if self.freeze_dense and not (self.token_readout or self.query_head
                                      or self.instance_head):
            raise ValueError("freeze_dense trains a readout only; enable one")
        if self.freeze_dense and self.backbone_freeze_epochs:
            raise ValueError("freeze_dense freezes the backbone for the whole "
                             "run; backbone_freeze_epochs must be 0")
        if self.instance_head and self.query_head:
            raise ValueError("instance_head and query_head are alternative "
                             "readouts; set one")
        if self.token_inputs not in ("trunk", "neck+trunk"):
            raise ValueError(f"token_inputs must be 'trunk' or 'neck+trunk', "
                             f"got {self.token_inputs!r}")
        if self.predict_tip_heatmap and self.tip_stride not in (
                self.out_stride, 4):
            raise ValueError(f"tip_stride must be out_stride ({self.out_stride}) "
                             f"or 4, got {self.tip_stride}")
        if self.tip_assign not in ("nearest", "hungarian"):
            raise ValueError(f"tip_assign must be 'nearest' or 'hungarian', "
                             f"got {self.tip_assign!r}")
        if self.tip_offset_weighting not in ("uniform", "gaussian"):
            raise ValueError(f"tip_offset_weighting must be 'uniform' or "
                             f"'gaussian', got {self.tip_offset_weighting!r}")
        if self.tip_pointer and not (self.query_head and self.predict_tip_heatmap):
            raise ValueError("tip_pointer requires query_head and "
                             "predict_tip_heatmap")
        if self.tip_pointer and self.tip_snap:
            raise ValueError("tip_pointer replaces tip_snap; set one")
        if self.tip_snap and not self.predict_tip_heatmap:
            raise ValueError("tip_snap requires predict_tip_heatmap")
        if self.fine_tokens and self.out_stride != 8:
            raise ValueError("fine_tokens takes the stride-4 detail under a "
                             "stride-8 cell; it needs out_stride 8")
        if self.query_head and not self.predict_ends:
            raise ValueError("query_head requires predict_ends")
        if self.instance_head and (self.embed_dim <= 0 or not self.predict_ends):
            raise ValueError("instance_head requires embed_dim > 0 and "
                             "predict_ends")
        if self.instance_dim % self.instance_heads:
            raise ValueError("instance_dim must be a multiple of "
                             "instance_heads")
        if self.claim_by_embedding and self.embed_dim <= 0:
            raise ValueError("claim_by_embedding requires embed_dim > 0")
        if self.match_gate_px <= 0:
            raise ValueError(f"match_gate_px must be positive, got "
                             f"{self.match_gate_px}")


def _fp32(conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run a regression output layer in float32 even under autocast.

    The offsets are in normalised coordinates, where one unit is the whole
    1024px image, and bf16's 8-bit mantissa holds an offset of 0.1 in steps of
    2^-11 (0.5px) and a half-length of 0.13 in steps of 2^-10 (1px) -- an error
    the readout would then average into every landing point.
    These layers are a few 3x3 convs over the head's output, so the cost of
    full precision is negligible. Outside autocast this is the plain call,
    which keeps the traced export graph free of autocast state.
    """
    dev = x.device.type
    if torch.is_autocast_enabled(dev):
        with torch.autocast(dev, enabled=False):
            return conv(x.float())
    return conv(x)


def _conv(cin: int, cout: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
        nn.GroupNorm(8, cout),
        nn.GELU(),
    )


class DenseDartNet(nn.Module):
    """Backbone -> FPN neck (plus a PAN bottom-up leg when ``use_pan``) -> one
    head at ``out_stride``."""

    def __init__(self, config: DenseDartConfig | None = None) -> None:
        super().__init__()
        cfg = config or DenseDartConfig()
        self.config = cfg

        self.backbone = FineBackbone(FineBackboneConfig(
            widths=tuple(cfg.backbone_widths),
            depths=tuple(cfg.backbone_depths),
            dilate_last=cfg.backbone_dilate_last,
            drop_path=cfg.backbone_drop_path,
        ))
        if cfg.backbone_weights:
            self._load_backbone(cfg.backbone_weights)
        if cfg.backbone_freeze_epochs > 0:
            self.backbone.freeze()

        widths = list(cfg.backbone_widths)
        self.target_level = {4: 0, 8: 1, 16: 2}[cfg.out_stride]
        # Laterals for every level at or below the target; the deeper ones are
        # carried down into it rather than the target being pushed up to them.
        self.lateral = nn.ModuleList(
            nn.Conv2d(w, cfg.head_width, 1)
            for w in widths[self.target_level:])
        self.smooth = nn.ModuleList(
            _conv(cfg.head_width, cfg.head_width)
            for _ in widths[self.target_level:-1])
        n_up = len(widths[self.target_level:]) - 1
        if cfg.use_pan and n_up > 0:
            # Stride-2 convs walking back up, each fused with the top-down map
            # at that level. The head reads the finest level either way; the
            # bottom-up leg exists so the coarse levels -- which the top-down
            # path then feeds back down -- carry fine detail rather than only
            # semantics.
            self.down = nn.ModuleList(
                nn.Conv2d(cfg.head_width, cfg.head_width, 3, stride=2,
                          padding=1) for _ in range(n_up))
            self.pan_smooth = nn.ModuleList(
                _conv(cfg.head_width, cfg.head_width) for _ in range(n_up))
        else:
            self.down = None

        head = []
        for _ in range(cfg.head_depth):
            head.append(_conv(cfg.head_width, cfg.head_width))
        self.head = nn.Sequential(*head)

        # Board keypoints get their OWN stack off the neck rather than sharing
        # the dart head's trunk: sharing features downstream of the neck lets
        # the board loss destabilise dart training. The backbone and neck are
        # shared, which is the point of a multi-task model, but nothing
        # downstream of them is.
        if cfg.predict_keypoints:
            kp = []
            c_in = cfg.head_width
            for _ in range(cfg.kp_head_depth):
                kp.append(_conv(c_in, cfg.kp_head_width))
                c_in = cfg.kp_head_width
            self.kp_trunk = nn.Sequential(*kp)
            self.kp_head = nn.Conv2d(cfg.kp_head_width, cfg.num_keypoints, 3,
                                     padding=1)
            nn.init.constant_(self.kp_head.bias, -4.0)
            # One shared sub-cell offset for all forty, as CenterNet does for
            # its classes: at a peak the offset belongs to whichever keypoint
            # peaked there, and two of these forty landing in the same cell
            # means the board is far enough away that the extra precision is
            # not what is limiting anyway.
            self.kp_offset = nn.Conv2d(cfg.kp_head_width, 2, 3, padding=1)
            nn.init.zeros_(self.kp_offset.weight)
            nn.init.zeros_(self.kp_offset.bias)
        else:
            self.kp_trunk = None

        # The tip heatmap gets its own stack off the neck too, for the same
        # reason as the keypoints.
        self.tip_neck_proj = self.tip_fine_proj = None
        if cfg.predict_tip_heatmap:
            tip = []
            c_in = cfg.head_width
            if cfg.tip_stride < cfg.out_stride:
                # The neck upsampled to stride 4 plus a lateral from the
                # stride-4 backbone stage, summed, as a top-down FPN step.
                self.tip_neck_proj = nn.Conv2d(cfg.head_width,
                                               cfg.tip_head_width, 1)
                self.tip_fine_proj = nn.Conv2d(widths[0], cfg.tip_head_width, 1)
                c_in = cfg.tip_head_width
            for _ in range(cfg.tip_head_depth):
                tip.append(_conv(c_in, cfg.tip_head_width))
                c_in = cfg.tip_head_width
            self.tip_trunk = nn.Sequential(*tip)
            self.tip_heat = nn.Conv2d(cfg.tip_head_width, 1, 3, padding=1)
            nn.init.constant_(self.tip_heat.bias, -4.0)
            self.tip_cell_offset = nn.Conv2d(cfg.tip_head_width, 2, 3, padding=1)
            nn.init.zeros_(self.tip_cell_offset.weight)
            nn.init.zeros_(self.tip_cell_offset.bias)
        else:
            self.tip_trunk = None

        self.fg_head = nn.Conv2d(cfg.head_width, 1, 3, padding=1)
        # Start sparse: foreground is ~1.6% of cells, and a head that begins by
        # calling everything a dart spends its first epochs unlearning that.
        nn.init.constant_(self.fg_head.bias, -4.0)

        # Offsets from the cell centre to the dart's landing point and to its
        # flight tip -- the same convention as centre_offset, so the readout
        # votes them with the same machinery.
        if cfg.predict_ends:
            self.ends_head = nn.Conv2d(cfg.head_width, 4, 3, padding=1)
            # Zeroed: every cell starts by naming its own centre, which is
            # within half a dart of the truth, rather than a random point.
            nn.init.zeros_(self.ends_head.weight)
            nn.init.zeros_(self.ends_head.bias)
        else:
            self.ends_head = None

        # Default (random) init, unlike the regression heads: an all-zero
        # embedding puts every dart's mean at the same point, where the push
        # term's distance has no direction to grow in.
        self.embed_head = (nn.Conv2d(cfg.head_width, cfg.embed_dim, 3, padding=1)
                           if cfg.embed_dim > 0 else None)
        self.token_proj = self.instance_head = self.query_head = None
        self.token_readout = None
        if cfg.token_readout:
            from darts_model.model.token_readout import TokenReadout
            self.token_readout = TokenReadout(cfg)
        self.fine_proj = None
        if cfg.instance_head or cfg.query_head:
            c_in = cfg.head_width * (2 if cfg.token_inputs == "neck+trunk" else 1)
            self.token_proj = nn.Conv2d(c_in, cfg.token_dim, 1)
            if cfg.fine_tokens:
                self.fine_proj = nn.Conv2d(widths[0], cfg.fine_dim, 1)
        if cfg.instance_head:
            from darts_model.model.instance import InstanceHead
            self.instance_head = InstanceHead(cfg)
        if cfg.query_head:
            from darts_model.model.queries import QueryHead
            self.query_head = QueryHead(cfg)
        #: Whether the per-dart readout reads the dense field detached; set
        #: per epoch by the training module (``readout_warmup_epochs``).
        self.readout_detached = False

        # centre offset (2), (cos, sin) (2), log half-length, log half-width
        self.box_head = nn.Conv2d(cfg.head_width, 6, 3, padding=1)
        nn.init.zeros_(self.box_head.weight)
        nn.init.zeros_(self.box_head.bias)
        with torch.no_grad():
            self.box_head.bias[2] = 1.0      # unit direction along +x
            self.box_head.bias[4] = math.log(0.076)
            self.box_head.bias[5] = math.log(0.017)

    def _load_backbone(self, path: str) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck.get("backbone_state_dict")
        if sd is None:
            for prefix in ("backbone.", "model.backbone."):
                sd = {k.split(prefix, 1)[1]: v
                      for k, v in ck["state_dict"].items()
                      if k.startswith(prefix)}
                if sd:
                    break
        # Strict: a checkpoint whose keys do not line up would otherwise load
        # nothing and train from random init, which looks like a bad result
        # rather than a missing pretraining step.
        if not sd:
            raise RuntimeError(f"{path} holds no backbone weights")
        self.backbone.load_state_dict(sd, strict=True)
        print(f"FineBackbone: loaded {len(sd)} tensors from {path}", flush=True)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        stages = self.backbone(images)
        if isinstance(stages, tuple):
            stages = stages[1]
        stages = list(stages)
        fine = stages[0]
        stages = stages[self.target_level:]

        # Top-down: upsample the coarse onto the fine and add. Never resample
        # fine levels down onto coarse ones, which does not blur fine
        # activations so much as drop them -- an isolated stride-4 response
        # between sample points measures norm 0.0000.
        feats = [lat(s) for lat, s in zip(self.lateral, stages)]
        top_down = [None] * len(feats)
        x = feats[-1]
        top_down[-1] = x
        for i in range(len(feats) - 2, -1, -1):
            x = feats[i] + F.interpolate(x, size=feats[i].shape[-2:],
                                         mode="bilinear", align_corners=False)
            x = self.smooth[i](x)
            top_down[i] = x

        if self.down is not None:
            # Bottom-up: walk back up fusing with the top-down map at each
            # level, then walk down once more so the finest level -- the one
            # the head reads -- sees what the coarse levels learned from it.
            up = top_down[0]
            bottom_up = [up]
            for i, (dn, sm) in enumerate(zip(self.down, self.pan_smooth)):
                up = sm(dn(up) + top_down[i + 1])
                bottom_up.append(up)
            x = bottom_up[-1]
            for i in range(len(bottom_up) - 2, -1, -1):
                x = bottom_up[i] + F.interpolate(
                    x, size=bottom_up[i].shape[-2:], mode="bilinear",
                    align_corners=False)

        neck = x
        x = self.head(x)
        box = _fp32(self.box_head, x)
        out_ends = {}
        if self.ends_head is not None:
            e = _fp32(self.ends_head, x)
            out_ends = {"tip_offset": e[:, 0:2], "flight_offset": e[:, 2:4]}
        if self.token_readout is not None:
            # The stage-3 tokens read the neck at their stride-8 parent cell.
            out_ends["neck"] = neck
        if self.embed_head is not None:
            out_ends["embedding"] = _fp32(self.embed_head, x)   # (B, E, h, w)
        if self.token_proj is not None:
            t = (torch.cat((neck, x), 1) if self.config.token_inputs == "neck+trunk"
                 else x)
            t = t.detach() if self.readout_detached else t
            out_ends["tokens"] = self.token_proj(t)             # (B, D, h, w)
        if self.fine_proj is not None:
            f = fine.detach() if self.readout_detached else fine
            out_ends["fine"] = self.fine_proj(f)                # (B, F, 2h, 2w)

        if self.tip_trunk is not None:
            n = neck.detach() if self.readout_detached else neck
            if self.tip_fine_proj is not None:
                f = fine.detach() if self.readout_detached else fine
                n = (self.tip_fine_proj(f)
                     + F.interpolate(self.tip_neck_proj(n), size=f.shape[-2:],
                                     mode="bilinear", align_corners=False))
            t = self.tip_trunk(n)
            # (B, h, w) at tip_stride: twice the detector's grid at stride 4.
            out_ends["tip_logits"] = self.tip_heat(t)[:, 0]
            # What the tip head saw at each cell: the pointer's peak features.
            out_ends["tip_features"] = t
            out_ends["tip_cell_offset"] = _fp32(self.tip_cell_offset, t)

        out_kp = {}
        if self.kp_trunk is not None:
            k = self.kp_trunk(neck)
            out_kp = {
                "kp_logits": self.kp_head(k),              # (B, K, h, w)
                "kp_offset": _fp32(self.kp_offset, k),     # (B, 2, h, w)
            }
        return {
            **out_kp,
            **out_ends,
            "fg_logits": self.fg_head(x)[:, 0],          # (B, h, w)
            "centre_offset": box[:, 0:2],                # (B, 2, h, w)
            # RAW, deliberately not normalised here. Any normalisation in the
            # forward puts a 1/|x| into the backward, and nothing supervises a
            # background cell's direction, so cells are free to drift through
            # zero. Clamping the norm caps the forward and not the gradient
            # (1/eps); putting eps inside the root caps it at 1/sqrt(eps),
            # which is still 1e4 per cell and sums over the whole map.
            #
            # The target is a unit vector and the loss is an L1 against it, so
            # the raw output is trained toward unit length anyway, with a
            # bounded gradient. Normalisation happens at readout, under
            # no_grad, where it cannot reach the weights.
            "direction": box[:, 2:4],                    # (B, 2, h, w)
            # Clamped, because the consumers exponentiate it. An unbounded
            # log-extent is an overflow path: an overfit can reach a half
            # extent of 661 (log 6.5) in 30 steps, and exp() of a runaway log
            # is inf, then NaN. Measured darts project to 18-268 px at 1024,
            # i.e. half-extents of 0.009-0.131 normalised, so these bounds sit
            # 9x below and 4x above anything real and cannot bind on data.
            "log_extent": box[:, 4:6].clamp(LOG_EXTENT_MIN, LOG_EXTENT_MAX),
        }

    def read_darts(self, out: dict, instance: torch.Tensor | None = None,
                   n_slots: int = 0,
                   subset: tuple[float, float] | None = None) -> dict:
        """The per-dart readout over a dense field from :meth:`forward`.

        Without ``instance``, seeds are picked in-graph and the planes are
        (B, instance_slots, ...), with ``score`` the vote density a slot must
        clear to be a dart. With ``instance`` (the ground-truth dart map) and
        its ``n_slots``, slot s is ground-truth dart s, and ``present`` says
        which exist.

        With ``query_head`` instead, the planes are (B, query_count, ...) and
        ``score`` is each query's confidence; there are no seeds to give.
        """
        from darts_model.model.instance import (
            candidate_count, gather_candidates, ground_truth_seeds,
            select_seeds,
        )
        cfg = self.config
        if self.readout_detached:
            # The readout's own inputs stay trainable.
            own = ("tokens", "fine", "tip_logits", "tip_cell_offset",
                   "tip_features")
            out = {k: v if k in own else v.detach() for k, v in out.items()}
        image_px = float(int(out["fg_logits"].shape[-1]) * cfg.out_stride)
        centre_scale = cfg.membership_centre_px / image_px
        if self.token_readout is not None:
            from darts_model.model.token_readout import gather_tokens
            return self.token_readout(gather_tokens(out, cfg.readout_tokens))
        cand = gather_candidates(out, candidate_count(cfg.out_stride),
                                 centre_scale, cfg.context_radius)
        if self.query_head is not None:
            if not cfg.tip_pointer:
                return self.query_head(cand)
            # Detached: the tip head is trained by its own losses only.
            from darts_model.model.tips import tip_peaks
            xy, score, idx = tip_peaks(out["tip_logits"].detach(),
                                       out["tip_cell_offset"].detach(),
                                       cfg.tip_peak_count, return_index=True)
            feat = out["tip_features"].detach().float()
            b, c = int(feat.shape[0]), int(feat.shape[1])
            p = int(idx.shape[1])
            feat = torch.gather(feat.reshape(b, c, -1), 2,
                                idx.unsqueeze(1).expand(b, c, p)).transpose(1, 2)
            res = self.query_head(cand, {"xy": xy, "score": score,
                                         "feature": feat})
            res["peaks_xy"], res["peaks_score"] = xy, score
            return res
        if instance is None:
            mu, score = select_seeds(cand, cfg.instance_slots, cfg, image_px)
            ok = (score >= cfg.peak_min_votes * cfg.fg_threshold).float()
            res = self.instance_head(cand, mu, ok)
            res["score"] = score
            return res
        mu, present = ground_truth_seeds(out, instance, n_slots, centre_scale,
                                         subset)
        res = self.instance_head(cand, mu, present.float())
        res["present"] = present
        return res


#: Bounds on the predicted log half-extent. See the forward for why.
LOG_EXTENT_MIN = math.log(0.001)
LOG_EXTENT_MAX = math.log(0.5)


def penalty_reduced_focal(logits, target, alpha: float = 2.0,
                          beta: float = 4.0):
    """CenterNet's focal loss, normalised by positive count.

    Chosen over weighted BCE because one positive in ~60 cells is an imbalance
    BCE can only be told about through a ``pos_weight`` somebody has to derive,
    and a wrongly derived one silently breaks the head.
    """
    # float32 and log-space, both deliberately.
    #
    # Clamping p to [1e-4, 1 - 1e-4] and taking (1 - p).log() is not safe
    # under bf16 autocast: bf16 has 8 mantissa bits, so the smallest step below
    # 1.0 is 2^-8 ~ 0.0039 and 0.9999 rounds to exactly 1.0. sigmoid saturates
    # there for any logit above roughly 6.2, and then (1 - p).log() is
    # log(0) = -inf. The 40-channel keypoint head crosses that line routinely.
    #
    # logsigmoid is exact for every finite input and needs no clamp at all, so
    # the loss is finite at any precision.
    logits = logits.float()
    target = target.float()
    log_p = F.logsigmoid(logits)
    log_1mp = F.logsigmoid(-logits)
    p = logits.sigmoid()
    pos = target.ge(1.0 - 1e-6).float()
    pos_loss = -((1.0 - p) ** alpha) * log_p * pos
    neg_loss = (-((1.0 - target) ** beta) * (p ** alpha)
                * log_1mp * (1.0 - pos))
    return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1.0)


def _dist(a: torch.Tensor, dim: int) -> torch.Tensor:
    """Euclidean norm with a gradient at zero.

    A dart covered by a single cell sits exactly on its own mean, and the
    plain norm's gradient there is 0/0.
    """
    return (a.square().sum(dim) + 1e-12).sqrt()


def slot_floor(cfg) -> float:
    """The score a per-dart slot must clear to be a dart: a query's
    confidence threshold, or a seed's Hough vote floor."""
    if cfg.query_head or getattr(cfg, "token_readout", False):
        return float(cfg.query_conf_threshold)
    return float(cfg.peak_min_votes * cfg.fg_threshold)


def token_feature_dim(cfg) -> int:
    """Channels of a candidate cell's ``feature``: the token projection, then
    the 2x2 stride-4 detail when ``fine_tokens``."""
    return cfg.token_dim + (4 * cfg.fine_dim if cfg.fine_tokens else 0)


def slot_count(cfg) -> int:
    """Per-dart slots the model reads out; 0 without a per-dart readout."""
    if cfg.query_head or getattr(cfg, "token_readout", False):
        return int(cfg.query_count)
    if cfg.instance_head:
        return int(cfg.instance_slots)
    return 0


def _huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    """Huber of a non-negative error: quadratic below ``delta``, linear
    above, continuous in value and slope."""
    return torch.where(x < delta, 0.5 * x.square() / delta, x - 0.5 * delta)


def embedding_loss(emb, inst, n_slots: int, pull_margin: float,
                   push_margin: float, include=None):
    """Discriminative instance-embedding loss (De Brabandere et al., 2017).

    ``emb`` (B, E, h, w); ``inst`` (B, h, w) with 0 for background and slot
    + 1 for a dart; ``include`` (B, h, w) bool, the dart cells that take
    part (default: all of them). Returns ``(pull, push, reg)``:

    * pull -- per dart, the mean over its cells of
      ``max(0, |e - mu| - pull_margin)^2``; then the mean over darts.
    * push -- per pair of darts in one frame, ``max(0, 2 * push_margin -
      |mu_a - mu_b|)^2``; then the mean over pairs. Zero for a batch with no
      frame holding two darts.
    * reg -- the mean ``|mu|`` over darts.

    Averaged per dart before over darts, so a dart of 20 cells weighs as
    much as one of 200: the short, foreshortened darts are where groups
    overlap most.
    """
    emb = emb.float()
    b, e, h, w = emb.shape
    zero = emb.sum() * 0.0
    member = inst.unsqueeze(1) == torch.arange(
        1, n_slots + 1, device=inst.device).view(1, -1, 1, 1)   # (B, S, h, w)
    if include is not None:
        member = member & include.unsqueeze(1)
    member = member.float()
    count = member.sum((2, 3))                                   # (B, S)
    present = count > 0
    if not bool(present.any()):
        return zero, zero, zero
    mu = torch.einsum("behw,bshw->bse", emb, member) / count.clamp(
        min=1.0).unsqueeze(-1)                                   # (B, S, E)

    # Each cell against its own dart's mean; cells of no dart get mu = 0 and
    # are masked out of the sum.
    mu_cell = torch.einsum("bshw,bse->behw", member, mu)
    hinge = (_dist(emb - mu_cell, 1).unsqueeze(1) - pull_margin).clamp(
        min=0.0).square()                                        # (B, 1, h, w)
    per_dart = (hinge * member).sum((2, 3)) / count.clamp(min=1.0)
    pull = per_dart[present].mean()

    gap = _dist(mu.unsqueeze(2) - mu.unsqueeze(1), -1)           # (B, S, S)
    pair = (present.unsqueeze(2) & present.unsqueeze(1)
            & torch.ones(n_slots, n_slots, dtype=torch.bool,
                         device=inst.device).triu(1))
    push = ((2.0 * push_margin - gap).clamp(min=0.0).square()[pair].mean()
            if bool(pair.any()) else zero)

    reg = _dist(mu, -1)[present].mean()
    return pull, push, reg


def keypoint_targets(kp, mask, hw, sigma: float):
    """Normalised keypoints -> per-keypoint Gaussian heatmaps and offsets.

    ``kp`` (B, K, 2) in [0, 1], ``mask`` (B, K). Returns the heatmap (B, K, h,
    w), the sub-cell offset (B, 2, h, w) and the cells the offset is defined at
    (B, h, w).

    Built on device each step rather than in the loader: it is a handful of
    exponentials on a grid the GPU is already holding, against sending
    B*40*128*128 floats over the bus every batch.
    """
    b, k, _ = kp.shape
    h, w = hw
    ys = torch.arange(h, device=kp.device).view(1, 1, h, 1) + 0.5
    xs = torch.arange(w, device=kp.device).view(1, 1, 1, w) + 0.5

    # Into cell units, where sigma is expressed.
    cx = kp[..., 0].view(b, k, 1, 1) * w
    cy = kp[..., 1].view(b, k, 1, 1) * h
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    heat = torch.exp(-d2 / (2.0 * sigma * sigma))
    heat = heat * mask.view(b, k, 1, 1).float()

    # The offset is supervised only at each keypoint's own cell, so it is
    # gathered there rather than painted over the map.
    ix = cx.view(b, k).long().clamp(0, w - 1)
    iy = cy.view(b, k).long().clamp(0, h - 1)
    off = torch.zeros((b, 2, h, w), device=kp.device)
    at = torch.zeros((b, h, w), dtype=torch.bool, device=kp.device)
    bi = torch.arange(b, device=kp.device).view(b, 1).expand(b, k)
    sel = mask
    if bool(sel.any()):
        # The keypoint's own cell is pinned to exactly 1.0, as CenterNet does.
        # Sampled at cell centres the Gaussian peaks just BELOW 1 -- 0.975 for
        # a keypoint sitting a third of a cell off centre -- and the focal loss
        # counts a cell as positive only at 1.0. Without the pin no cell is
        # ever a positive, the loss is its negative term alone, and the head
        # learns nothing.
        ki = torch.arange(k, device=kp.device).view(1, k).expand(b, k)
        heat[bi[sel], ki[sel], iy[sel], ix[sel]] = 1.0
        off[bi[sel], 0, iy[sel], ix[sel]] = (
            cx.view(b, k)[sel] - ix[sel].float() - 0.5)
        off[bi[sel], 1, iy[sel], ix[sel]] = (
            cy.view(b, k)[sel] - iy[sel].float() - 0.5)
        at[bi[sel], iy[sel], ix[sel]] = True
    return heat, off, at


def cell_centres(h: int, w: int, device) -> torch.Tensor:
    """Normalised (x, y) centre of every cell.  (2, h, w)."""
    ys, xs = torch.meshgrid(
        (torch.arange(h, device=device) + 0.5) / h,
        (torch.arange(w, device=device) + 0.5) / w,
        indexing="ij",
    )
    return torch.stack([xs, ys], dim=0)


def decode_boxes(out: dict, stride_hw: tuple[int, int]) -> dict:
    """Per-cell predictions -> absolute boxes.

    Offsets are cell-relative, so two cells on one dart emit *different*
    numbers that decode to the *same* box.  Voting therefore has to decode
    first and cluster after; averaging the raw channels would be wrong.
    """
    h, w = stride_hw
    centres = cell_centres(h, w, out["centre_offset"].device)[None]
    d = out["direction"]
    # Normalised here and nowhere else. This runs at readout, outside the
    # gradient path, so 1/|x| cannot reach the weights.
    d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-6)
    dec = {
        "centre": centres + out["centre_offset"],     # (B, 2, h, w) normalised
        "direction": d,
        "extent": out["log_extent"].exp(),
    }
    # Absolute, by the same cell-centre-plus-offset rule as the box centre, so
    # every cell on a dart names the same two points and the accumulator can
    # average them.
    if "tip_offset" in out:
        dec["tip_point"] = centres + out["tip_offset"]
        dec["flight_point"] = centres + out["flight_offset"]
    if "embedding" in out:
        dec["embedding"] = out["embedding"]
    return dec


def match_points(gt: torch.Tensor, pred: torch.Tensor, gate: float):
    """One-to-one assignment of predictions to ground truth within a gate.

    ``gt`` (G, 2) and ``pred`` (P, 2) in the same units as ``gate``. Returns
    ``(gt_idx, pred_idx, dist)`` for the matched pairs only.

    Optimal (Hungarian) rather than greedy or per-GT nearest: a pair outside
    the gate costs more than every in-gate pair combined, so the assignment
    first maximises the number of in-gate matches and then minimises their
    total distance. A per-GT argmin would let one prediction serve several
    darts and hide a missed dart in a tight group.
    """
    from scipy.optimize import linear_sum_assignment

    empty = torch.zeros(0, dtype=torch.long)
    if gt.shape[0] == 0 or pred.shape[0] == 0:
        return empty, empty, torch.zeros(0)
    d = torch.cdist(gt.double().cpu(), pred.double().cpu())
    big = gate * (min(d.shape) + 1)
    cost = torch.where(d <= gate, d, torch.full_like(d, big))
    gi, pi = linear_sum_assignment(cost.numpy())
    gi = torch.as_tensor(gi, dtype=torch.long)
    pi = torch.as_tensor(pi, dtype=torch.long)
    ok = d[gi, pi] <= gate
    return gi[ok], pi[ok], d[gi, pi][ok].float()


class DenseDartLitModule(L.LightningModule):
    """Training wrapper.

    Three masked regressions plus a focal foreground term.  No matcher, no
    assignment, no confidence head: which cell supervises which dart is given by
    the silhouette, and "how many darts" is answered at readout by counting
    peaks rather than by a head whose meaning drifts when the backbone moves.

    Single-device only. The non-finite batch skip returns ``None`` from
    ``training_step``, which would deadlock DDP, and the validation metrics
    are ratios and quantiles over per-dart lists that are not reduced across
    ranks; :meth:`setup` refuses a multi-process trainer rather than report
    rank-local numbers.
    """

    #: The monitor for checkpoint ranking. Logged at the end of EVERY
    #: validation epoch, so a checkpoint callback can neither fail to find it
    #: nor rank an epoch on a value left over from an earlier one.
    MONITOR = "val/landing_px_error_penalised"
    #: The validation metrics logged unconditionally, i.e. the only ones safe
    #: to rank checkpoints on. Everything else is logged only when defined
    #: (errors only when something matched, precision only when something was
    #: detected, and so on), and Lightning would keep an epoch-old value in its
    #: place.
    RANKABLE_METRICS = (MONITOR, "val/total_loss")

    def __init__(self, config: DenseDartConfig,
                 gpu_augment: GPUAugment | None = None) -> None:
        """``gpu_augment`` is a built ``GPUAugment``, e.g. from
        ``darts_model.data.gpu_augment.gpu_augment_from_config``. The default
        normalises only, with no sensor noise, which is what validation,
        export and tests need."""
        super().__init__()
        self.save_hyperparameters({"config": vars(config)})
        self.config = config
        self.model = DenseDartNet(config)
        if config.init_weights:
            self._load_full(config.init_weights)
        if config.freeze_dense:
            for name, p in self.model.named_parameters():
                if not name.startswith(self.TRAINED_WHEN_FROZEN):
                    p.requires_grad_(False)

        # ALWAYS present, because it carries the normalisation as well as the
        # noise -- and normalisation is part of the model's input contract, not
        # an augmentation.
        #
        # It lives on this module, not a DataModule, because
        # `darts_model/cli/train.py` hands Lightning plain DataLoaders, so a
        # DataModule's `on_after_batch_transfer` would never run. The backbone
        # is pretrained through the same GPUAugment (see
        # `darts_model/cli/pretrain.py`), and the two pipelines must match.
        if gpu_augment is None:
            from darts_model.data.gpu_augment import GPUAugment
            gpu_augment = GPUAugment(enabled=False)
        self.gpu_augment = gpu_augment

        self._skipped = 0
        # Split, because the aggregate cannot say whether the LOSS went
        # non-finite or the GRADIENTS did -- a different cause and a different
        # fix -- or which module produced it.
        self._skipped_loss = 0
        self._skipped_grad = 0
        self._nonfinite_params: dict[str, int] = {}
        #: global_step at which the backbone was unfrozen; anchors its LR
        #: re-warmup, and is checkpointed so a resume does not restart it.
        self._unfreeze_step: int | None = None
        self._rewarm_steps = 0
        self._lr_lambda = None
        self._warmup_steps = 0
        #: Optimizer steps of the embedding weight's ramp; set with the
        #: schedule, since only then is the epoch length known. 0 (outside a
        #: trainer) means full weight.
        self._embed_ramp_steps = 0
        self.on_validation_epoch_start()

    def setup(self, stage: str) -> None:
        tr = getattr(self, "_trainer", None)
        if tr is not None and tr.world_size > 1:
            raise NotImplementedError(
                f"DenseDartLitModule is single-device (world size "
                f"{tr.world_size} requested): the non-finite batch skip would "
                "deadlock DDP and the validation metrics are not reduced "
                "across ranks. Train with devices: 1.")

    # ------------------------------------------------------------------ loss

    def _targets(self, batch, hw):
        """Per-cell box targets, gathered from each cell's own dart.

        Every foreground cell is supervised toward the WHOLE box of the dart it
        sits on, so cells at the tip and at the flight carry identical angle and
        extents, and centre offsets differing by exactly their displacement.
        That identity is what makes them votes for one point.

        Background cells gather dart 0's entry, which may be padding; every
        consumer masks them out.
        """
        h, w = hw
        inst = batch["instance"]                       # (B, h, w), 0 = bg
        box = batch["dart_box"]                        # (B, MAX_DARTS, 6)
        idx = (inst - 1).clamp(min=0)
        B = inst.shape[0]
        flat = box.gather(1, idx.view(B, -1, 1).expand(-1, -1, 6))
        return flat.view(B, h, w, 6).permute(0, 3, 1, 2)   # (B, 6, h, w)

    def _gather_ends(self, batch, hw):
        """Per-cell (landing_xy, flight_xy) of the dart that cell belongs to.

        The same gather as _targets: every cell of a dart carries that dart's
        two endpoints, so cells at opposite ends emit offsets differing by
        exactly their separation and vote for the same pair of points.
        """
        h, w = hw
        inst = batch["instance"]
        ends = batch["dart_ends"]
        idx = (inst - 1).clamp(min=0)
        B = inst.shape[0]
        flat = ends.gather(1, idx.view(B, -1, 1).expand(-1, -1, 4))
        return flat.view(B, h, w, 4).permute(0, 3, 1, 2)

    @staticmethod
    def _box_keep(batch, keep):
        """``keep`` restricted to cells whose dart has a usable box orientation.

        Batches without ``dart_box_mask`` treat every box as usable.
        """
        if "dart_box_mask" not in batch:
            return keep, keep.sum().clamp(min=1).float()
        inst = batch["instance"]
        B = inst.shape[0]
        idx = (inst - 1).clamp(min=0).view(B, -1)
        ok = batch["dart_box_mask"].gather(1, idx).view_as(inst)
        box_keep = keep & ok
        return box_keep, box_keep.sum().clamp(min=1).float()

    @staticmethod
    def _masked_l1(pred, target, keep, n):
        """Mean absolute error over the ``keep`` cells, per channel.

        The target is zeroed outside the mask before the difference rather than
        the product masked after it: a masked-out target may be padding holding
        anything, and NaN * 0 is NaN in the forward and in the gradient.
        ``n`` is the batch-wide count of kept cells, so a frame with no darts
        contributes nothing and an all-empty batch gives exactly zero.
        """
        m = keep[:, None]
        target = torch.where(m, target, torch.zeros_like(target))
        return ((pred - target).abs() * m).sum() / n / pred.shape[1]

    def _step(self, batch, stage: str):
        # A frozen dense model runs without gradients: the readout reads it
        # and trains on its own.
        frozen = torch.no_grad() if self.config.freeze_dense else nullcontext()
        with frozen:
            out = self.model(batch["image"].float())
        h, w = out["fg_logits"].shape[-2:]
        fg = batch["foreground"]
        keep = fg > 0
        n = keep.sum().clamp(min=1).float()
        tgt = self._targets(batch, (h, w))
        centres = cell_centres(h, w, fg.device)[None]
        cfg = self.config

        losses = {"fg": penalty_reduced_focal(
            out["fg_logits"], fg, cfg.focal_alpha, cfg.focal_gamma)}
        losses["centre"] = self._masked_l1(
            out["centre_offset"], tgt[:, 0:2] - centres, keep, n)
        # Direction and extents skip darts seen end-on, whose box is a
        # fallback; their centre and ends are still supervised.
        box_keep, n_box = self._box_keep(batch, keep)
        losses["dir"] = self._masked_l1(out["direction"], tgt[:, 2:4],
                                        box_keep, n_box)
        losses["size"] = self._masked_l1(
            out["log_extent"], tgt[:, 4:6].clamp(min=1e-6).log(), box_keep, n_box)
        total = (losses["fg"] * cfg.fg_weight
                 + losses["centre"] * cfg.centre_weight
                 + losses["dir"] * cfg.dir_weight
                 + losses["size"] * cfg.size_weight)

        if cfg.predict_ends and "dart_ends" in batch:
            # Same masked-L1 shape as the box terms, against the renderer's own
            # two points rather than anything derived from the box.
            ends = self._gather_ends(batch, (h, w))
            losses["tip"] = self._masked_l1(
                out["tip_offset"], ends[:, 0:2] - centres, keep, n)
            losses["flight"] = self._masked_l1(
                out["flight_offset"], ends[:, 2:4] - centres, keep, n)
            total = (total + losses["tip"] * cfg.tip_weight
                     + losses["flight"] * cfg.flight_weight)

        if cfg.embed_dim > 0:
            include = keep
            if "instance_purity" in batch:
                include = include & (batch["instance_purity"]
                                     >= cfg.embed_min_purity)
            pull, push, reg = embedding_loss(
                out["embedding"], batch["instance"], batch["dart_box"].shape[1],
                cfg.embed_pull_margin, cfg.embed_push_margin, include)
            losses["embed_pull"], losses["embed_push"] = pull, push
            ramp = 1.0
            if stage == "train" and self._embed_ramp_steps > 0:
                ramp = min((self.global_step + 1) / self._embed_ramp_steps, 1.0)
            total = total + ramp * cfg.embed_weight * (
                pull + push + cfg.embed_reg_weight * reg)

        if cfg.query_head and "dart_ends" in batch:
            from darts_model.model.queries import query_losses
            darts = self.model.read_darts(out)
            scale = float(batch["image"].shape[-1])
            q_losses, tip_px, match = query_losses(darts, batch, cfg, scale,
                                                   _huber, return_match=True)
            if cfg.tip_pointer:
                q_losses["pointer"] = self._pointer_loss(darts, batch, match,
                                                         scale, stage)
                total = total + cfg.tip_pointer_weight * q_losses["pointer"]
            losses.update(q_losses)
            total = (total
                     + cfg.query_conf_weight * q_losses["q_conf"]
                     + cfg.instance_weight * q_losses["q_tip"]
                     + cfg.instance_flight_weight * q_losses["q_flight"]
                     + cfg.query_centre_weight * q_losses["q_centre"]
                     + cfg.query_dir_weight * q_losses["q_dir"]
                     + cfg.query_size_weight * q_losses["q_size"])
            if stage == "val":
                # The matched queries' landing error, whatever their
                # confidence: localisation apart from detection.
                self._gt_seed_err.append(tip_px.cpu())

        if cfg.token_readout and "dart_ends" in batch:
            from darts_model.model.token_readout import (
                gather_tokens, readout_losses,
            )
            tok = gather_tokens(out, cfg.readout_tokens)
            pred = self.model.token_readout(tok)
            r, tip_px, _ = readout_losses(pred, tok, batch, cfg,
                                          float(batch["image"].shape[-1]),
                                          _huber)
            losses.update(r)
            total = (total + cfg.readout_conf_weight * r["r_conf"]
                     + cfg.readout_tip_weight * r["r_tip"]
                     + cfg.readout_estimate_weight * r["r_estimate"]
                     + cfg.readout_flight_weight * r["r_flight"]
                     + cfg.readout_dart_attn_weight * r["r_dart_attn"]
                     + cfg.readout_tip_attn_weight * r["r_tip_attn"])
            if stage == "val":
                self._gt_seed_err.append(tip_px.cpu())

        if cfg.instance_head and "dart_ends" in batch:
            # Slot s is ground-truth dart s; its seed is the mean embedding of
            # a random part of the dart in training, of all of it in
            # validation, where it measures the head apart from seed-picking.
            n_slots = batch["dart_ends"].shape[1]
            darts = self.model.read_darts(
                out, batch["instance"], n_slots,
                cfg.instance_seed_subset if stage == "train" else None)
            ok = darts["present"] & batch["dart_mask"] & (darts["support"] > 1e-3)
            scale = float(batch["image"].shape[-1])
            ends = batch["dart_ends"].float()
            tip_px = _dist((darts["tip"] - ends[..., 0:2]) * scale, -1)
            flight_px = _dist((darts["flight"] - ends[..., 2:4]) * scale, -1)
            zero = tip_px.sum() * 0.0
            losses["inst_tip"] = (_huber(tip_px[ok], cfg.instance_huber_px).mean()
                                  if bool(ok.any()) else zero)
            losses["inst_flight"] = (
                _huber(flight_px[ok], cfg.instance_huber_px).mean()
                if bool(ok.any()) else zero)
            total = (total + cfg.instance_weight * losses["inst_tip"]
                     + cfg.instance_flight_weight * losses["inst_flight"])
            if stage == "val":
                self._gt_seed_err.append(tip_px[ok].detach().cpu())

        if cfg.predict_tip_heatmap and "dart_ends" in batch:
            from darts_model.model.tips import tip_targets
            heat, off, at = tip_targets(batch["dart_ends"][..., 0:2].float(),
                                        batch["dart_mask"],
                                        tuple(out["tip_logits"].shape[-2:]),
                                        cfg.tip_sigma, cfg.tip_offset_radius)
            losses["tip_heat"] = penalty_reduced_focal(
                out["tip_logits"], heat, cfg.focal_alpha, cfg.focal_gamma)
            if cfg.tip_offset_weighting == "gaussian":
                from darts_model.model.tips import weighted_offset_l1
                losses["tip_cell_offset"] = weighted_offset_l1(
                    out["tip_cell_offset"].float(), off, heat * at.float())
            else:
                losses["tip_cell_offset"] = self._masked_l1(
                    out["tip_cell_offset"], off, at,
                    at.sum().clamp(min=1).float())
            total = (total + losses["tip_heat"] * cfg.tip_heat_weight
                     + losses["tip_cell_offset"] * cfg.tip_offset_weight)

        if cfg.predict_keypoints and "keypoints" in batch:
            kp_heat, kp_off, kp_at = keypoint_targets(
                batch["keypoints"], batch["keypoint_mask"], (h, w),
                cfg.kp_sigma)
            losses["kp"] = penalty_reduced_focal(
                out["kp_logits"], kp_heat, cfg.focal_alpha, cfg.focal_gamma)
            losses["kp_off"] = self._masked_l1(
                out["kp_offset"], kp_off, kp_at,
                kp_at.sum().clamp(min=1).float())
            total = (total + losses["kp"] * cfg.kp_weight
                     + losses["kp_off"] * cfg.kp_offset_weight)

        if stage == "train" and not torch.isfinite(total):
            # Skip the batch rather than let it reach the optimiser: a single
            # non-finite step makes the WEIGHTS non-finite, and the run is
            # lost from healthy losses with no warning.
            self._skipped += 1
            self._skipped_loss += 1
            bad = [k for k, v in losses.items() if not torch.isfinite(v)]
            rank_zero_info(
                f"[nonfinite] step {self.global_step}: LOSS "
                f"{bad or ['total only']} -- batch skipped "
                f"({self._skipped_loss} loss, {self._skipped_grad} grad)")
            self.log("train/skipped_batches", float(self._skipped))
            self.log("train/skipped_loss", float(self._skipped_loss))
            return None

        self.log(f"{stage}/total_loss", total, prog_bar=True)
        for k, v in losses.items():
            self.log(f"{stage}/{k}_loss", v)
        if stage == "val":
            self._accumulate_metrics(out, batch, (h, w))
            if cfg.predict_keypoints and "keypoints" in batch:
                self._accumulate_keypoints(out, batch, (h, w))
        return total

    def on_before_optimizer_step(self, optimizer) -> None:
        """Log the pre-clip gradient norm, and refuse a non-finite step.

        The norm is logged because without it a spike is invisible: gradient
        clipping hides the magnitude, and the loss curve stays flat right up to
        the step that destroys the weights.

        The magnitude is logged on the SKIPPED steps too, as
        train/skipped_grad_finite_norm. Logging it only on steps that pass the
        check would make "were the failing steps spiking?" unanswerable --
        selection on the outcome, which is how a gradient explosion would hide.
        The norm over the still-finite parameters says whether the rest of the
        network was blowing up alongside whatever went non-finite.

        Per-parameter norms and maxima are reduced on device and copied to the
        host once, as one small tensor: a sync per parameter would stall the
        GPU hundreds of times a step.
        """
        named = [(name, p.grad.detach()) for name, p in self.named_parameters()
                 if p.grad is not None]
        if not named:
            return
        grads = [g for _, g in named]
        norms = torch.stack([n.float() for n in torch._foreach_norm(grads)])
        maxes = torch.stack([g.abs().amax().float() for g in grads])
        norms, maxes = torch.stack([norms, maxes]).cpu()
        # A parameter's norm is non-finite exactly when one of its entries is.
        bad = ~torch.isfinite(norms)
        good = ~bad
        total = float(norms[good].square().sum()) ** 0.5
        finite_max = float(maxes[good].max()) if bool(good.any()) else 0.0
        if not bool(bad.any()):
            self.log("train/grad_norm", total)
            return

        # Named, not just counted: without the parameter name there is no way
        # to locate the origin. Every one is recorded, because the FIRST
        # non-finite parameter in iteration order is not necessarily where it
        # originated.
        names = [name for (name, _), b in zip(named, bad.tolist()) if b]
        for name in names:
            self._nonfinite_params[name] = self._nonfinite_params.get(name, 0) + 1
        self._skipped += 1
        self._skipped_grad += 1
        rank_zero_info(
            f"[nonfinite] step {self.global_step}: GRAD in {len(names)} "
            f"param(s), first {names[0]!r} -- batch skipped "
            f"({self._skipped_loss} loss, {self._skipped_grad} grad)")
        self.log("train/skipped_batches", float(self._skipped))
        self.log("train/skipped_grad", float(self._skipped_grad))
        # What the rest of the network was doing at the moment of failure.
        self.log("train/skipped_grad_finite_norm", total)
        self.log("train/skipped_grad_finite_max", finite_max)
        optimizer.zero_grad(set_to_none=True)

    # --------------------------------------------------------------- metrics

    SEP_BUCKETS = ((0.0, 30.0), (30.0, 60.0), (60.0, 120.0),
                   (120.0, float("inf")))

    @torch.no_grad()
    def _accumulate_metrics(self, out, batch, hw):
        """Detection metrics from the READOUT, not from the dense field.

        Scoring the field directly would flatter the model: it would measure
        per-cell regression and never exercise voting, which is where
        detections actually come from.

        Detections are matched one-to-one to ground-truth darts on the landing
        point within ``match_gate_px`` (:func:`match_points`). Every error is
        over matched pairs; a missed dart is counted against recall and charged
        the gate in the penalised error, never silently left out. Frames with
        no darts count their detections as false positives.
        """
        from darts_model.model.hough import detect
        h, w = hw
        cfg = self.config
        self._accumulate_centre_vs_distance(out, batch, hw)
        if "dart_ends" in batch:
            self._accumulate_cluster_cells(out, batch, hw)
        dec = decode_boxes(out, (h, w))
        fg_prob = out["fg_logits"].float().sigmoid()
        scale = float(batch["image"].shape[-1])
        gate = cfg.match_gate_px
        has_ends = "dart_ends" in batch

        # Every readout the field supports, scored on the same weights; the
        # first is the one the config ships, and the checkpoint monitor.
        def hough(by_embedding):
            return lambda b: detect({k: v[b] for k, v in dec.items()},
                                    fg_prob[b], cfg,
                                    claim_by_embedding=by_embedding)

        readouts = {}
        if cfg.instance_head:
            readouts["per_dart"] = self._per_dart_detections(out)
        if cfg.query_head:
            readouts["queries"] = self._per_dart_detections(out)
        if cfg.token_readout:
            readouts["tokens"] = self._per_dart_detections(out)
        modes = [cfg.claim_by_embedding, not cfg.claim_by_embedding]
        for by_embedding in modes[:2 if "embedding" in out else 1]:
            name = "embedding_claim" if by_embedding else "centre_claim"
            readouts[name] = hough(by_embedding)
        if cfg.predict_tip_heatmap and cfg.token_readout:
            # Stage 2's readout on the same frozen weights: Hough with the tip
            # peaks assigned, alongside plain Hough.
            readouts["centre_claim_assigned"] = self._snapped(
                readouts["centre_claim"], out, scale)
            self._accumulate_tip_peaks(out, batch, scale)
        elif cfg.predict_tip_heatmap:
            # The shipped readout, snapped or not as configured, and the other
            # choice alongside it.
            first = next(iter(readouts))
            plain = readouts.pop(first)
            if cfg.tip_pointer:
                # The pointer's choice; the estimate it chose from; and the
                # hand-set snap applied to that estimate, on the same weights.
                est = self._per_dart_detections(out, key="tip_estimate")
                ordered = {first: plain, f"{first}_estimate": est,
                           f"{first}_snapped": self._snapped(est, out, scale)}
            else:
                snapped = self._snapped(plain, out, scale)
                ordered = {first: snapped if cfg.tip_snap else plain,
                           f"{first}_{'unsnapped' if cfg.tip_snap else 'snapped'}":
                               plain if cfg.tip_snap else snapped}
            readouts = {**ordered, **readouts}
            self._accumulate_tip_peaks(out, batch, scale)
        primary, *others = readouts

        for b in range(fg_prob.shape[0]):
            sel = batch["dart_mask"][b]
            got = readouts[primary](b)
            self._n_det += len(got)
            n_gt = int(sel.sum())
            if n_gt == 0:
                self._n_empty += 1
                self._n_empty_det += len(got)
                continue
            self._n_gt += n_gt

            box = batch["dart_box"][b][sel].float()
            gt_box_tip = (box[:, 0:2] - box[:, 2:4] * box[:, 4:5]) * scale
            if has_ends:
                ends = batch["dart_ends"][b][sel].float() * scale
                gt_land, gt_flight = ends[:, 0:2], ends[:, 2:4]
            else:
                gt_land = gt_box_tip
                gt_flight = (box[:, 0:2] + box[:, 2:4] * box[:, 4:5]) * scale

            # Distance from each dart to its nearest neighbour, so errors can
            # be read against how tightly the darts are grouped.
            if n_gt > 1:
                d = torch.cdist(gt_land, gt_land)
                d.fill_diagonal_(float("inf"))
                sep = d.amin(dim=1)
            else:
                sep = gt_land.new_full((1,), float("inf"))

            for name in others:
                other = readouts[name](b)
                err = torch.full((n_gt,), float(gate))
                if other:
                    pl = torch.stack([g["tip"] for g in other]).float() * scale
                    gi, _, dist = match_points(gt_land, pl, gate)
                    err[gi] = dist
                self._land_err_alt.setdefault(name, []).append(err)

            land_err = torch.full((n_gt,), float("nan"))
            if got:
                pred_land = torch.stack([g["tip"] for g in got]).float() * scale
                gi, pi, dist = match_points(gt_land, pred_land, gate)
                self._n_matched += int(gi.numel())
                land_err[gi] = dist
                if gi.numel():
                    pf = torch.stack([g["flight"] for g in got]).float() * scale
                    pb = torch.stack([g["box_tip"] for g in got]).float() * scale
                    gi_d, pi_d = gi.to(pf.device), pi.to(pf.device)
                    self._flight_err.append(
                        (pf[pi_d] - gt_flight[gi_d]).norm(dim=-1).cpu())
                    # Box-derived tips only for darts with a usable box.
                    if "dart_box_mask" in batch:
                        ok = batch["dart_box_mask"][b][sel].to(pf.device)[gi_d]
                        gi_d, pi_d = gi_d[ok], pi_d[ok]
                    self._box_tip_err.append(
                        (pb[pi_d] - gt_box_tip[gi_d]).norm(dim=-1).cpu())
            self._land_err.append(land_err)
            self._sep.append(sep.cpu())

    def _pointer_loss(self, darts, batch, match, scale: float, stage: str):
        """Cross-entropy of each matched query's choice against the peak that
        detects its dart's tip (queries.pointer_targets), or keep when none
        does; in validation, also how the choices went."""
        from darts_model.model.queries import pointer_targets
        cfg = self.config
        f, q, d = match
        if not f.numel():
            return darts["pointer_logits"].sum() * 0.0
        ends = batch["dart_ends"][..., 0:2].float()
        target = pointer_targets(darts["peaks_xy"], darts["peaks_score"], ends,
                                 batch["dart_mask"], f, d,
                                 2.0 * cfg.tip_stride / scale)
        logits = darts["pointer_logits"][f, q]
        loss = F.cross_entropy(logits, target)
        if stage == "val":
            with torch.no_grad():
                pick = logits.argmax(-1)
                tips = ends[f]
                gap = torch.cdist(ends[f, d].unsqueeze(1), tips)[:, 0] * scale
                gap = torch.where(batch["dart_mask"][f], gap,
                                  torch.full_like(gap, float("inf")))
                gap.scatter_(1, d.view(-1, 1), float("inf"))
                clustered = gap.amin(-1) < self.CLUSTER_PX
                self._ptr["n"] += int(f.numel())
                self._ptr["correct"] += int((pick == target).sum())
                self._ptr["keep"] += int((pick == 0).sum())
                self._ptr["target_keep"] += int((target == 0).sum())
                self._ptr["n_cl"] += int(clustered.sum())
                self._ptr["correct_cl"] += int(((pick == target) & clustered).sum())
                # Two queries of one frame putting most weight on one peak.
                all_pick = darts["pointer_logits"].argmax(-1)       # (B, Q)
                conf = darts["score"] >= slot_floor(cfg)
                for b in range(all_pick.shape[0]):
                    chosen = all_pick[b][conf[b] & (all_pick[b] > 0)]
                    self._ptr["shared"] += int(chosen.numel()
                                               - chosen.unique().numel())
        return loss

    @torch.no_grad()
    def _snapped(self, readout, out, scale: float):
        """``readout`` with each detection's landing point snapped to the tip
        heatmap (:func:`tips.snap_to_tips`); counts the snaps."""
        from darts_model.model.tips import snap_to_tips, tip_peaks
        cfg = self.config
        xy, score = tip_peaks(out["tip_logits"], out["tip_cell_offset"],
                              cfg.tip_peak_count)
        cache = {}

        def frame(b):
            if b not in cache:
                dets = readout(b)
                if dets:
                    pts = torch.stack([d["tip"] for d in dets]).float()
                    if cfg.tip_assign == "hungarian":
                        from darts_model.model.tips import assign_tips
                        new, found = assign_tips(pts, xy[b], score[b], cfg,
                                                 scale)
                        new = new[None]
                    else:
                        dirs = torch.stack([d["direction"] for d in dets]).float()[None]
                        new, found = snap_to_tips(pts[None], dirs, xy[b:b + 1],
                                                  score[b:b + 1], cfg, scale)
                    dets = [{**d, "tip": new[0, i]} for i, d in enumerate(dets)]
                    self._n_snap += int(found.sum())
                    self._n_snap_tried += len(dets)
                cache[b] = dets
            return cache[b]
        return frame

    @torch.no_grad()
    def _accumulate_tip_peaks(self, out, batch, scale: float):
        """How well the tip heatmap alone places the true landing points:
        for each dart, the nearest peak clearing ``tip_snap_score`` within
        ``tip_snap_px``, if any."""
        from darts_model.model.tips import tip_peaks
        cfg = self.config
        xy, score = tip_peaks(out["tip_logits"], out["tip_cell_offset"],
                              cfg.tip_peak_count)
        gt = batch["dart_ends"][..., 0:2].float()
        d = torch.cdist(gt, xy) * scale                         # (B, D, P)
        d = torch.where(score.unsqueeze(1) >= cfg.tip_snap_score, d,
                        torch.full_like(d, float("inf")))
        near = d.amin(-1)[batch["dart_mask"]]
        self._tip_peak_n += int(near.numel())
        hit = near <= cfg.tip_snap_px
        self._tip_peak_err.append(near[hit].cpu())

    @torch.no_grad()
    def _per_dart_detections(self, out, key: str = "tip"):
        """The per-dart readout as Hough-style detection lists, one per frame,
        with the landing point taken from ``key``.

        A slot is a dart when its score clears :func:`slot_floor`;
        detections come strongest first.
        """
        cfg = self.config
        r = self.model.read_darts(out)
        floor = slot_floor(cfg)
        box_tip = r["centre"] - r["direction"] * r["extent"][..., :1]
        frames = []
        for b in range(r["score"].shape[0]):
            order = torch.argsort(r["score"][b], descending=True).tolist()
            frames.append([
                {"tip": r[key][b, s], "flight": r["flight"][b, s],
                 "box_tip": box_tip[b, s], "direction": r["direction"][b, s],
                 "score": float(r["score"][b, s])}
                for s in order if float(r["score"][b, s]) >= floor])
        return lambda b: frames[b]

    @torch.no_grad()
    def _accumulate_centre_vs_distance(self, out, batch, hw):
        """How badly a cell locates the box centre, against how far away it is.

        A diagnostic for receptive field: darts project to 18-268 px, so a long
        one spans ~33 cells at stride 8 while the head's two 3x3 convs see ~5.
        If that bound bites, a cell far from its box centre cannot know where
        the centre is, and the error grows with distance. If the error is flat
        in distance, receptive field is not the limit and wider-context blocks
        (SPPF/C2PSA) would buy nothing.

        Measured per foreground cell, in pixels, from the dense field rather
        than the readout: voting averages the per-cell errors away, which is
        precisely what we need to see here.
        """
        h, w = hw
        fg = batch["foreground"]
        keep = fg > 0
        if not bool(keep.any()):
            return
        tgt = self._targets(batch, (h, w))
        centres = cell_centres(h, w, fg.device)[None]

        def pick(t):
            return t.permute(0, 2, 3, 1)[keep]

        scale = float(batch["image"].shape[-1])
        tc = pick(tgt[:, 0:2])
        pc = pick(centres + out["centre_offset"])
        # centres is (1, 2, h, w) and only broadcasts when added to something;
        # picked on its own it would keep batch 1 against a batch-B mask.
        cc = pick(centres.expand(fg.shape[0], -1, -1, -1))
        self._dist.append((cc - tc).norm(dim=-1).mul(scale).cpu())
        self._derr.append((pc - tc).norm(dim=-1).mul(scale).cpu())

    #: A dart with another dart's landing point closer than this, in pixels,
    #: is "clustered" for the per-cell identity metrics.
    CLUSTER_PX = 60.0

    @torch.no_grad()
    def _accumulate_cluster_cells(self, out, batch, hw):
        """How often a clustered dart's cells describe a neighbouring dart.

        Over the predicted-foreground cells of darts within ``CLUSTER_PX`` of
        another, counts the cells whose predicted landing point lies nearer a
        neighbour's landing point than their own dart's -- a cell describing
        the wrong dart. With an embedding, also the cells whose embedding lies
        nearer a neighbour's mean embedding than their own dart's: whether the
        embedding separates the darts the regressions confuse.

        From the dense field, not the readout: it measures what the cells say,
        before grouping decides which of them are heard.
        """
        h, w = hw
        cfg = self.config
        inst = batch["instance"]
        scale = float(batch["image"].shape[-1])
        cell = (out["fg_logits"].float().sigmoid() >= cfg.fg_threshold) & (inst > 0)
        if "tip_offset" not in out or not bool(cell.any()):
            return
        centres = cell_centres(h, w, inst.device)[None]
        tip = (centres + out["tip_offset"].float()).permute(0, 2, 3, 1)  # (B,h,w,2)
        land = batch["dart_ends"][..., 0:2].float()                  # (B, S, 2)
        valid = batch["dart_mask"]
        n_slots = land.shape[1]
        eye = torch.eye(n_slots, dtype=torch.bool, device=land.device)
        dd = torch.cdist(land, land) * scale
        dd = dd.masked_fill(eye | ~valid[:, None, :], float("inf"))
        clustered = (dd.amin(-1) < self.CLUSTER_PX) & valid          # (B, S)

        slot = (inst - 1).clamp(min=0)
        sel = cell & clustered.gather(1, slot.view(slot.shape[0], -1)).view_as(inst)
        if not bool(sel.any()):
            return
        bi = torch.nonzero(sel, as_tuple=True)[0]
        own = slot[sel]
        other_ok = valid[bi] & ~eye[own]                             # (N, S)

        def wrong(d):
            """d (N, S): distance to each slot's reference."""
            d_own = d.gather(1, own[:, None])[:, 0]
            d_other = d.masked_fill(~other_ok, float("inf")).amin(1)
            return int((d_other < d_own).sum())

        self._cl_cells += int(sel.sum())
        self._cl_wrong_tip += wrong((tip[sel][:, None] - land[bi]).norm(dim=-1))

        if "embedding" in out:
            emb = out["embedding"].float()
            member = (inst.unsqueeze(1) == torch.arange(
                1, n_slots + 1, device=inst.device).view(1, -1, 1, 1)) & cell[:, None]
            member = member.float()
            mu = torch.einsum("behw,bshw->bse", emb, member) / member.sum(
                (2, 3)).clamp(min=1.0).unsqueeze(-1)
            e = emb.permute(0, 2, 3, 1)[sel]                         # (N, E)
            self._cl_wrong_emb += wrong((e[:, None] - mu[bi]).norm(dim=-1))

    @torch.no_grad()
    def _accumulate_keypoints(self, out, batch, hw):
        """Pixel error of the 40 board corners, read out as the head will be.

        Argmax per channel plus the predicted sub-cell offset -- not a
        soft-argmax over the whole map, which would flatter the model by
        letting a broad, badly-peaked heatmap average its way to the right
        answer.
        """
        mask = batch["keypoint_mask"]
        if not bool(mask.any()):
            return
        h, w = hw
        logits = out["kp_logits"]
        b, k = logits.shape[:2]
        flat = logits.flatten(2).argmax(dim=2)              # (B, K)
        iy, ix = flat // w, flat % w
        off = out["kp_offset"].float()
        bi = torch.arange(b, device=logits.device).view(b, 1).expand(b, k)
        px = (ix.float() + 0.5 + off[bi, 0, iy, ix]) / w
        py = (iy.float() + 0.5 + off[bi, 1, iy, ix]) / h
        tgt = batch["keypoints"]
        scale = float(batch["image"].shape[-1])
        err = torch.stack([px - tgt[..., 0], py - tgt[..., 1]], dim=-1)
        self._kp_err.append((err.norm(dim=-1) * scale)[mask].cpu())

    def on_validation_epoch_start(self) -> None:
        self._n_gt = self._n_det = self._n_matched = 0
        self._n_empty = self._n_empty_det = 0
        self._land_err: list[torch.Tensor] = []   # per GT dart, NaN = missed
        self._sep: list[torch.Tensor] = []        # per GT dart
        self._flight_err: list[torch.Tensor] = []  # per matched dart
        self._box_tip_err: list[torch.Tensor] = []  # per matched dart
        self._kp_err: list[torch.Tensor] = []
        #: Per readout other than the monitored one: per GT dart, gate = missed.
        self._land_err_alt: dict[str, list[torch.Tensor]] = {}
        #: Per-dart readout from ground-truth seeds, per dart.
        self._gt_seed_err: list[torch.Tensor] = []
        #: Tip heatmap: darts with a peak near them, and that peak's error.
        self._tip_peak_n = 0
        self._tip_peak_err: list[torch.Tensor] = []
        #: Snaps made by the snapped readout, out of detections offered.
        self._n_snap = self._n_snap_tried = 0
        #: Peak pointer: matched queries, choices of the detecting peak, of
        #: keep, targets that were keep, the same for clustered darts, and
        #: confident queries sharing a peak.
        self._ptr = dict.fromkeys(("n", "correct", "keep", "target_keep",
                                   "n_cl", "correct_cl", "shared"), 0)
        self._cl_cells = self._cl_wrong_tip = self._cl_wrong_emb = 0
        self._dist: list[torch.Tensor] = []
        self._derr: list[torch.Tensor] = []

    def on_validation_epoch_end(self) -> None:
        gate = self.config.match_gate_px
        land = torch.cat(self._land_err) if self._land_err else torch.zeros(0)
        hit = ~torch.isnan(land)

        # The checkpoint monitor, logged unconditionally. Mean over EVERY
        # ground-truth dart of its matched landing error, or the gate if it was
        # missed, so a model cannot improve it by failing to detect the hard
        # darts. With no darts in the epoch at all there is nothing to rank,
        # and the gate is the neutral worst case.
        penalised = (torch.where(hit, land, torch.full_like(land, gate)).mean()
                     if land.numel() else torch.tensor(gate))
        self.log(self.MONITOR, penalised, prog_bar=True)

        if self._n_gt:
            self.log("val/recall", self._n_matched / self._n_gt, prog_bar=True)
            self.log("val/detections_per_dart", self._n_det / self._n_gt)
        if self._n_det:
            self.log("val/precision", self._n_matched / self._n_det)
        if self._n_empty:
            # Separate from the monitor: false positives on an empty board are
            # a different failure from missing or mislocating a dart, and
            # folding them in would make the two trade against each other.
            self.log("val/fp_per_empty_frame", self._n_empty_det / self._n_empty)

        if bool(hit.any()):
            # Scoring reads the landing point: the renderer's own entry point,
            # which for a dart pointing near the camera projects inside the
            # silhouette, up to 27px from any end of the box.
            m = land[hit]
            self.log("val/landing_px_error", m.mean(), prog_bar=True)
            self.log("val/landing_px_p90", m.quantile(0.9))
            self.log("val/landing_px_max", m.max())
        if self._tip_peak_n:
            found = torch.cat(self._tip_peak_err)
            self.log("val/tip_peak_found", found.numel() / self._tip_peak_n)
            if found.numel():
                # The heatmap's own precision, on the darts it found.
                self.log("val/tip_peak_px", found.mean())
        if self._n_snap_tried:
            self.log("val/tip_snap_rate", self._n_snap / self._n_snap_tried)
        if self._ptr["n"]:
            p = self._ptr
            self.log("val/pointer_correct", p["correct"] / p["n"])
            self.log("val/pointer_keep", p["keep"] / p["n"])
            self.log("val/pointer_target_keep", p["target_keep"] / p["n"])
            self.log("val/pointer_shared_peaks", float(p["shared"]))
            if p["n_cl"]:
                self.log("val/pointer_correct_clustered",
                         p["correct_cl"] / p["n_cl"])
        for name, errs in self._land_err_alt.items():
            self.log(f"val/landing_px_error_penalised_{name}",
                     torch.cat(errs).mean())
        if self._gt_seed_err:
            # The per-dart head with seeds it cannot get wrong, or the queries
            # as matched in training: against the monitor, how much of the
            # error is in deciding which slots are darts.
            gt_seed = torch.cat(self._gt_seed_err)
            if gt_seed.numel():
                self.log("val/landing_px_error_gt_seeds", gt_seed.mean())
        if self._cl_cells:
            # The share of a clustered dart's cells that describe a neighbour:
            # 16.5% before the embedding existed, against 0.5% for isolated
            # darts.
            self.log("val/cluster_cells_wrong_dart",
                     self._cl_wrong_tip / self._cl_cells)
            if self.config.embed_dim > 0:
                self.log("val/cluster_cells_wrong_embedding",
                         self._cl_wrong_emb / self._cl_cells)
        if self._flight_err:
            self.log("val/flight_px_error", torch.cat(self._flight_err).mean())
        if self._box_tip_err:
            # The box's own end against the ground-truth box's end: how well the
            # box is localised, independent of the landing-point head.
            self.log("val/tip_px_error", torch.cat(self._box_tip_err).mean())

        if land.numel():
            sep = torch.cat(self._sep)
            for lo, hi in self.SEP_BUCKETS:
                name = f"{int(lo)}_{int(hi) if hi != float('inf') else 'inf'}"
                sel = (sep >= lo) & (sep < hi)
                self.log(f"val/n_sep_{name}", sel.float().sum())
                if bool(sel.any()):
                    self.log(f"val/recall_sep_{name}", hit[sel].float().mean())
                if bool((sel & hit).any()):
                    self.log(f"val/landing_px_sep_{name}",
                             land[sel & hit].mean())

        # Centre error bucketed by how far the voting cell sits from the centre
        # it is voting for. Rising with distance => receptive field is the
        # limit; flat => it is not, and SPPF/C2PSA are not worth building.
        if self._dist:
            dist = torch.cat(self._dist)
            derr = torch.cat(self._derr)
            edges = [0.0, 10.0, 25.0, 50.0, 100.0, float("inf")]
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = (dist >= lo) & (dist < hi)
                hi_name = "inf" if hi == float("inf") else int(hi)
                name = f"{int(lo)}_{hi_name}px"
                self.log(f"val/n_cells_dist_{name}", sel.float().sum())
                if bool(sel.any()):
                    self.log(f"val/centre_px_dist_{name}", derr[sel].mean())

        if self._kp_err:
            kp = torch.cat(self._kp_err)
            self.log("val/kp_px_error", kp.mean(), prog_bar=True)
            # The mean hides the failure that matters. A board pose fitted to
            # forty corners survives a few loose ones and collapses on a
            # confident outlier, so the tail is the number to watch.
            self.log("val/kp_px_p90", kp.quantile(0.9))
            self.log("val/kp_px_max", kp.max())

        self.on_validation_epoch_start()

    # ----------------------------------------------------------------- hooks

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    #: The per-dart readout's own weights: their own optimizer group, at
    #: ``readout_lr_factor``.
    READOUT_PREFIXES = ("token_proj.", "fine_proj.", "instance_head.",
                        "query_head.", "tip_", "token_readout.")

    #: What trains under ``freeze_dense``: the readouts' own weights. The tip
    #: head is stage 2's and stays frozen with the rest.
    TRAINED_WHEN_FROZEN = ("token_proj.", "fine_proj.", "instance_head.",
                           "query_head.", "token_readout.")

    #: Heads that may be absent from an ``init_weights`` checkpoint.
    NEW_HEAD_PREFIXES = ("embed_head.", "token_proj.", "fine_proj.",
                         "instance_head.", "query_head.", "tip_",
                         "token_readout.")

    def _load_full(self, path: str) -> None:
        """Every model weight from a previous run; no optimizer state.

        Loud about what it found. Silently loading nothing looks exactly like
        a successful fine-tune that simply does not improve.
        """
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = {k.split("model.", 1)[1]: v
              for k, v in ck["state_dict"].items() if k.startswith("model.")}
        if not sd:
            raise RuntimeError(
                f"init_weights: no 'model.' tensors in {path} -- refusing to "
                f"start a fine-tune from randomly initialised weights")
        result = self.model.load_state_dict(sd, strict=False)
        print(f"init_weights: loaded {len(sd)} tensors from {path} "
              f"(epoch {ck.get('epoch')})", flush=True)
        # A head the checkpoint predates starts from its own initialisation;
        # that is the point of adding one to a trained model. Anything else
        # missing is an architecture mismatch.
        fresh = [k for k in result.missing_keys
                 if k.startswith(self.NEW_HEAD_PREFIXES)]
        missing = [k for k in result.missing_keys if k not in fresh]
        if fresh:
            print(f"  {len(fresh)} tensors of heads the checkpoint predates "
                  f"start from initialisation: {fresh}", flush=True)
        if missing:
            raise RuntimeError(
                f"init_weights: {len(missing)} tensors NOT in the "
                f"checkpoint, e.g. {missing[:4]}. The architecture "
                f"does not match; fine-tuning would mix trained and random "
                f"weights.")
        if result.unexpected_keys:
            print(f"  {len(result.unexpected_keys)} unused checkpoint tensors "
                  f"(ignored): {result.unexpected_keys[:3]}", flush=True)

    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int) -> dict:
        """Normalisation always; sensor noise only while training.

        On the module rather than a DataModule because this trainer passes
        DataLoaders directly, so a DataModule hook would never run.
        """
        images = batch["image"]
        # The buffers do not follow the module automatically on the first
        # batch of a resumed run, so they are moved to meet the data.
        if self.gpu_augment.mean.device != images.device:
            self.gpu_augment.to(images.device)
        tr = getattr(self, "_trainer", None)
        batch["image"] = self.gpu_augment(
            images, training=bool(tr is not None and tr.training))
        return batch

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["backbone_unfreeze_step"] = self._unfreeze_step

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self._unfreeze_step = checkpoint.get("backbone_unfreeze_step")

    #: AdamW beta1 after warmup.
    BETA1 = 0.9

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Warm beta1 alongside the LR, as YOLO warms SGD momentum, and hold it
        at exactly ``BETA1`` afterwards."""
        nw = self._warmup_steps
        f = min(self.global_step / nw, 1.0) if nw else 1.0
        b1 = self.config.warmup_beta1 + (self.BETA1 - self.config.warmup_beta1) * f
        for g in optimizer.param_groups:
            g["betas"] = (self.BETA1 if f >= 1.0 else b1, g["betas"][1])
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def _backbone_lr_multiplier(self, step: int) -> float:
        """The backbone group's re-warmup, applied on top of the schedule.

        1 when the backbone was never frozen. While frozen it is 0 (nothing
        is updated anyway); from the unfreeze it ramps linearly to 1 over
        ``backbone_rewarmup_epochs``.
        """
        if self.config.backbone_freeze_epochs <= 0:
            return 1.0
        if self._unfreeze_step is None:
            return 0.0
        if self._rewarm_steps <= 0:
            return 1.0
        return min(max((step - self._unfreeze_step) / self._rewarm_steps, 0.0),
                   1.0)

    def on_train_epoch_start(self) -> None:
        cfg = self.config
        if cfg.freeze_dense:
            # Frozen parts in eval mode, so the backbone's drop-path is off
            # and the readout reads the model it will be shipped with.
            for name, child in self.model.named_children():
                if not (name + ".").startswith(self.TRAINED_WHEN_FROZEN):
                    child.eval()
        detached = self.current_epoch < cfg.readout_warmup_epochs
        if detached != self.model.readout_detached:
            print(f"[epoch {self.current_epoch}] per-dart readout reads the "
                  f"dense field {'detached' if detached else 'attached'}",
                  flush=True)
        self.model.readout_detached = detached
        if not cfg.backbone_freeze_epochs:
            return
        # `>=`, not `==`: requires_grad is not in state_dict, so a resumed run
        # rebuilds frozen and an equality test would never fire again.
        if (self.current_epoch >= cfg.backbone_freeze_epochs
                and self.model.backbone.is_frozen):
            self.model.backbone.unfreeze()
            resumed = self._unfreeze_step is not None
            if not resumed:
                self._unfreeze_step = self.global_step
            step = self.global_step
            end = self._unfreeze_step + self._rewarm_steps
            # The LR the next backbone step actually takes, read from the
            # optimizer rather than recomputed, and where the re-warmup ends.
            tr = getattr(self, "_trainer", None)
            now = next((g["lr"] for opt in (tr.optimizers if tr else [])
                        for g in opt.param_groups
                        if g.get("name") == "backbone"), float("nan"))
            lam = self._lr_lambda or (lambda _s: 1.0)
            peak = cfg.lr * cfg.backbone_lr_factor * lam(end)
            print(f"[epoch {self.current_epoch}] backbone unfrozen"
                  f"{' (resumed)' if resumed else ''} at step {step}: lr "
                  f"{now:.2e}, re-warming to {peak:.2e} at step {end}",
                  flush=True)

    def configure_optimizers(self):
        cfg = self.config
        # No weight decay on 1-D parameters: norm weights and biases, and
        # ConvNeXt's layer-scale gamma. Decaying a norm's scale or a layer
        # scale toward zero shrinks the block it gates, which is a change of
        # architecture rather than a regulariser.
        groups: dict[str, list] = {"backbone": [], "backbone_no_decay": [],
                                   "head": [], "head_no_decay": [],
                                   "readout": [], "readout_no_decay": []}
        for name, p in self.model.named_parameters():
            if not p.requires_grad and self.config.freeze_dense:
                continue
            part = ("backbone" if name.startswith("backbone.")
                    else "readout" if name.startswith(self.READOUT_PREFIXES)
                    else "head")
            groups[part + ("_no_decay" if p.ndim <= 1 else "")].append(p)
        param_groups = []
        for gname, params in groups.items():
            if not params:
                continue
            is_bb = gname.startswith("backbone")
            factor = (cfg.backbone_lr_factor if is_bb
                      else cfg.readout_lr_factor if gname.startswith("readout")
                      else 1.0)
            param_groups.append({
                "name": gname, "params": params,
                "lr": cfg.lr * factor,
                "weight_decay": 0.0 if gname.endswith("no_decay")
                else cfg.weight_decay,
            })
        opt = torch.optim.AdamW(param_groups, betas=(cfg.warmup_beta1, 0.999))

        # Per optimizer step, not per epoch.
        #
        # From the config, NOT from `estimated_stepping_batches`: the dataset
        # is an IterableDataset with no `__len__`, so Lightning reports 1.
        inferred = cfg.steps_per_epoch <= 0
        per_epoch = cfg.steps_per_epoch
        if inferred:
            per_epoch = max(
                int(self.trainer.estimated_stepping_batches)
                // max(cfg.max_epochs, 1), 1)
        total = max(per_epoch * cfg.max_epochs, 1)
        # Only when the number was GUESSED and the guess is degenerate. An
        # explicit 1 is somebody's smoke test and is theirs to make; an
        # inferred 1 is Lightning reporting on a dataset it cannot measure.
        # Loud, because the silent version trains a whole run at the cosine
        # floor and every metric merely looks disappointing.
        if inferred and per_epoch <= 1:
            raise RuntimeError(
                "cannot determine the LR schedule length: steps_per_epoch was "
                f"not set and the trainer estimates {per_epoch} step(s) per "
                "epoch, which for an IterableDataset means it could not "
                "measure the dataset at all. Set config.steps_per_epoch. "
                "Refusing to train a full run at the cosine floor.")
        # The floor of 100 is YOLO's, and it is what keeps a short smoke run
        # from having effectively no warmup at all.
        nw = max(int(round(cfg.warmup_epochs * per_epoch)), 100)
        nw = min(nw, max(total - 1, 1))
        self._warmup_steps = nw
        self._rewarm_steps = int(round(cfg.backbone_rewarmup_epochs * per_epoch))
        self._embed_ramp_steps = (int(round(cfg.embed_warmup_epochs * per_epoch))
                                  if cfg.embed_dim > 0 else 0)
        lrf = cfg.final_lr_factor

        def lam(step: int) -> float:
            if step < nw:
                # From ~0, not from 0.01x: with the backbone unfrozen from the
                # first step there is no frozen phase to absorb a large initial
                # update, so the ramp has to start at the bottom.
                return (step + 1) / nw
            p = (step - nw) / max(total - nw, 1)
            return lrf + (1.0 - lrf) * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))

        self._lr_lambda = lam

        def backbone_lam(step: int) -> float:
            return lam(step) * self._backbone_lr_multiplier(step)

        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, [backbone_lam if g["name"].startswith("backbone") else lam
                  for g in param_groups])
        print(f"[optim] warmup {nw} steps (~{cfg.warmup_epochs} epochs of "
              f"{per_epoch}), cosine to {cfg.lr * lrf:.2e} over {total}"
              + (f"; backbone frozen {cfg.backbone_freeze_epochs} epochs, then "
                 f"re-warmed over {self._rewarm_steps} steps"
                 if cfg.backbone_freeze_epochs > 0 else ""),
              flush=True)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}
