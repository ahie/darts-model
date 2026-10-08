"""Per-dart readout by learned queries over the foreground cells.

A fixed set of plain learned queries -- one per dart a turn can hold --
cross-attends to the kept cells, with the foreground score as an attention
bias, and each query reads out one dart: a confidence, the landing point, the
flight point and the oriented box. Training matches queries to darts with the
Hungarian algorithm on landing distance plus confidence.

What makes this a dart readout rather than a generic set predictor is where
the points come from. A query does not regress coordinates; it points. Its
landing point is

    sum_i attn_i * (pos_i + tip_offset_i + h(token_i, query))

an attention-weighted mean over cells of each cell's position, the cell's
own dense estimate, and a correction conditioned on the query. The
correction is the point of the design. A cell's own estimate answers "where
is the tip of the dart I think I am on", and next to a crossing that can be
the neighbour's tip, though the cell sits on this dart's -- the error the
dense readout makes. Conditioned on the query, the cell instead answers
"where is the tip of THIS query's dart, seen from here": the query decides
which dart, the cell contributes its local evidence and its exact position.
The correction starts at zero, so a fresh head reads out the attention-weighted
mean of the dense estimates.

Why the old failure modes of query detectors should not return:

* The trunk is converged and the foreground head trained, so queries learn on
  stable features rather than chasing a moving backbone, which is where query
  roles used to drift.
* Each query has its own output branch. Shared branches made queries
  symmetric, the assignment noisy and the confidence stick at its prior.
* The matching cost includes confidence; without it the assignment was
  unstable.
* Plain learned queries, no anchors or reference points: there is no spatial
  prior for where a dart lands, and anchored variants lost every time.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from darts_model.model.instance import (
    DART_SCALE, DELTA_SCALE, MEMBERSHIP_FLOOR, _Block,
)

#: Octaves of the absolute Fourier position encoding over the unit image. The
#: finest has a period of 1/64 of the image, 16px at 1024: attention needs
#: positions to relate cells, while precision comes from the pointer, which
#: uses each cell's exact position.
POSITION_OCTAVES = 8


def _fourier(x: torch.Tensor, octaves: int) -> torch.Tensor:
    f = (2.0 ** torch.arange(octaves, device=x.device, dtype=x.dtype)) * math.pi
    a = (x.unsqueeze(-1) * f).flatten(-2)
    return torch.cat((a.sin(), a.cos()), dim=-1)


class QueryHead(nn.Module):
    """Kept cells -> ``query_count`` darts, each with a confidence."""

    def __init__(self, cfg) -> None:
        super().__init__()
        d, n = cfg.instance_dim, cfg.query_count
        from darts_model.model.detector import token_feature_dim
        self.cfg = cfg
        self.feat = nn.Linear(token_feature_dim(cfg), d)
        self.pos = nn.Linear(2 * 2 * POSITION_OCTAVES, d)
        # Relative tip, flight and centre, and the direction.
        self.geom = nn.Linear(8, d)
        self.encode = _Block(d, cfg.instance_heads)
        self.queries = nn.Parameter(torch.randn(n, d))
        self.self_attn = nn.ModuleList(_Block(d, cfg.instance_heads)
                                       for _ in range(cfg.instance_layers))
        self.cross_attn = nn.ModuleList(_Block(d, cfg.instance_heads)
                                        for _ in range(cfg.instance_layers))
        self.norm = nn.LayerNorm(d)
        self.point_q = nn.Linear(d, d)
        self.point_k = nn.Linear(d, d)
        # One output branch per query.
        self.conf = nn.ModuleList(nn.Linear(d, 1) for _ in range(n))
        self.correct = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 6))
            for _ in range(n))
        for c in self.correct:
            nn.init.zeros_(c[-1].weight)
            nn.init.zeros_(c[-1].bias)
        self.peak_pointer = (PeakPointer(cfg)
                             if getattr(cfg, "tip_pointer", False) else None)

    def forward(self, cand: dict, peaks: dict | None = None) -> dict:
        """``cand`` from :func:`instance.gather_candidates`, sorted by
        foreground score. Returns (B, Q, ...) planes: ``logit`` and ``score``
        (the confidence), ``tip`` and ``flight`` (the readout), ``base_tip``
        (the attention-weighted mean of the dense estimates the correction
        starts from), the box ``centre``, ``direction`` and ``extent``, and
        ``attn`` (B, Q, M), the pointer weights.

        With ``peaks`` (see :class:`PeakPointer`) and a pointer, ``tip`` is
        the pointer's choice among the estimate and the peaks;
        ``tip_estimate`` is the estimate and ``pointer_logits`` the choice.

        Runs in float32 whatever the autocast state: it averages positions
        whose bf16 step is ~0.5px.
        """
        dev = cand["score"].device.type
        if torch.is_autocast_enabled(dev):
            with torch.autocast(dev, enabled=False):
                return self._forward(
                    {k: v.float() for k, v in cand.items()},
                    None if peaks is None else
                    {k: v.float() for k, v in peaks.items()})
        return self._forward(cand, peaks)

    def _forward(self, cand: dict, peaks: dict | None = None) -> dict:
        cfg = self.cfg
        b = int(cand["score"].shape[0])
        m = min(cfg.query_tokens, int(cand["score"].shape[1]))
        n = cfg.query_count
        if "context" in cand:
            # The cells around the silhouette too, so re-ranked by the
            # max-pooled score before the top m are taken.
            _, order = torch.topk(cand["context"], m, dim=1, sorted=True)
            c = {k: torch.gather(v, 1, order.view(b, m, *([1] * (v.dim() - 2)))
                                 .expand(b, m, *v.shape[2:]))
                 for k, v in cand.items()}
        else:
            c = {k: v[:, :m] for k, v in cand.items()}   # top-m by foreground

        # Foreground as the mask, soft: a bias of log(fg) on every attention
        # over the cells -- of the context score where there is one, so the
        # queries see past the silhouette. It gates what the queries see; it
        # is not trained by them. The pointer always takes the silhouette's
        # own score: coordinates come only from dart cells.
        point_bias = c["score"].detach().clamp(min=MEMBERSHIP_FLOOR).log()
        bias = (c["context"].detach().clamp(min=MEMBERSHIP_FLOOR).log()
                if "context" in c else point_bias)                  # (B, M)

        pos = c["pos"]
        rel = lambda p: (p - pos) / DART_SCALE  # noqa: E731
        tokens = (self.feat(c["feature"])
                  + self.pos(_fourier(pos, POSITION_OCTAVES))
                  + self.geom(torch.cat((rel(c["tip"]), rel(c["flight"]),
                                         rel(c["centre"]), c["direction"]),
                                        -1)))
        x = self.encode(tokens, tokens, bias)

        q = self.queries.unsqueeze(0).expand(b, n, -1)
        none = bias.new_zeros(b, n)
        for sa, ca in zip(self.self_attn, self.cross_attn):
            # Among the queries first: how two queries learn not to describe
            # the same dart.
            q = sa(q, q, none)
            q = ca(q, x, bias)
        q = self.norm(q)

        d = int(q.shape[-1])
        logits = (self.point_q(q) @ self.point_k(x).transpose(1, 2)
                  / math.sqrt(d) + point_bias.unsqueeze(1))          # (B, Q, M)
        attn = logits.softmax(-1)

        pair = torch.cat((x.unsqueeze(1).expand(b, n, m, d),
                          q.unsqueeze(2).expand(b, n, m, d)), -1)
        corr = torch.stack([self.correct[j](pair[:, j]) for j in range(n)],
                           1) * DELTA_SCALE                           # (B,Q,M,6)

        def point(base: torch.Tensor, k: int) -> torch.Tensor:
            v = base.unsqueeze(1) + corr[..., k:k + 2]
            return (attn.unsqueeze(-1) * v).sum(2)

        def mean(v: torch.Tensor) -> torch.Tensor:
            return (attn.unsqueeze(-1) * v.unsqueeze(1)).sum(2)

        dsum = mean(c["direction"])
        direction = dsum / dsum.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        logit = torch.cat([self.conf[j](q[:, j]) for j in range(n)], -1)
        tip = point(c["tip"], 0)
        extra = {}
        if self.peak_pointer is not None and peaks is not None:
            landing, plog = self.peak_pointer(q, tip, direction, peaks)
            extra = {"tip_estimate": tip, "pointer_logits": plog}
            tip = landing
        return {
            **extra,
            "logit": logit,
            "score": torch.sigmoid(logit),
            "tip": tip,
            "flight": point(c["flight"], 2),
            "centre": point(c["centre"], 4),
            "base_tip": mean(c["tip"]),
            "direction": direction,
            "extent": mean(c["extent"]),
            "attn": attn,
        }


#: Initial bias of the pointer's "keep my estimate" option. With the pair
#: scores starting at zero, a fresh pointer starts as the readout it extends:
#: e^10 leaves each of up to 16 peaks ~5e-5 of the weight, so even peaks of
#: other darts hundreds of pixels away pull the landing point well under a
#: pixel (e^6 still moved it ~5px). The choice trains regardless: the
#: cross-entropy's gradient on the target option is 1 - its weight.
KEEP_BIAS = 10.0
#: Finer scale for pair geometry, normalised: ~10px at 1024. Distances also go
#: in at DART_SCALE; the two together resolve both a dart's length and the
#: pixels the choice turns on.
NEAR_SCALE = 0.01


class PeakPointer(nn.Module):
    """Learned assignment of tip-heatmap peaks to queries.

    Each query chooses among keeping its own landing estimate and each of the
    tip head's peaks, by a softmax over a "keep" logit from the query and a
    logit per (query, peak) pair from the query, the peak's features and the
    peak's position relative to the query's estimate, along and across its
    axis. The landing point is the weighted mean of the estimate and the
    peaks' sub-pixel positions, so as the choice sharpens it becomes the
    chosen peak exactly.

    It replaces a hand-set snap -- nearest peak within 8px of the estimate
    and 4px of its axis -- with what the query knows of its whole dart and
    of the other queries: how far to reach, in which direction, and which of
    two close tips is its own. Fixed shapes, so it exports in-graph.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        d = cfg.instance_dim
        self.peak = nn.Linear(cfg.tip_head_width + 2 * 2 * POSITION_OCTAVES + 1, d)
        self.pair = nn.Sequential(nn.Linear(2 * d + 7, d), nn.GELU(),
                                  nn.Linear(d, 1))
        self.keep = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.zeros_(self.pair[-1].weight)
        nn.init.zeros_(self.pair[-1].bias)
        nn.init.zeros_(self.keep[-1].weight)
        nn.init.constant_(self.keep[-1].bias, KEEP_BIAS)

    def forward(self, q, est, axis, peaks: dict):
        """``q`` (B, Q, d), ``est`` and ``axis`` (B, Q, 2); ``peaks`` holds
        ``xy`` (B, P, 2), ``score`` (B, P) and ``feature`` (B, P, F). Returns
        ``(landing (B, Q, 2), logits (B, Q, 1 + P))``, option 0 being keep."""
        b, n = int(q.shape[0]), int(q.shape[1])
        p = int(peaks["score"].shape[1])
        log_s = peaks["score"].clamp(min=1e-6).log().unsqueeze(-1)
        tok = self.peak(torch.cat((peaks["feature"],
                                   _fourier(peaks["xy"], POSITION_OCTAVES),
                                   log_s), -1))                    # (B, P, d)
        delta = peaks["xy"].unsqueeze(1) - est.unsqueeze(2)       # (B, Q, P, 2)
        u = axis.unsqueeze(2)
        along = (delta * u).sum(-1)
        across = delta[..., 0] * u[..., 1] - delta[..., 1] * u[..., 0]
        dist = (delta.square().sum(-1) + 1e-12).sqrt()
        geom = torch.stack((along / DART_SCALE, across / DART_SCALE,
                            dist / DART_SCALE, along / NEAR_SCALE,
                            across / NEAR_SCALE, dist / NEAR_SCALE,
                            log_s.squeeze(-1).unsqueeze(1).expand(b, n, p)), -1)
        d = int(q.shape[-1])
        pair = torch.cat((q.unsqueeze(2).expand(b, n, p, d),
                          tok.unsqueeze(1).expand(b, n, p, d), geom), -1)
        logit_p = self.pair(pair)[..., 0]                         # (B, Q, P)
        logit_p = torch.where(peaks["score"].unsqueeze(1) > 0, logit_p,
                              torch.full_like(logit_p, -1e4))
        logits = torch.cat((self.keep(q), logit_p), -1)           # (B, Q, 1+P)
        w = logits.softmax(-1)
        landing = (w[..., :1] * est
                   + (w[..., 1:].unsqueeze(-1) * peaks["xy"].unsqueeze(1)).sum(2))
        return landing, logits


def pointer_targets(peaks_xy: torch.Tensor, peaks_score: torch.Tensor,
                    gt_tip: torch.Tensor, gt_ok: torch.Tensor, f, d,
                    radius: float) -> torch.Tensor:
    """Ground-truth choice for each matched (frame, dart): 1 + the index of
    the peak that detects that dart's tip, or 0 (keep) when none does.

    A peak detects a tip when it lies within ``radius`` (normalised) of it
    and nearer to it than to any other dart's tip in the frame. This only
    labels training data from the ground truth; nothing at inference uses
    it.
    """
    xy = peaks_xy[f]                                              # (N, P, 2)
    ok = peaks_score[f] > 0
    tips = gt_tip[f]                                              # (N, D, 2)
    dist_all = torch.cdist(xy, tips)                              # (N, P, D)
    dist_all = torch.where(gt_ok[f].unsqueeze(1), dist_all,
                           torch.full_like(dist_all, float("inf")))
    own = torch.gather(dist_all, 2, d.view(-1, 1, 1).expand(-1, xy.shape[1], 1))[..., 0]
    others = dist_all.scatter(2, d.view(-1, 1, 1).expand(-1, xy.shape[1], 1),
                              float("inf")).amin(-1)
    valid = ok & (own <= radius) & (own < others)
    own = torch.where(valid, own, torch.full_like(own, float("inf")))
    best = own.argmin(-1)
    return torch.where(valid.any(-1), best + 1, torch.zeros_like(best))


def match_queries(pred_tip: torch.Tensor, pred_score: torch.Tensor,
                  gt_tip: torch.Tensor, gt_ok: torch.Tensor,
                  cost_px: float, cost_conf: float, scale: float):
    """Hungarian assignment of queries to darts, per frame.

    Cost is ``cost_px`` per pixel of landing distance (L1) minus
    ``cost_conf`` times the query's confidence. Returns (frame, query, dart)
    index tensors over every matched pair.
    """
    from scipy.optimize import linear_sum_assignment

    tip = pred_tip.detach().float().cpu()
    score = pred_score.detach().float().cpu()
    gt = gt_tip.detach().float().cpu()
    ok = gt_ok.detach().cpu()
    fb, qb, db = [], [], []
    for f in range(tip.shape[0]):
        darts = torch.nonzero(ok[f]).flatten()
        if not darts.numel():
            continue
        l1 = (tip[f].unsqueeze(1) - gt[f, darts].unsqueeze(0)).abs().sum(-1)
        cost = cost_px * l1 * scale - cost_conf * score[f].unsqueeze(1)
        qi, di = linear_sum_assignment(cost.numpy())
        fb += [f] * len(qi)
        qb += qi.tolist()
        db += darts[di].tolist()
    t = lambda v: torch.as_tensor(np.asarray(v, dtype=np.int64),  # noqa: E731
                                  device=pred_tip.device)
    return t(fb), t(qb), t(db)


def query_losses(pred: dict, batch: dict, cfg, scale: float,
                 huber, return_match: bool = False):
    """The query head's losses, and the matched landing errors in pixels;
    with ``return_match``, also the (frame, query, dart) matching.

    Confidence is a BCE over every query: 1 for a query matched to a dart, 0
    otherwise. The matched queries' landing, flight and box centre are Huber
    in pixels; direction and log-extent are L1, on darts with a usable box.
    """
    ends = batch["dart_ends"].float()
    gt_ok = batch["dart_mask"]
    f, q, d = match_queries(pred["tip"], pred["score"], ends[..., 0:2], gt_ok,
                            cfg.query_cost_px, cfg.query_cost_conf, scale)
    target = torch.zeros_like(pred["logit"])
    target[f, q] = 1.0
    losses = {"q_conf": F.binary_cross_entropy_with_logits(pred["logit"],
                                                           target)}
    zero = pred["tip"].sum() * 0.0
    if not f.numel():
        for k in ("q_tip", "q_flight", "q_centre", "q_dir", "q_size"):
            losses[k] = zero
        if return_match:
            return losses, zero.new_zeros(0), (f, q, d)
        return losses, zero.new_zeros(0)

    def px(a, b):
        return ((a - b).square().sum(-1) + 1e-12).sqrt() * scale

    tip_px = px(pred["tip"][f, q], ends[f, d, 0:2])
    losses["q_tip"] = huber(tip_px, cfg.instance_huber_px).mean()
    losses["q_flight"] = huber(px(pred["flight"][f, q], ends[f, d, 2:4]),
                               cfg.instance_huber_px).mean()
    box = batch["dart_box"].float()[f, d]
    ok = (batch["dart_box_mask"][f, d] if "dart_box_mask" in batch
          else torch.ones_like(f, dtype=torch.bool))
    losses["q_centre"] = huber(px(pred["centre"][f, q], box[:, 0:2]),
                               cfg.instance_huber_px).mean()
    if bool(ok.any()):
        losses["q_dir"] = (pred["direction"][f, q][ok] - box[ok, 2:4]).abs().mean()
        losses["q_size"] = (pred["extent"][f, q][ok].clamp(min=1e-6).log()
                            - box[ok, 4:6].clamp(min=1e-6).log()).abs().mean()
    else:
        losses["q_dir"] = losses["q_size"] = zero
    if return_match:
        return losses, tip_px.detach(), (f, q, d)
    return losses, tip_px.detach()
