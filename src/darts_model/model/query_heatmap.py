"""Query heatmap readout: one heatmap per learned query over the stride-4 tip
grid, a confidence for each, Hungarian-matched to the darts.

The class-agnostic tip heatmap already finds tips to about a pixel; what it
cannot say is which peak belongs to which dart, nor where a hidden tip is. So
each query here draws its own heatmap over the same grid, from the same
features:

* Queries: plain learned queries -- no anchors or reference points -- that
  attend to each other and to the neck's coarsest level (stride 32 with the
  PAN, 32x32 at 1024) and nothing finer: which dart a query takes is the most
  global decision the readout makes, and the heatmaps carry the detail. The
  cross attention is what makes them depend on the frame: learned queries
  that only attended to each other would be the same for every image.
* Heatmaps: a small MLP over every (query, cell) pair answers "is this cell
  the tip of *my* dart". Each cell has an embedding of the tip head's
  features, the neck and a Fourier encoding of its own position, mixed over
  its 3x3 neighbourhood so it can express where in the cell a tip lies; the
  query shifts the MLP's hidden layer and supplies its output weights. The tip
  head's features say where *a* tip is but look alike at every dart's tip, and
  convolutional features carry no absolute position, so a plain dot product
  between a query and such cells cannot single one tip out; the position lets
  a query that has located its dart target that place, and the nonlinearity
  lets query and cell interact. The tip head's output logits are deliberately
  not added: they are the same for every query and far sharper than a
  query's own term, so every query's softmax would collapse onto the same
  strongest tip.
* Decoding: the argmax cell, refined two ways. ``tip`` is a soft-argmax over a
  small window around it, the sub-cell position read off the query's own
  distribution; restricted to the window, a far tail cannot bias it and two
  darts further apart than the window cannot be averaged. ``tip_hard`` is the
  argmax cell plus the tip head's trained sub-cell offset at that cell.
  Flight, box centre, direction and extent are the dense field's estimates at
  the tip cell's stride-8 parent.
* Training (:func:`heatmap_losses`): each dart's target is the tip heatmap's
  Gaussian, centred on the exact landing point and normalised into a
  distribution, without the pin to 1.0 that only the focal loss needs. KL to
  each query's softmax over the whole grid; an L1 on the decoded landing
  point -- the window soft-argmax around the query's own argmax, exactly as
  read out -- for the queries whose argmax lies within a cell of their
  dart's tip; and a confidence BCE. The matching cost is confidence, the
  cross-entropy and the decoded landing distance.

Coinciding tips need nothing special: two queries may peak on the same cell.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from darts_model.model.instance import _Block
from darts_model.model.queries import POSITION_OCTAVES, _fourier


def local_softargmax(flat: torch.Tensor, cy: torch.Tensor, cx: torch.Tensor,
                     hw: tuple[int, int], radius: int) -> torch.Tensor:
    """The softmax-weighted mean position over the window of half-width
    ``radius`` cells around (``cy``, ``cx``), in cells (cell centres at +0.5).

    ``flat`` (..., h*w) logits; ``cy``, ``cx`` (...) integer cells. Cells
    outside the grid take no weight. Returns (..., 2) as (x, y).
    """
    h, w = hw
    r = torch.arange(-radius, radius + 1, device=flat.device)
    dy, dx = torch.meshgrid(r, r, indexing="ij")
    yy = cy.unsqueeze(-1) + dy.flatten()                   # (..., K)
    xx = cx.unsqueeze(-1) + dx.flatten()
    inside = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
    index = yy.clamp(0, h - 1) * w + xx.clamp(0, w - 1)
    v = torch.gather(flat, -1, index)
    v = torch.where(inside, v, torch.full_like(v, -1e4))
    p = v.softmax(-1)
    return torch.stack(((p * (xx.to(p.dtype) + 0.5)).sum(-1),
                        (p * (yy.to(p.dtype) + 0.5)).sum(-1)), -1)


def gaussian_targets(ends: torch.Tensor, mask: torch.Tensor,
                     hw: tuple[int, int], sigma: float) -> torch.Tensor:
    """Each dart's landing point as a distribution over the grid: the
    Gaussian of ``sigma`` cells around the exact point, sampled at cell
    centres and normalised. ``ends`` (B, D, 2) normalised; returns
    (B, D, h*w), zero for masked darts."""
    h, w = hw
    ys = torch.arange(h, device=ends.device, dtype=torch.float32) + 0.5
    xs = torch.arange(w, device=ends.device, dtype=torch.float32) + 0.5
    cx = ends[..., 0].float() * w
    cy = ends[..., 1].float() * h
    d2 = ((xs.view(1, 1, 1, w) - cx[..., None, None]) ** 2
          + (ys.view(1, 1, h, 1) - cy[..., None, None]) ** 2)
    g = torch.exp(-d2 / (2.0 * sigma * sigma)).flatten(2)
    g = g / g.sum(-1, keepdim=True).clamp(min=1e-12)
    return g * mask.unsqueeze(-1).float()


class HeatmapReadout(nn.Module):
    """The dense field -> ``query_count`` darts with a confidence each."""

    def __init__(self, cfg) -> None:
        super().__init__()
        d, n, heads = cfg.heatmap_dim, cfg.query_count, cfg.heatmap_heads
        p = cfg.heatmap_pixel_dim
        self.cfg = cfg
        self.memory = nn.Linear(cfg.head_width, d)
        self.pos = nn.Linear(2 * 2 * POSITION_OCTAVES, d)
        self.queries = nn.Parameter(torch.randn(n, d))
        self.self_attn = nn.ModuleList(_Block(d, heads)
                                       for _ in range(cfg.heatmap_layers))
        self.cross_attn = nn.ModuleList(_Block(d, heads)
                                        for _ in range(cfg.heatmap_layers))
        self.norm = nn.LayerNorm(d)
        self.pixel_tip = nn.Conv2d(cfg.tip_head_width, p, 1)
        self.pixel_neck = nn.Conv2d(cfg.head_width, p, 1)
        self.pixel_pos = nn.Linear(2 * 2 * POSITION_OCTAVES, p)
        # 3x3: a cell's sub-cell position relative to a tip is read from its
        # neighbourhood, as the tip head's own offset conv reads it.
        self.pixel_out = nn.Conv2d(p, p, 3, padding=1)
        # The (query, cell) MLP: the cell's hidden units, shifted by the
        # query's, then the query's own output weights.
        hidden = cfg.heatmap_hidden
        self.cell_hidden = nn.Conv1d(p, hidden, 1)
        self.query_hidden = nn.Linear(d, hidden)
        self.query_out = nn.Linear(d, hidden + 1)
        # One confidence branch per query.
        self.conf = nn.ModuleList(nn.Linear(d, 1) for _ in range(n))

    def forward(self, out: dict) -> dict:
        """Planes (B, Q, ...): ``logit``/``score`` (confidence), ``tip``
        (window soft-argmax), ``tip_hard`` (argmax cell plus the tip head's
        offset), ``flight``, ``centre``, ``direction``, ``extent``, all
        normalised, and the query heatmaps ``heat_logits`` (B, Q, h*w).

        Runs in float32 whatever the autocast state.
        """
        dev = out["tip_logits"].device.type
        if torch.is_autocast_enabled(dev):
            with torch.autocast(dev, enabled=False):
                return self._forward(
                    {k: v.float() if torch.is_tensor(v) and v.is_floating_point()
                     else v for k, v in out.items()})
        return self._forward(out)

    def _forward(self, out: dict) -> dict:
        from darts_model.model.detector import decode_boxes

        cfg = self.cfg
        neck, feat = out["neck"], out["tip_features"]
        b = int(neck.shape[0])
        h, w = int(feat.shape[-2]), int(feat.shape[-1])

        mem = out["neck_coarse"]
        mh, mw = int(mem.shape[-2]), int(mem.shape[-1])
        ys = (torch.arange(mh, device=mem.device, dtype=mem.dtype) + 0.5) / mh
        xs = (torch.arange(mw, device=mem.device, dtype=mem.dtype) + 0.5) / mw
        grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1).view(1, -1, 2)
        mem = (self.memory(mem.flatten(2).transpose(1, 2))
               + self.pos(_fourier(grid, POSITION_OCTAVES)))      # (B, M, d)

        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        none_q = q.new_zeros(b, int(q.shape[1]))
        none_m = mem.new_zeros(b, int(mem.shape[1]))
        for sa, ca in zip(self.self_attn, self.cross_attn):
            q = sa(q, q, none_q)
            q = ca(q, mem, none_m)
        q = self.norm(q)

        cell_y = (torch.arange(h, device=feat.device, dtype=feat.dtype) + 0.5) / h
        cell_x = (torch.arange(w, device=feat.device, dtype=feat.dtype) + 0.5) / w
        cells = torch.stack(torch.meshgrid(cell_x, cell_y, indexing="xy"), -1)
        pos = self.pixel_pos(_fourier(cells, POSITION_OCTAVES))   # (h, w, P)
        pixel = (self.pixel_tip(feat) + pos.permute(2, 0, 1).unsqueeze(0)
                 + F.interpolate(self.pixel_neck(neck), size=(h, w),
                                 mode="bilinear", align_corners=False))
        pixel = self.pixel_out(F.gelu(pixel)).flatten(2)          # (B, P, h*w)

        hidden = F.gelu(self.cell_hidden(pixel).unsqueeze(1)
                        + self.query_hidden(q).unsqueeze(-1))     # (B, Q, H, h*w)
        wout = self.query_out(q)                                  # (B, Q, H + 1)
        nh = int(hidden.shape[2])
        heat = (torch.einsum("bqh,bqhk->bqk", wout[..., :nh], hidden)
                / math.sqrt(nh) + wout[..., nh:])                 # (B, Q, h*w)

        index = heat.argmax(-1)                                   # (B, Q)
        cy = torch.div(index, w, rounding_mode="floor")
        cx = index - cy * w
        size = torch.tensor([w, h], device=heat.device, dtype=heat.dtype)
        soft = local_softargmax(heat, cy, cx, (h, w), cfg.heatmap_window) / size

        offset = out["tip_cell_offset"].flatten(2)                # (B, 2, h*w)
        at = torch.gather(offset, 2, index.unsqueeze(1).expand(-1, 2, -1))
        hard = (torch.stack((cx, cy), -1).to(heat.dtype) + 0.5
                + at.transpose(1, 2)) / size

        # The dense field's estimates at the tip cell's stride-8 parent.
        fh, fw = (int(s) for s in out["fg_logits"].shape[-2:])
        dec = decode_boxes(out, (fh, fw))
        parent = ((cy * fh) // h) * fw + (cx * fw) // w

        def at_parent(plane):                                     # (B, C, fh, fw)
            c = int(plane.shape[1])
            return torch.gather(plane.flatten(2), 2,
                                parent.unsqueeze(1).expand(-1, c, -1)).transpose(1, 2)

        logit = torch.cat([self.conf[j](q[:, j])
                           for j in range(int(q.shape[1]))], -1)
        return {
            "logit": logit,
            "score": torch.sigmoid(logit),
            "tip": soft,
            "tip_hard": hard,
            "flight": at_parent(dec["flight_point"]),
            "centre": at_parent(dec["centre"]),
            "direction": at_parent(dec["direction"]),
            "extent": at_parent(dec["extent"]),
            "heat_logits": heat,
            "cell": torch.stack((cx, cy), -1),                    # (B, Q, 2)
            "grid_size": (h, w),
        }


def match_heatmaps(pred: dict, ce: torch.Tensor, gt_tip: torch.Tensor,
                   gt_ok: torch.Tensor, cfg, scale: float):
    """Hungarian assignment of queries to darts, per frame, on confidence,
    the heatmap cross-entropy ``ce`` (B, Q, D) and the decoded landing
    distance. Returns (frame, query, dart) index tensors."""
    from scipy.optimize import linear_sum_assignment

    tip = pred["tip"].detach().float().cpu()
    score = pred["score"].detach().float().cpu()
    ce = ce.detach().float().cpu()
    gt = gt_tip.detach().float().cpu()
    ok = gt_ok.detach().cpu()
    fb, qb, db = [], [], []
    for f in range(tip.shape[0]):
        darts = torch.nonzero(ok[f]).flatten()
        if not darts.numel():
            continue
        l1 = (tip[f].unsqueeze(1) - gt[f, darts].unsqueeze(0)).abs().sum(-1)
        cost = (cfg.query_cost_px * l1 * scale
                + cfg.heatmap_cost_ce * ce[f][:, darts]
                - cfg.query_cost_conf * score[f].unsqueeze(1))
        qi, di = linear_sum_assignment(cost.numpy())
        fb += [f] * len(qi)
        qb += qi.tolist()
        db += darts[di].tolist()
    t = lambda v: torch.as_tensor(np.asarray(v, dtype=np.int64),  # noqa: E731
                                  device=pred["tip"].device)
    return t(fb), t(qb), t(db)


def heatmap_losses(pred: dict, batch: dict, cfg, scale: float):
    """The readout's losses, the matched landing errors in pixels, the
    matching, and per-pair statistics.

    ``h_conf``: BCE over every query, 1 when matched. For each matched
    (query, dart): ``h_kl``, the KL from the dart's Gaussian to the query's
    softmax over the grid. ``h_l1``: the L1 in pixels of the decoded landing
    point, for the matched queries whose argmax lies within a cell of their
    dart's tip (Chebyshev). The decode's window is centred on the argmax, so
    only a query peaking on its own tip has a window that can reach it; a
    query peaking elsewhere is left to the KL and the matching, and a window
    placed around the tip instead would measure the shared features' shape
    there, not the query's peak.

    The statistics, per matched pair: ``on_peak`` (the argmax is within a
    cell of the tip) and both decodes' errors in pixels, ``soft_px`` and
    ``offset_px``.
    """
    h, w = pred["grid_size"]
    ends = batch["dart_ends"].float()
    mask = batch["dart_mask"]
    target = gaussian_targets(ends[..., 0:2], mask, (h, w), cfg.tip_sigma)
    logp = pred["heat_logits"].float().log_softmax(-1)              # (B, Q, K)
    ce = -torch.einsum("bdk,bqk->bqd", target, logp)
    f, q, d = match_heatmaps(pred, ce, ends[..., 0:2], mask, cfg, scale)

    conf = torch.zeros_like(pred["logit"])
    conf[f, q] = 1.0
    losses = {"h_conf": F.binary_cross_entropy_with_logits(pred["logit"], conf)}
    zero = pred["logit"].sum() * 0.0
    if not f.numel():
        losses.update(h_kl=zero, h_l1=zero)
        empty = zero.new_zeros(0)
        stats = {"on_peak": empty.bool(), "soft_px": empty, "offset_px": empty}
        return losses, empty, (f, q, d), stats

    t = target[f, d]
    entropy = -(t * t.clamp(min=1e-12).log()).sum(-1)
    losses["h_kl"] = (ce[f, q, d] - entropy).mean()

    gt = ends[f, d, 0:2]
    size = torch.tensor([w, h], device=gt.device)
    tip_cell = (gt * size).floor().long()
    on_peak = ((pred["cell"][f, q] - tip_cell).abs() <= 1).all(-1)
    l1 = (pred["tip"][f, q] - gt).abs().sum(-1) * scale
    losses["h_l1"] = l1[on_peak].mean() if bool(on_peak.any()) else zero

    def px(a):
        return ((a - gt).square().sum(-1) + 1e-12).sqrt() * scale

    tip_px = px(pred["tip"][f, q])
    stats = {"on_peak": on_peak.detach(), "soft_px": tip_px.detach(),
             "offset_px": px(pred["tip_hard"][f, q]).detach()}
    return losses, tip_px.detach(), (f, q, d), stats
