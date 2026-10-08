"""Tip heatmap: where landing points are, at sub-pixel precision, for snapping.

Every dart cell regresses its dart's landing point, and those estimates know
the dart's line far better than where along it the point ends: on isolated
darts the error spreads 2.6px along the axis against 0.9px across it, and
even the cells sitting on the tip are ~2px out. The board corners, from the
same backbone at the same stride, come out at 0.26-0.30px. They are read as
a heatmap -- is the corner in this cell? -- plus a sub-cell offset trained
only at the corner's own cell, rather than as a regression every cell of a
long object has to agree on.

This is that head for tips, as a refinement rather than a readout. It is
class-agnostic, one channel for any number of darts, CenterNet-style: an
independent sigmoid per cell, a penalty-reduced focal loss, and per-dart
Gaussians combined by max, never summed, so every tip's own cell is a
maximum of the target however close another tip is. The readout still
finds the darts and places each one; a dart's landing point then snaps to
the nearest confident peak close to its estimate and its axis, and keeps the
estimate where there is none.

Every landing point is painted, hidden ones included: an entry covered by
the dart's own barrel is still fixed by the visible dart, and how
confidently the head can place it is for training to find out.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

#: Peaks scoring under this are reported as score 0 at (0, 0). Below it the
#: top-k order among near-equal noise differs between export backends, and
#: nothing snaps there anyway.
PEAK_FLOOR = 0.05


def tip_targets(ends: torch.Tensor, mask: torch.Tensor, hw, sigma: float,
                offset_radius: int = 0):
    """Landing points -> the tip heatmap target, sub-cell offsets and where
    the offsets are defined.

    ``ends`` (B, D, 2) normalised landing points, ``mask`` (B, D). Returns
    ``(heat (B, h, w), offset (B, 2, h, w), at (B, h, w))``. The heat is the
    max over darts of each dart's Gaussian, with each tip's own cell pinned
    to exactly 1.0 (see ``keypoint_targets`` for why the pin is needed).

    The offset is the vector from a cell's centre to the tip, in cells, and
    is defined on every cell within ``offset_radius`` (Chebyshev) of a tip's
    own cell -- not only on that cell, as CenterNet does. A peak one cell off
    then still points at the tip. That was measured to matter: at stride 4,
    37% of peaks landed one cell from the tip's own, 2.5px out, where the
    tip's own cell would have read 0.9px. A cell near two tips takes the
    nearer.
    """
    from darts_model.model.detector import keypoint_targets
    heat, offset, at = keypoint_targets(ends, mask, hw, sigma)
    heat = heat.amax(dim=1)
    if offset_radius <= 0:
        return heat, offset, at
    h, w = hw
    cx = ends[..., 0] * w                                        # (B, D) cells
    cy = ends[..., 1] * h
    xs = torch.arange(w, device=ends.device, dtype=ends.dtype)
    ys = torch.arange(h, device=ends.device, dtype=ends.dtype)
    dx = cx[..., None, None] - (xs.view(1, 1, 1, w) + 0.5)      # (B, D, 1, w)
    dy = cy[..., None, None] - (ys.view(1, 1, h, 1) + 0.5)      # (B, D, h, 1)
    r = float(offset_radius)
    near = (((cx.floor()[..., None, None] - xs.view(1, 1, 1, w)).abs() <= r)
            & ((cy.floor()[..., None, None] - ys.view(1, 1, h, 1)).abs() <= r)
            & mask[..., None, None])                             # (B, D, h, w)
    d2 = torch.where(near, dx.square() + dy.square(),
                     torch.full_like(near, float("inf"), dtype=ends.dtype))
    k = d2.argmin(dim=1, keepdim=True)                           # nearest tip
    dx = torch.gather(dx.expand_as(d2), 1, k)[:, 0]
    dy = torch.gather(dy.expand_as(d2), 1, k)[:, 0]
    at = near.any(dim=1)
    offset = torch.stack((dx, dy), 1) * at.unsqueeze(1)
    return heat, offset, at


def tip_peaks(logits: torch.Tensor, offset: torch.Tensor, k: int,
              return_index: bool = False):
    """The ``k`` strongest local maxima of the tip heatmap.

    ``logits`` (B, h, w), ``offset`` (B, 2, h, w) in cells. Returns
    ``(xy (B, k, 2) normalised, score (B, k))``; a cell is a peak when it
    equals the 3x3 max around it. Anything else, and any peak under
    ``PEAK_FLOOR``, scores 0 at (0, 0). With ``return_index``, also each
    peak's flat cell index (B, k), to gather features at.
    """
    b, h, w = (int(x) for x in logits.shape)
    p = torch.sigmoid(logits.float())
    pooled = F.max_pool2d(p.unsqueeze(1), 3, stride=1, padding=1)[:, 0]
    peak = p * (p >= pooled).float() * (p >= PEAK_FLOOR).float()
    n = h * w
    k = min(k, n)
    score, idx = torch.topk(peak.reshape(b, n), k, dim=1, sorted=True)
    off = torch.gather(offset.float().reshape(b, 2, n), 2,
                       idx.unsqueeze(1).expand(b, 2, k))
    ix = (idx % w).float()
    iy = torch.div(idx, w, rounding_mode="floor").float()
    xy = torch.stack(((ix + 0.5 + off[:, 0]) / w, (iy + 0.5 + off[:, 1]) / h),
                     dim=-1)
    xy = xy * (score > 0).float().unsqueeze(-1)
    if return_index:
        return xy, score, idx
    return xy, score


def weighted_offset_l1(pred: torch.Tensor, target: torch.Tensor,
                       weight: torch.Tensor) -> torch.Tensor:
    """Mean absolute offset error per channel, weighted per cell.

    ``weight`` (B, h, w) is zero where the offset is undefined. Normalised by
    the summed weight, so a tip's own cell (weight 1 under the heatmap
    target) keeps most of the loss while its neighbours (0.61 and 0.37 at a
    one-cell Gaussian) still learn to point at the tip. Averaging uniformly
    over the 3x3 instead gave the own cell a ninth of the weight, and its
    reading fell from 0.94 to 1.50px.
    """
    w = weight.unsqueeze(1)
    target = torch.where(w > 0, target, torch.zeros_like(target))
    return ((pred - target).abs() * w).sum() / w.sum().clamp(min=1e-6) / pred.shape[1]


def snap_to_tips(points: torch.Tensor, direction: torch.Tensor,
                 peaks_xy: torch.Tensor, peaks_score: torch.Tensor, cfg,
                 image_px: float):
    """Move each landing point to the nearest acceptable tip peak.

    ``points`` and ``direction`` (B, S, 2): each dart's landing estimate and
    its unit axis; ``peaks_xy`` (B, P, 2) and ``peaks_score`` (B, P) from
    :func:`tip_peaks`. A peak is acceptable for a dart when it scores at
    least ``tip_snap_score``, lies within ``tip_snap_px`` of the estimate and
    within ``tip_snap_axis_px`` of the dart's axis through it. Returns
    ``(snapped (B, S, 2), found (B, S))``; a dart with no acceptable peak
    keeps its estimate. Fixed shapes, so it exports in-graph.
    """
    d = (peaks_xy.unsqueeze(1) - points.unsqueeze(2)) * image_px   # (B,S,P,2)
    dist = (d.square().sum(-1) + 1e-12).sqrt()
    u = direction.unsqueeze(2)
    across = (d[..., 0] * u[..., 1] - d[..., 1] * u[..., 0]).abs()
    ok = ((peaks_score.unsqueeze(1) >= cfg.tip_snap_score).float()
          * (dist <= cfg.tip_snap_px).float()
          * (across <= cfg.tip_snap_axis_px).float())
    # Nearest acceptable peak: unacceptable ones are pushed past every
    # acceptable distance.
    big = 4.0 * cfg.tip_snap_px + 1.0
    i = torch.argmin(dist + (1.0 - ok) * big, dim=-1)               # (B, S)
    b, s = int(i.shape[0]), int(i.shape[1])
    chosen = torch.gather(peaks_xy, 1, i.unsqueeze(-1).expand(b, s, 2))
    found = torch.gather(ok, 2, i.unsqueeze(-1))[..., 0]
    snapped = found.unsqueeze(-1) * chosen + (1.0 - found.unsqueeze(-1)) * points
    return snapped, found


def assign_tips(points: torch.Tensor, peaks_xy: torch.Tensor,
                peaks_score: torch.Tensor, cfg, image_px: float):
    """Give detections tip peaks one-to-one, by the Hungarian algorithm.

    ``points`` (N, 2) are one frame's landing estimates, ``peaks_xy`` (P, 2)
    and ``peaks_score`` (P,) its peaks. Peaks scoring at least
    ``tip_snap_score`` are assigned to estimates so as to minimise the total
    distance, never further than ``tip_assign_gate_px``, each peak to at most
    one dart. Returns ``(assigned (N, 2), found (N,))``; a dart left without a
    peak keeps its estimate.

    Global rather than each dart taking its nearest peak: a dart's score
    depends only on the set of landing points, not on which flight a tip
    belongs to, so in a tight group it is enough that every dart gets a
    distinct tip. A dart whose estimate landed 20-30px off can still be given
    a tip -- possibly its neighbour's, in which case the neighbour gets the
    other -- where a nearest-peak snap within a few pixels gives it none, or
    gives two darts the same one.
    """
    from scipy.optimize import linear_sum_assignment

    out = points.clone()
    found = torch.zeros(points.shape[0], dtype=torch.bool)
    ok = peaks_score >= cfg.tip_snap_score
    if not bool(ok.any()) or points.shape[0] == 0:
        return out, found
    cand = peaks_xy[ok]
    dist = (torch.cdist(points.double(), cand.double()) * image_px).cpu()
    gate = float(cfg.tip_assign_gate_px)
    cost = torch.where(dist <= gate, dist, torch.full_like(dist, 1e9))
    rows, cols = linear_sum_assignment(cost.numpy())
    for r, c in zip(rows.tolist(), cols.tolist()):
        if float(dist[r, c]) <= gate:
            out[r] = cand[c].to(out.dtype)
            found[r] = True
    return out, found
