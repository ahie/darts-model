"""Turn a dense field of per-cell boxes into detections, by voting.

Every foreground cell decodes the box of the dart it sits on, so all ~83 cells
covering one dart vote for the same box.  Detections are the peaks those votes
accumulate into; angle and extent are averaged over each peak's voters.

Two properties this buys, both of which a query-based readout has to learn:

* **Duplicate suppression is arithmetic.** Cells agreeing on a box produce one
  peak, whatever their number.  Nothing has to discover that two queries
  describe one object.
* **Overlap needs no rule.** A cell in the intersection of two darts either
  commits to one -- a good vote -- or hedges, and hedging cells disagree with
  each other, so they scatter between the peaks instead of accumulating into a
  false one.  This is the property voting exists for, and it removes the
  FCOS-style "assign the ambiguous cell to the smaller object" rule, which
  would not apply anyway since darts are all one size.

Inference only: no gradient passes through here, so it is free to be
non-differentiable, and nothing about duplicate suppression can destabilise
training.
"""
from __future__ import annotations

import torch


def vote(centres: torch.Tensor, weights: torch.Tensor, bins: int):
    """Accumulate weighted votes over a ``bins x bins`` grid of the unit square.

    ``centres`` is (N, 2) in [0, 1]; ``weights`` is (N,).  Returns the float32
    accumulator (bins, bins) indexed [y, x].
    """
    acc = torch.zeros(bins * bins, device=centres.device,
                      dtype=torch.float32)
    if centres.numel() == 0:
        return acc.view(bins, bins)
    ix = (centres[:, 0] * bins).long().clamp(0, bins - 1)
    iy = (centres[:, 1] * bins).long().clamp(0, bins - 1)
    # float32 whatever the scores arrive in: a bf16 sum of ~80 votes near 0.9
    # rounds in steps of 0.25-0.5, enough to create or break a tie.
    acc.index_add_(0, iy * bins + ix, weights.float())
    return acc.view(bins, bins)


#: The eight neighbours of a bin, as (dy, dx). Those before the centre in
#: raster order are the ones an equal value loses to.
_NEIGHBOURS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                    if (dy, dx) != (0, 0))


def local_maxima(acc: torch.Tensor, min_value: float):
    """Bins that are the strict maximum of their own 3x3 neighbourhood and
    clear a floor.

    Ties are broken by raster order: a bin must be ``>=`` every neighbour and
    ``>`` every neighbour that precedes it (the row above, and the bin to its
    left). That is a strict total order on (value, -raster index), so no two
    peaks are ever within one bin of each other and a plateau of equal votes
    yields one peak, not several.

    This is the whole of non-maximum suppression here: eight shifted
    comparisons on the accumulator, not an IoU loop over predicted boxes and
    not a learned behaviour.
    """
    h, w = acc.shape
    padded = torch.nn.functional.pad(acc[None, None], (1, 1, 1, 1),
                                     value=float("-inf"))[0, 0]
    peak = acc >= min_value
    for dy, dx in _NEIGHBOURS:
        nb = padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        before = dy < 0 or (dy == 0 and dx < 0)
        peak &= (acc > nb) if before else (acc >= nb)
    return peak


def detect(decoded: dict, fg_prob: torch.Tensor, cfg,
           claim_by_embedding: bool | None = None) -> list[dict]:
    """One image's dense field -> a list of detections.

    ``decoded`` holds ``centre`` (2, h, w), ``direction`` (2, h, w) and
    ``extent`` (2, h, w), plus ``tip_point`` and ``flight_point`` (2, h, w) when
    the head predicts the two ends and ``embedding`` (E, h, w) when it predicts
    one; ``fg_prob`` is (h, w). ``claim_by_embedding`` overrides the config's.

    Each detection carries two versions of each end:

    * ``tip`` / ``flight`` -- the landing point and flight tip. Voted from the
      predicted ends when the head has them (``tip_is_predicted``), else the
      box's ends. ``tip`` is the point scoring reads.
    * ``box_tip`` / ``box_flight`` -- always the ends of the voted box's
      centreline, whatever the head predicts.

    Procedure:

    1. Every cell over ``fg_threshold`` votes its predicted centre, weighted by
       its foreground score, into bins of ``vote_bin_px``.
    2. Peaks are :func:`local_maxima` over ``peak_min_votes * fg_threshold``.
    3. Peaks are visited in descending accumulator value, ties in ascending
       raster order. Each claims the not-yet-claimed voters whose bin lies in
       its 3x3 neighbourhood. A peak that claims fewer than ``peak_min_votes``
       is dropped and claims nothing, so its voters stay available to later
       peaks. Every voter therefore counts toward at most one detection.
    4. With ``claim_by_embedding``, step 3's claims are only each peak's core.
       The core's fg-weighted mean embedding is the peak's seed, and every
       voter -- claimed in step 3 or not -- is regrouped to its nearest seed,
       if within ``embed_push_margin``. A peak left with fewer than
       ``peak_min_votes`` is dropped.
    5. Each accepted peak's box, ends and score are the fg-weighted means over
       its claimed voters.
    """
    h, w = fg_prob.shape
    keep = fg_prob >= cfg.fg_threshold
    if not bool(keep.any()):
        return []
    if claim_by_embedding is None:
        claim_by_embedding = getattr(cfg, "claim_by_embedding", False)
    if claim_by_embedding and "embedding" not in decoded:
        raise ValueError("claim_by_embedding needs an embedding in the field")

    has_ends = "tip_point" in decoded
    if has_ends:
        tip_pt = decoded["tip_point"].permute(1, 2, 0)[keep]   # (N, 2)
        fl_pt = decoded["flight_point"].permute(1, 2, 0)[keep]
    centre = decoded["centre"].permute(1, 2, 0)[keep]          # (N, 2)
    direction = decoded["direction"].permute(1, 2, 0)[keep]    # (N, 2)
    extent = decoded["extent"].permute(1, 2, 0)[keep]          # (N, 2)
    score = fg_prob[keep].float()                              # (N,)

    # The accumulator is deliberately NOT the feature grid. Predicted centres
    # are continuous -- sub-cell precision is the entire reason for regressing
    # an offset -- and binning at the feature stride would merge the closest
    # decile of dart pairs, measured at 2.5 feature cells apart.
    bins = max(int(round(1.0 / (cfg.vote_bin_px / (w * cfg.out_stride)))), 1)
    acc = vote(centre, score, bins)
    peaks = local_maxima(acc, cfg.peak_min_votes * cfg.fg_threshold)
    if not bool(peaks.any()):
        return []

    ix = (centre[:, 0] * bins).long().clamp(0, bins - 1)
    iy = (centre[:, 1] * bins).long().clamp(0, bins - 1)
    ys, xs = torch.nonzero(peaks, as_tuple=True)       # raster order
    # Stable sort, so equal values keep raster order.
    order = torch.argsort(acc[ys, xs], descending=True, stable=True)
    claimed = torch.zeros_like(score, dtype=torch.bool)
    groups = []
    for k in order.tolist():
        py, px = int(ys[k]), int(xs[k])
        # Voters of this peak and of the ring around it: a peak's support
        # straddles bin edges, so taking the bin alone would both understate
        # the count and bias the average toward whichever side of the edge the
        # true centre fell on. Voters already claimed by a stronger peak are
        # not counted twice.
        m = ((ix - px).abs() <= 1) & ((iy - py).abs() <= 1) & ~claimed
        if int(m.sum()) < cfg.peak_min_votes:
            continue
        claimed |= m
        groups.append(m)

    if claim_by_embedding and groups:
        # A centre vote only says where a cell thinks its dart's middle is; a
        # cell whose centre is off by a few bins is lost to its own dart even
        # when everything else about it is right. The embedding is trained to
        # say which dart the cell is on, so it regroups every voter.
        emb = decoded["embedding"].permute(1, 2, 0)[keep].float()   # (N, E)
        seeds = torch.stack([
            (emb[m] * score[m, None]).sum(0) / score[m].sum().clamp(min=1e-6)
            for m in groups])                                       # (P, E)
        d = torch.cdist(emb, seeds)                                 # (N, P)
        near, which = d.min(dim=1)
        ok = near <= cfg.embed_push_margin
        groups = [g for g in (ok & (which == j) for j in range(len(seeds)))
                  if int(g.sum()) >= cfg.peak_min_votes]

    out = []
    for m in groups:
        n = int(m.sum())
        wgt = score[m]
        tot = wgt.sum().clamp(min=1e-6)
        c = (centre[m] * wgt[:, None]).sum(0) / tot
        d = (direction[m] * wgt[:, None]).sum(0)
        d = d / d.norm().clamp(min=1e-6)
        e = (extent[m] * wgt[:, None]).sum(0) / tot
        box_tip = c - d * e[0]
        box_flight = c + d * e[0]
        # The box's end is not the landing point in general. For a dart
        # pointing near the camera the entry projects into the INTERIOR of the
        # silhouette -- its own barrel and flight cover where it went in --
        # measured a median 1.4px from the silhouette's end but 8.9px, up to
        # 27, for the shortest-projecting decile. No end of any box can
        # represent that, and the landing point is what scoring uses.
        if has_ends:
            tip = (tip_pt[m] * wgt[:, None]).sum(0) / tot
            flight = (fl_pt[m] * wgt[:, None]).sum(0) / tot
        else:
            tip, flight = box_tip, box_flight
        out.append({
            "centre": c,
            "direction": d,
            "half_length": e[0],
            "half_width": e[1],
            "tip": tip,
            "flight": flight,
            "box_tip": box_tip,
            "box_flight": box_flight,
            "tip_is_predicted": has_ends,
            "votes": n,
            "score": float(tot / max(n, 1)),
        })
    out.sort(key=lambda r: -r["votes"])
    return out
