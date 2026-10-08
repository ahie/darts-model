"""Stage-3 readout: learned queries over stride-4 tokens that carry the dense
tip supervision. Hough voting and peak assignment, done in-graph and learned.

The earlier attention readouts were asked to localise from stride-8 tokens
and a weighted average of the dense regression's ~2px estimates, with one
landing loss per dart and absolute positions added once. The stride-4 tip
heatmap, densely supervised at every cell, places a tip to ~1px. So here the
tokens ARE the heatmap's cells, and the queries only decide which dart and
which tip:

* Tokens: the stride-4 cells on a dart or near a tip -- the top
  ``readout_tokens`` by the larger of the upsampled foreground score and the
  tip heatmap -- each carrying the tip head's features, the neck's features
  at its stride-8 parent cell, its tip logit and sub-cell tip offset, and the
  parent cell's dense estimates.
* Encoder: self-attention among the tokens restricted to a window, with a
  learned relative-position bias per head. Telling apart the cells of two
  touching darts is a relation between nearby cells, which absolute
  encodings added once do not give.
* Decoder: plain learned queries -- no anchors or reference points -- each
  with its own output branch, attending to the tokens.
* Outputs per query: a confidence; a *dart* pointer over tokens whose values
  are the dense estimates of the cells' dart (landing, flight, centre,
  direction, extent) -- the long-range estimate that covers hidden tips; and
  a *tip* pointer over [use that estimate, token 1 ... token M] whose token
  values are each token's position plus its own trained sub-cell tip offset,
  so a query pointing at its tip's cells inherits the heatmap's precision.
* Training (:func:`readout_losses`): Hungarian matching on the landing point plus
  confidence, as before; landing, estimate and flight losses in pixels; and
  two cross-entropies that supervise the routing directly -- the dart
  pointer's mass on the matched dart's own cells, and the tip pointer's
  weight on the tip's own cell.

Coinciding tips need nothing special: two queries may point at the same
cells.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from darts_model.model.instance import DART_SCALE, MEMBERSHIP_FLOOR, _Block
from darts_model.model.queries import POSITION_OCTAVES, _fourier


class _LocalAttention(nn.Module):
    """Self-attention among tokens on a grid, within a square window, with a
    learned bias per head for every offset in it."""

    def __init__(self, d: int, heads: int, window: int) -> None:
        super().__init__()
        self.heads, self.window = heads, window
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        side = 2 * window + 1
        self.rel = nn.Parameter(torch.zeros(heads, side * side))

    def forward(self, x, grid, extra):
        """``x`` (N, M, d); ``grid`` (N, M, 2) integer cell coordinates;
        ``extra`` (N, M) an additive per-key bias."""
        n, m, d = (int(v) for v in x.shape)
        h, dh, r = self.heads, d // self.heads, self.window
        q, k, v = self.qkv(x).view(n, m, 3, h, dh).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))           # (N,h,M,dh)
        off = grid.unsqueeze(1) - grid.unsqueeze(2)                # (N,M,M,2)
        inside = (off.abs() <= r).all(-1)
        o = off.clamp(-r, r) + r
        index = (o[..., 1] * (2 * r + 1) + o[..., 0]).reshape(n, m * m)
        bias = torch.gather(self.rel.unsqueeze(0).expand(n, -1, -1), 2,
                            index.unsqueeze(1).expand(n, h, m * m))
        bias = bias.view(n, h, m, m)
        bias = torch.where(inside.unsqueeze(1), bias,
                           torch.full_like(bias, -1e4))
        a = q @ k.transpose(-1, -2) / math.sqrt(dh) + bias + extra[:, None, None, :]
        out = (a.softmax(-1) @ v).transpose(1, 2).reshape(n, m, d)
        return self.o(out)


class _LocalBlock(nn.Module):
    def __init__(self, d: int, heads: int, window: int) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = _LocalAttention(d, heads, window)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                 nn.Linear(2 * d, d))

    def forward(self, x, grid, extra):
        x = x + self.attn(self.n1(x), grid, extra)
        return x + self.mlp(self.n2(x))


def gather_tokens(out: dict, k: int) -> dict:
    """The top-``k`` stride-4 cells by max(upsampled foreground, tip
    heatmap), as (B, k, ...) planes.

    Positions are normalised cell centres at the tip grid; ``grid`` is the
    integer cell coordinate and ``grid_size`` the grid's (w, h); ``parent``
    the flat index of the stride-8 cell the token lies in, at which the dense
    outputs and the neck are read.
    """
    tip_logits = out["tip_logits"].float()
    b, h4, w4 = (int(v) for v in tip_logits.shape)
    h8, w8 = (int(v) for v in out["fg_logits"].shape[-2:])
    f = h4 // h8
    tip_p = torch.sigmoid(tip_logits)
    fg_p = torch.sigmoid(out["fg_logits"].float())
    fg_up = F.interpolate(fg_p.unsqueeze(1), size=(h4, w4), mode="nearest")[:, 0]
    sel = torch.maximum(fg_up, tip_p).reshape(b, h4 * w4)
    n4 = h4 * w4
    k = min(k, n4)
    score, idx = torch.topk(sel, k, dim=1, sorted=True)
    # Cast back to integers explicitly: CoreML lowers integer division and
    # modulo through floats, and these index gathers.
    ix = (idx % w4).long()
    iy = (idx // w4).long()
    parent = ((iy // f) * w8 + ix // f).long()

    def at4(t):                                     # (B, C, h4, w4) -> (B, k, C)
        c = int(t.shape[1])
        return torch.gather(t.float().reshape(b, c, n4), 2,
                            idx.unsqueeze(1).expand(b, c, k)).transpose(1, 2)

    def at8(t):                                     # (B, C, h8, w8) -> (B, k, C)
        c = int(t.shape[1])
        return torch.gather(t.float().reshape(b, c, h8 * w8), 2,
                            parent.unsqueeze(1).expand(b, c, k)).transpose(1, 2)

    pos = torch.stack(((ix.float() + 0.5) / w4, (iy.float() + 0.5) / h4), -1)
    pc = torch.stack((((parent % w8).float() + 0.5) / w8,
                      ((parent // w8).float() + 0.5) / h8), -1)
    d = at8(out["direction"])
    off = at4(out["tip_cell_offset"])
    return {
        "score": score, "pos": pos,
        "grid": torch.stack((ix, iy), -1),
        "grid_size": torch.tensor([w4, h4], device=idx.device),
        "parent": parent,
        "tip_logit": torch.gather(tip_logits.reshape(b, n4), 1, idx),
        "fg_logit": at8(out["fg_logits"].unsqueeze(1))[..., 0],
        # The token's own tip reading: its cell centre plus its sub-cell offset.
        "tip_point": torch.stack(((ix.float() + 0.5 + off[..., 0]) / w4,
                                  (iy.float() + 0.5 + off[..., 1]) / h4), -1),
        # The dense estimates of the token's dart, from its parent cell.
        "est_tip": pc + at8(out["tip_offset"]),
        "est_flight": pc + at8(out["flight_offset"]),
        "est_centre": pc + at8(out["centre_offset"]),
        "est_direction": d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6),
        "est_extent": at8(out["log_extent"]).exp(),
        "tip_feature": at4(out["tip_features"]),
        "neck_feature": at8(out["neck"]),
    }


class TokenReadout(nn.Module):
    """Stride-4 tokens -> ``query_count`` darts with a confidence each."""

    def __init__(self, cfg) -> None:
        super().__init__()
        d, n, heads = cfg.readout_dim, cfg.query_count, cfg.readout_heads
        self.cfg = cfg
        # tip features, neck features, tip logit, fg logit, own tip offset
        # (2), and the dense estimates relative to the token (tip, flight,
        # centre: 6) and its direction (2).
        c_in = cfg.tip_head_width + cfg.head_width + 2 + 2 + 6 + 2
        self.embed = nn.Sequential(nn.Linear(c_in, d), nn.GELU(),
                                   nn.Linear(d, d))
        self.pos = nn.Linear(2 * 2 * POSITION_OCTAVES, d)
        self.encode = nn.ModuleList(
            _LocalBlock(d, heads, cfg.readout_window)
            for _ in range(cfg.readout_enc_layers))
        self.queries = nn.Parameter(torch.randn(n, d))
        self.self_attn = nn.ModuleList(_Block(d, heads)
                                       for _ in range(cfg.readout_dec_layers))
        self.cross_attn = nn.ModuleList(_Block(d, heads)
                                        for _ in range(cfg.readout_dec_layers))
        self.norm = nn.LayerNorm(d)
        self.dart_q, self.dart_k = nn.Linear(d, d), nn.Linear(d, d)
        self.tip_q, self.tip_k = nn.Linear(d, d), nn.Linear(d, d)
        # One output branch per query.
        self.conf = nn.ModuleList(nn.Linear(d, 1) for _ in range(n))
        self.keep = nn.ModuleList(nn.Linear(d, 1) for _ in range(n))

    def forward(self, tok: dict) -> dict:
        """``tok`` from :func:`gather_tokens`. Returns (B, Q, ...) planes:
        ``logit``/``score`` (confidence), ``tip`` (the landing point, the
        tip pointer's blend), ``tip_hard`` (its single top option's value),
        ``tip_estimate`` (the dart pointer's dense estimate), ``flight``,
        ``centre``, ``direction``, ``extent``, and the pointers' ``dart_attn``
        (B, Q, M) and ``tip_attn`` (B, Q, 1 + M), option 0 being the estimate.

        Runs in float32 whatever the autocast state.
        """
        dev = tok["score"].device.type
        if torch.is_autocast_enabled(dev):
            with torch.autocast(dev, enabled=False):
                return self._forward(
                    {k: v.float() if v.is_floating_point() else v
                     for k, v in tok.items()})
        return self._forward(tok)

    def _forward(self, t: dict) -> dict:
        cfg = self.cfg
        b, m = int(t["score"].shape[0]), int(t["score"].shape[1])
        n = cfg.query_count
        pos = t["pos"]
        rel = lambda p: (p - pos) / DART_SCALE  # noqa: E731
        x = self.embed(torch.cat((
            t["tip_feature"], t["neck_feature"],
            t["tip_logit"].unsqueeze(-1), t["fg_logit"].unsqueeze(-1),
            rel(t["tip_point"]), rel(t["est_tip"]), rel(t["est_flight"]),
            rel(t["est_centre"]), t["est_direction"]), -1))
        x = x + self.pos(_fourier(pos, POSITION_OCTAVES))
        # How much a token is a dart or a tip at all: a soft mask on every
        # attention over the tokens.
        bias = t["score"].clamp(min=MEMBERSHIP_FLOOR).log()          # (B, M)
        for blk in self.encode:
            x = blk(x, t["grid"], bias)

        q = self.queries.unsqueeze(0).expand(b, n, -1)
        none = bias.new_zeros(b, n)
        for sa, ca in zip(self.self_attn, self.cross_attn):
            q = sa(q, q, none)
            q = ca(q, x, bias)
        q = self.norm(q)
        d = int(q.shape[-1])

        dart_logits = (self.dart_q(q) @ self.dart_k(x).transpose(1, 2)
                       / math.sqrt(d) + bias.unsqueeze(1))           # (B, Q, M)
        dart = dart_logits.softmax(-1)

        def mean(v):
            return (dart.unsqueeze(-1) * v.unsqueeze(1)).sum(2)

        est = mean(t["est_tip"])
        tip_bias = torch.sigmoid(t["tip_logit"]).clamp(min=MEMBERSHIP_FLOOR).log()
        tip_logits = (self.tip_q(q) @ self.tip_k(x).transpose(1, 2)
                      / math.sqrt(d) + tip_bias.unsqueeze(1))        # (B, Q, M)
        keep = torch.cat([self.keep[j](q[:, j]) for j in range(n)], -1)
        tip_all = torch.cat((keep.unsqueeze(-1), tip_logits), -1)    # (B, Q, 1+M)
        a = tip_all.softmax(-1)
        landing = (a[..., :1] * est
                   + (a[..., 1:].unsqueeze(-1) * t["tip_point"].unsqueeze(1)).sum(2))
        # The hard reading: the single highest-weighted option's value, the
        # estimate or one token's tip reading, rather than the blend. Read
        # alongside the blend so the two can be compared on device; on held-out
        # frames it is better for visible tips and worse for hidden ones.
        values = torch.cat((est.unsqueeze(2),
                            t["tip_point"].unsqueeze(1).expand(b, n, m, 2)), 2)
        pick = tip_all.argmax(-1)                                    # (B, Q)
        hard = torch.gather(values, 2, pick.view(b, n, 1, 1).expand(b, n, 1, 2))[:, :, 0]

        dsum = mean(t["est_direction"])
        logit = torch.cat([self.conf[j](q[:, j]) for j in range(n)], -1)
        return {
            "logit": logit,
            "score": torch.sigmoid(logit),
            "tip": landing,
            "tip_hard": hard,
            "tip_estimate": est,
            "flight": mean(t["est_flight"]),
            "centre": mean(t["est_centre"]),
            "direction": dsum / dsum.norm(dim=-1, keepdim=True).clamp(min=1e-6),
            "extent": mean(t["est_extent"]),
            "dart_logits": dart_logits,
            "tip_logits": tip_all,
        }


def readout_losses(pred: dict, tok: dict, batch: dict, cfg, scale: float,
                   huber):
    """Stage-3 losses, the matched landing errors in pixels, and the matching.

    Matching and confidence as for the query readout. For each matched
    (query, dart): Huber in pixels on the landing point, the dart pointer's
    estimate and the flight point; the dart pointer's log-mass on the tokens
    whose parent cell belongs to that dart; and the tip pointer's
    cross-entropy against the tip's own cell.

    The tip's own cell, not any of the 3x3 around it: those all point at the
    tip, but the own cell reads it best (~0.9px against ~1.8 for its
    neighbours), and a target satisfied by spreading over the nine left
    isolated darts at 2.5px where the own cell alone gives 1.4. When the own
    cell was not selected as a token the target falls back to the 3x3, and
    when none of those was, to the estimate.
    """
    from darts_model.model.queries import match_queries

    ends = batch["dart_ends"].float()
    f, q, d = match_queries(pred["tip"], pred["score"], ends[..., 0:2],
                            batch["dart_mask"], cfg.query_cost_px,
                            cfg.query_cost_conf, scale)
    target = torch.zeros_like(pred["logit"])
    target[f, q] = 1.0
    losses = {"r_conf": F.binary_cross_entropy_with_logits(pred["logit"],
                                                           target)}
    zero = pred["tip"].sum() * 0.0
    names = ("r_tip", "r_estimate", "r_flight", "r_dart_attn", "r_tip_attn")
    if not f.numel():
        losses.update(dict.fromkeys(names, zero))
        return losses, zero.new_zeros(0), (f, q, d)

    def px(a, b):
        return ((a - b).square().sum(-1) + 1e-12).sqrt() * scale

    gt_tip, gt_flight = ends[f, d, 0:2], ends[f, d, 2:4]
    tip_px = px(pred["tip"][f, q], gt_tip)
    losses["r_tip"] = huber(tip_px, cfg.instance_huber_px).mean()
    losses["r_estimate"] = huber(px(pred["tip_estimate"][f, q], gt_tip),
                                 cfg.instance_huber_px).mean()
    losses["r_flight"] = huber(px(pred["flight"][f, q], gt_flight),
                               cfg.instance_huber_px).mean()

    # Dart pointer: mass on the matched dart's own tokens.
    inst = batch["instance"].reshape(batch["instance"].shape[0], -1)
    own = torch.gather(inst, 1, tok["parent"])[f] == (d + 1).unsqueeze(-1)
    logp = pred["dart_logits"][f, q].log_softmax(-1)
    has = own.any(-1)
    mass = torch.logsumexp(logp.masked_fill(~own, -1e4), -1)
    losses["r_dart_attn"] = (-mass[has]).mean() if bool(has.any()) else zero

    # Tip pointer: the tip's own cell; else the 3x3 around it; else the
    # estimate.
    tip_cell = (gt_tip * tok["grid_size"].float()).floor()
    gap = (tok["grid"][f].float() - tip_cell.unsqueeze(1)).abs()        # (N,M,2)
    own_cell = (gap == 0).all(-1)
    near = (gap <= 1).all(-1)
    logp_tip = pred["tip_logits"][f, q].log_softmax(-1)
    tokens = logp_tip[..., 1:]
    own_mass = torch.logsumexp(tokens.masked_fill(~own_cell, -1e4), -1)
    near_mass = torch.logsumexp(tokens.masked_fill(~near, -1e4), -1)
    tip_mass = torch.where(own_cell.any(-1), own_mass,
                           torch.where(near.any(-1), near_mass, logp_tip[..., 0]))
    losses["r_tip_attn"] = (-tip_mass).mean()
    return losses, tip_px.detach(), (f, q, d)
