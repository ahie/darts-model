"""Per-dart readout: every visible cell of one dart, read together.

The dense head predicts each dart's landing point from every cell on it, but
each cell answers from its own receptive field, and Hough voting averages
those answers rather than combining what the cells see. A flight cell 80px
from the tip can only extrapolate, and the average of eighty extrapolations
is not an inference from the whole dart. This module is the inference from
the whole dart:

1. Candidates: the top-K cells by foreground score, with their features,
   positions and per-cell estimates gathered -- the cells the exported graph
   already keeps.
2. Seeds, one per dart. In training, each ground-truth dart's mean embedding
   over a random subset of its cells. At inference, picked in-graph: the cell
   with the densest neighbourhood of agreeing centre votes, its core's mean
   embedding as the seed, then every cell near that seed suppressed, S times
   over. Fixed S, so the graph has static shapes and exports whole.
3. Soft membership per seed: foreground score times the cell's probability
   of being on that dart rather than on any other dart in the frame or on
   none -- a softmax over the frame's seeds of a Gaussian in embedding
   distance, with "no dart" a fixed option at the push margin, so a cell is
   never counted twice and a cell near no seed counts for none. The width
   matters more than the form: with an embedding not yet at its push margin, a
   neighbour's cells carried 17% of a dart's own mass at sigma 0.75, pulling
   the mean 6px -> 16px; 0.5 roughly halves that, and once means sit the
   push margin apart it is negligible either way. Membership enters every
   attention layer as an additive bias, so it is differentiable in the
   embedding: a cell that leaks into a dart's pool and drags its landing point
   off is pushed out by the landing loss itself.

   Distance is in a joint space: the embedding, plus the cell's predicted box
   centre in units of ``membership_centre_px``. Position already separates
   darts that are far apart -- the Hough readout never needed help there --
   and leaving that to the embedding alone merged two darts 300px apart whose
   embeddings had not yet been pushed apart. The embedding decides where the
   centres crowd together, which is the case it exists for.
4. Per dart, the M most-member cells become tokens -- features, position
   relative to the dart's centroid, and the cell's own estimates -- and pass
   through one self-attention layer (the shaft line reaches the tip cells,
   the tip reaches the flight cells) and cross-attention from a query seeded
   with the dart's pooled tokens.
5. The landing point is the membership-weighted mean of the cells' own
   estimates plus a learned correction that starts at zero, so a model this
   is added to starts exactly at its soft-membership Hough readout.

No learned query slots and no matcher: which dart a slot describes is fixed by
its seed, not by an assignment that can drift.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Typical dart half-length in normalised image coordinates (darts project to
#: 18-268px at 1024, a half-length of 0.009-0.131). Relative positions and
#: offsets are divided by it, so the tokens see O(1) geometry.
DART_SCALE = 0.07
#: Output unit of the learned correction, normalised: ~10px at 1024. The
#: correction layer starts at zero; this only sets its conditioning.
DELTA_SCALE = 0.01
#: Octaves of the Fourier position encoding. At DART_SCALE the finest is
#: ~2.2px at 1024, below the cell size at stride 8.
FOURIER_OCTAVES = 6
#: Floor on a membership weight inside log(): a cell with no membership is
#: masked by a bias of about -9, not -inf, so no attention row is ever empty.
MEMBERSHIP_FLOOR = 1e-4


def candidate_count(out_stride: int) -> int:
    """Cells kept per frame at ``out_stride``: 1024 at stride 8, scaled with
    cell area. A dart covers ~83 cells at stride 8, so three darts are ~250
    and nine would still fit."""
    return max(1, round(1024 * (8 / out_stride) ** 2))


def gather_candidates(out: dict, k: int, centre_scale: float = 0.0,
                      context_radius: int = 0) -> dict:
    """The top-``k`` cells by foreground logit, as (B, k, C) planes.

    Positions and per-cell estimates are absolute, in normalised image
    coordinates, by the same ``(i + 0.5) / n`` cell-centre convention as
    ``cell_centres``. ``identity``, when the field has an embedding, is the
    joint space seeds and membership work in: the embedding, then the
    predicted centre over ``centre_scale``. ``feature`` is the cell's token
    projection, followed by the 2x2 stride-4 ``fine`` vectors under it when
    the field has them. With ``context_radius``, ``context`` is the
    foreground score max-pooled over that many cells around each one.
    Gradients flow through every estimate, so a loss on the per-dart readout
    reaches the per-cell heads that fed it.
    """
    fg = out["fg_logits"].float()
    # Python ints, not traced sizes: the graph is specialised to one input
    # size, and CoreML cannot lower a size that stays a tensor.
    b, h, w = (int(x) for x in fg.shape)
    n = h * w
    k = min(k, n)
    logit, idx = torch.topk(fg.reshape(b, n), k, dim=1, sorted=True)

    def g(name: str) -> torch.Tensor:
        t = out[name].float()
        c = int(t.shape[1])
        t = torch.gather(t.reshape(b, c, n), 2, idx.unsqueeze(1).expand(b, c, k))
        return t.transpose(1, 2)

    ix = (idx % w).float()
    iy = torch.div(idx, w, rounding_mode="floor").float()
    pos = torch.stack(((ix + 0.5) / w, (iy + 0.5) / h), dim=-1)
    d = g("direction")
    centre = pos + g("centre_offset")
    feature = g("tokens")
    if "fine" in out:
        # Space-to-depth: the 2x2 stride-4 vectors under each cell, stacked
        # as channels of that cell.
        f = out["fine"].float()
        cf = int(f.shape[1])
        f = f[..., :2 * h, :2 * w].reshape(b, cf, h, 2, w, 2)
        f = f.permute(0, 1, 3, 5, 2, 4).reshape(b, 4 * cf, h * w)
        f = torch.gather(f, 2, idx.unsqueeze(1).expand(b, 4 * cf, k))
        feature = torch.cat((feature, f.transpose(1, 2)), -1)
    cand = {
        "score": torch.sigmoid(logit),
        "pos": pos,
        "centre": centre,
        "tip": pos + g("tip_offset"),
        "flight": pos + g("flight_offset"),
        # Normalised here as in decode_boxes: outside the dense loss's path.
        "direction": d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6),
        "extent": g("log_extent").exp(),
        "feature": feature,
    }
    if context_radius > 0:
        r = int(context_radius)
        ctx = F.max_pool2d(torch.sigmoid(fg).unsqueeze(1), 2 * r + 1, stride=1,
                           padding=r).reshape(b, n)
        cand["context"] = torch.gather(ctx, 1, idx)
    if "embedding" in out:
        emb = g("embedding")
        cand["embedding"] = emb
        cand["identity"] = torch.cat((emb, centre / centre_scale), dim=-1)
    return cand


def _sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise squared distance, (..., N, C) x (..., M, C) -> (..., N, M).

    Explicit broadcasting rather than ``torch.cdist``, which neither export
    backend lowers.
    """
    return (a.unsqueeze(-2) - b.unsqueeze(-3)).square().sum(-1)


def select_seeds(cand: dict, n_slots: int, cfg, image_px: float):
    """Pick up to ``n_slots`` darts in-graph. Returns ``(mu, score)``.

    ``mu`` (B, S, I) is each slot's seed in the identity space and ``score``
    (B, S) its
    vote density: the foreground-weighted count of kept cells whose predicted
    box centre lies within ``seed_radius_px`` of the seed cell's -- the same
    quantity the Hough accumulator peaks on. A slot is a dart when the score
    clears ``peak_min_votes * fg_threshold``, the Hough floor.

    Each pick suppresses the cells it accounts for: those voting within the
    radius of its centre, and those whose identity lies within
    ``embed_push_margin`` of its seed. The second is the one that matters; a
    dart's cells whose centre votes strayed are still its cells. Unrolled
    ``n_slots`` times, so the graph is static.
    """
    score = cand["score"]
    voting = (score >= cfg.fg_threshold).float()
    wgt = score * voting                                       # (B, K)
    r2 = (cfg.seed_radius_px / image_px) ** 2
    near = (_sq_dist(cand["centre"], cand["centre"]) <= r2).float()  # (B,K,K)
    density = (near * wgt.unsqueeze(1)).sum(-1)
    density = torch.where(voting > 0, density, torch.full_like(density, -1.0))
    emb = cand["identity"]
    margin2 = cfg.embed_push_margin ** 2
    k = int(density.shape[1])
    mus, scores = [], []
    for _ in range(n_slots):
        best, i = density.max(dim=1)                            # (B,)
        core = torch.gather(near, 1, i.view(-1, 1, 1).expand(-1, 1, k))[:, 0]
        core = core * wgt                                       # (B, K)
        mu = (core.unsqueeze(-1) * emb).sum(1) / core.sum(1, keepdim=True).clamp(
            min=1e-6)
        mus.append(mu)
        scores.append(best)
        close = ((emb - mu.unsqueeze(1)).square().sum(-1) <= margin2).float()
        # A sum rather than a boolean or: CoreML has no converter for the
        # latter.
        density = torch.where((core > 0).float() + close > 0,
                              torch.full_like(density, -1.0), density)
    return torch.stack(mus, 1), torch.stack(scores, 1)


def ground_truth_seeds(out: dict, instance: torch.Tensor, n_slots: int,
                       centre_scale: float,
                       subset: tuple[float, float] | None = None):
    """Each ground-truth dart's mean identity over its labelled cells.

    ``instance`` (B, h, w) holds slot + 1 per dart cell. Returns ``(mu,
    present)``, (B, S, I) and (B, S), in the identity space of
    :func:`gather_candidates`. With ``subset``, each dart's mean is
    over a random fraction of its cells drawn from that range, as an
    inference seed is the mean of a peak's core rather than of the whole
    dart; a draw that keeps no cell falls back to all of them.
    """
    h, w = (int(x) for x in instance.shape[-2:])
    ys, xs = torch.meshgrid((torch.arange(h, device=instance.device) + 0.5) / h,
                            (torch.arange(w, device=instance.device) + 0.5) / w,
                            indexing="ij")
    centre = torch.stack((xs, ys))[None] + out["centre_offset"].float()
    emb = torch.cat((out["embedding"].float(), centre / centre_scale), dim=1)
    member = instance.unsqueeze(1) == torch.arange(
        1, n_slots + 1, device=instance.device).view(1, -1, 1, 1)  # (B,S,h,w)
    present = member.flatten(2).any(-1)
    m = member
    if subset is not None:
        lo, hi = subset
        frac = torch.empty(member.shape[:2] + (1, 1),
                           device=emb.device).uniform_(lo, hi)
        drawn = member & (torch.rand(member.shape, device=emb.device) < frac)
        empty = ~drawn.flatten(2).any(-1)
        m = torch.where(empty[..., None, None], member, drawn)
    m = m.float()
    mu = torch.einsum("behw,bshw->bse", emb, m) / m.sum((2, 3)).clamp(
        min=1.0).unsqueeze(-1)
    return mu, present


class _Attention(nn.Module):
    """Multi-head attention with an additive per-key bias.

    Written out rather than ``nn.MultiheadAttention``, whose fused kernel
    does not survive tracing for either export backend.
    """

    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)

    def forward(self, q_in, kv_in, bias):
        n, lq, d = (int(x) for x in q_in.shape)
        lk = int(kv_in.shape[1])
        h, dh = self.heads, d // self.heads
        q = self.q(q_in).view(n, lq, h, dh).transpose(1, 2)
        k, v = self.kv(kv_in).view(n, lk, 2, h, dh).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        a = q @ k.transpose(-1, -2) / math.sqrt(dh) + bias[:, None, None, :]
        o = (a.softmax(-1) @ v).transpose(1, 2).reshape(n, lq, d)
        return self.o(o)


class _Block(nn.Module):
    """Pre-norm attention then MLP, both residual."""

    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.nk = nn.LayerNorm(d)
        self.attn = _Attention(d, heads)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                 nn.Linear(2 * d, d))

    def forward(self, q, kv, bias):
        q = q + self.attn(self.n1(q), self.nk(kv), bias)
        return q + self.mlp(self.n2(q))


def _fourier(x: torch.Tensor) -> torch.Tensor:
    f = (2.0 ** torch.arange(FOURIER_OCTAVES, device=x.device,
                             dtype=x.dtype)) * math.pi
    a = (x.unsqueeze(-1) * f).flatten(-2)
    return torch.cat((a.sin(), a.cos()), dim=-1)


class InstanceHead(nn.Module):
    """Candidates and seeds -> one landing point and flight point per seed."""

    def __init__(self, cfg) -> None:
        super().__init__()
        d = cfg.instance_dim
        self.cfg = cfg
        from darts_model.model.detector import token_feature_dim
        self.feat = nn.Linear(token_feature_dim(cfg), d)
        self.pos = nn.Linear(2 * 2 * FOURIER_OCTAVES, d)
        # Relative tip, relative flight, direction.
        self.geom = nn.Linear(6, d)
        self.encode = _Block(d, cfg.instance_heads)
        self.decode = nn.ModuleList(_Block(d, cfg.instance_heads)
                                    for _ in range(cfg.instance_layers))
        self.norm = nn.LayerNorm(d)
        self.delta = nn.Linear(d, 4)
        # Zero: the readout starts as the soft-membership mean of the cells'
        # own estimates, i.e. no worse than the Hough readout it extends.
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, cand: dict, mu: torch.Tensor,
                slot_ok: torch.Tensor) -> dict:
        """``cand`` from :func:`gather_candidates`, ``mu`` (B, S, I), and
        ``slot_ok`` (B, S): which seeds are darts. Only those compete for
        cells; a slot that is not a dart still reads out, from what is left.

        Returns (B, S, ...) planes: ``tip`` and ``flight`` (the readout),
        ``base_tip`` (the mean it corrects), the box ``centre``, ``direction``
        and ``extent`` as membership-weighted means, and ``support``, the
        summed membership.

        Runs in float32 whatever the autocast state: it averages positions
        whose bf16 step is ~0.5px.
        """
        dev = mu.device.type
        if torch.is_autocast_enabled(dev):
            with torch.autocast(dev, enabled=False):
                return self._forward({k: v.float() for k, v in cand.items()},
                                     mu.float(), slot_ok)
        return self._forward(cand, mu, slot_ok)

    def _forward(self, cand: dict, mu: torch.Tensor,
                 slot_ok: torch.Tensor) -> dict:
        cfg = self.cfg
        b, s = int(mu.shape[0]), int(mu.shape[1])
        k = int(cand["score"].shape[1])
        m = min(cfg.instance_tokens, k)

        # Foreground gates membership but is not trained by it: whether a cell
        # is a dart is the fg head's question, which dart is the embedding's.
        voting = (cand["score"] >= cfg.fg_threshold).float()
        fg = (cand["score"] * voting).detach()
        d2 = (cand["identity"].unsqueeze(1) - mu.unsqueeze(2)).square().sum(-1)
        two_var = 2.0 * cfg.membership_sigma ** 2
        logit = -d2 / two_var                                   # (B, S, K)
        # Seeds that are not darts take no part. A large negative rather than
        # -inf, so the softmax stays finite on every backend.
        logit = torch.where(slot_ok.unsqueeze(-1) > 0, logit,
                            torch.full_like(logit, -1e4))
        none = torch.full_like(logit[:, :1], -cfg.embed_push_margin ** 2
                               / two_var)
        p = torch.cat((logit, none), dim=1).softmax(dim=1)[:, :s]
        member = fg.unsqueeze(1) * p                            # (B, S, K)
        wt, ti = torch.topk(member, m, dim=-1)                  # (B, S, M)

        def take(name: str) -> torch.Tensor:
            x = cand[name]
            c = int(x.shape[-1])
            x = x.unsqueeze(1).expand(b, s, k, c)
            return torch.gather(x, 2, ti.unsqueeze(-1).expand(b, s, m, c))

        sw = wt.sum(-1, keepdim=True).clamp(min=1e-6)           # (B, S, 1)

        def mean(x: torch.Tensor) -> torch.Tensor:
            return (wt.unsqueeze(-1) * x).sum(2) / sw

        pos, tip, flight = take("pos"), take("tip"), take("flight")
        direction = take("direction")
        centroid = mean(pos)
        base_tip, base_flight = mean(tip), mean(flight)
        rel = lambda p: (p - centroid.unsqueeze(2)) / DART_SCALE  # noqa: E731

        tokens = (self.feat(take("feature"))
                  + self.pos(_fourier(rel(pos)))
                  + self.geom(torch.cat((rel(tip), rel(flight), direction), -1)))
        tokens = tokens.view(b * s, m, -1)
        bias = wt.clamp(min=MEMBERSHIP_FLOOR).log().view(b * s, m)

        x = self.encode(tokens, tokens, bias)
        w = wt.view(b * s, m, 1)
        q = (w * x).sum(1, keepdim=True) / sw.view(b * s, 1, 1)
        for blk in self.decode:
            q = blk(q, x, bias)
        delta = self.delta(self.norm(q)).view(b, s, 4) * DELTA_SCALE

        dsum = (wt.unsqueeze(-1) * direction).sum(2)
        return {
            "tip": base_tip + delta[..., 0:2],
            "flight": base_flight + delta[..., 2:4],
            "base_tip": base_tip,
            "centre": mean(take("centre")),
            "direction": dsum / dsum.norm(dim=-1, keepdim=True).clamp(min=1e-6),
            "extent": mean(take("extent")),
            "support": sw[..., 0],
        }
