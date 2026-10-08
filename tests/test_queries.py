"""The query readout: pointing, query-conditioned correction, Hungarian
matching, its training path and its export."""
from dataclasses import dataclass

import pytest
import torch

from darts_model.export.common import (
    SLOT_OUTPUTS, effective_topk, output_names, readout_contract,
)
from darts_model.export.coreml import OUTPUTS, ExportWrapper
from darts_model.model.detector import (
    DenseDartConfig, DenseDartLitModule, _huber,
)
from darts_model.model.queries import QueryHead, match_queries, query_losses

from test_dense_dart import _batch, _cfg


@dataclass
class _Cfg:
    token_dim: int = 8
    fine_tokens: bool = False
    fine_dim: int = 0
    instance_dim: int = 16
    instance_heads: int = 4
    instance_layers: int = 1
    query_count: int = 2
    query_tokens: int = 64


def _candidates():
    """Two darts of 10 cells, tips at x=0.20 and x=0.70. Four of dart A's
    cells next to its tip carry dart B's landing point as their own estimate
    -- the frame-5 error."""
    n = 20
    pos = torch.zeros(1, n, 2)
    pos[0, :10] = torch.tensor([0.25, 0.25]) + torch.randn(10, 2) * 0.01
    pos[0, 10:] = torch.tensor([0.75, 0.75]) + torch.randn(10, 2) * 0.01
    tip = torch.zeros(1, n, 2)
    tip[0, :10] = torch.tensor([0.20, 0.25])
    tip[0, 10:] = torch.tensor([0.70, 0.75])
    tip[0, :4] = torch.tensor([0.70, 0.75])        # confused cells of A
    return {
        "score": torch.full((1, n), 0.9), "pos": pos, "centre": pos.clone(),
        "tip": tip, "flight": tip + 0.1,
        "direction": torch.tensor([1.0, 0.0]).expand(1, n, 2).clone(),
        "extent": torch.full((1, n, 2), 0.05),
        "feature": torch.randn(1, n, 8),
    }


def test_a_fresh_head_points_at_the_weighted_mean_of_the_dense_estimates():
    torch.manual_seed(0)
    cand = _candidates()
    out = QueryHead(_Cfg())(cand)
    assert torch.allclose(out["tip"], out["base_tip"])
    want = (out["attn"].unsqueeze(-1) * cand["tip"].unsqueeze(1)).sum(2)
    assert torch.allclose(out["tip"], want, atol=1e-6)
    assert torch.allclose(out["attn"].sum(-1), torch.ones(1, 2))


def test_the_query_conditioned_correction_overrides_a_wrong_cell():
    """Cells whose own estimate names the neighbour's tip are corrected
    per cell, given the query: dart A is read out at A's tip although four of
    its cells say otherwise."""
    torch.manual_seed(0)
    cand = _candidates()
    head = QueryHead(_Cfg())
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    gt = torch.tensor([[[0.20, 0.25], [0.70, 0.75]]])
    for _ in range(300):
        out = head(cand)
        loss = (out["tip"] - gt).abs().sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    out = head(cand)
    assert float((out["tip"] - gt).norm(dim=-1).max()) * 256 < 1.0


def test_matching_takes_the_nearest_query_and_values_confidence():
    tip = torch.tensor([[[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]]])
    score = torch.tensor([[0.9, 0.9, 0.9]])
    gt = torch.tensor([[[0.79, 0.8], [0.21, 0.2], [0.0, 0.0]]])
    ok = torch.tensor([[True, True, False]])
    f, q, d = match_queries(tip, score, gt, ok, 0.1, 1.0, 256.0)
    assert sorted(zip(q.tolist(), d.tolist())) == [(0, 1), (2, 0)]
    # Two queries equally far: the confident one is taken.
    tip = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])
    f, q, d = match_queries(tip, torch.tensor([[0.1, 0.9]]),
                            torch.tensor([[[0.5, 0.5]]]),
                            torch.tensor([[True]]), 0.1, 1.0, 256.0)
    assert q.tolist() == [1]


def _qcfg(**kw):
    base = dict(query_head=True, token_dim=8, instance_dim=16,
                instance_layers=1, fg_threshold=0.3, peak_min_votes=4)
    base.update(kw)
    return _cfg(**base)


def test_losses_are_finite_on_frames_without_darts():
    torch.manual_seed(0)
    lit = DenseDartLitModule(_qcfg())
    batch = _batch()
    batch["dart_mask"][:] = False
    with torch.no_grad():
        out = lit.model(batch["image"].float())
    losses, err = query_losses(lit.model.read_darts(out), batch, lit.config,
                               256.0, _huber)
    assert all(torch.isfinite(v) for v in losses.values())
    assert err.numel() == 0
    assert float(losses["q_tip"]) == 0.0


def test_config_takes_one_per_dart_readout():
    with pytest.raises(ValueError, match="alternative"):
        DenseDartConfig(query_head=True, instance_head=True, embed_dim=4)


def test_the_query_readout_trains_end_to_end():
    """Fit one batch; the confident queries are the two darts, at their
    landing points, and the spare query is not confident."""
    torch.manual_seed(0)
    lit = DenseDartLitModule(_qcfg(instance_weight=0.05,
                                   instance_flight_weight=0.02,
                                   query_conf_weight=0.5))
    lit.train()
    lit.log = lambda *a, **k: None
    batch = _batch()
    opt = torch.optim.Adam(lit.parameters(), lr=3e-3)
    for _ in range(400):
        loss = lit._step(batch, "train")
        opt.zero_grad()
        loss.backward()
        opt.step()
    lit.eval()
    with torch.no_grad():
        r = lit.model.read_darts(lit.model(batch["image"].float()))
    for b in range(2):
        keep = r["score"][b] >= lit.config.query_conf_threshold
        assert int(keep.sum()) == 2
        tips = r["tip"][b][keep] * 256
        gt = batch["dart_ends"][b, :2, 0:2] * 256
        assert float(torch.cdist(gt, tips).amin(1).max()) < 3.0


def test_validation_ranks_the_queries_and_logs_hough_alongside():
    torch.manual_seed(0)
    lit = DenseDartLitModule(_qcfg()).eval()
    logged = {}
    lit.log = lambda n, v, *a, **k: logged.__setitem__(n, float(v))
    with torch.no_grad():
        lit.model.fg_head.bias.fill_(10.0)
        lit._step(_batch(), "val")
    lit.on_validation_epoch_end()
    for name in ("val/landing_px_error_penalised",
                 "val/landing_px_error_penalised_centre_claim",
                 "val/landing_px_error_gt_seeds", "val/q_conf_loss"):
        assert name in logged, name


def test_a_finetune_may_add_the_queries_to_a_plain_detector(tmp_path):
    trained = DenseDartLitModule(_cfg())
    path = tmp_path / "plain.ckpt"
    torch.save({"state_dict": trained.state_dict()}, path)
    lit = DenseDartLitModule(_qcfg(init_weights=str(path)))
    assert torch.equal(lit.model.box_head.weight, trained.model.box_head.weight)


def _wrapper():
    torch.manual_seed(0)
    cfg = _qcfg(head_width=16, head_depth=1, kp_head_width=16, kp_head_depth=1)
    lit = DenseDartLitModule(cfg).eval()
    with torch.no_grad():
        for p in lit.model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        lit.model.fg_head.bias.fill_(0.0)
    return cfg, lit.model, ExportWrapper(
        lit.model, topk=effective_topk(256, cfg.out_stride)).eval()


def test_the_exported_graph_emits_the_queries():
    cfg, net, wrapper = _wrapper()
    image = torch.randint(0, 256, (1, 3, 256, 256)).float()
    with torch.no_grad():
        outs = dict(zip(output_names(cfg), wrapper(image)))
        want = net.read_darts(net((image - wrapper.norm_mean)
                                  / wrapper.norm_std))
        traced = torch.jit.trace(wrapper, image, strict=False)(image)
        exported = torch.export.export(wrapper, (image,)).module()(image)
    assert list(outs) == OUTPUTS + SLOT_OUTPUTS
    assert outs["slot_score"].shape == (1, cfg.query_count)
    assert torch.allclose(outs["slot_score"], want["score"])
    assert torch.allclose(outs["slot_tip"], want["tip"])
    for a, b, c in zip(outs.values(), traced, exported):
        assert torch.allclose(a, b, atol=1e-5)
        assert torch.allclose(a, c, atol=1e-5)


def test_the_contract_reads_slots_by_confidence():
    cfg = _qcfg()
    c = readout_contract(cfg, 256, 1024)
    assert c["darts"]["slots"] == cfg.query_count
    assert c["darts"]["min_score"] == pytest.approx(cfg.query_conf_threshold)


def test_a_detached_readout_cannot_move_the_dense_model():
    """During the readout warmup the query losses train the head and its
    token projection, and nothing upstream."""
    torch.manual_seed(0)
    lit = DenseDartLitModule(_qcfg())
    lit.log = lambda *a, **k: None
    with torch.no_grad():
        lit.model.fg_head.bias.fill_(10.0)
    lit.model.readout_detached = True
    batch = _batch()
    out = lit.model(batch["image"].float())
    losses, _ = query_losses(lit.model.read_darts(out), batch, lit.config,
                             256.0, _huber)
    (losses["q_tip"] + losses["q_conf"]).backward()
    for name, p in lit.model.named_parameters():
        if name.startswith(("query_head.", "token_proj.")):
            continue
        assert p.grad is None or float(p.grad.abs().max()) == 0.0, name
    assert lit.model.token_proj.weight.grad is not None


def test_the_readout_trains_in_its_own_optimizer_group():
    lit = DenseDartLitModule(_qcfg(readout_lr_factor=10.0, steps_per_epoch=10))
    opt = lit.configure_optimizers()["optimizer"]
    lr = {g["name"]: g["lr"] for g in opt.param_groups}
    assert lr["readout"] == pytest.approx(10 * lr["head"])
    names = {id(p): n for n, p in lit.model.named_parameters()}
    readout = [names[id(p)] for g in opt.param_groups
               if g["name"].startswith("readout") for p in g["params"]]
    assert readout and all(n.startswith(("query_head.", "token_proj."))
                           for n in readout)


# ------------------------------------------------------- richer tokens

def _rich(**kw):
    base = dict(token_inputs="neck+trunk", token_dim=16, fine_tokens=True,
                fine_dim=4, context_radius=1)
    base.update(kw)
    return _qcfg(**base)


def test_the_fine_detail_is_the_2x2_stride4_block_under_each_cell():
    from darts_model.model.instance import gather_candidates
    b, h, w = 1, 4, 4
    out = {"fg_logits": torch.randn(b, h, w),
           "tokens": torch.randn(b, 3, h, w),
           "fine": torch.arange(2 * 4 * h * w, dtype=torch.float).view(
               b, 2, 2 * h, 2 * w)}
    for name in ("centre_offset", "tip_offset", "flight_offset", "direction",
                 "log_extent"):
        out[name] = torch.zeros(b, 2, h, w)
    cand = gather_candidates(out, h * w)
    flat = torch.topk(out["fg_logits"].reshape(b, -1), h * w).indices[0]
    for j, i in enumerate(flat.tolist()):
        iy, ix = divmod(i, w)
        block = out["fine"][0, :, 2 * iy:2 * iy + 2, 2 * ix:2 * ix + 2]
        assert torch.equal(cand["feature"][0, j, 3:], block.reshape(-1))
        assert torch.equal(cand["feature"][0, j, :3], out["tokens"][0, :, iy, ix])


def test_context_cells_inform_the_queries_but_are_never_pointed_at():
    """A cell just outside the silhouette enters the attention through the
    max-pooled score, but the pointer, biased by the silhouette's own score,
    gives it no weight."""
    from darts_model.model.instance import gather_candidates
    torch.manual_seed(0)
    cfg = _rich()
    b, h, w = 1, 8, 8
    fg = torch.full((b, h, w), -12.0)
    fg[0, 3:5, 3:5] = 6.0                               # the silhouette
    out = {"fg_logits": fg, "tokens": torch.randn(b, cfg.token_dim, h, w),
           "fine": torch.randn(b, cfg.fine_dim, 2 * h, 2 * w)}
    for name in ("centre_offset", "tip_offset", "flight_offset", "direction",
                 "log_extent"):
        out[name] = torch.randn(b, 2, h, w) * 0.01
    cand = gather_candidates(out, h * w, context_radius=1)
    ring = (cand["context"][0] > 0.5) & (cand["score"][0] < 0.01)
    assert int(ring.sum()) == 16 - 4                    # the 4x4 ring round 2x2
    r = QueryHead(cfg)(cand)
    # The pointer's weights are over the tokens re-ranked by context.
    m = r["attn"].shape[-1]
    order = torch.topk(cand["context"], m, dim=1).indices[0]
    ring_tokens = ring[order]
    assert float(r["attn"][0][:, ring_tokens].sum(-1).max()) < 1e-3


def test_rich_tokens_train_and_export():
    torch.manual_seed(0)
    cfg = _rich(head_width=16, head_depth=1, kp_head_width=16, kp_head_depth=1)
    lit = DenseDartLitModule(cfg)
    lit.log = lambda *a, **k: None
    batch = _batch()
    loss = lit._step(batch, "train")
    loss.backward()
    assert lit.model.fine_proj.weight.grad is not None
    lit.eval()
    wrapper = ExportWrapper(lit.model, topk=effective_topk(256, 8)).eval()
    image = torch.randint(0, 256, (1, 3, 256, 256)).float()
    with torch.no_grad():
        want = wrapper(image)
        traced = torch.jit.trace(wrapper, image, strict=False)(image)
        exported = torch.export.export(wrapper, (image,)).module()(image)
    for a, b, c in zip(want, traced, exported):
        assert torch.allclose(a, b, atol=1e-5)
        assert torch.allclose(a, c, atol=1e-5)


def test_the_detached_readout_keeps_its_own_projections_trainable():
    torch.manual_seed(0)
    lit = DenseDartLitModule(_rich())
    with torch.no_grad():
        lit.model.fg_head.bias.fill_(10.0)
    lit.model.readout_detached = True
    batch = _batch()
    out = lit.model(batch["image"].float())
    losses, _ = query_losses(lit.model.read_darts(out), batch, lit.config,
                             256.0, _huber)
    (losses["q_tip"] + losses["q_conf"]).backward()
    for name, p in lit.model.named_parameters():
        if name.startswith(("query_head.", "token_proj.", "fine_proj.")):
            continue
        assert p.grad is None or float(p.grad.abs().max()) == 0.0, name
    assert lit.model.fine_proj.weight.grad is not None


def test_rich_token_settings_are_validated():
    with pytest.raises(ValueError, match="out_stride 8"):
        DenseDartConfig(fine_tokens=True, out_stride=4)
    with pytest.raises(ValueError, match="token_inputs"):
        DenseDartConfig(token_inputs="backbone")
