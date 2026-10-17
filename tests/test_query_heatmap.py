"""The query heatmap readout: decoding, targets, losses, freezing."""
import pytest
import torch

from darts_model.model.detector import DenseDartConfig, DenseDartLitModule
from darts_model.model.query_heatmap import (
    gaussian_targets, heatmap_losses, local_softargmax,
)

from test_dense_dart import _batch, _cfg


def _hcfg(**kw):
    base = dict(predict_tip_heatmap=True, tip_stride=4, tip_head_width=8,
                tip_sigma=1.0, tip_offset_radius=1,
                tip_offset_weighting="gaussian", heatmap_readout=True,
                heatmap_dim=32, heatmap_heads=4, heatmap_layers=2,
                heatmap_pixel_dim=16,
                fg_threshold=0.3, peak_min_votes=4)
    base.update(kw)
    return _cfg(**base)


def test_the_window_soft_argmax_reads_a_sub_cell_peak():
    h, w = 16, 16
    ys, xs = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5,
                            indexing="ij")
    x0, y0 = 5.3, 7.8
    logits = (-((xs - x0) ** 2 + (ys - y0) ** 2) / (2 * 0.6 ** 2)).flatten()
    # A distant, strong mode outside the window must not pull it.
    logits.view(h, w)[1, 14] = 0.0
    got = local_softargmax(logits, torch.tensor(7), torch.tensor(5), (h, w), 2)
    assert torch.allclose(got, torch.tensor([x0, y0]), atol=0.05)


def test_the_window_takes_no_weight_off_the_grid():
    flat = torch.zeros(16)                      # a flat 4x4 map
    got = local_softargmax(flat, torch.tensor(0), torch.tensor(0), (4, 4), 2)
    # Only the 3x3 inside the grid at the corner counts: their mean is 1.5.
    assert torch.allclose(got, torch.tensor([1.5, 1.5]))


def test_targets_are_distributions_centred_on_the_exact_point():
    ends = torch.tensor([[[0.33, 0.61], [0.5, 0.5]]])
    mask = torch.tensor([[True, False]])
    t = gaussian_targets(ends, mask, (32, 32), 1.0)
    assert torch.allclose(t[0, 0].sum(), torch.tensor(1.0))
    assert float(t[0, 1].abs().sum()) == 0.0
    ys, xs = torch.meshgrid(torch.arange(32) + 0.5, torch.arange(32) + 0.5,
                            indexing="ij")
    mean = torch.stack(((t[0, 0] * xs.flatten()).sum(),
                        (t[0, 0] * ys.flatten()).sum()))
    assert torch.allclose(mean, torch.tensor([0.33 * 32, 0.61 * 32]), atol=1e-3)


def test_config_is_validated():
    with pytest.raises(ValueError, match="alternative readouts"):
        DenseDartConfig(heatmap_readout=True, token_readout=True,
                        predict_tip_heatmap=True, tip_stride=4)
    with pytest.raises(ValueError, match="heatmap_readout requires"):
        DenseDartConfig(heatmap_readout=True, predict_tip_heatmap=True,
                        tip_stride=8)


def test_losses_are_finite_on_frames_without_darts():
    torch.manual_seed(0)
    lit = DenseDartLitModule(_hcfg())
    batch = _batch()
    batch["dart_mask"][:] = False
    with torch.no_grad():
        out = lit.model(batch["image"].float())
    losses, err, _, stats = heatmap_losses(lit.model.heatmap_readout(out),
                                           batch, lit.config, 256.0)
    assert stats["on_peak"].numel() == 0
    assert all(torch.isfinite(v) for v in losses.values())
    assert err.numel() == 0


def test_the_heatmap_readout_trains_end_to_end():
    torch.manual_seed(0)
    lit = DenseDartLitModule(_hcfg())
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
        gt = batch["dart_ends"][b, :2, 0:2] * 256
        for key in ("tip", "tip_hard"):
            tips = r[key][b][keep] * 256
            assert float(torch.cdist(gt, tips).amin(1).max()) < 1.5, key


def test_a_frozen_dense_model_does_not_move(tmp_path):
    torch.manual_seed(0)
    trained = DenseDartLitModule(_hcfg(heatmap_readout=False))
    path = tmp_path / "stage2.ckpt"
    torch.save({"state_dict": trained.state_dict()}, path)
    lit = DenseDartLitModule(_hcfg(freeze_dense=True, init_weights=str(path)))
    lit.log = lambda *a, **k: None
    trainable = {n for n, p in lit.model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("heatmap_readout.") for n in trainable)
    before = {n: p.detach().clone() for n, p in lit.model.named_parameters()}
    opt = torch.optim.Adam([p for p in lit.parameters() if p.requires_grad],
                           lr=1e-2)
    loss = lit._step(_batch(), "train")
    opt.zero_grad()
    loss.backward()
    opt.step()
    for n, p in lit.model.named_parameters():
        if n.startswith("heatmap_readout."):
            continue
        assert torch.equal(p, before[n]), n


def test_validation_logs_both_decodes_against_stage_two():
    torch.manual_seed(0)
    lit = DenseDartLitModule(_hcfg(tip_assign="hungarian")).eval()
    logged = {}
    lit.log = lambda n, v, *a, **k: logged.__setitem__(n, float(v))
    with torch.no_grad():
        lit._step(_batch(), "val")
    lit.on_validation_epoch_end()
    for name in ("val/landing_px_error_penalised",
                 "val/landing_px_error_penalised_heatmaps_offset",
                 "val/landing_px_error_penalised_centre_claim",
                 "val/landing_px_error_penalised_centre_claim_assigned",
                 "val/h_kl_loss", "val/h_l1_loss",
                 "val/heatmap_on_peak_share"):
        assert name in logged, name


def test_the_landing_l1_counts_only_queries_peaking_on_their_tip():
    """A query whose argmax is on another dart's tip has a decode window that
    cannot reach its own tip; the L1 leaves it to the KL and the matching."""
    from types import SimpleNamespace

    h = w = 16
    cfg = SimpleNamespace(tip_sigma=1.0, heatmap_window=3, query_cost_px=0.1,
                          query_cost_conf=1.0, heatmap_cost_ce=1.0)
    ends = torch.tensor([[[5.3 / w, 5.6 / h, 0.0, 0.0],
                          [12.4 / w, 11.2 / h, 0.0, 0.0]]])
    batch = {"dart_ends": ends, "dart_mask": torch.tensor([[True, True]])}
    heat = torch.full((1, 2, h * w), -10.0)
    heat[0, 0, 5 * w + 5] = 5.0      # query 0 peaks on dart 0's tip cell
    heat[0, 1, 5 * w + 6] = 5.0      # query 1 also peaks next to dart 0
    tip = torch.tensor([[[5.5 / w, 5.5 / h], [6.5 / w, 5.5 / h]]])
    pred = {"heat_logits": heat, "grid_size": (h, w),
            "logit": torch.zeros(1, 2), "score": torch.full((1, 2), 0.5),
            "tip": tip, "tip_hard": tip,
            "cell": torch.tensor([[[5, 5], [6, 5]]])}
    losses, _, (f, q, d), stats = heatmap_losses(pred, batch, cfg, float(w * 4))
    # Query 1 is matched to dart 1 but peaks a dart away: off its peak.
    by_dart = {int(dd): bool(o) for dd, o in zip(d, stats["on_peak"])}
    assert by_dart == {0: True, 1: False}
    # So the L1 is query 0's decode error alone: |0.2| + |0.1| cells, 4px each.
    assert torch.allclose(losses["h_l1"], torch.tensor(1.2), atol=1e-4)


def test_the_queries_can_attend_to_the_stride_8_neck():
    torch.manual_seed(0)
    lit = DenseDartLitModule(_hcfg(heatmap_memory="neck")).eval()
    with torch.no_grad():
        out = lit.model(_batch()["image"].float())
        r = lit.model.read_darts(out)
    assert r["tip"].shape == (2, lit.config.query_count, 2)
    with pytest.raises(ValueError, match="heatmap_memory"):
        _hcfg(heatmap_memory="fine")


def test_votes_are_splatted_bilinearly_at_their_points():
    from darts_model.model.query_heatmap import vote_map
    # Two cells vote: one exactly on cell (3, 2)'s centre, one between four
    # cells; a third votes off the grid and is dropped.
    pts = torch.tensor([[[3.5 / 8, 5.0 / 8, 1.2], [2.5 / 8, 6.0 / 8, 0.5]]])
    wt = torch.tensor([[1.0, 2.0, 7.0]])
    v = vote_map(pts, wt, (8, 8))[0, 0]
    assert torch.isclose(v[2, 3], torch.tensor(1.0))
    assert torch.allclose(v[5:7, 4:6], torch.full((2, 2), 0.5))
    assert torch.isclose(v.sum(), torch.tensor(3.0))


def test_the_readout_runs_with_the_vote_map():
    torch.manual_seed(0)
    lit = DenseDartLitModule(_hcfg(heatmap_votes=True)).eval()
    with torch.no_grad():
        r = lit.model.read_darts(lit.model(_batch()["image"].float()))
    assert r["tip"].shape == (2, lit.config.query_count, 2)
