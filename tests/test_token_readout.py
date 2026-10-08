"""Stage 3: the token readout, its losses, freezing and export."""
import pytest
import torch

from darts_model.export.common import effective_topk, output_names
from darts_model.export.coreml import ExportWrapper
from darts_model.model.detector import DenseDartConfig, DenseDartLitModule, _huber
from darts_model.model.token_readout import (
    _LocalAttention, gather_tokens, readout_losses,
)

from test_dense_dart import _batch, _cfg


def _rcfg(**kw):
    base = dict(predict_tip_heatmap=True, tip_stride=4, tip_head_width=8,
                tip_sigma=1.0, tip_offset_radius=1,
                tip_offset_weighting="gaussian", token_readout=True,
                readout_tokens=256, readout_dim=32, readout_heads=4,
                readout_enc_layers=1, readout_dec_layers=2, readout_window=4,
                fg_threshold=0.3, peak_min_votes=4)
    base.update(kw)
    return _cfg(**base)


def _out(b=1, h=8, w=8, tip_width=4, neck=6):
    """A dense field with one silhouette cell and one tip cell lit."""
    out = {"fg_logits": torch.full((b, h, w), -9.0),
           "tip_logits": torch.full((b, 2 * h, 2 * w), -9.0),
           "tip_cell_offset": torch.zeros(b, 2, 2 * h, 2 * w),
           "tip_features": torch.randn(b, tip_width, 2 * h, 2 * w),
           "neck": torch.randn(b, neck, h, w)}
    for k in ("centre_offset", "tip_offset", "flight_offset", "log_extent"):
        out[k] = torch.zeros(b, 2, h, w)
    out["direction"] = torch.zeros(b, 2, h, w)
    out["direction"][:, 0] = 1.0
    out["fg_logits"][0, 3, 5] = 5.0
    out["tip_logits"][0, 9, 4] = 5.0
    out["tip_cell_offset"][0, :, 9, 4] = torch.tensor([0.25, -0.25])
    out["tip_offset"][0, :, 3, 5] = torch.tensor([0.1, 0.0])
    return out


def test_tokens_are_the_dart_and_tip_cells_with_their_dense_readings():
    out = _out()
    tok = gather_tokens(out, 8)
    # The stride-8 silhouette cell (3, 5) covers four stride-4 cells; the tip
    # cell (9, 4) is one more: five tokens above the floor.
    assert int((tok["score"][0] > 0.5).sum()) == 5
    i = [j for j in range(8) if tok["grid"][0, j].tolist() == [4, 9]][0]
    assert torch.allclose(tok["tip_point"][0, i],
                          torch.tensor([(4.5 + 0.25) / 16, (9.5 - 0.25) / 16]))
    # A silhouette token reads its parent cell's dense estimate.
    j = [j for j in range(8) if tok["grid"][0, j].tolist() == [10, 6]][0]
    assert int(tok["parent"][0, j]) == 3 * 8 + 5
    assert torch.allclose(tok["est_tip"][0, j], torch.tensor([5.5 / 8 + 0.1, 3.5 / 8]))
    assert tok["grid_size"].tolist() == [16, 16]


def test_local_attention_never_reaches_outside_its_window():
    torch.manual_seed(0)
    attn = _LocalAttention(8, 2, window=2)
    x = torch.randn(1, 3, 8)
    grid = torch.tensor([[[0, 0], [1, 1], [10, 10]]])
    with torch.no_grad():
        base = attn(x, grid, torch.zeros(1, 3))
        x2 = x.clone()
        x2[0, 2] += 100.0                              # change the far token
        moved = attn(x2, grid, torch.zeros(1, 3))
    assert torch.allclose(base[0, :2], moved[0, :2], atol=1e-5)


def test_losses_are_finite_on_frames_without_darts():
    torch.manual_seed(0)
    lit = DenseDartLitModule(_rcfg())
    batch = _batch()
    batch["dart_mask"][:] = False
    with torch.no_grad():
        out = lit.model(batch["image"].float())
    tok = gather_tokens(out, lit.config.readout_tokens)
    losses, err, _ = readout_losses(lit.model.token_readout(tok), tok, batch,
                                    lit.config, 256.0, _huber)
    assert all(torch.isfinite(v) for v in losses.values())
    assert err.numel() == 0


def test_config_is_validated():
    with pytest.raises(ValueError, match="alternative readouts"):
        DenseDartConfig(token_readout=True, query_head=True,
                        predict_tip_heatmap=True)
    with pytest.raises(ValueError, match="requires predict_tip_heatmap"):
        DenseDartConfig(token_readout=True)
    with pytest.raises(ValueError, match="readout only"):
        DenseDartConfig(freeze_dense=True)


def test_the_token_readout_trains_end_to_end():
    torch.manual_seed(0)
    lit = DenseDartLitModule(_rcfg(query_conf_weight=0.5))
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
        assert float(torch.cdist(gt, tips).amin(1).max()) < 1.5


def test_a_frozen_dense_model_does_not_move(tmp_path):
    torch.manual_seed(0)
    trained = DenseDartLitModule(_rcfg(token_readout=False))
    path = tmp_path / "stage2.ckpt"
    torch.save({"state_dict": trained.state_dict()}, path)
    lit = DenseDartLitModule(_rcfg(freeze_dense=True, init_weights=str(path)))
    lit.log = lambda *a, **k: None
    trainable = {n for n, p in lit.model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("token_readout.") for n in trainable)
    before = {n: p.detach().clone() for n, p in lit.model.named_parameters()}
    opt = torch.optim.Adam([p for p in lit.parameters() if p.requires_grad],
                           lr=1e-2)
    loss = lit._step(_batch(), "train")
    opt.zero_grad()
    loss.backward()
    opt.step()
    for n, p in lit.model.named_parameters():
        if n.startswith("token_readout."):
            continue
        assert torch.equal(p, before[n]), n
    names = {g["name"] for g in DenseDartLitModule(
        _rcfg(freeze_dense=True, steps_per_epoch=10)).configure_optimizers()[
            "optimizer"].param_groups}
    assert names <= {"readout", "readout_no_decay"}


def test_validation_compares_against_stage_two(monkeypatch):
    import darts_model.model.hough as hough
    batch = _batch()

    def detect(decoded, fg_prob, cfg, claim_by_embedding=None):
        return [{"tip": batch["dart_ends"][0, i, 0:2] + 0.02,
                 "flight": batch["dart_ends"][0, i, 2:4],
                 "box_tip": batch["dart_ends"][0, i, 0:2],
                 "direction": torch.tensor([1.0, 0.0]), "votes": 9,
                 "score": 0.9} for i in range(2)]
    monkeypatch.setattr(hough, "detect", detect)
    torch.manual_seed(0)
    lit = DenseDartLitModule(_rcfg(tip_assign="hungarian")).eval()
    logged = {}
    lit.log = lambda n, v, *a, **k: logged.__setitem__(n, float(v))
    with torch.no_grad():
        lit._step(batch, "val")
    lit.on_validation_epoch_end()
    for name in ("val/landing_px_error_penalised",
                 "val/landing_px_error_penalised_centre_claim",
                 "val/landing_px_error_penalised_centre_claim_assigned",
                 "val/r_dart_attn_loss", "val/r_tip_attn_loss"):
        assert name in logged, name


def test_the_token_readout_exports():
    torch.manual_seed(0)
    cfg = _rcfg(head_width=16, head_depth=1, kp_head_width=16, kp_head_depth=1)
    lit = DenseDartLitModule(cfg).eval()
    with torch.no_grad():
        for p in lit.model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        lit.model.fg_head.bias.fill_(0.0)
    wrapper = ExportWrapper(lit.model, topk=effective_topk(256, 8)).eval()
    image = torch.randint(0, 256, (1, 3, 256, 256)).float()
    with torch.no_grad():
        outs = dict(zip(output_names(cfg), wrapper(image)))
        r = lit.model.read_darts(lit.model((image - wrapper.norm_mean)
                                           / wrapper.norm_std))
        traced = torch.jit.trace(wrapper, image, strict=False)(image)
        exported = torch.export.export(wrapper, (image,)).module()(image)
    assert outs["slot_score"].shape == (1, cfg.query_count)
    assert torch.allclose(outs["slot_tip"], r["tip"])
    for a, b, c in zip(outs.values(), traced, exported):
        assert torch.allclose(a, b, atol=1e-5)
        assert torch.allclose(a, c, atol=1e-5)


def test_the_tip_pointer_is_trained_onto_the_tips_own_cell():
    """Weight spread over the tip's 3x3 costs more than weight on its own
    cell, though all nine point at the tip."""
    tok = {"grid": torch.tensor([[[4, 4], [5, 4], [4, 5], [9, 9]]]),
           "grid_size": torch.tensor([16, 16]),
           "parent": torch.zeros(1, 4, dtype=torch.long)}
    batch = {"dart_ends": torch.tensor([[[4.5 / 16, 4.5 / 16, 0.9, 0.9]]]),
             "dart_mask": torch.tensor([[True]]),
             "instance": torch.ones(1, 8, 8, dtype=torch.long)}

    def pred(tip_logits):
        return {"logit": torch.tensor([[5.0]]), "score": torch.tensor([[0.99]]),
                "tip": torch.tensor([[[4.5 / 16, 4.5 / 16]]]),
                "tip_estimate": torch.tensor([[[4.5 / 16, 4.5 / 16]]]),
                "flight": torch.tensor([[[0.9, 0.9]]]),
                "dart_logits": torch.zeros(1, 1, 4),
                "tip_logits": torch.tensor([[tip_logits]])}

    cfg = DenseDartConfig()
    focused, _, _ = readout_losses(pred([-9.0, 9.0, -9.0, -9.0, -9.0]), tok,
                                   batch, cfg, 256.0, _huber)
    spread, _, _ = readout_losses(pred([-9.0, 0.0, 0.0, 0.0, -9.0]), tok,
                                  batch, cfg, 256.0, _huber)
    assert float(focused["r_tip_attn"]) < 0.01
    assert float(spread["r_tip_attn"]) > 1.0


def test_the_hard_slot_is_the_top_options_value():
    torch.manual_seed(0)
    lit = DenseDartLitModule(_rcfg()).eval()
    with torch.no_grad():
        lit.model.fg_head.bias.fill_(10.0)
        out = lit.model(torch.randn(1, 3, 256, 256))
        tok = gather_tokens(out, lit.config.readout_tokens)
        r = lit.model.token_readout(tok)
    pick = r["tip_logits"].argmax(-1)[0]
    for q, j in enumerate(pick.tolist()):
        want = r["tip_estimate"][0, q] if j == 0 else tok["tip_point"][0, j - 1]
        assert torch.allclose(r["tip_hard"][0, q], want)
    assert output_names(lit.config)[-3:] == ["slot_tip_hard", "tip_xy", "tip_score"]
