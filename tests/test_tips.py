"""The tip heatmap: its targets, peaks, the snap, training and export."""
from dataclasses import dataclass

import pytest
import torch

from darts_model.export.common import (
    TIP_OUTPUTS, effective_topk, keypoint_module_names, output_names,
    readout_contract,
)
from darts_model.export.coreml import OUTPUTS, SLOT_OUTPUTS, ExportWrapper
from darts_model.model.detector import DenseDartConfig, DenseDartLitModule
from darts_model.model.tips import snap_to_tips, tip_peaks, tip_targets

from test_dense_dart import _batch, _cfg


def test_targets_take_the_max_of_the_darts_never_the_sum():
    """Two tips one cell apart: each tip's cell is exactly 1 and no cell
    exceeds 1, so each tip is a maximum of the target."""
    ends = torch.tensor([[[4.5 / 16, 4.5 / 16], [5.5 / 16, 4.5 / 16]]])
    heat, off, at = tip_targets(ends, torch.tensor([[True, True]]), (16, 16), 1.0)
    assert float(heat.max()) == 1.0
    assert float(heat[0, 4, 4]) == 1.0 and float(heat[0, 4, 5]) == 1.0
    assert int(at.sum()) == 2


def test_peaks_carry_the_sub_cell_offset_and_drop_noise():
    logits = torch.full((1, 16, 16), -9.0)
    logits[0, 3, 4] = 3.0
    logits[0, 10, 12] = 2.0
    logits[0, 0, 0] = -4.0                      # 0.018: under the floor
    offset = torch.zeros(1, 2, 16, 16)
    offset[0, :, 3, 4] = torch.tensor([0.25, -0.1])
    xy, score = tip_peaks(logits, offset, 4)
    assert (score > 0).sum() == 2
    assert torch.allclose(xy[0, 0], torch.tensor([(4.5 + 0.25) / 16,
                                                  (3.5 - 0.1) / 16]))
    assert torch.all(xy[0, 2:] == 0)


@dataclass
class _Snap:
    tip_snap_score: float = 0.3
    tip_snap_px: float = 8.0
    tip_snap_axis_px: float = 4.0


def test_a_point_snaps_to_the_nearest_peak_on_its_axis():
    peaks = torch.tensor([[[0.500, 0.506],      # 6px ahead along the axis
                           [0.505, 0.500],      # nearer, but 5px off the axis
                           [0.600, 0.600]]])    # far
    score = torch.tensor([[0.9, 0.9, 0.9]])
    pts = torch.tensor([[[0.5, 0.5]]])
    axis = torch.tensor([[[0.0, 1.0]]])
    new, found = snap_to_tips(pts, axis, peaks, score, _Snap(), 1000.0)
    assert found.tolist() == [[1.0]]
    assert torch.allclose(new[0, 0], peaks[0, 0])


def test_without_an_acceptable_peak_the_estimate_stands():
    pts = torch.tensor([[[0.5, 0.5]]])
    axis = torch.tensor([[[0.0, 1.0]]])
    weak = (torch.tensor([[[0.5, 0.502]]]), torch.tensor([[0.1]]))
    far = (torch.tensor([[[0.5, 0.52]]]), torch.tensor([[0.9]]))
    for peaks, score in (weak, far):
        new, found = snap_to_tips(pts, axis, peaks, score, _Snap(), 1000.0)
        assert found.tolist() == [[0.0]]
        assert torch.equal(new, pts)


def test_snapping_requires_the_head():
    with pytest.raises(ValueError, match="predict_tip_heatmap"):
        DenseDartConfig(tip_snap=True)


def _tcfg(**kw):
    base = dict(predict_tip_heatmap=True, tip_head_width=16, tip_snap=True)
    base.update(kw)
    return _cfg(**base)


def test_the_tip_head_learns_the_landing_points():
    """Fit one batch: the heatmap's peaks land on the two landing points,
    sub-pixel at 256px, through the offset."""
    torch.manual_seed(0)
    lit = DenseDartLitModule(_tcfg())
    lit.train()
    lit.log = lambda *a, **k: None
    batch = _batch()
    opt = torch.optim.Adam(lit.parameters(), lr=3e-3)
    for _ in range(300):
        loss = lit._step(batch, "train")
        opt.zero_grad()
        loss.backward()
        opt.step()
    lit.eval()
    with torch.no_grad():
        out = lit.model(batch["image"].float())
    xy, score = tip_peaks(out["tip_logits"], out["tip_cell_offset"], 8)
    for b in range(2):
        found = xy[b][score[b] >= 0.3] * 256
        gt = batch["dart_ends"][b, :2, 0:2] * 256
        assert found.shape[0] == 2
        assert float(torch.cdist(gt, found).amin(1).max()) < 1.0


def test_validation_logs_snapped_and_unsnapped(monkeypatch):
    """One detection near each dart, from a stand-in readout, so there is
    something to snap."""
    import darts_model.model.hough as hough
    batch = _batch()

    def detect(decoded, fg_prob, cfg, claim_by_embedding=None):
        return [{"tip": batch["dart_ends"][0, i, 0:2] + 0.004,
                 "flight": batch["dart_ends"][0, i, 2:4],
                 "box_tip": batch["dart_ends"][0, i, 0:2],
                 "direction": torch.tensor([1.0, 0.0]), "votes": 9,
                 "score": 0.9} for i in range(2)]
    monkeypatch.setattr(hough, "detect", detect)
    torch.manual_seed(0)
    lit = DenseDartLitModule(_tcfg()).eval()
    logged = {}
    lit.log = lambda n, v, *a, **k: logged.__setitem__(n, float(v))
    with torch.no_grad():
        lit._step(batch, "val")
    lit.on_validation_epoch_end()
    for name in ("val/landing_px_error_penalised",
                 "val/landing_px_error_penalised_centre_claim_unsnapped",
                 "val/tip_peak_found", "val/tip_snap_rate"):
        assert name in logged, name


def test_a_finetune_may_add_the_tip_head(tmp_path):
    trained = DenseDartLitModule(_cfg())
    path = tmp_path / "plain.ckpt"
    torch.save({"state_dict": trained.state_dict()}, path)
    DenseDartLitModule(_tcfg(init_weights=str(path)))


def test_the_tip_head_is_kept_fp32_with_the_keypoints():
    net = DenseDartLitModule(_tcfg(predict_keypoints=True)).model
    names = keypoint_module_names(net)
    assert {"tip_trunk", "tip_heat", "tip_cell_offset"} <= set(names)


def _wrapper(**kw):
    torch.manual_seed(0)
    cfg = _tcfg(head_width=16, head_depth=1, kp_head_width=16, kp_head_depth=1,
                **kw)
    lit = DenseDartLitModule(cfg).eval()
    with torch.no_grad():
        for p in lit.model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        lit.model.fg_head.bias.fill_(0.0)
        lit.model.tip_heat.bias.fill_(0.0)
    return cfg, lit.model, ExportWrapper(
        lit.model, topk=effective_topk(256, cfg.out_stride)).eval()


def test_the_exported_graph_emits_the_peaks():
    cfg, net, wrapper = _wrapper()
    image = torch.randint(0, 256, (1, 3, 256, 256)).float()
    with torch.no_grad():
        outs = dict(zip(output_names(cfg), wrapper(image)))
        dense = net((image - wrapper.norm_mean) / wrapper.norm_std)
    assert list(outs) == OUTPUTS + TIP_OUTPUTS
    xy, score = tip_peaks(dense["tip_logits"], dense["tip_cell_offset"],
                          cfg.tip_peak_count)
    assert torch.equal(outs["tip_score"], score)
    assert torch.allclose(outs["tip_xy"], xy)
    assert readout_contract(cfg, 256, 1024)["tips"]["slots_snapped"] is False


def test_slots_are_snapped_in_the_graph_and_it_exports():
    cfg, net, wrapper = _wrapper(query_head=True, token_dim=8, instance_dim=16,
                                 instance_layers=1)
    image = torch.randint(0, 256, (1, 3, 256, 256)).float()
    with torch.no_grad():
        outs = dict(zip(output_names(cfg), wrapper(image)))
        dense = net((image - wrapper.norm_mean) / wrapper.norm_std)
        darts = net.read_darts(dense)
        peaks = tip_peaks(dense["tip_logits"], dense["tip_cell_offset"],
                          cfg.tip_peak_count)
        want, _ = snap_to_tips(darts["tip"], darts["direction"], *peaks, cfg,
                               256.0)
        traced = torch.jit.trace(wrapper, image, strict=False)(image)
        exported = torch.export.export(wrapper, (image,)).module()(image)
    assert list(outs) == OUTPUTS + SLOT_OUTPUTS + TIP_OUTPUTS
    assert torch.allclose(outs["slot_tip"], want)
    for a, b, c in zip(outs.values(), traced, exported):
        assert torch.allclose(a, b, atol=1e-5)
        assert torch.allclose(a, c, atol=1e-5)
    assert readout_contract(cfg, 256, 1024)["tips"]["slots_snapped"] is True


# ------------------------------------------------------------ stride 4

def test_a_stride4_tip_head_runs_at_twice_the_detectors_grid():
    lit = DenseDartLitModule(_tcfg(tip_stride=4, tip_sigma=2.0))
    with torch.no_grad():
        out = lit.model(torch.randn(1, 3, 256, 256))
    assert out["fg_logits"].shape[-2:] == (32, 32)
    assert out["tip_logits"].shape[-2:] == (64, 64)
    assert out["tip_cell_offset"].shape[-2:] == (64, 64)


def test_a_stride4_tip_head_learns_the_landing_points():
    torch.manual_seed(0)
    lit = DenseDartLitModule(_tcfg(tip_stride=4, tip_sigma=2.0))
    lit.train()
    lit.log = lambda *a, **k: None
    batch = _batch()
    opt = torch.optim.Adam(lit.parameters(), lr=3e-3)
    for _ in range(300):
        loss = lit._step(batch, "train")
        opt.zero_grad()
        loss.backward()
        opt.step()
    lit.eval()
    with torch.no_grad():
        out = lit.model(batch["image"].float())
    xy, score = tip_peaks(out["tip_logits"], out["tip_cell_offset"], 8)
    for b in range(2):
        found = xy[b][score[b] >= 0.3] * 256
        gt = batch["dart_ends"][b, :2, 0:2] * 256
        assert found.shape[0] == 2
        assert float(torch.cdist(gt, found).amin(1).max()) < 1.0


def test_a_stride4_tip_head_exports_and_stays_fp32():
    cfg, net, wrapper = _wrapper(tip_stride=4, tip_sigma=2.0)
    assert {"tip_fine_proj", "tip_neck_proj"} <= set(keypoint_module_names(net))
    image = torch.randint(0, 256, (1, 3, 256, 256)).float()
    with torch.no_grad():
        want = wrapper(image)
        traced = torch.jit.trace(wrapper, image, strict=False)(image)
        exported = torch.export.export(wrapper, (image,)).module()(image)
    for a, b, c in zip(want, traced, exported):
        assert torch.allclose(a, b, atol=1e-5)
        assert torch.allclose(a, c, atol=1e-5)


def test_tip_stride_is_validated():
    with pytest.raises(ValueError, match="tip_stride"):
        DenseDartConfig(predict_tip_heatmap=True, tip_stride=2)


# ------------------------------------------------- neighbourhood offsets

def test_every_cell_around_a_tip_points_at_it():
    tip = torch.tensor([[[5.3 / 16, 7.8 / 16]]])
    _, off, at = tip_targets(tip, torch.tensor([[True]]), (16, 16), 1.0,
                             offset_radius=1)
    assert int(at.sum()) == 9
    ys, xs = torch.nonzero(at[0], as_tuple=True)
    for y, x in zip(ys.tolist(), xs.tolist()):
        pointed = torch.tensor([x + 0.5, y + 0.5]) + off[0, :, y, x]
        assert torch.allclose(pointed, torch.tensor([5.3, 7.8]), atol=1e-5)


def test_a_cell_between_two_tips_points_at_the_nearer():
    tips = torch.tensor([[[4.4 / 16, 4.5 / 16], [6.4 / 16, 4.5 / 16]]])
    _, off, at = tip_targets(tips, torch.tensor([[True, True]]), (16, 16), 1.0,
                             offset_radius=1)
    # Cell x=5 is within one cell of both tips' cells (4 and 6); its centre
    # (5.5) is 1.1 from the first tip and 0.9 from the second.
    assert bool(at[0, 4, 5])
    assert float(5.5 + off[0, 0, 4, 5]) == pytest.approx(6.4, abs=1e-5)


def test_radius_zero_trains_the_tips_own_cell_only():
    tip = torch.tensor([[[5.3 / 16, 7.8 / 16]]])
    _, off, at = tip_targets(tip, torch.tensor([[True]]), (16, 16), 1.0)
    assert int(at.sum()) == 1 and bool(at[0, 7, 5])


def test_a_masked_dart_trains_no_offsets():
    tip = torch.tensor([[[5.3 / 16, 7.8 / 16]]])
    _, _, at = tip_targets(tip, torch.tensor([[False]]), (16, 16), 1.0,
                           offset_radius=1)
    assert int(at.sum()) == 0


# ------------------------------------------------------------ hungarian

@dataclass
class _Assign:
    tip_snap_score: float = 0.3
    tip_assign_gate_px: float = 40.0


def test_assignment_is_one_to_one_and_global():
    """Two darts whose estimates both sit nearest the same peak: nearest-peak
    snapping would give it to both; the assignment gives each a distinct
    tip, minimising the total distance."""
    from darts_model.model.tips import assign_tips
    est = torch.tensor([[0.500, 0.500], [0.510, 0.500]])
    peaks = torch.tensor([[0.505, 0.500], [0.530, 0.500]])
    out, found = assign_tips(est, peaks, torch.tensor([0.9, 0.9]), _Assign(), 1000.0)
    assert found.tolist() == [True, True]
    assert torch.allclose(out, peaks)


def test_assignment_respects_the_gate_and_the_score():
    from darts_model.model.tips import assign_tips
    est = torch.tensor([[0.5, 0.5], [0.2, 0.2]])
    peaks = torch.tensor([[0.5, 0.52], [0.2, 0.3], [0.21, 0.2]])
    score = torch.tensor([0.9, 0.9, 0.1])            # the near one is weak
    out, found = assign_tips(est, peaks, score, _Assign(), 1000.0)
    # Dart 0's peak is 20px away: inside the gate. Dart 1's only confident
    # peak is 100px away: outside it, so the estimate stands.
    assert found.tolist() == [True, False]
    assert torch.allclose(out[0], peaks[0]) and torch.equal(out[1], est[1])


def test_validation_ranks_the_assigned_readout(monkeypatch):
    import darts_model.model.hough as hough
    batch = _batch()

    def detect(decoded, fg_prob, cfg, claim_by_embedding=None):
        return [{"tip": batch["dart_ends"][0, i, 0:2] + 0.02,
                 "flight": batch["dart_ends"][0, i, 2:4],
                 "box_tip": batch["dart_ends"][0, i, 0:2],
                 "direction": torch.tensor([1.0, 0.0]), "votes": 9,
                 "score": 0.9} for i in range(2)]
    monkeypatch.setattr(hough, "detect", detect)
    lit = DenseDartLitModule(_tcfg(tip_assign="hungarian")).eval()
    logged = {}
    lit.log = lambda n, v, *a, **k: logged.__setitem__(n, float(v))
    with torch.no_grad():
        lit._step(batch, "val")
    lit.on_validation_epoch_end()
    assert "val/landing_px_error_penalised_centre_claim_unsnapped" in logged
