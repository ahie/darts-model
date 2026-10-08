"""The learned peak pointer and the weighted tip-offset loss."""
from dataclasses import dataclass

import pytest
import torch

from darts_model.export.common import effective_topk, output_names
from darts_model.export.coreml import ExportWrapper
from darts_model.model.detector import DenseDartConfig, DenseDartLitModule
from darts_model.model.queries import PeakPointer, pointer_targets
from darts_model.model.tips import weighted_offset_l1

from test_dense_dart import _batch, _cfg


@dataclass
class _Cfg:
    instance_dim: int = 16
    tip_head_width: int = 8


def _peaks(xy, score=None):
    xy = torch.tensor([xy], dtype=torch.float)
    p = xy.shape[1]
    return {"xy": xy, "score": torch.full((1, p), 0.9) if score is None
            else torch.tensor([score]), "feature": torch.zeros(1, p, 8)}


def test_a_fresh_pointer_keeps_the_estimate():
    torch.manual_seed(0)
    ptr = PeakPointer(_Cfg())
    est = torch.tensor([[[0.5, 0.5]]])
    peaks = _peaks([[0.5 + 0.01 * i, 0.4] for i in range(16)])
    landing, logits = ptr(torch.randn(1, 1, 16), est, torch.tensor([[[0.0, 1.0]]]),
                          peaks)
    assert float(logits.softmax(-1)[0, 0, 0]) > 0.999
    assert float((landing - est).norm()) * 1024 < 0.5


def test_empty_peak_slots_take_no_weight():
    ptr = PeakPointer(_Cfg())
    with torch.no_grad():
        ptr.keep[-1].bias.fill_(-20.0)                 # anything but keep
    peaks = _peaks([[0.6, 0.6], [0.0, 0.0]], score=[0.9, 0.0])
    landing, _ = ptr(torch.randn(1, 1, 16), torch.tensor([[[0.5, 0.5]]]),
                     torch.tensor([[[0.0, 1.0]]]), peaks)
    assert torch.allclose(landing, torch.tensor([[[0.6, 0.6]]]), atol=1e-5)


def test_the_pointer_learns_to_prefer_peaks_along_the_axis():
    """A dart's estimate is uncertain along its axis and tight across it.
    Each sample offers the true tip somewhere along the axis and a decoy
    nearer, but off the axis: the pointer must learn the anisotropy that a
    radius cannot express."""
    torch.manual_seed(0)
    ptr = PeakPointer(_Cfg())
    opt = torch.optim.Adam(ptr.parameters(), lr=3e-3)
    q = torch.zeros(64, 1, 16)

    def sample():
        n = 64
        axis = torch.nn.functional.normalize(torch.randn(n, 2), dim=-1)
        normal = torch.stack((-axis[:, 1], axis[:, 0]), -1)
        est = 0.3 + 0.4 * torch.rand(n, 2)
        along = (torch.rand(n, 1) * 8 + 2) / 1024 * torch.sign(torch.randn(n, 1))
        decoy = (torch.rand(n, 1) * 1.5 + 2.5) / 1024
        true = est + axis * along
        fake = est + normal * decoy * torch.sign(torch.randn(n, 1))
        swap = torch.rand(n) < 0.5                    # either slot order
        xy = torch.where(swap.view(n, 1, 1), torch.stack((fake, true), 1),
                         torch.stack((true, fake), 1))
        target = torch.where(swap, 2, 1)
        peaks = {"xy": xy, "score": torch.full((n, 2), 0.9),
                 "feature": torch.zeros(n, 2, 8)}
        return est.unsqueeze(1), axis.unsqueeze(1), peaks, target

    for _ in range(1500):
        est, axis, peaks, target = sample()
        _, logits = ptr(q, est, axis, peaks)
        loss = torch.nn.functional.cross_entropy(logits[:, 0], target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    est, axis, peaks, target = sample()
    with torch.no_grad():
        _, logits = ptr(q, est, axis, peaks)
    assert float((logits[:, 0].argmax(-1) == target).float().mean()) > 0.95


def test_targets_name_the_peak_that_detects_each_tip():
    peaks_xy = torch.tensor([[[0.500, 0.505], [0.520, 0.500], [0.900, 0.900]]])
    score = torch.tensor([[0.9, 0.9, 0.9]])
    tips = torch.tensor([[[0.500, 0.500], [0.524, 0.500], [0.100, 0.100]]])
    ok = torch.tensor([[True, True, True]])
    f = torch.tensor([0, 0, 0])
    d = torch.tensor([0, 1, 2])
    t = pointer_targets(peaks_xy, score, tips, ok, f, d, 8 / 1024)
    # Dart 0: the peak 5px away. Dart 1: the peak 4px away. Dart 2: none
    # within reach, so keep.
    assert t.tolist() == [1, 2, 0]


def test_a_peak_nearer_another_tip_is_not_this_darts():
    peaks_xy = torch.tensor([[[0.506, 0.500]]])
    score = torch.tensor([[0.9]])
    tips = torch.tensor([[[0.500, 0.500], [0.508, 0.500]]])
    t = pointer_targets(peaks_xy, score, tips, torch.tensor([[True, True]]),
                        torch.tensor([0]), torch.tensor([0]), 8 / 1024)
    assert t.tolist() == [0]


def test_the_weighted_offset_loss_favours_the_tips_own_cell():
    pred = torch.zeros(1, 2, 3, 3)
    target = torch.zeros(1, 2, 3, 3)
    target[0, :, 1, 1] = 1.0                       # error 1 at the own cell
    w = torch.tensor([[[0.37, 0.61, 0.37], [0.61, 1.0, 0.61],
                       [0.37, 0.61, 0.37]]])
    weighted = weighted_offset_l1(pred, target, w)
    uniform = weighted_offset_l1(pred, target, (w > 0).float())
    # The own cell's share of the loss: 1/4.92 weighted, 1/9 uniform.
    assert float(weighted) == pytest.approx(1.0 / float(w.sum()))
    assert float(weighted) > 1.75 * float(uniform)


def _pcfg(**kw):
    base = dict(query_head=True, token_dim=8, instance_dim=16, instance_layers=1,
                fg_threshold=0.3, peak_min_votes=4, predict_tip_heatmap=True,
                tip_head_width=8, tip_stride=4, tip_sigma=1.0,
                tip_offset_radius=1, tip_offset_weighting="gaussian",
                tip_pointer=True)
    base.update(kw)
    return _cfg(**base)


def test_the_pointer_needs_queries_and_tips_and_replaces_the_snap():
    with pytest.raises(ValueError, match="tip_pointer requires"):
        DenseDartConfig(tip_pointer=True)
    with pytest.raises(ValueError, match="replaces tip_snap"):
        DenseDartConfig(tip_pointer=True, tip_snap=True, query_head=True,
                        predict_tip_heatmap=True)


def test_the_pointed_readout_trains_end_to_end():
    torch.manual_seed(0)
    lit = DenseDartLitModule(_pcfg(instance_weight=0.05, query_conf_weight=0.5,
                                   tip_pointer_weight=0.5))
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
        assert float(torch.cdist(gt, tips).amin(1).max()) < 1.0


def test_validation_logs_the_pointer_and_its_alternatives():
    torch.manual_seed(0)
    lit = DenseDartLitModule(_pcfg()).eval()
    logged = {}
    lit.log = lambda n, v, *a, **k: logged.__setitem__(n, float(v))
    with torch.no_grad():
        lit.model.fg_head.bias.fill_(10.0)
        lit.model.tip_heat.bias.fill_(0.0)
        lit._step(_batch(), "val")
    lit.on_validation_epoch_end()
    for name in ("val/landing_px_error_penalised",
                 "val/landing_px_error_penalised_queries_estimate",
                 "val/landing_px_error_penalised_queries_snapped",
                 "val/pointer_correct", "val/pointer_keep",
                 "val/pointer_shared_peaks", "val/pointer_loss"):
        assert name in logged, name


def test_the_pointed_readout_exports():
    torch.manual_seed(0)
    cfg = _pcfg(head_width=16, head_depth=1, kp_head_width=16, kp_head_depth=1)
    lit = DenseDartLitModule(cfg).eval()
    with torch.no_grad():
        for p in lit.model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        lit.model.fg_head.bias.fill_(0.0)
        lit.model.tip_heat.bias.fill_(0.0)
    wrapper = ExportWrapper(lit.model, topk=effective_topk(256, 8)).eval()
    image = torch.randint(0, 256, (1, 3, 256, 256)).float()
    with torch.no_grad():
        outs = dict(zip(output_names(cfg), wrapper(image)))
        r = lit.model.read_darts(lit.model((image - wrapper.norm_mean)
                                           / wrapper.norm_std))
        traced = torch.jit.trace(wrapper, image, strict=False)(image)
        exported = torch.export.export(wrapper, (image,)).module()(image)
    assert torch.allclose(outs["slot_tip"], r["tip"])
    for a, b, c in zip(outs.values(), traced, exported):
        assert torch.allclose(a, b, atol=1e-5)
        assert torch.allclose(a, c, atol=1e-5)
