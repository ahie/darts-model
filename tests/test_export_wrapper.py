"""ExportWrapper's compact outputs are the training-time readout, relocated.

The apps see only the top-K cells and the 40 keypoint argmaxes. These tests
hold them to `decode_boxes`, `hough.detect` and the argmax keypoint readout on
the same weights, so the in-graph gather, cell-centre convention and
normalisation cannot drift from what the metrics measure.
"""
from __future__ import annotations

import pytest
import torch

from darts_model.export.common import ExportError, effective_topk
from darts_model.export.coreml import OUTPUTS, ExportWrapper
from darts_model.model.detector import DenseDartConfig, DenseDartNet, decode_boxes
from darts_model.model.hough import detect

SIZE = 256


def _tiny(**kw) -> DenseDartConfig:
    base = dict(
        backbone_widths=(16, 24, 32, 48), backbone_depths=(1, 1, 1, 1),
        head_width=16, head_depth=1, backbone_weights="",
        predict_ends=True, predict_keypoints=True,
        kp_head_width=16, kp_head_depth=1,
        # Low enough that a perturbed model votes somewhere.
        fg_threshold=0.3, peak_min_votes=2)
    base.update(kw)
    return DenseDartConfig(**base)


def _perturbed(cfg: DenseDartConfig) -> DenseDartNet:
    """Every weight moved off its init, so the zero-initialised box and ends
    heads produce distinct per-cell values."""
    torch.manual_seed(0)
    net = DenseDartNet(cfg).eval()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        net.fg_head.bias.fill_(0.0)
    return net


def _run():
    cfg = _tiny()
    net = _perturbed(cfg)
    wrapper = ExportWrapper(net, topk=effective_topk(SIZE, cfg.out_stride)).eval()
    torch.manual_seed(1)
    image = torch.randint(0, 256, (1, 3, SIZE, SIZE)).float()
    with torch.no_grad():
        outs = dict(zip(OUTPUTS, wrapper(image)))
        dense = net((image - wrapper.norm_mean) / wrapper.norm_std)
    return cfg, outs, dense


def _scatter(plane: torch.Tensor, idx: torch.Tensor, h: int, w: int):
    """(1, K, C) at flat cell indices -> (C, h, w)."""
    c = plane.shape[-1]
    out = torch.full((c, h * w), float("nan"))
    out[:, idx] = plane[0].T
    return out.view(c, h, w)


def test_per_cell_outputs_equal_decode_boxes() -> None:
    cfg, outs, dense = _run()
    h, w = dense["fg_logits"].shape[-2:]
    assert h * w == outs["dart_score"].shape[1], "256px should keep every cell"
    dec = {k: v[0] for k, v in decode_boxes(dense, (h, w)).items()}

    # topk on the same tensor: the same order the wrapper used.
    _, idx = torch.topk(dense["fg_logits"].reshape(-1), h * w, sorted=True)
    pairs = {"dart_centre": "centre", "dart_direction": "direction",
             "dart_extent": "extent", "dart_tip": "tip_point",
             "dart_flight": "flight_point"}
    for name, key in pairs.items():
        got = _scatter(outs[name], idx, h, w)
        assert torch.allclose(got, dec[key], atol=1e-6), (
            f"{name}: max delta {(got - dec[key]).abs().max():.3e}")
    score = _scatter(outs["dart_score"][..., None], idx, h, w)[0]
    assert torch.allclose(score, dense["fg_logits"][0].sigmoid(), atol=1e-6)
    assert torch.all(outs["dart_score"][0][:-1] >= outs["dart_score"][0][1:])


def test_hough_readout_is_the_same_on_either_side() -> None:
    """The app's readout over the compact outputs equals the training one."""
    cfg, outs, dense = _run()
    h, w = dense["fg_logits"].shape[-2:]
    _, idx = torch.topk(dense["fg_logits"].reshape(-1), h * w, sorted=True)
    rebuilt = {
        "centre": _scatter(outs["dart_centre"], idx, h, w),
        "direction": _scatter(outs["dart_direction"], idx, h, w),
        "extent": _scatter(outs["dart_extent"], idx, h, w),
        "tip_point": _scatter(outs["dart_tip"], idx, h, w),
        "flight_point": _scatter(outs["dart_flight"], idx, h, w),
    }
    fg = _scatter(outs["dart_score"][..., None], idx, h, w)[0]

    want = detect({k: v[0] for k, v in decode_boxes(dense, (h, w)).items()},
                  dense["fg_logits"][0].sigmoid(), cfg)
    got = detect(rebuilt, fg, cfg)
    assert want, "the perturbed model should produce at least one detection"
    assert len(got) == len(want)
    for a, b in zip(got, want):
        assert set(a) == set(b)
        for k in a:
            if isinstance(a[k], torch.Tensor):
                assert torch.allclose(a[k], b[k], atol=1e-6), k
            else:
                assert a[k] == pytest.approx(b[k], abs=1e-6), k


def test_keypoints_are_the_argmax_readout() -> None:
    cfg, outs, dense = _run()
    logits, off = dense["kp_logits"][0], dense["kp_offset"][0]
    nk, h, w = logits.shape
    flat = logits.reshape(nk, -1)
    peak, i = flat.max(dim=1)
    iy, ix = i // w, i % w
    want = torch.stack([(ix + 0.5 + off[0, iy, ix]) / w,
                        (iy + 0.5 + off[1, iy, ix]) / h], dim=-1)
    assert torch.allclose(outs["kp_xy"][0], want, atol=1e-6)
    assert torch.allclose(outs["kp_conf"][0], peak.sigmoid(), atol=1e-6)
    assert outs["kp_xy"].shape == (1, cfg.num_keypoints, 2)


def test_small_inputs_clamp_k() -> None:
    cfg = _tiny()
    k = effective_topk(128, cfg.out_stride)
    assert k == 16 * 16
    wrapper = ExportWrapper(_perturbed(cfg), topk=k).eval()
    with torch.no_grad():
        score = wrapper(torch.zeros(1, 3, 128, 128))[0]
    assert score.shape == (1, k)


def test_sizes_off_the_stride_are_refused() -> None:
    with pytest.raises(ExportError, match="multiple"):
        effective_topk(1000 + 4, 8)


@pytest.mark.parametrize("missing", ["predict_ends", "predict_keypoints"])
def test_wrapper_requires_both_heads(missing: str) -> None:
    net = DenseDartNet(_tiny(**{missing: False}))
    with pytest.raises(ExportError, match=missing):
        ExportWrapper(net)
