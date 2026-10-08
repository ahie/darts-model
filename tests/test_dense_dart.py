"""The dense readout, end to end on a fixed batch.

The load-bearing test is the overfit one. The design's stated failure mode is
"cells on the same dart predict inconsistent boxes, so the accumulator has no
peak" -- and that is checkable without a training run: fit one batch hard, then
read detections out through the real voting path and see whether they land.
"""
import math

import pytest
import torch


from darts_model.data.detect_dataset import MAX_DARTS
from darts_model.model.detector import (
    DenseDartConfig, DenseDartLitModule, DenseDartNet, cell_centres,
    decode_boxes, keypoint_targets, penalty_reduced_focal,
)
from darts_model.model.hough import detect
from darts_model.cli.train import build_config


def _cfg(**kw):
    base = dict(backbone_widths=(16, 24, 32, 48), backbone_depths=(1, 1, 1, 1),
                head_width=32, backbone_freeze_epochs=0, max_epochs=10,
                warmup_epochs=2, out_stride=8)
    base.update(kw)
    return DenseDartConfig(**base)


def _batch(B=2, SZ=256, stride=8):
    """A batch whose silhouette cells lie INSIDE the box each one votes for.

    Real data has this by construction -- positives are the dart's own pixels
    -- and a fixture that violated it would ask cells to vote for boxes they
    are not on. `test_every_foreground_cell_lies_inside_its_own_box` holds
    this property.

    Each dart also gets a realistic VOTE BUDGET. The measured floor on real
    data is 17 silhouette cells per dart; below it, whether a dart produces a
    peak at all becomes platform-dependent. 27 and 24 cells here.

    Grid is 32x32, cell centre (i + 0.5) / 32, coordinates (x, y).
    dart 1: centre (0.42, 0.42) along +x -> x in [0.28, 0.56], y in [0.37,0.47]
    dart 2: centre (0.60, 0.75) along +y -> y in [0.61, 0.89], x in [0.55,0.65]
    """
    h = w = SZ // stride
    inst = torch.zeros(B, h, w, dtype=torch.long)
    inst[:, 12:15, 9:18] = 1          # 3 x 9 = 27 cells inside dart 1's box
    inst[:, 20:28, 18:21] = 2         # 8 x 3 = 24 cells inside dart 2's
    box = torch.zeros(B, MAX_DARTS, 6)
    box[:, 0] = torch.tensor([0.42, 0.42, 1.0, 0.0, 0.140, 0.050])
    box[:, 1] = torch.tensor([0.60, 0.75, 0.0, 1.0, 0.140, 0.050])
    # The two ends the renderer reports. Deliberately NOT the box's ends: the
    # landing point sits a little inside, which is the whole reason they are
    # predicted separately, and a fixture where they coincided could not tell
    # the two conventions apart.
    ends = torch.zeros(B, MAX_DARTS, 4)
    ends[:, 0] = torch.tensor([0.42 - 0.125, 0.42, 0.42 + 0.140, 0.42])
    ends[:, 1] = torch.tensor([0.60, 0.75 - 0.125, 0.60, 0.75 + 0.140])
    return {
        "image": torch.randn(B, 3, SZ, SZ),
        "instance": inst,
        "foreground": (inst > 0).float(),
        "dart_box": box,
        "dart_ends": ends,
        "dart_mask": torch.tensor([[True, True, False]] * B),
    }


def test_every_foreground_cell_lies_inside_its_own_box():
    """The invariant the box loss needs, checked on the fixture.

    Real silhouette positives satisfy it by construction; a hand-built fixture
    does not automatically.
    """
    b = _batch()
    inst, box = b["instance"], b["dart_box"]
    h, w = inst.shape[-2:]
    centres = cell_centres(h, w, inst.device)
    for d in (1, 2):
        cx, cy, cos, sin, hl, hw = box[0, d - 1].tolist()
        sel = inst[0] == d
        assert sel.any()
        dx = centres[0][sel] - cx
        dy = centres[1][sel] - cy
        along = (dx * cos + dy * sin).abs()          # along the dart's axis
        across = (-dx * sin + dy * cos).abs()        # across it
        assert float(along.max()) <= hl, f"dart {d} overruns its length"
        assert float(across.max()) <= hw, f"dart {d} overruns its width"
        # Above the measured real-data floor of 17 cells, so the readout is
        # not being asked to find a peak from fewer votes than it ever sees.
        assert int(sel.sum()) >= 17, f"dart {d} has too few votes"


# ------------------------------------------------------------------- config

def test_loader_rejects_unknown_keys():
    with pytest.raises(SystemExit, match="unknown or misplaced config keys"):
        build_config({"head": {"head_widht": 128}})


@pytest.mark.parametrize("raw, where", [
    # Read from experiment: only; under training: it would be ignored.
    ({"training": {"log_every_n_steps": 10}}, "training.log_every_n_steps"),
    # A trainer key is read from training: only.
    ({"head": {"batch_size": 8}}, "head.batch_size"),
    ({"experiment": {"sead": 1}}, "experiment.sead"),
    ({"head": {"lr": 1e-3}, "training": {"lr": 1e-4}}, "training.lr"),
])
def test_loader_rejects_misplaced_keys(raw, where):
    with pytest.raises(SystemExit, match=where.replace(".", r"\.")):
        build_config(raw)


def test_loader_tolerates_trainer_keys():
    cfg = build_config({"training": {"lr": 1e-4, "batch_size": 8,
                                     "precision": "bf16-mixed"}})
    assert cfg.lr == pytest.approx(1e-4)


def test_rejects_a_stride_the_backbone_cannot_serve():
    with pytest.raises(ValueError, match="out_stride"):
        DenseDartConfig(out_stride=7)


# -------------------------------------------------------------------- model

def test_output_is_one_map_at_the_configured_stride():
    net = DenseDartNet(_cfg(out_stride=8)).eval()
    with torch.no_grad():
        out = net(torch.randn(1, 3, 256, 256))
    assert out["fg_logits"].shape == (1, 32, 32)
    assert out["centre_offset"].shape == (1, 2, 32, 32)


def test_direction_is_unit_after_decode_not_in_the_forward():
    """The head emits a raw vector; normalisation happens at readout, outside
    the gradient path. Normalising in the forward puts a 1/|x| into the
    backward and nothing supervises a background cell's direction."""
    net = DenseDartNet(_cfg()).eval()
    with torch.no_grad():
        out = net(torch.randn(1, 3, 256, 256))
        dec = decode_boxes(out, out["fg_logits"].shape[-2:])
    n = dec["direction"].norm(dim=1)
    assert torch.allclose(n, torch.ones_like(n), atol=1e-5)


def test_foreground_starts_sparse():
    """A head that begins by calling every cell a dart spends its first epochs
    unlearning that; foreground is ~1.6% of cells."""
    net = DenseDartNet(_cfg()).eval()
    with torch.no_grad():
        out = net(torch.randn(2, 3, 256, 256))
    assert out["fg_logits"].sigmoid().mean() < 0.1


def test_decode_is_cell_relative():
    """Two cells on one dart emit different offsets that decode to the same
    absolute box. Averaging the raw channels instead would be wrong."""
    h = w = 8
    out = {"centre_offset": torch.zeros(1, 2, h, w),
           "direction": torch.zeros(1, 2, h, w),
           "log_extent": torch.zeros(1, 2, h, w)}
    centres = cell_centres(h, w, out["centre_offset"].device)
    # each cell asked to point at (0.5, 0.5)
    out["centre_offset"][0] = 0.5 - centres
    dec = decode_boxes(out, (h, w))
    assert torch.allclose(dec["centre"][0],
                          torch.full((2, h, w), 0.5), atol=1e-6)


# ---------------------------------------------------------------------- fit

def test_overfits_one_batch_and_reads_out_through_voting():
    """The design's stated failure mode, checked without a training run.

    If cells on one dart cannot agree on a box, the accumulator has no peak and
    the whole readout fails -- so fitting a single batch and then detecting
    through the REAL voting path is the cheapest falsification available.
    """
    torch.manual_seed(0)
    lit = DenseDartLitModule(_cfg())
    lit.train()
    lit.log = lambda *a, **k: None
    batch = _batch()
    opt = torch.optim.Adam(lit.parameters(), lr=3e-3)
    first = None
    for i in range(150):
        loss = lit._step(batch, "train")
        if i == 0:
            first = float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    final = float(loss.detach())
    assert final < first * 0.25, f"{first:.3f} -> {final:.3f}"

    lit.eval()
    with torch.no_grad():
        out = lit.model(batch["image"].float())
        dec = decode_boxes(out, out["fg_logits"].shape[-2:])
        got = detect({k: v[0] for k, v in dec.items()},
                     out["fg_logits"][0].sigmoid(), lit.config)
    assert len(got) == 2, f"expected 2 darts, got {len(got)}"
    found = sorted(float(g["centre"][1]) for g in got)
    assert found[0] == pytest.approx(0.42, abs=0.05)
    assert found[1] == pytest.approx(0.75, abs=0.05)


def test_an_untrained_model_is_charged_the_gate_for_every_missed_dart():
    """An untrained head detects nothing. Its landing error over matches is
    undefined and not logged, but the monitor is: every dart missed, each
    charged the gate. Skipping unmatched darts instead would score a model
    that misses the hard darts BETTER than one that finds them."""
    lit = DenseDartLitModule(_cfg())
    logged = {}
    lit.log = lambda n, v, **k: logged.__setitem__(n, float(v))
    lit.eval()
    with torch.no_grad():
        lit.on_validation_epoch_start()
        lit._step(_batch(), "val")
        lit.on_validation_epoch_end()
    assert logged["val/recall"] == 0.0
    assert logged["val/detections_per_dart"] == 0.0
    assert logged[DenseDartLitModule.MONITOR] == lit.config.match_gate_px
    assert "val/landing_px_error" not in logged
    assert "val/tip_px_error" not in logged
    assert "val/precision" not in logged


def test_a_fitted_model_reports_recall_and_the_separation_buckets():
    torch.manual_seed(0)
    lit = DenseDartLitModule(_cfg())
    lit.train()
    lit.log = lambda *a, **k: None
    batch = _batch()
    opt = torch.optim.Adam(lit.parameters(), lr=3e-3)
    for _ in range(150):
        loss = lit._step(batch, "train")
        opt.zero_grad(); loss.backward(); opt.step()

    logged = {}
    lit.log = lambda n, v, **k: logged.__setitem__(n, float(v))
    lit.eval()
    with torch.no_grad():
        lit.on_validation_epoch_start()
        lit._step(batch, "val")
        lit.on_validation_epoch_end()
    assert logged["val/recall"] > 0.9
    assert logged["val/precision"] > 0.9
    assert "val/tip_px_error" in logged
    assert any(k.startswith("val/n_sep_") for k in logged)
    assert any(k.startswith("val/recall_sep_") for k in logged)
    # Every dart matched, so the penalised error is the plain one.
    assert logged[DenseDartLitModule.MONITOR] == pytest.approx(
        logged["val/landing_px_error"])


# ------------------------------------------------------- numerical safety

def test_direction_gradient_is_bounded_at_zero():
    """A cell's raw direction may drift through zero -- nothing supervises the
    background -- and `x / |x|.clamp(eps)` has gradient 1/eps = 1e6 there, a
    spike that can destroy the weights in one step from healthy losses."""
    net = DenseDartNet(_cfg()).eval()
    with torch.no_grad():
        net.box_head.weight.zero_()
        net.box_head.bias.zero_()          # direction channels exactly (0, 0)
    x = torch.randn(1, 3, 256, 256)
    out = net(x)
    out["direction"].sum().backward()
    g = net.box_head.bias.grad
    assert torch.isfinite(g).all()
    assert float(g.abs().max()) < 1e5, f"gradient spike: {float(g.abs().max()):.3g}"


def test_a_non_finite_batch_is_skipped_not_propagated():
    """One bad batch must not be able to destroy the weights."""
    lit = DenseDartLitModule(_cfg())
    lit.train()
    logged = {}
    lit.log = lambda n, v, **k: logged.__setitem__(n, float(v))
    batch = _batch()
    batch["image"][0, 0, 0, 0] = float("nan")
    assert lit._step(batch, "train") is None
    assert logged.get("train/skipped_batches", 0) >= 1


def test_non_finite_gradients_do_not_reach_the_optimiser():
    lit = DenseDartLitModule(_cfg())
    lit.train()
    logged = {}
    lit.log = lambda n, v, **k: logged.__setitem__(n, float(v))
    for p in lit.parameters():
        p.grad = torch.zeros_like(p)
    list(lit.parameters())[0].grad[0] = float("inf")
    opt = torch.optim.SGD(lit.parameters(), lr=1e-3)
    lit.on_before_optimizer_step(opt)
    assert all(p.grad is None for p in lit.parameters())
    assert logged.get("train/skipped_batches", 0) >= 1


def test_grad_norm_is_logged_when_finite():
    """Without this the spike is invisible: clipping hides the magnitude and
    the loss curve stays flat until the step that kills the run."""
    lit = DenseDartLitModule(_cfg())
    lit.train()
    logged = {}
    lit.log = lambda n, v, **k: logged.__setitem__(n, float(v))
    for p in lit.parameters():
        p.grad = torch.full_like(p, 0.01)
    lit.on_before_optimizer_step(torch.optim.SGD(lit.parameters(), lr=1e-3))
    assert logged["train/grad_norm"] > 0


def test_log_extent_is_clamped_so_exp_cannot_overflow():
    """The NaN path. `log_extent` is exponentiated by both the box loss and the
    readout, so an unbounded one is an overflow waiting to happen -- an
    unclamped overfit can drive a half-extent to 661 in 30 steps."""
    net = DenseDartNet(_cfg())
    with torch.no_grad():
        net.box_head.bias[4:6] = 50.0      # a runaway, far past anything real
    out = net(torch.randn(1, 3, 256, 256))
    ext = out["log_extent"].exp()
    assert torch.isfinite(ext).all()
    assert float(ext.max()) <= 0.5 + 1e-6


def test_centre_error_is_reported_against_distance_from_the_cell():
    """The diagnostic for whether receptive field is the limit.

    A cell 100 px from its box centre has to predict a 100 px offset, and the
    head's two 3x3 convs see ~5 cells. If error grows with distance the field
    is the bound; if it is flat it is not. Logged per bucket so the answer
    arrives with the run rather than after it.
    """
    lit = DenseDartLitModule(_cfg())
    logged = {}
    lit.log = lambda n, v, **k: logged.__setitem__(n, float(v))
    lit.eval()
    with torch.no_grad():
        lit.on_validation_epoch_start()
        lit._step(_batch(), "val")
        lit.on_validation_epoch_end()

    buckets = [k for k in logged if k.startswith("val/n_cells_dist_")]
    assert len(buckets) == 5, buckets
    # The fixture's cells span 0 to ~a half-length from their centres, so the
    # near buckets must be populated -- an all-empty histogram would mean the
    # distance was computed in the wrong units and would silently log nothing.
    assert sum(logged[k] for k in buckets) > 0
    assert any(k.startswith("val/centre_px_dist_") for k in logged)


def test_a_nonfinite_loss_and_a_nonfinite_gradient_are_told_apart():
    """One counter cannot answer the question that matters.

    A single `train/skipped_batches` cannot say whether the LOSS went
    non-finite or the GRADIENTS did -- different causes, different fixes -- or
    which module produced it. So they are separate counters, and the gradient
    path names the offending parameter.
    """
    lit = DenseDartLitModule(_cfg())
    logged = {}
    lit.log = lambda n, v, **k: logged.__setitem__(n, float(v))

    # A non-finite LOSS: poison the image so the forward carries NaN through.
    batch = _batch()
    batch["image"][0, 0, 0, 0] = float("nan")
    assert lit._step(batch, "train") is None, "a NaN loss must skip the batch"
    assert logged["train/skipped_loss"] == 1.0
    assert logged["train/skipped_batches"] == 1.0
    assert "train/skipped_grad" not in logged

    # A non-finite GRADIENT with a perfectly finite loss.
    lit2 = DenseDartLitModule(_cfg())
    logged2 = {}
    lit2.log = lambda n, v, **k: logged2.__setitem__(n, float(v))
    loss = lit2._step(_batch(), "train")
    loss.backward()
    named = dict(lit2.named_parameters())
    target = "model.box_head.weight"
    named[target].grad[0, 0, 0, 0] = float("inf")
    lit2.on_before_optimizer_step(torch.optim.SGD(lit2.parameters(), lr=0.0))

    assert logged2["train/skipped_grad"] == 1.0
    assert "train/skipped_loss" not in logged2
    # grad_norm must NOT be logged for a skipped step -- a norm computed over a
    # set containing inf is not a number anyone should plot.
    assert "train/grad_norm" not in logged2
    assert target in lit2._nonfinite_params, lit2._nonfinite_params

    # ...but the MAGNITUDE must still be recorded, over the parameters that are
    # still finite. Without this the only steps carrying a magnitude are the
    # ones that did not fail, so "were the failing steps spiking?" cannot be
    # answered from the logs -- which is exactly how a gradient explosion would
    # hide. That is selection on the outcome, not evidence of calm.
    assert logged2["train/skipped_grad_finite_norm"] > 0.0
    assert logged2["train/skipped_grad_finite_max"] > 0.0
    for k in ("train/skipped_grad_finite_norm", "train/skipped_grad_finite_max"):
        assert torch.isfinite(torch.tensor(logged2[k])), k


def test_pathological_dart_geometry_does_not_amplify_the_gradient():
    """Rare geometry is not a route to a gradient explosion, by construction.

    Worth holding as a test because it is a natural hypothesis for skipped
    batches -- overlapped darts blowing up near convergence -- and it is wrong
    for a structural reason: L1's derivative is +/-1 and focal's is bounded, so
    d(loss)/d(head output) is capped by the loss weights however wrong the
    target is, and the normaliser `m.sum()` is batch-wide so one bad image
    cannot amplify by shrinking the denominator.

    Measured on a FITTED model: a cold one disagrees with everything, so its
    gradients say nothing about what a converged one does.

    What this does NOT cover: d(output)/d(weights). Amplification through the
    activations is still possible, but that is a property of the IMAGE, not of
    the dart layout, and no arrangement of boxes here can produce it.
    """
    torch.manual_seed(0)
    lit = DenseDartLitModule(_cfg())
    lit.train()
    lit.log = lambda *a, **k: None
    opt = torch.optim.Adam(lit.parameters(), lr=3e-3)
    for _ in range(120):
        loss = lit._step(_batch(), "train")
        opt.zero_grad()
        loss.backward()
        opt.step()

    def grad_norm(batch):
        lit.zero_grad(set_to_none=True)
        out = lit._step(batch, "train")
        assert out is not None, "loss went non-finite on a finite target"
        out.backward()
        n = sum(float(p.grad.norm()) ** 2
                for p in lit.parameters() if p.grad is not None)
        return n ** 0.5

    base = grad_norm(_batch())

    coincident = _batch()
    coincident["dart_box"][:, 1] = coincident["dart_box"][:, 0].clone()
    coincident["dart_box"][:, 1, 0] += 0.002        # 2 px apart at 1024
    coincident["instance"][:, 12:14, 9:14] = 2      # contested cells

    occluded = _batch()
    occluded["instance"][occluded["instance"] == 2] = 0
    occluded["instance"][0, 20, 18] = 2             # one visible cell left

    degenerate = _batch()
    degenerate["dart_box"][:, 1, 4:6] = 1e-7        # end-on, box collapses

    for name, b in (("coincident", coincident), ("occluded", occluded),
                    ("degenerate", degenerate)):
        got = grad_norm(b)
        assert got < base * 3.0, f"{name}: {got:.2f} vs clean {base:.2f}"


def test_a_nonfinite_target_is_caught_as_a_loss_failure_not_a_gradient_one():
    """The discriminator that tells a bad label from a bad backward.

    A NaN reaching the loss from the DATA must land in the loss counter, so
    that the gradient counter means what it says.
    """
    lit = DenseDartLitModule(_cfg())
    logged = {}
    lit.log = lambda n, v, **k: logged.__setitem__(n, float(v))
    batch = _batch()
    batch["dart_box"][:, 1, 0] = float("nan")
    assert lit._step(batch, "train") is None
    assert logged["train/skipped_loss"] == 1.0
    assert "train/skipped_grad" not in logged


# ------------------------------------------------------------ keypoints

def _kp_batch(B=2, SZ=256, stride=8, K=40):
    """A batch carrying the 40 double-bed corners as well as the darts."""
    b = _batch(B=B, SZ=SZ, stride=stride)
    # Spread around a ring, which is where they actually live, and keep them
    # off the very edge so none is masked out by the in-frame test.
    ang = torch.arange(K, dtype=torch.float32) / K * 2 * math.pi
    kp = torch.stack([0.5 + 0.33 * ang.cos(), 0.5 + 0.33 * ang.sin()], dim=-1)
    b["keypoints"] = kp.unsqueeze(0).repeat(B, 1, 1)
    b["keypoint_mask"] = torch.ones(B, K, dtype=torch.bool)
    return b


def test_keypoint_targets_peak_on_the_keypoint_and_carry_the_subcell_offset():
    """A cell is 8px wide, so the heatmap alone cannot locate a corner better
    than that; the offset is what the remaining precision comes from."""
    h = w = 32
    kp = torch.tensor([[[0.2571, 0.7143]]])          # deliberately off-centre
    mask = torch.ones(1, 1, dtype=torch.bool)
    heat, off, at = keypoint_targets(kp, mask, (h, w), sigma=2.0)

    assert heat.shape == (1, 1, h, w)
    peak = heat[0, 0].flatten().argmax()
    iy, ix = int(peak) // w, int(peak) % w
    assert ix == int(0.2571 * w) and iy == int(0.7143 * h)
    # Exactly 1.0, not merely the sampled Gaussian's value there: the focal
    # loss counts a cell as positive only at 1.0, and sampled at cell centres
    # the exponential peaks below it, so without pinning nothing is ever a
    # positive and the head trains on the negative term alone.
    assert float(heat[0, 0, iy, ix]) == 1.0

    # Offset recovers the exact position, which is the whole point of it.
    assert bool(at[0, iy, ix])
    x = (ix + 0.5 + float(off[0, 0, iy, ix])) / w
    y = (iy + 0.5 + float(off[0, 1, iy, ix])) / h
    assert x == pytest.approx(0.2571, abs=1e-5)
    assert y == pytest.approx(0.7143, abs=1e-5)


def test_a_masked_keypoint_is_not_supervised_anywhere():
    """An off-image corner must produce no target at all. Clamping it to the
    edge instead would be a confident label on the wrong pixel, which trains
    worse than no label."""
    heat, off, at = keypoint_targets(
        torch.tensor([[[0.5, 0.5]]]), torch.zeros(1, 1, dtype=torch.bool),
        (32, 32), sigma=2.0)
    assert float(heat.max()) == 0.0
    assert float(off.abs().max()) == 0.0
    assert not bool(at.any())


def test_the_keypoint_head_is_decoupled_from_the_dart_head():
    """Sharing features downstream of the neck lets the board loss
    destabilise dart training. Backbone and neck are shared on purpose;
    nothing after them is."""
    net = DenseDartNet(_cfg(predict_keypoints=True))
    kp_params = {id(p) for p in net.kp_trunk.parameters()}
    kp_params |= {id(p) for p in net.kp_head.parameters()}
    kp_params |= {id(p) for p in net.kp_offset.parameters()}
    dart_params = {id(p) for p in net.head.parameters()}
    dart_params |= {id(p) for p in net.fg_head.parameters()}
    dart_params |= {id(p) for p in net.box_head.parameters()}
    assert not (kp_params & dart_params)

    # And the keypoint loss must not reach the dart head's weights at all.
    lit = DenseDartLitModule(_cfg(predict_keypoints=True))
    lit.log = lambda *a, **k: None
    out = lit.model(_kp_batch()["image"].float())
    h, w = out["fg_logits"].shape[-2:]
    b = _kp_batch()
    heat, _, _ = keypoint_targets(b["keypoints"], b["keypoint_mask"], (h, w), 2.0)
    lit.zero_grad()
    out["kp_logits"].mul(heat).sum().backward()
    for name, prm in lit.model.named_parameters():
        if name.startswith(("head.", "fg_head.", "box_head.")):
            assert prm.grad is None or float(prm.grad.abs().sum()) == 0.0, name


def test_overfits_the_forty_board_corners():
    """The readout is argmax plus offset, exactly as inference does it -- not a
    soft-argmax, which would let a broad, badly-peaked heatmap average its way
    to the right answer and report a precision the head does not have."""
    torch.manual_seed(0)
    lit = DenseDartLitModule(_cfg(predict_keypoints=True))
    lit.train()
    lit.log = lambda *a, **k: None
    batch = _kp_batch()
    opt = torch.optim.Adam(lit.parameters(), lr=3e-3)
    for _ in range(220):
        loss = lit._step(batch, "train")
        opt.zero_grad()
        loss.backward()
        opt.step()

    lit.eval()
    logged = {}
    lit.log = lambda n, v, **k: logged.__setitem__(n, float(v))
    with torch.no_grad():
        lit.on_validation_epoch_start()
        lit._step(batch, "val")
        lit.on_validation_epoch_end()

    assert "val/kp_px_error" in logged
    # A cell is 8px at this stride; beating that means the offset is working
    # rather than the heatmap alone.
    assert logged["val/kp_px_error"] < 8.0, logged["val/kp_px_error"]
    assert logged["val/kp_px_max"] < 40.0, logged["val/kp_px_max"]


def test_the_defaults_build_the_released_model():
    """Both optional heads are on in the shipped model and the exporters read
    their outputs, so the defaults build them."""
    cfg = DenseDartConfig()
    assert cfg.predict_ends and cfg.predict_keypoints
    out = DenseDartNet(_cfg())(_batch()["image"].float())
    for key in ("fg_logits", "centre_offset", "direction", "log_extent",
                "tip_offset", "flight_offset", "kp_logits", "kp_offset"):
        assert key in out, key


def test_the_optional_heads_can_be_turned_off():
    out = DenseDartNet(_cfg(predict_ends=False, predict_keypoints=False))(
        _batch()["image"].float())
    assert not {"tip_offset", "flight_offset", "kp_logits"} & set(out)
    lit = DenseDartLitModule(_cfg(predict_ends=False, predict_keypoints=False))
    lit.log = lambda *a, **k: None
    assert lit._step(_kp_batch(), "train") is not None


# ----------------------------------------------------------------- ends

def test_every_cell_of_a_dart_votes_for_the_same_two_points():
    """The property the accumulator depends on.

    Cells at opposite ends of one dart emit DIFFERENT offsets -- each is
    relative to its own cell -- that decode to the same pair of absolute
    points. Averaging the raw channels instead would be wrong.
    """
    h = w = 8
    out = {"centre_offset": torch.zeros(1, 2, h, w),
           "direction": torch.zeros(1, 2, h, w),
           "log_extent": torch.zeros(1, 2, h, w)}
    centres = cell_centres(h, w, out["centre_offset"].device)
    out["direction"][:, 0] = 1.0
    out["tip_offset"] = (torch.tensor([0.3, 0.4]).view(1, 2, 1, 1)
                         - centres[None])
    out["flight_offset"] = (torch.tensor([0.8, 0.9]).view(1, 2, 1, 1)
                            - centres[None])
    dec = decode_boxes(out, (h, w))
    assert torch.allclose(dec["tip_point"][0],
                          torch.tensor([0.3, 0.4]).view(2, 1, 1)
                          .expand(2, h, w), atol=1e-6)
    assert torch.allclose(dec["flight_point"][0],
                          torch.tensor([0.8, 0.9]).view(2, 1, 1)
                          .expand(2, h, w), atol=1e-6)


def test_the_readout_prefers_the_predicted_tip_over_the_box_end():
    """When the head offers a landing point, voting must use it.

    Falling back to the box end would throw away the one output that handles a
    dart pointing at the camera, where the entry projects into the interior of
    the silhouette and no end of any box reaches it.
    """
    h = w = 16
    centres = cell_centres(h, w, torch.device("cpu"))
    out = {"centre_offset": torch.full((1, 2, h, w), 0.0),
           "direction": torch.zeros(1, 2, h, w),
           "log_extent": torch.log(torch.full((1, 2, h, w), 0.05))}
    out["direction"][:, 0] = 1.0
    out["centre_offset"] = (torch.tensor([0.5, 0.5]).view(1, 2, 1, 1)
                            - centres[None])
    # A landing point deliberately NOT at the box end (which is 0.45, 0.50).
    land = torch.tensor([0.37, 0.53])
    out["tip_offset"] = land.view(1, 2, 1, 1) - centres[None]
    out["flight_offset"] = (torch.tensor([0.55, 0.50]).view(1, 2, 1, 1)
                            - centres[None])

    fg = torch.zeros(h, w)
    fg[6:10, 6:10] = 1.0
    dec = decode_boxes(out, (h, w))
    got = detect({k: v[0] for k, v in dec.items()}, fg, _cfg())
    assert len(got) == 1
    assert got[0]["tip_is_predicted"]
    assert torch.allclose(got[0]["tip"], land, atol=1e-4), got[0]["tip"]
    # ...and it is genuinely different from what the box would have given.
    box_end = got[0]["centre"] - got[0]["direction"] * got[0]["half_length"]
    assert float((box_end - land).norm()) > 0.05


def test_the_ends_are_learnable_and_reported_separately():
    torch.manual_seed(0)
    lit = DenseDartLitModule(_cfg(predict_ends=True))
    lit.train()
    lit.log = lambda *a, **k: None
    batch = _kp_batch()
    opt = torch.optim.Adam(lit.parameters(), lr=3e-3)
    for _ in range(200):
        loss = lit._step(batch, "train")
        opt.zero_grad()
        loss.backward()
        opt.step()

    lit.eval()
    logged = {}
    lit.log = lambda n, v, **k: logged.__setitem__(n, float(v))
    with torch.no_grad():
        lit.on_validation_epoch_start()
        lit._step(batch, "val")
        lit.on_validation_epoch_end()

    assert "val/landing_px_error" in logged
    assert "val/flight_px_error" in logged
    # The box's own end against the ground-truth box's end, reported
    # alongside: it measures the box, not the landing head.
    assert "val/tip_px_error" in logged
    assert logged["val/landing_px_error"] < 25.0, logged["val/landing_px_error"]


def test_focal_loss_survives_a_saturating_logit_in_bfloat16():
    """Clamping p to [1e-4, 1 - 1e-4] is not a guard under bf16.

    bf16 has 8 mantissa bits, so the smallest step below 1.0 is 2^-8 and
    0.9999 rounds to exactly 1.0 -- the upper bound does nothing, and
    (1 - p).log() becomes log(0). sigmoid saturates there for any logit past
    ~6.2, which 40 keypoint channels reach constantly.
    """
    # 1 - 1e-4 really is not representable, which is the whole premise.
    assert float(torch.tensor(1.0 - 1e-4, dtype=torch.bfloat16)) == 1.0

    target = torch.zeros(1, 4, 8, 8)
    target[0, 0, 4, 4] = 1.0
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        logits = torch.full((1, 4, 8, 8), 12.0, dtype=dtype)
        loss = penalty_reduced_focal(logits, target)
        assert torch.isfinite(loss), f"{dtype}: {loss}"
    # Large negative saturates the other end.
    for dtype in (torch.bfloat16, torch.float32):
        logits = torch.full((1, 4, 8, 8), -40.0, dtype=dtype)
        assert torch.isfinite(penalty_reduced_focal(logits, target))


def test_focal_loss_still_agrees_with_the_clamped_form_where_that_was_valid():
    """The log-space form computes the same loss as the naive clamped one.

    Checked in float32 and away from saturation, the regime where the clamped
    form is correct -- otherwise this would just be comparing one formula
    against itself.
    """
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 16, 16) * 1.5
    target = torch.rand(2, 3, 16, 16).clamp(max=0.99)
    target[0, 0, 5, 5] = 1.0

    p = logits.sigmoid().clamp(1e-4, 1.0 - 1e-4)
    pos = target.ge(1.0 - 1e-6).float()
    old = ((-((1.0 - p) ** 2.0) * p.log() * pos).sum()
           + (-((1.0 - target) ** 4.0) * (p ** 2.0)
              * (1.0 - p).log() * (1.0 - pos)).sum()) / pos.sum().clamp(min=1.0)
    assert float(penalty_reduced_focal(logits, target)) == pytest.approx(
        float(old), rel=1e-5)
