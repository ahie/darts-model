"""Training-loop behaviour of the detector module: empty frames, matching,
checkpoint monitoring, the frozen backbone and the optimiser setup."""
from __future__ import annotations

import math
import warnings

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from darts_model.data.detect_dataset import MAX_DARTS
from darts_model.model.backbone import FineBackbone, FineBackboneConfig
from darts_model.model.detector import (
    DenseDartConfig, DenseDartLitModule, DenseDartNet, match_points,
)


def _cfg(**kw):
    base = dict(backbone_widths=(16, 24, 32, 48), backbone_depths=(1, 1, 1, 1),
                head_width=16, head_depth=1, kp_head_width=16, kp_head_depth=1,
                backbone_freeze_epochs=0, max_epochs=10, warmup_epochs=2,
                steps_per_epoch=10)
    base.update(kw)
    return DenseDartConfig(**base)


def _frame(n_darts: int, sz: int = 128, stride: int = 8) -> dict:
    """One sample as the dataset yields it, with 0-2 darts."""
    h = sz // stride
    inst = torch.zeros(h, h, dtype=torch.long)
    box = torch.zeros(MAX_DARTS, 6)
    ends = torch.zeros(MAX_DARTS, 4)
    mask = torch.zeros(MAX_DARTS, dtype=torch.bool)
    darts = [((0.3, 0.3), slice(4, 6), slice(2, 8)),
             ((0.7, 0.7), slice(10, 12), slice(8, 14))][:n_darts]
    for i, ((cx, cy), ys, xs) in enumerate(darts):
        inst[ys, xs] = i + 1
        box[i] = torch.tensor([cx, cy, 1.0, 0.0, 0.2, 0.06])
        ends[i] = torch.tensor([cx - 0.18, cy, cx + 0.2, cy])
        mask[i] = True
    k = torch.arange(40, dtype=torch.float32) / 40 * 2 * math.pi
    return {
        "image": torch.randint(0, 256, (3, sz, sz), dtype=torch.uint8),
        "instance": inst,
        "foreground": (inst > 0).float(),
        "dart_box": box,
        "dart_ends": ends,
        "dart_mask": mask,
        "keypoints": torch.stack([0.5 + 0.3 * k.cos(), 0.5 + 0.3 * k.sin()], -1),
        "keypoint_mask": torch.ones(40, dtype=torch.bool),
    }


def _batch(counts) -> dict:
    frames = [_frame(n) for n in counts]
    b = {k: torch.stack([f[k] for f in frames]) for k in frames[0]}
    b["image"] = b["image"].float()
    return b


def _logging(lit) -> dict:
    logged: dict = {}
    lit.log = lambda n, v, **k: logged.__setitem__(
        n, float(v.detach()) if torch.is_tensor(v) else float(v))
    return logged


# ------------------------------------------------------------ empty frames

@pytest.mark.parametrize("counts", [(0, 0), (0, 2), (1, 0)])
def test_losses_are_finite_with_zero_dart_frames(counts):
    lit = DenseDartLitModule(_cfg())
    logged = _logging(lit)
    loss = lit._step(_batch(counts), "train")
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    for name, p in lit.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), name
    if counts == (0, 0):
        # No dart cells anywhere: every masked term is exactly zero, while the
        # foreground and keypoint terms still supervise.
        for k in ("centre", "dir", "size", "tip", "flight"):
            assert logged[f"train/{k}_loss"] == 0.0, k
        assert logged["train/kp_loss"] > 0.0


def test_padding_in_masked_dart_slots_cannot_poison_the_loss():
    """A masked-out slot may hold anything; NaN * 0 would still be NaN."""
    lit = DenseDartLitModule(_cfg())
    lit.log = lambda *a, **k: None
    b = _batch((0, 1))
    b["dart_box"][0] = float("nan")
    b["dart_box"][1, 1:] = float("nan")
    b["dart_ends"][0] = float("nan")
    loss = lit._step(b, "train")
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in lit.parameters()
               if p.grad is not None)


def test_validation_on_an_all_empty_epoch_logs_the_monitor_as_the_gate():
    lit = DenseDartLitModule(_cfg())
    logged = _logging(lit)
    lit.eval()
    with torch.no_grad():
        lit.on_validation_epoch_start()
        lit._step(_batch((0, 0)), "val")
        lit.on_validation_epoch_end()
    assert logged[DenseDartLitModule.MONITOR] == lit.config.match_gate_px
    assert logged["val/fp_per_empty_frame"] == 0.0
    assert "val/recall" not in logged
    for v in logged.values():
        assert math.isfinite(v)


def test_detections_on_an_empty_frame_are_false_positives(monkeypatch):
    import darts_model.model.hough as hough

    fake = [{"tip": torch.tensor([0.5, 0.5]), "flight": torch.tensor([0.6, 0.5]),
             "box_tip": torch.tensor([0.5, 0.5])}] * 2
    monkeypatch.setattr(hough, "detect", lambda *a, **k: fake)
    lit = DenseDartLitModule(_cfg())
    logged = _logging(lit)
    lit.eval()
    with torch.no_grad():
        lit.on_validation_epoch_start()
        lit._step(_batch((0, 0, 0)), "val")
        lit.on_validation_epoch_end()
    assert logged["val/fp_per_empty_frame"] == 2.0
    assert logged["val/precision"] == 0.0


# ---------------------------------------------------------------- matching

def test_one_prediction_cannot_serve_two_darts():
    gt = torch.tensor([[100.0, 100.0], [110.0, 100.0]])
    pred = torch.tensor([[105.0, 100.0]])
    gi, pi, d = match_points(gt, pred, gate=40.0)
    assert gi.numel() == 1 and pi.tolist() == [0]


def test_matching_maximises_matches_before_minimising_distance():
    # Nearest-first would pair gt0 with pred1 (1px) and strand gt1 (pred0 is
    # 33px from it, outside the gate). The optimum matches both.
    gt = torch.tensor([[0.0, 0.0], [18.0, 0.0]])
    pred = torch.tensor([[-15.0, 0.0], [1.0, 0.0]])
    gi, pi, d = match_points(gt, pred, gate=20.0)
    assert sorted(zip(gi.tolist(), pi.tolist())) == [(0, 0), (1, 1)]


def test_matches_outside_the_gate_are_misses():
    gi, pi, d = match_points(torch.tensor([[0.0, 0.0]]),
                             torch.tensor([[50.0, 0.0]]), gate=40.0)
    assert gi.numel() == 0


def test_a_missed_dart_costs_the_gate_and_is_not_dropped(monkeypatch):
    """One dart found 3px off, one missed: the plain landing error is 3, the
    penalised one (3 + gate) / 2, recall 1/2."""
    import darts_model.model.hough as hough

    b = _batch((2,))
    scale = b["image"].shape[-1]
    land = b["dart_ends"][0, 0, 0:2]
    hit = land + torch.tensor([3.0 / scale, 0.0])
    monkeypatch.setattr(hough, "detect", lambda *a, **k: [
        {"tip": hit, "flight": b["dart_ends"][0, 0, 2:4],
         "box_tip": b["dart_box"][0, 0, 0:2]}])
    lit = DenseDartLitModule(_cfg())
    logged = _logging(lit)
    lit.eval()
    with torch.no_grad():
        lit.on_validation_epoch_start()
        lit._step(b, "val")
        lit.on_validation_epoch_end()
    gate = lit.config.match_gate_px
    assert logged["val/recall"] == 0.5
    assert logged["val/precision"] == 1.0
    assert logged["val/landing_px_error"] == pytest.approx(3.0, abs=1e-3)
    assert logged[DenseDartLitModule.MONITOR] == pytest.approx(
        (3.0 + gate) / 2, abs=1e-3)


# ------------------------------------------------------------- the trainer

class _DS(Dataset):
    def __init__(self, counts):
        self.counts = counts

    def __len__(self):
        return len(self.counts)

    def __getitem__(self, i):
        return _frame(self.counts[i])


class _Recorder(L.Callback):
    """The monitor's value as each validation run leaves it, with any value
    from an earlier epoch removed first, so a stale one cannot pass."""

    def __init__(self):
        self.seen = []

    def on_validation_start(self, trainer, module):
        trainer.callback_metrics.pop(DenseDartLitModule.MONITOR, None)

    def on_validation_end(self, trainer, module):
        self.seen.append(trainer.callback_metrics.get(DenseDartLitModule.MONITOR))


def test_fit_with_an_untrained_model_ranks_checkpoints_every_epoch(tmp_path):
    """An untrained model detects nothing. ModelCheckpoint raises if its
    monitor is missing, so the monitor has to be logged every epoch
    regardless -- including with a frozen backbone and empty frames."""
    torch.manual_seed(0)
    lit = DenseDartLitModule(_cfg(steps_per_epoch=2, max_epochs=2,
                                  backbone_freeze_epochs=1))
    rec = _Recorder()
    trainer = L.Trainer(
        max_epochs=2, accelerator="cpu", devices=1, logger=False,
        enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[rec, L.pytorch.callbacks.ModelCheckpoint(
            dirpath=tmp_path, monitor=DenseDartLitModule.MONITOR, mode="min",
            save_top_k=3, save_last=True)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.fit(lit, DataLoader(_DS([1, 2, 0, 1]), batch_size=2),
                    DataLoader(_DS([2, 0, 1, 1]), batch_size=2))
    assert trainer.current_epoch == 2
    assert len(rec.seen) == 2
    assert all(v is not None for v in rec.seen)
    assert float(rec.seen[-1]) == lit.config.match_gate_px
    assert len(list(tmp_path.glob("*.ckpt"))) >= 2


def test_multi_process_training_is_refused():
    lit = DenseDartLitModule(_cfg())

    class _T:
        world_size = 2

    lit._trainer = _T()
    with pytest.raises(NotImplementedError, match="single-device"):
        lit.setup("fit")


# ---------------------------------------------------------- frozen backbone

def _drop_path_active(bb: FineBackbone) -> bool:
    return any(m.training for m in bb.modules()
               if type(m).__name__ == "DropPath")


def test_a_frozen_backbone_stays_in_eval_mode():
    """Stochastic depth must not keep perturbing features the head is being
    fitted against while nothing in the backbone can learn."""
    bb = FineBackbone(FineBackboneConfig(widths=(8, 8, 8, 8),
                                         depths=(1, 1, 1, 1), drop_path=0.5))
    bb.freeze()
    bb.train()                      # what Lightning does every epoch
    assert not bb.training and not _drop_path_active(bb)
    x = torch.randn(2, 3, 64, 64)
    assert torch.equal(bb(x)[0], bb(x)[0])

    bb.unfreeze()                   # restores the mode the parent asked for
    assert bb.training and _drop_path_active(bb)
    bb.eval()
    bb.freeze()
    bb.unfreeze()
    assert not bb.training


def test_the_detector_keeps_its_frozen_backbone_in_eval_mode():
    net = DenseDartNet(_cfg(backbone_freeze_epochs=3))
    net.train()
    assert not net.backbone.training
    assert net.head.training
    net.backbone.unfreeze()
    net.train()
    assert net.backbone.training


# ------------------------------------------------------------------- optim

def _optim(lit):
    class _T:
        estimated_stepping_batches = 100
    lit._trainer = _T()
    got = lit.configure_optimizers()
    return got["optimizer"], got["lr_scheduler"]["scheduler"]


def test_one_dimensional_parameters_get_no_weight_decay():
    lit = DenseDartLitModule(_cfg())
    opt, _ = _optim(lit)
    for g in opt.param_groups:
        for p in g["params"]:
            assert (g["weight_decay"] == 0.0) == (p.ndim <= 1), g["name"]
    gammas = {id(p) for n, p in lit.model.named_parameters()
              if n.endswith(".gamma")}
    assert gammas
    for g in opt.param_groups:
        if any(id(p) in gammas for p in g["params"]):
            assert g["weight_decay"] == 0.0
    assert sum(len(g["params"]) for g in opt.param_groups) == len(
        list(lit.model.parameters()))


def test_the_backbone_rewarms_after_the_unfreeze():
    cfg = _cfg(backbone_freeze_epochs=4, backbone_rewarmup_epochs=2,
               steps_per_epoch=10, max_epochs=20)
    lit = DenseDartLitModule(cfg)
    opt, sched = _optim(lit)
    lams = dict(zip((g["name"] for g in opt.param_groups), sched.lr_lambdas))
    base = lams["head"]
    assert lams["backbone"](10) == 0.0           # still frozen
    lit._unfreeze_step = 40
    assert lams["backbone"](40) == 0.0
    assert lams["backbone"](50) == pytest.approx(0.5 * base(50))
    assert lams["backbone"](60) == pytest.approx(base(60))
    assert lams["backbone"](150) == pytest.approx(base(150))
    assert lams["backbone_no_decay"](50) == pytest.approx(0.5 * base(50))
    # Never frozen: no re-warmup at all.
    lit2 = DenseDartLitModule(_cfg(backbone_freeze_epochs=0))
    opt2, sched2 = _optim(lit2)
    lams2 = dict(zip((g["name"] for g in opt2.param_groups), sched2.lr_lambdas))
    assert lams2["backbone"](5) == pytest.approx(lams2["head"](5))


def test_the_unfreeze_step_survives_a_resume():
    lit = DenseDartLitModule(_cfg(backbone_freeze_epochs=2))
    lit._unfreeze_step = 123
    ck: dict = {}
    lit.on_save_checkpoint(ck)
    lit2 = DenseDartLitModule(_cfg(backbone_freeze_epochs=2))
    lit2.on_load_checkpoint(ck)
    assert lit2._unfreeze_step == 123


def test_beta1_ends_the_warmup_at_exactly_point_nine(monkeypatch):
    lit = DenseDartLitModule(_cfg())
    opt, _ = _optim(lit)
    nw = lit._warmup_steps
    monkeypatch.setattr(L.LightningModule, "optimizer_step",
                        lambda *a, **k: None)
    seen = []
    for step in (0, nw - 1, nw, nw + 5):
        monkeypatch.setattr(DenseDartLitModule, "global_step",
                            property(lambda self, s=step: s))
        lit.optimizer_step(0, 0, opt, None)
        seen.append(opt.param_groups[0]["betas"][0])
    assert seen[0] == pytest.approx(lit.config.warmup_beta1)
    assert seen[1] < 0.9
    assert seen[2] == 0.9 and seen[3] == 0.9


# ------------------------------------------------------------------ config

def test_pan_with_a_dilated_last_stage_is_rejected():
    with pytest.raises(ValueError, match="use_pan"):
        DenseDartConfig(use_pan=True, backbone_dilate_last=True)
    DenseDartNet(_cfg(use_pan=False, backbone_dilate_last=True))(
        torch.randn(1, 3, 128, 128))


# --------------------------------------------------------------- precision

def test_regression_outputs_are_float32_under_bf16_autocast():
    net = DenseDartNet(_cfg()).eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = net(torch.randn(1, 3, 128, 128))
    for k in ("centre_offset", "direction", "log_extent", "tip_offset",
              "flight_offset", "kp_offset"):
        assert out[k].dtype == torch.float32, k
    # The trunk itself still runs in bf16; only the output layers are lifted.
    assert out["fg_logits"].dtype == torch.bfloat16


def test_end_on_darts_are_left_out_of_the_orientation_loss() -> None:
    """A cell of an end-on dart supervises centre and ends, not direction or
    size: its box is a fallback whose orientation is not the dart's."""
    from darts_model.model.detector import DenseDartLitModule
    keep = torch.tensor([[[True, True, False]]])
    batch = {"instance": torch.tensor([[[1, 2, 0]]]),
             "dart_box_mask": torch.tensor([[True, False, False]])}
    box_keep, n = DenseDartLitModule._box_keep(batch, keep)
    assert box_keep.tolist() == [[[True, False, False]]]
    assert float(n) == 1.0
    del batch["dart_box_mask"]
    box_keep, _ = DenseDartLitModule._box_keep(batch, keep)
    assert box_keep.tolist() == keep.tolist()
