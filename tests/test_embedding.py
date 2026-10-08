"""The instance embedding: its loss, the readout that groups by it, and adding
it to a trained model."""
from dataclasses import dataclass

import pytest
import torch

from darts_model.model.detector import (
    DenseDartConfig, DenseDartLitModule, decode_boxes, embedding_loss,
)
from darts_model.model.hough import detect

from test_dense_dart import _batch, _cfg


def _inst():
    inst = torch.zeros(1, 4, 4, dtype=torch.long)
    inst[0, 0, :2] = 1
    inst[0, 2, :2] = 2
    return inst


def _emb(a, b):
    """Dart 1's cells at ``a``, dart 2's at ``b``, background at the origin."""
    e = torch.zeros(1, 2, 4, 4)
    e[0, :, 0, :2] = torch.tensor(a, dtype=torch.float)[:, None]
    e[0, :, 2, :2] = torch.tensor(b, dtype=torch.float)[:, None]
    return e


def test_separated_coherent_darts_cost_nothing():
    pull, push, _ = embedding_loss(_emb((0.0, 0.0), (5.0, 0.0)), _inst(), 3,
                                   0.5, 1.5)
    assert float(pull) == 0.0 and float(push) == 0.0


def test_darts_closer_than_twice_the_push_margin_are_pushed():
    _, push, _ = embedding_loss(_emb((0.0, 0.0), (1.0, 0.0)), _inst(), 3,
                                0.5, 1.5)
    assert float(push) == pytest.approx((3.0 - 1.0) ** 2)


def test_a_cell_off_its_darts_mean_is_pulled_past_the_margin_only():
    e = _emb((0.0, 0.0), (5.0, 0.0))
    e[0, 0, 0, 0] = 2.0                 # dart 1: cells at x=2 and x=0, mean 1
    pull, _, _ = embedding_loss(e, _inst(), 3, 0.5, 1.5)
    # Each of dart 1's cells is 1 from the mean: (1 - 0.5)^2; dart 2 costs 0.
    assert float(pull) == pytest.approx(0.25 / 2)


def test_background_and_excluded_cells_do_not_take_part():
    e = _emb((0.0, 0.0), (5.0, 0.0))
    e[0, :, 3, 3] = 100.0               # background
    e[0, 0, 0, 0] = 100.0               # dart 1, but excluded
    include = torch.ones(1, 4, 4, dtype=torch.bool)
    include[0, 0, 0] = False
    pull, push, _ = embedding_loss(e, _inst(), 3, 0.5, 1.5, include)
    assert float(pull) == 0.0 and float(push) == 0.0


def test_a_single_dart_frame_has_no_push_and_a_finite_gradient():
    """One dart of one cell sits exactly on its own mean, where a plain norm's
    gradient is 0/0."""
    inst = torch.zeros(1, 4, 4, dtype=torch.long)
    inst[0, 1, 1] = 1
    e = torch.randn(1, 3, 4, 4, requires_grad=True)
    pull, push, reg = embedding_loss(e, inst, 3, 0.5, 1.5)
    (pull + push + reg).backward()
    assert float(push.detach()) == 0.0
    assert torch.isfinite(e.grad).all()


def test_an_empty_batch_gives_exact_zeros():
    e = torch.randn(2, 3, 4, 4, requires_grad=True)
    pull, push, reg = embedding_loss(e, torch.zeros(2, 4, 4, dtype=torch.long),
                                     3, 0.5, 1.5)
    assert float(pull + push + reg) == 0.0
    (pull + push + reg).backward()


def test_push_is_within_a_frame_only():
    """Darts in different frames of a batch are never compared: identity is
    only ever needed to tell apart darts seen together."""
    inst = torch.zeros(2, 4, 4, dtype=torch.long)
    inst[:, 0, :2] = 1
    e = torch.zeros(2, 2, 4, 4)                     # both darts at the origin
    _, push, _ = embedding_loss(e, inst, 3, 0.5, 1.5)
    assert float(push) == 0.0


def test_the_embedding_trains_and_separates_the_darts():
    torch.manual_seed(0)
    lit = DenseDartLitModule(_cfg(embed_dim=4))
    lit.train()
    lit.log = lambda *a, **k: None
    batch = _batch()
    opt = torch.optim.Adam(lit.parameters(), lr=3e-3)
    for _ in range(150):
        loss = lit._step(batch, "train")
        opt.zero_grad()
        loss.backward()
        opt.step()
    lit.eval()
    with torch.no_grad():
        out = lit.model(batch["image"].float())
        pull, push, _ = embedding_loss(out["embedding"], batch["instance"], 3,
                                       0.5, 1.5)
    assert float(pull) < 0.05 and float(push) < 0.05


def test_claim_by_embedding_requires_the_head():
    with pytest.raises(ValueError, match="embed_dim"):
        DenseDartConfig(claim_by_embedding=True)


# ------------------------------------------------------------------ readout

@dataclass
class _Cfg:
    out_stride: int = 8
    vote_bin_px: float = 4.0
    fg_threshold: float = 0.5
    peak_min_votes: int = 8
    embed_push_margin: float = 1.5
    claim_by_embedding: bool = True


def _two_darts_one_strayed():
    """Two darts of 20 cells each, far apart; six of dart A's cells vote a
    centre five bins off A's peak but carry A's embedding and A's tip.

    Centre-bin grouping loses those six -- they fall outside every peak's 3x3
    neighbourhood and are too few to make a peak of their own. Grouping by
    embedding returns them to A.
    """
    h = w = 32
    fg = torch.zeros(h, w)
    centre = torch.zeros(2, h, w)
    direction = torch.zeros(2, h, w)
    direction[0] = 1.0
    extent = torch.full((2, h, w), 0.05)
    tip = torch.zeros(2, h, w)
    emb = torch.zeros(2, h, w)
    for row, (cx, cy, tx, e) in ((4, (0.25, 0.25, 0.20, 0.0)),
                                 (20, (0.75, 0.75, 0.70, 5.0))):
        for k in range(20):
            yy, xx = row + k // 10, 4 + k % 10
            fg[yy, xx] = 0.9
            centre[:, yy, xx] = torch.tensor([cx, cy])
            tip[:, yy, xx] = torch.tensor([tx, cy])
            emb[:, yy, xx] = torch.tensor([e, 0.0])
    for k in range(6):                          # strays of dart A
        yy, xx = 8, 4 + k
        fg[yy, xx] = 0.9
        centre[:, yy, xx] = torch.tensor([0.25 + 5 * 4 / 256, 0.25])
        tip[:, yy, xx] = torch.tensor([0.20, 0.25])
        emb[:, yy, xx] = torch.tensor([0.1, 0.0])
    field = {"centre": centre, "direction": direction, "extent": extent,
             "tip_point": tip, "flight_point": tip.clone(), "embedding": emb}
    return field, fg


def test_embedding_grouping_recovers_cells_whose_centre_vote_strayed():
    field, fg = _two_darts_one_strayed()
    by_centre = detect(field, fg, _Cfg(), claim_by_embedding=False)
    by_embed = detect(field, fg, _Cfg())
    assert sorted(d["votes"] for d in by_centre) == [20, 20]
    assert sorted(d["votes"] for d in by_embed) == [20, 26]


def test_a_cell_far_from_every_seed_joins_no_dart():
    field, fg = _two_darts_one_strayed()
    field["embedding"][:, 8, 4:10] = torch.tensor([2.5, 9.0])[:, None]
    got = detect(field, fg, _Cfg())
    assert sorted(d["votes"] for d in got) == [20, 20]


def test_embedding_grouping_without_an_embedding_is_an_error():
    field, fg = _two_darts_one_strayed()
    del field["embedding"]
    with pytest.raises(ValueError, match="embedding"):
        detect(field, fg, _Cfg())


def test_decode_passes_the_embedding_through():
    lit = DenseDartLitModule(_cfg(embed_dim=4)).eval()
    with torch.no_grad():
        out = lit.model(torch.randn(1, 3, 256, 256))
    dec = decode_boxes(out, out["fg_logits"].shape[-2:])
    assert dec["embedding"].shape == (1, 4, 32, 32)


def test_validation_logs_both_groupings_and_the_cluster_metrics():
    lit = DenseDartLitModule(_cfg(embed_dim=4)).eval()
    logged = {}
    lit.log = lambda name, value, *a, **k: logged.__setitem__(name, value)
    batch = _batch()
    # Bring the two darts' landing points within the cluster distance.
    batch["dart_ends"][:, 1, 0:2] = batch["dart_ends"][:, 0, 0:2] + 0.01
    # A head that fires everywhere, so every dart cell is measured.
    with torch.no_grad():
        lit.model.fg_head.bias.fill_(10.0)
        lit._step(batch, "val")
    lit.on_validation_epoch_end()
    assert "val/landing_px_error_penalised_embedding_claim" in logged
    assert 0.0 <= float(logged["val/cluster_cells_wrong_dart"]) <= 1.0
    assert 0.0 <= float(logged["val/cluster_cells_wrong_embedding"]) <= 1.0


# ------------------------------------------------------------------ init

def test_a_finetune_may_add_the_embedding_head_to_a_trained_model(tmp_path):
    trained = DenseDartLitModule(_cfg())
    path = tmp_path / "trained.ckpt"
    torch.save({"state_dict": trained.state_dict(), "epoch": 7}, path)
    lit = DenseDartLitModule(_cfg(embed_dim=4, init_weights=str(path)))
    w = trained.model.box_head.weight
    assert torch.equal(lit.model.box_head.weight, w)


def test_a_finetune_still_refuses_any_other_missing_tensor(tmp_path):
    trained = DenseDartLitModule(_cfg(predict_keypoints=False))
    path = tmp_path / "trained.ckpt"
    torch.save({"state_dict": trained.state_dict()}, path)
    with pytest.raises(RuntimeError, match="NOT in the checkpoint"):
        DenseDartLitModule(_cfg(embed_dim=4, init_weights=str(path)))


def test_the_embedding_term_ramps_in_during_training_only():
    lit = DenseDartLitModule(_cfg(embed_dim=4))
    lit.log = lambda *a, **k: None
    batch = _batch()
    torch.manual_seed(0)
    full = float(lit._step(batch, "train"))
    lit._embed_ramp_steps = 1000                 # step 0 of 1000: weight 1e-3
    torch.manual_seed(0)
    ramped = float(lit._step(batch, "train"))
    torch.manual_seed(0)
    val = float(lit._step(batch, "val"))
    assert ramped < full
    assert val == pytest.approx(full)
