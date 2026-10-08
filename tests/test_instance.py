"""The per-dart readout: in-graph seed picking, soft membership, the head,
its training path and its export."""
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from darts_model.export.common import (
    SLOT_OUTPUTS, check_slots, effective_topk, output_names, readout_contract,
)
from darts_model.export.coreml import OUTPUTS, ExportWrapper
from darts_model.model.detector import DenseDartConfig, DenseDartLitModule
from darts_model.model.instance import (
    InstanceHead, gather_candidates, ground_truth_seeds, select_seeds,
)

from test_dense_dart import _batch, _cfg


#: Centre scale of the identity space at 256px: 40px.
SCALE = 40 / 256


def _identity(cand):
    cand["identity"] = torch.cat((cand["embedding"], cand["centre"] / SCALE), -1)
    return cand


def _seed(*pairs):
    """Seeds from (embedding, centre) pairs, as one (1, S, I) batch."""
    return torch.tensor([[list(e) + [c / SCALE for c in xy] for e, xy in pairs]])


A = ((0.0, 0.0), (0.25, 0.25))
B = ((5.0, 0.0), (0.75, 0.75))


@dataclass
class _Cfg:
    fg_threshold: float = 0.5
    peak_min_votes: int = 4
    seed_radius_px: float = 6.0
    embed_push_margin: float = 1.5
    membership_sigma: float = 0.5
    token_dim: int = 8
    fine_tokens: bool = False
    fine_dim: int = 0
    instance_dim: int = 16
    instance_heads: int = 4
    instance_layers: int = 1
    instance_tokens: int = 32


def _candidates(strays: bool = True):
    """Two darts of 10 cells each, at different centres and embeddings, plus
    (optionally) 4 cells of dart A whose centre votes strayed 20px but whose
    embedding is A's. Image is 256px; coordinates normalised."""
    rows = []
    for k in range(10):
        rows.append(((0.25, 0.25), (0.20, 0.25), (0.0, 0.0)))     # A
    for k in range(10):
        rows.append(((0.75, 0.75), (0.70, 0.75), (5.0, 0.0)))     # B
    if strays:
        for k in range(4):
            rows.append(((0.25 + 20 / 256, 0.25), (0.20, 0.25), (0.1, 0.0)))
    n = len(rows)
    centre = torch.tensor([r[0] for r in rows]).unsqueeze(0)
    tip = torch.tensor([r[1] for r in rows]).unsqueeze(0)
    emb = torch.tensor([r[2] for r in rows]).unsqueeze(0)
    pos = centre + torch.randn(1, n, 2) * 0.01
    return _identity({
        "score": torch.full((1, n), 0.9), "pos": pos, "centre": centre,
        "tip": tip, "flight": tip + 0.1,
        "direction": torch.tensor([1.0, 0.0]).expand(1, n, 2).clone(),
        "extent": torch.full((1, n, 2), 0.05), "embedding": emb,
        "feature": torch.randn(1, n, 8),
    })


def test_seeds_find_each_dart_once():
    torch.manual_seed(0)
    cand = _candidates()
    mu, score = select_seeds(cand, 3, _Cfg(), 256.0)
    floor = _Cfg.peak_min_votes * _Cfg.fg_threshold
    real = score[0] >= floor
    # A and B, and NOT the strays: their centres are far from A's, but their
    # embedding is A's, so A's pick suppresses them.
    assert int(real.sum()) == 2
    found = sorted(float(m[0]) for m in mu[0][real])
    assert found == pytest.approx([0.0, 5.0], abs=1e-6)


def test_without_an_embedding_match_strays_would_make_their_own_seed():
    """The control for the test above: strays carrying an embedding of their
    own are a dart of their own."""
    torch.manual_seed(0)
    cand = _candidates()
    cand["embedding"][0, 20:] = torch.tensor([0.0, 9.0])
    _identity(cand)
    _, score = select_seeds(cand, 3, _Cfg(), 256.0)
    assert int((score[0] >= _Cfg.peak_min_votes * _Cfg.fg_threshold).sum()) == 3


def test_cells_under_the_fg_threshold_neither_seed_nor_vote():
    torch.manual_seed(0)
    cand = _candidates(strays=False)
    cand["score"][0, 10:] = 0.2                     # dart B below threshold
    _, score = select_seeds(cand, 2, _Cfg(), 256.0)
    assert int((score[0] >= _Cfg.peak_min_votes * _Cfg.fg_threshold).sum()) == 1


def test_a_fresh_head_reads_out_the_weighted_mean_of_its_darts_cells():
    """Zero-initialised correction: the readout is the membership-weighted
    mean of the cells' own landing points, and a far-away dart's cells carry
    no weight in it."""
    torch.manual_seed(0)
    cand = _candidates()
    head = InstanceHead(_Cfg())
    out = head(cand, _seed(A, B), torch.ones(1, 2))
    assert torch.allclose(out["tip"], out["base_tip"])
    assert torch.allclose(out["tip"][0, 0], torch.tensor([0.20, 0.25]), atol=1e-5)
    assert torch.allclose(out["tip"][0, 1], torch.tensor([0.70, 0.75]), atol=1e-5)


def test_the_landing_loss_pushes_a_leaked_cell_out_of_the_dart():
    """Membership is differentiable in the embedding: a neighbour's cell that
    sits close to dart A's seed, dragging A's landing point toward B's, gets
    a gradient that moves its embedding away from A's seed."""
    torch.manual_seed(0)
    cand = _candidates(strays=False)
    # One of B's cells, whose embedding AND centre vote both sit next to A's.
    cand["embedding"][0, 10] = torch.tensor([0.3, 0.0])
    cand["centre"][0, 10] = torch.tensor([0.25, 0.25])
    cand["embedding"].requires_grad_(True)
    _identity(cand)
    head = InstanceHead(_Cfg())
    out = head(cand, _seed(A, B), torch.ones(1, 2))
    err = (out["tip"][0, 0] - torch.tensor([0.20, 0.25])).norm()
    err.backward()
    g = cand["embedding"].grad[0, 10]
    # Gradient descent moves the embedding along -g: away from the seed at 0.
    assert float(-g[0]) > 0


def test_darts_at_the_push_margin_take_none_of_each_others_cells():
    """Two darts whose seeds sit twice the push margin apart -- where the
    embedding loss puts them -- keep their cells to themselves."""
    torch.manual_seed(0)
    cand = _candidates(strays=False)
    # B's cells centred on A's too: only the embedding tells them apart.
    cand["embedding"][0, 10:] = torch.tensor([3.0, 0.0])
    cand["centre"][0, 10:] = torch.tensor([0.25, 0.25])
    _identity(cand)
    head = InstanceHead(_Cfg())
    out = head(cand, _seed(A, ((3.0, 0.0), (0.25, 0.25))), torch.ones(1, 2))
    assert torch.allclose(out["tip"][0, 0], torch.tensor([0.20, 0.25]), atol=1e-4)
    assert torch.allclose(out["tip"][0, 1], torch.tensor([0.70, 0.75]), atol=1e-4)


def test_a_cell_is_never_counted_twice():
    """A cell's membership over every dart and "no dart" sums to one, so a
    duplicate seed splits a dart's cells rather than reading them twice, and
    a seed that is not a dart takes none of them."""
    torch.manual_seed(0)
    cand = _candidates(strays=False)
    head = InstanceHead(_Cfg())
    mu = _seed(A, A)                                        # a duplicate
    both = head(cand, mu, torch.ones(1, 2))
    one = head(cand, mu, torch.tensor([[1.0, 0.0]]))
    fg_a = float(cand["score"][0, :10].sum())
    assert float(both["support"][0].sum()) <= fg_a + 1e-4
    assert float(both["support"][0, 0]) == pytest.approx(
        float(both["support"][0, 1]))
    assert float(one["support"][0, 0]) > 1.5 * float(both["support"][0, 0])


def test_ground_truth_seeds_fall_back_to_the_whole_dart():
    out = {"embedding": torch.zeros(1, 2, 4, 4),
           "centre_offset": torch.zeros(1, 2, 4, 4)}
    out["embedding"][0, 0, 1, 1] = 3.0
    inst = torch.zeros(1, 4, 4, dtype=torch.long)
    inst[0, 1, 1] = 2                                # one cell of dart 2
    torch.manual_seed(0)
    for _ in range(20):
        mu, present = ground_truth_seeds(out, inst, 3, 0.5,
                                         subset=(0.01, 0.02))
        assert present.tolist() == [[False, True, False]]
        # Embedding, then the cell's centre (0.375, 0.375) over the scale.
        assert mu[0, 1].tolist() == pytest.approx([3.0, 0.0, 0.75, 0.75])


def test_far_apart_darts_stay_apart_whatever_their_embedding():
    """Two darts 0.5 of the image apart with the SAME embedding: position
    alone separates them, in seeding and in membership."""
    torch.manual_seed(0)
    cand = _candidates(strays=False)
    cand["embedding"][0, 10:] = 0.0
    _identity(cand)
    mu, score = select_seeds(cand, 3, _Cfg(), 256.0)
    real = score[0] >= _Cfg.peak_min_votes * _Cfg.fg_threshold
    assert int(real.sum()) == 2
    out = InstanceHead(_Cfg())(cand, mu, real[None].float())
    tips = sorted(out["tip"][0][real][:, 0].tolist())
    assert tips == pytest.approx([0.20, 0.70], abs=1e-4)


# ------------------------------------------------------------- the model

def _icfg(**kw):
    base = dict(embed_dim=4, instance_head=True, token_dim=8, instance_dim=16,
                instance_layers=1, fg_threshold=0.3, peak_min_votes=4)
    base.update(kw)
    return _cfg(**base)


def test_config_requires_the_embedding_and_the_ends():
    with pytest.raises(ValueError, match="instance_head requires"):
        DenseDartConfig(instance_head=True)
    with pytest.raises(ValueError, match="instance_head requires"):
        DenseDartConfig(instance_head=True, embed_dim=4, predict_ends=False)


def test_the_per_dart_readout_trains_end_to_end():
    """Fit one batch, then read the darts out through the in-graph seeds --
    the path validation and the exported model take."""
    torch.manual_seed(0)
    lit = DenseDartLitModule(_icfg(instance_weight=0.05,
                                   instance_flight_weight=0.02))
    lit.train()
    logged = {}
    lit.log = lambda n, v, *a, **k: logged.__setitem__(n, float(v))
    batch = _batch()
    opt = torch.optim.Adam(lit.parameters(), lr=3e-3)
    for _ in range(400):
        loss = lit._step(batch, "train")
        opt.zero_grad()
        loss.backward()
        opt.step()
    lit.eval()
    with torch.no_grad():
        out = lit.model(batch["image"].float())
        r = lit.model.read_darts(out)
    floor = lit.config.peak_min_votes * lit.config.fg_threshold
    for b in range(2):
        tips = r["tip"][b][r["score"][b] >= floor] * 256
        assert tips.shape[0] == 2
        gt = batch["dart_ends"][b, :2, 0:2] * 256
        err = torch.cdist(gt, tips).amin(1)
        assert float(err.max()) < 3.0, err


def test_validation_ranks_the_per_dart_readout_and_logs_the_others():
    torch.manual_seed(0)
    lit = DenseDartLitModule(_icfg()).eval()
    logged = {}
    lit.log = lambda n, v, *a, **k: logged.__setitem__(n, float(v))
    with torch.no_grad():
        lit.model.fg_head.bias.fill_(10.0)
        lit._step(_batch(), "val")
    lit.on_validation_epoch_end()
    for name in ("val/landing_px_error_penalised",
                 "val/landing_px_error_penalised_centre_claim",
                 "val/landing_px_error_penalised_embedding_claim",
                 "val/landing_px_error_gt_seeds"):
        assert name in logged, name


def test_a_finetune_may_add_the_per_dart_head(tmp_path):
    trained = DenseDartLitModule(_cfg(embed_dim=4))
    path = tmp_path / "embed.ckpt"
    torch.save({"state_dict": trained.state_dict()}, path)
    lit = DenseDartLitModule(_icfg(init_weights=str(path)))
    assert torch.equal(lit.model.embed_head.weight,
                       trained.model.embed_head.weight)


# ---------------------------------------------------------------- export

def _wrapper():
    torch.manual_seed(0)
    cfg = _icfg(head_width=16, head_depth=1, kp_head_width=16,
                kp_head_depth=1)
    lit = DenseDartLitModule(cfg).eval()
    with torch.no_grad():
        for p in lit.model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        lit.model.fg_head.bias.fill_(0.0)
    return cfg, lit.model, ExportWrapper(
        lit.model, topk=effective_topk(256, cfg.out_stride)).eval()


def test_the_exported_graph_emits_the_per_dart_readout():
    cfg, net, wrapper = _wrapper()
    image = torch.randint(0, 256, (1, 3, 256, 256)).float()
    with torch.no_grad():
        outs = dict(zip(output_names(cfg), wrapper(image)))
        dense = net((image - wrapper.norm_mean) / wrapper.norm_std)
        want = net.read_darts(dense)
    assert list(outs) == OUTPUTS + SLOT_OUTPUTS
    assert outs["slot_score"].shape == (1, cfg.instance_slots)
    assert torch.allclose(outs["slot_tip"], want["tip"])
    assert torch.allclose(outs["slot_flight"], want["flight"])


def test_the_exported_graph_traces_and_exports():
    """Both converters start from a trace: jit.trace for CoreML,
    torch.export for LiteRT. Static shapes all the way through."""
    cfg, _, wrapper = _wrapper()
    image = torch.randint(0, 256, (1, 3, 256, 256)).float()
    with torch.no_grad():
        want = wrapper(image)
        traced = torch.jit.trace(wrapper, image, strict=False)
        got = traced(image)
        exported = torch.export.export(wrapper, (image,)).module()(image)
    for a, b, c in zip(want, got, exported):
        assert torch.allclose(a, b, atol=1e-5)
        assert torch.allclose(a, c, atol=1e-5)


def test_the_contract_says_how_to_read_the_slots():
    cfg = _icfg()
    c = readout_contract(cfg, 256, 1024)
    assert c["darts"]["slots"] == cfg.instance_slots
    assert c["darts"]["min_score"] == pytest.approx(
        cfg.peak_min_votes * cfg.fg_threshold)
    assert c["output_shapes"]["slot_tip"] == [1, cfg.instance_slots, 2]


def _slots(scores, tips):
    return {"slot_score": np.array([scores], float),
            "slot_tip": np.array([tips], float),
            "slot_flight": np.array([tips], float) + 0.1}


def test_reordered_slots_compare_equal():
    ref = _slots([9.0, 8.9, 0.0], [[0.1, 0.1], [0.5, 0.5], [0.0, 0.0]])
    got = _slots([8.9, 9.0, 0.0], [[0.5, 0.5], [0.1, 0.1], [0.3, 0.3]])
    assert check_slots(ref, got, 256, 2.0, 1.0, log=lambda *a: None) == 0


def test_a_moved_or_lost_dart_is_a_failure():
    ref = _slots([9.0, 8.9], [[0.1, 0.1], [0.5, 0.5]])
    moved = _slots([9.0, 8.9], [[0.1, 0.1], [0.5, 0.52]])
    lost = _slots([9.0, 1.0], [[0.1, 0.1], [0.5, 0.5]])
    assert check_slots(ref, moved, 256, 2.0, 1.0, log=lambda *a: None) > 0
    assert check_slots(ref, lost, 256, 2.0, 1.0, log=lambda *a: None) > 0
