"""The pieces both exporters and the visualiser share: the fp16 selector, the
checkpoint loader, the set-wise comparison and the output contract."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from darts_model.export.common import (
    ExportError,
    KeypointFp32Selector,
    check_against_reference,
    keypoint_module_names,
    load_checkpoint,
    match_cells,
    readout_contract,
    set_distance,
    uses_keypoint_weights,
    verification_frame,
)
from darts_model.model.detector import DenseDartConfig, DenseDartLitModule

KP = ["kp_trunk", "kp_head", "kp_offset"]


# ------------------------------------------------------------ fp16 selector

@pytest.mark.parametrize("op, inputs", [
    # The conv itself is anonymous; its weight constant carries the path.
    ("conv_17", ["input_231", "model_kp_trunk_0_0_weight"]),
    ("conv_18", ["x", "model_kp_head_weight_to_fp16", "model_kp_head_bias"]),
    ("add_3", ["mul_2", "model.kp_trunk.0.1.bias"]),
    ("conv_19", ["x", "self_model_kp_offset_weight"]),
    # Named after the traced variable.
    ("kp_offset", ["x"]),
])
def test_keypoint_ops_are_kept_fp32(op, inputs) -> None:
    assert uses_keypoint_weights(op, inputs, KP)


@pytest.mark.parametrize("op, inputs", [
    ("conv_3", ["input_1", "model_head_0_0_weight"]),
    ("conv_4", ["x", "model_box_head_weight"]),
    # A token must match whole: `kpl`, `skp_head2` and `kp_headroom` are not
    # the keypoint head.
    ("reduce_max_1", ["kpl"]),
    ("conv_5", ["model_skp_head2_weight"]),
    ("conv_6", ["model_kp_headroom_weight"]),
    ("gather_1", []),
])
def test_other_ops_may_be_fp16(op, inputs) -> None:
    assert not uses_keypoint_weights(op, inputs, KP)


def test_module_names_come_from_the_network() -> None:
    net = DenseDartLitModule(DenseDartConfig(
        backbone_widths=(16, 24, 32, 48), backbone_depths=(1, 1, 1, 1),
        head_width=16, head_depth=1, predict_keypoints=True,
        kp_head_width=16, kp_head_depth=1)).model
    assert sorted(keypoint_module_names(net)) == sorted(KP)


def test_selector_reads_mil_like_ops_and_records_what_it_kept() -> None:
    """Shaped like coremltools' Operation: `name`, and `inputs` mapping to a
    Var or a tuple of Vars, each with a `name` and a producing `op`."""
    def var(name):
        return SimpleNamespace(name=name, op=SimpleNamespace(name=name))

    kp_conv = SimpleNamespace(name="conv_9", inputs={
        "x": var("gelu_4"), "weight": var("model_kp_trunk_0_0_weight")})
    other = SimpleNamespace(name="conv_2", inputs={
        "x": var("gelu_1"), "weight": var("model_head_0_0_weight")})
    concat = SimpleNamespace(name="concat_1", inputs={
        "values": (var("a"), var("model_kp_head_bias"))})

    sel = KeypointFp32Selector(KP)
    assert sel(kp_conv) is False       # stays fp32
    assert sel(other) is True          # may be fp16
    assert sel(concat) is False
    assert sel.kept == ["conv_9", "concat_1"]


# ------------------------------------------------------------ loader

def _tiny_raw() -> dict:
    return {
        "backbone": {"backbone_widths": [16, 24, 32, 48],
                     "backbone_depths": [1, 1, 1, 1],
                     "backbone_weights": "does/not/exist.ckpt",
                     "init_weights": "nor/this.ckpt"},
        "head": {"head_width": 16, "head_depth": 1, "predict_ends": True,
                 "predict_keypoints": True, "kp_head_width": 16,
                 "kp_head_depth": 1},
        "data": {"image_size": 256},
    }


def _write(tmp_path, raw, state_dict):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(raw))
    ck = tmp_path / "m.ckpt"
    torch.save({"state_dict": state_dict, "epoch": 3, "global_step": 9}, ck)
    return str(ck), str(cfg)


def _state_dict(raw) -> dict:
    from darts_model.cli.train import build_config
    cfg = build_config(raw)
    cfg.backbone_weights = cfg.init_weights = ""
    return DenseDartLitModule(cfg).state_dict()


def test_loader_loads_every_weight_and_ignores_init_paths(tmp_path) -> None:
    raw = _tiny_raw()
    sd = _state_dict(raw)
    ck, cfg = _write(tmp_path, raw, sd)
    loaded = load_checkpoint(ck, cfg, log=lambda *_: None)
    for k, v in loaded.lit.state_dict().items():
        assert torch.equal(v, sd[k]), k
    assert loaded.epoch == 3


def test_loader_refuses_a_missing_model_weight(tmp_path) -> None:
    raw = _tiny_raw()
    sd = _state_dict(raw)
    sd.pop(next(k for k in sd if k.startswith("model.kp_head")))
    ck, cfg = _write(tmp_path, raw, sd)
    with pytest.raises(ExportError, match="missing"):
        load_checkpoint(ck, cfg, log=lambda *_: None)


def test_loader_refuses_a_config_for_another_network(tmp_path) -> None:
    """A checkpoint without the keypoint head, loaded through a config that
    asks for one, must not draw random-init corners."""
    raw = _tiny_raw()
    small = json.loads(json.dumps(raw))
    small["head"]["predict_keypoints"] = False
    ck, cfg = _write(tmp_path, raw, _state_dict(small))
    with pytest.raises(ExportError, match="missing"):
        load_checkpoint(ck, cfg, log=lambda *_: None)


def test_loader_reports_unexpected_keys(tmp_path) -> None:
    raw = _tiny_raw()
    sd = _state_dict(raw)
    sd["model.stale_head.weight"] = torch.zeros(1)
    ck, cfg = _write(tmp_path, raw, sd)
    lines = []
    load_checkpoint(ck, cfg, log=lines.append)
    assert any("unexpected" in line for line in lines)


def test_no_sample_and_no_renderer_fails_before_anything(monkeypatch) -> None:
    import darts_model.renderer as r

    def unavailable():
        raise ImportError("dartboard_renderer is not importable")
    monkeypatch.setattr(r, "import_renderer", unavailable)
    with pytest.raises(ExportError, match="--sample"):
        verification_frame(None, 256, {}, log=lambda *_: None)


# ------------------------------------------------------------ comparison

def _outputs(rng, k=64, kept=20, nk=40):
    score = np.sort(rng.uniform(0.0, 0.45, k))[::-1].copy()
    score[:kept] = np.linspace(0.95, 0.6, kept)
    out = {"dart_score": score[None]}
    for n in ("dart_centre", "dart_tip", "dart_flight"):
        out[n] = rng.uniform(0, 1, (1, k, 2))
    d = rng.normal(size=(1, k, 2))
    out["dart_direction"] = d / np.linalg.norm(d, axis=-1, keepdims=True)
    out["dart_extent"] = rng.uniform(0.01, 0.1, (1, k, 2))
    out["kp_xy"] = rng.uniform(0, 1, (1, nk, 2))
    out["kp_conf"] = rng.uniform(0, 1, (1, nk))
    return out


def _permute_ranks(out, perm):
    return {n: (v[:, perm] if v.shape[1] == len(perm) and n.startswith("dart")
                else v) for n, v in out.items()}


def test_reordered_ranks_compare_equal() -> None:
    """topk returns near-equal scores in another order on another backend;
    the comparison must see the same cells, not rank i against rank i."""
    rng = np.random.default_rng(0)
    ref = _outputs(rng)
    perm = np.arange(64)
    perm[:20] = rng.permutation(20)
    got = _permute_ranks(ref, perm)

    m = match_cells(ref, got, 0.5)
    assert m.kept_ref == m.kept_got == 20
    assert max(m.worst.values()) == 0.0
    assert check_against_reference(ref, got, 1024, 0.5,
                                   log=lambda *_: None) == 0
    assert set_distance(got["dart_tip"], ref["dart_tip"],
                        ref["dart_score"][0] >= 0.5) == 0.0


def test_a_moved_plane_is_reported() -> None:
    rng = np.random.default_rng(1)
    ref = _outputs(rng)
    got = {n: v.copy() for n, v in ref.items()}
    got["dart_tip"][0, 3] += 0.01        # ~10px at 1024
    m = match_cells(ref, got, 0.5)
    assert m.worst["dart_tip"] == pytest.approx(np.hypot(0.01, 0.01))
    assert m.worst["dart_centre"] == 0.0
    assert check_against_reference(ref, got, 1024, 0.5,
                                   log=lambda *_: None) == 1


def test_keypoint_confidence_loss_is_a_failure() -> None:
    rng = np.random.default_rng(2)
    ref = _outputs(rng)
    got = {n: v.copy() for n, v in ref.items()}
    got["kp_conf"][0, 35] -= 0.5
    assert check_against_reference(ref, got, 1024, 0.5,
                                   log=lambda *_: None) == 1


# ------------------------------------------------------------ contract

def test_contract_carries_the_readout() -> None:
    cfg = DenseDartConfig(predict_ends=True, predict_keypoints=True)
    c = readout_contract(cfg, 1024, 1024)
    json.dumps(c)
    assert c["input_size"] == 1024 and c["topk"] == 1024
    assert c["grid"] == [128, 128]
    assert c["readout"] == {"vote_bin_px": cfg.vote_bin_px,
                            "fg_threshold": cfg.fg_threshold,
                            "peak_min_votes": cfg.peak_min_votes,
                            "vote_bins": 256}
    assert len(c["kp_names"]) == 40
    assert c["kp_names"][0] == "double_outer_20_1"
    assert c["kp_names"][20] == "double_inner_20_1"
    assert c["output_shapes"]["dart_tip"] == [1, 1024, 2]
    assert c["output_shapes"]["kp_conf"] == [1, 40]


def test_topk_scales_with_cell_area() -> None:
    from darts_model.export.common import effective_topk, topk_for_stride
    assert topk_for_stride(8) == 1024
    assert topk_for_stride(4) == 4096
    assert topk_for_stride(16) == 256
    assert effective_topk(1024, 4) == 4096
    assert effective_topk(128, 4) == 1024      # clamped to the 32x32 grid


def test_everything_downstream_of_topk_stays_fp32() -> None:
    """The readout -- positions plus offsets, seeds, membership, weighted
    means -- runs after the topk over the cells, and stays fp32 whatever its
    name."""
    def op(name, op_type, *parents):
        return SimpleNamespace(name=name, op_type=op_type, inputs={
            f"x{i}": SimpleNamespace(name=p.name + "_out", op=p)
            for i, p in enumerate(parents)})

    image = op("conv_1", "conv")
    topk = op("topk_0", "topk", image)
    gathered = op("gather_3", "gather_along_axis", image, topk)
    readout = op("add_7", "add", gathered)
    head = op("conv_2", "conv", image)

    sel = KeypointFp32Selector(KP)
    assert sel(head) is True
    assert sel(readout) is False
    assert sel(topk) is False
    assert sel.readout_kept == ["add_7", "topk_0"]
    assert sel.kept == []
