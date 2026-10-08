"""The board homography behind the ``uv`` pretraining target, and the dart
instance map behind the detector's foreground."""
from __future__ import annotations

import math

import numpy as np
import pytest

from darts_model.board_geometry import SEGMENT_ORDER
from darts_model.data.targets import (
    FLIGHT,
    METAL,
    TIP,
    WIRE,
    board_homography,
    board_uv_at,
    build_dense_targets,
)
from test_data_synthetic import dart, synthetic_frame

# Where renderer/src/annotation.cpp places the keypoints (RING_RADII_BU in
# constants.h, 1 BU = 100 mm): doubles on the outer double wire, triples on
# the inner treble wire.
DOUBLE_OUTER_MM = 170.0
TRIPLE_INNER_MM = 99.0


def _annotation(scale: float, offset: tuple[float, float]) -> dict:
    """Keypoints as a frontal camera sees them: board mm -> pixels by an
    isotropic scale and a shift."""
    ox, oy = offset
    kps = [{"name": "center", "x": ox, "y": oy}]
    for i, seg in enumerate(SEGMENT_ORDER):
        th = i * 2.0 * math.pi / 20.0
        for ring, r in (("double", DOUBLE_OUTER_MM), ("triple", TRIPLE_INNER_MM)):
            kps.append({"name": f"{ring}_{seg}",
                        "x": ox + scale * r * math.sin(th),
                        "y": oy + scale * r * math.cos(th)})
    return {"board_keypoints": kps}


def test_uv_recovers_board_millimetres_exactly() -> None:
    """Every keypoint is consistent with one similarity transform, so the fit
    must invert it exactly. A radius that disagrees with the renderer's
    stretches the fit instead, and the uv target with it."""
    scale, offset = 2.5, (512.0, 480.0)
    H = board_homography(_annotation(scale, offset))
    assert H is not None

    mm = np.array([[0.0, 0.0], [0.0, 50.0], [120.0, 0.0], [0.0, 170.0], [-150.0, 90.0]])
    px = offset[0] + scale * mm[:, 0]
    py = offset[1] + scale * mm[:, 1]
    uv = board_uv_at(H, px, py)
    np.testing.assert_allclose(uv.T, mm, atol=1e-6)


def _cell(a: np.ndarray, row: int, col: int, f: int = 8) -> np.ndarray:
    return a[row * f:(row + 1) * f, col * f:(col + 1) * f]


def test_wire_crossing_a_dart_keeps_the_cell_on_the_dart() -> None:
    """A barrel cell with one column of visible wire. Priority reduction makes
    it a wire cell with no instance -- intended for the pretraining class map
    -- but the detector's instance map must still put it on the dart."""
    cls = np.full((64, 64), 1, np.uint8)
    inst = np.zeros((64, 64), np.uint8)
    _cell(cls, 2, 3)[:] = METAL
    _cell(inst, 2, 3)[:] = 1
    _cell(cls, 2, 3)[:, 3] = WIRE
    _cell(inst, 2, 3)[:, 3] = 0
    _, ann = synthetic_frame(64, darts=[dart(28, 16)], seg_class=cls,
                             seg_instance=inst)
    t = build_dense_targets(ann, 64, out_stride=8)
    assert t["seg_class"][2, 3] == WIRE and t["seg_instance"][2, 3] == 0
    assert t["dart_instance"][2, 3] == 1
    assert t["dart_instance"].sum() == 1


def test_dart_instance_is_the_majority_dart_not_the_priority_winner() -> None:
    cls = np.full((64, 64), 1, np.uint8)
    inst = np.zeros((64, 64), np.uint8)
    # 24 barrel pixels of dart 1 against 40 flight pixels of dart 2: the
    # barrel wins by priority, dart 2 by count.
    _cell(cls, 4, 4)[:3] = METAL
    _cell(inst, 4, 4)[:3] = 1
    _cell(cls, 4, 4)[3:] = FLIGHT
    _cell(inst, 4, 4)[3:] = 2
    # An even split goes to the lower id.
    _cell(cls, 5, 5)[:] = FLIGHT
    _cell(inst, 5, 5)[:4] = 3
    _cell(inst, 5, 5)[4:] = 2
    # One tip pixel on bare board is enough to make a dart cell.
    _cell(cls, 1, 6)[7, 7] = TIP
    _cell(inst, 1, 6)[7, 7] = 1
    _, ann = synthetic_frame(64, darts=[dart(8, 8)] * 3, seg_class=cls,
                             seg_instance=inst)
    t = build_dense_targets(ann, 64, out_stride=8)
    assert t["seg_instance"][4, 4] == 1
    assert t["dart_instance"][4, 4] == 2
    assert t["dart_instance"][5, 5] == 2
    assert t["dart_instance"][1, 6] == 1
    assert (t["dart_instance"] > 0).sum() == 3
    # The winner's share of the cell's dart pixels.
    p = t["dart_instance_purity"]
    assert p[4, 4] == pytest.approx(40 / 64)
    assert p[5, 5] == pytest.approx(0.5)
    assert p[1, 6] == 1.0 and p[0, 0] == 0.0


def test_height_is_unprojected_at_the_subsample_the_depth_came_from() -> None:
    """The renderer's reduction takes depth from one subsample of each pixel.
    Unprojected at that subsample's centre the board sits at height 0;
    unprojected at the pixel centre it does not, on an oblique view."""
    from test_data_synthetic import BOARD_Z_BU, _camera, _project

    size, ss = 64, 2
    _, ann = synthetic_frame(size)
    V, M = _camera(size, size)
    PV = M @ V
    rng = np.random.default_rng(0)
    sub = rng.integers(0, ss * ss, (size, size)).astype(np.uint8)
    sdy, sdx = np.divmod(sub.astype(np.int64), ss)
    ys, xs = np.mgrid[0:size, 0:size]
    nx = 2 * (xs + (sdx + 0.5) / ss).ravel() / size - 1
    ny = 2 * (ys + (sdy + 0.5) / ss).ravel() / size - 1
    inv = np.linalg.inv(PV)
    a, b = (inv @ np.vstack([nx, ny, np.full_like(nx, z), np.ones_like(nx)])
            for z in (-1.0, 1.0))
    a, b = a[:3] / a[3], b[:3] / b[3]
    hit = a + (BOARD_Z_BU - a[2]) / (b[2] - a[2]) * (b - a)
    ann["depth"] = _project(PV, hit, size, size)[2].reshape(size, size).astype(np.float32)

    without = build_dense_targets(ann, size, out_stride=4)["height"]
    ann["seg_subpixel"], ann["supersample"] = sub, ss
    with_sub = build_dense_targets(ann, size, out_stride=4)["height"]
    assert np.abs(with_sub).max() < 1e-3
    assert np.abs(without).max() > 10 * np.abs(with_sub).max()
