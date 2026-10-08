"""Synthetic renderer frames for the data tests, and the checks that they are
faithful enough to build targets from.

A frame here has the shape ``render_frame`` returns (see
renderer/src/python_bindings.cpp): a uint8 image and an annotation with
``board_keypoints``, ``darts``, ``camera_params`` and the ``seg_ids`` and
``depth`` buffers. The camera views the board obliquely, and depth is the board
plane's, computed through the same projection the renderer uses.
"""
from __future__ import annotations

import math

import numpy as np

from darts_model.board_geometry import RING_RADII_MM, SEGMENT_ORDER
from darts_model.data.targets import (
    BOARD_Z_BU,
    BOARDFACE,
    build_dense_targets,
    canonical_keypoints,
)

NEAR, FAR, FOCAL_MM, DISTANCE_BU, TILT_DEG = 0.1, 100.0, 50.0, 10.0, 40.0
#: H_APERTURE_MM in renderer/src/constants.h, as camera_params reports it.
H_APERTURE_MM = 20.955


def _keypoints_mm() -> dict[str, tuple[float, float]]:
    """All 81 keypoints on the board plane, as renderer/src/annotation.cpp
    places them: the double-bed corners half a segment past each centre."""
    out = dict(canonical_keypoints())
    for i, seg in enumerate(SEGMENT_ORDER):
        th = (i + 0.5) * 2.0 * math.pi / 20.0
        pair = f"{seg}_{SEGMENT_ORDER[(i + 1) % 20]}"
        for ring in ("double_outer", "double_inner"):
            r = RING_RADII_MM[ring]
            out[f"{ring}_{pair}"] = (r * math.sin(th), r * math.cos(th))
    return out


def _camera(width: int, height: int):
    """View and OpenGL projection as cameraProjection in
    renderer/src/board_transforms.h builds them, with Vulkan's Y flip.

    The camera looks at the board centre from TILT_DEG off its axis. Oblique
    on purpose: head-on, every pixel of the board is at one view depth, and
    the height target would not notice a wrong field of view.
    """
    a = math.radians(TILT_DEG)
    target = np.array([0.0, 0.0, BOARD_Z_BU])
    eye = target + DISTANCE_BU * np.array([0.0, -math.sin(a), math.cos(a)])
    fwd = (target - eye) / np.linalg.norm(target - eye)
    side = np.cross(fwd, [0.0, 1.0, 0.0])
    side /= np.linalg.norm(side)
    up = np.cross(side, fwd)
    V = np.eye(4)
    V[0, :3], V[1, :3], V[2, :3] = side, up, -fwd
    V[:3, 3] = -V[:3, :3] @ eye
    aspect = width / height
    fov_x = 2.0 * math.atan(H_APERTURE_MM / (2.0 * FOCAL_MM))
    fov_y = 2.0 * math.atan(math.tan(fov_x / 2.0) / aspect)
    t = 1.0 / math.tan(fov_y / 2.0)
    M = np.zeros((4, 4))
    M[0, 0], M[1, 1] = t / aspect, -t
    M[2, 2] = -(FAR + NEAR) / (FAR - NEAR)
    M[2, 3] = -(2.0 * FAR * NEAR) / (FAR - NEAR)
    M[3, 2] = -1.0
    return V, M


def _project(PV: np.ndarray, world: np.ndarray, width: int, height: int):
    clip = PV @ np.vstack([world, np.ones(world.shape[1])])
    ndc = clip[:3] / clip[3]
    return (ndc[0] + 1) / 2 * width, (ndc[1] + 1) / 2 * height, ndc[2]


def _board_depth(PV: np.ndarray, width: int, height: int) -> np.ndarray:
    """Depth of the board plane at every pixel centre, in the renderer's
    OpenGL [-1, 1] convention."""
    ys, xs = np.mgrid[0:height, 0:width]
    nx = 2 * (xs.ravel() + 0.5) / width - 1
    ny = 2 * (ys.ravel() + 0.5) / height - 1
    inv = np.linalg.inv(PV)
    ends = []
    for z in (-1.0, 1.0):
        w = inv @ np.vstack([nx, ny, np.full_like(nx, z), np.ones_like(nx)])
        ends.append(w[:3] / w[3])
    a, b = ends
    s = (BOARD_Z_BU - a[2]) / (b[2] - a[2])
    hit = a + s * (b - a)
    depth = _project(PV, hit, width, height)[2]
    return depth.reshape(height, width).astype(np.float32)


def synthetic_frame(size: int = 64, darts: list[dict] | None = None,
                    seg_class: np.ndarray | None = None,
                    seg_instance: np.ndarray | None = None,
                    image_value: int = 0, height: int | None = None):
    """``(image, annotation)`` as ``Renderer.render_frame`` returns them.

    ``size`` is the width, and the height too unless ``height`` is given.
    ``seg_class``/``seg_instance`` default to bare board face. Darts are dicts
    as :func:`dart` makes them.
    """
    width, height = size, height or size
    V, M = _camera(width, height)
    PV = M @ V
    names, mm = zip(*_keypoints_mm().items())
    world = np.array([[x / 100.0, y / 100.0, BOARD_Z_BU] for x, y in mm]).T
    px, py, _ = _project(PV, world, width, height)
    kps = [{"name": n, "x": float(x), "y": float(y)} for n, x, y in zip(names, px, py)]

    cls = (np.full((height, width), BOARDFACE, np.uint8) if seg_class is None
           else seg_class.astype(np.uint8))
    inst = (np.zeros((height, width), np.uint8) if seg_instance is None
            else seg_instance.astype(np.uint8))
    annotation = {
        "frame_id": 0,
        "image_path": "rgb/000000.jpg",
        "board_keypoints": kps,
        "darts": list(darts or []),
        "camera_params": {
            "focal_length": FOCAL_MM,
            "h_aperture_mm": H_APERTURE_MM,
            "resolution": [width, height],
            "view_matrix": V.flatten(order="F").tolist(),
            "near": NEAR, "far": FAR,
        },
        "metadata": {"num_darts": len(darts or []), "synthetic": True,
                     "generator": "vulkan_renderer"},
        "depth": _board_depth(PV, width, height),
        "seg_ids": np.stack([cls, inst], -1),
    }
    image = np.full((height, width, 3), image_value, np.uint8)
    return image, annotation


def dart(x: float, y: float, length: float = 20.0, width: float = 4.0,
         end_on: bool = False) -> dict:
    """A dart record pointing down the image from its tip at (x, y), with the
    renderer's corner order: tip-left, flight-left, flight-right, tip-right."""
    hw = width / 2
    return {"x": x, "y": y, "flight_x": x, "flight_y": y + length,
            "flight_in_front": False,
            "box_x": [x - hw, x - hw, x + hw, x + hw],
            "box_y": [y, y + length, y + length, y],
            "box_end_on": end_on,
            "score_zone": "S20"}


def test_synthetic_board_sits_at_height_zero() -> None:
    """The depth buffer here unprojects onto the board plane, so the height
    target is zero everywhere. Checks the builder as much as the target."""
    _, ann = synthetic_frame(64)
    t = build_dense_targets(ann, 64, out_stride=4)
    assert t is not None
    np.testing.assert_allclose(t["height"], 0.0, atol=1e-4)
    assert t["board_mask"].all()


def test_synthetic_uv_matches_the_board_plane() -> None:
    """The pixel holding the centre keypoint maps to uv (0, 0), to within the
    half pixel between the keypoint and that pixel's centre."""
    _, ann = synthetic_frame(64)
    t = build_dense_targets(ann, 64, out_stride=1)
    centre = next(k for k in ann["board_keypoints"] if k["name"] == "center")
    cx, cy = int(centre["x"]), int(centre["y"])
    np.testing.assert_allclose(t["uv"][:, cy, cx], 0.0, atol=0.03)


def test_non_square_frame_uses_the_horizontal_aperture() -> None:
    """The focal length is quoted against the horizontal aperture. Treating
    it as vertical changes nothing on a square frame and lifts a wide frame's
    board off the plane."""
    _, ann = synthetic_frame(96, height=48)
    t = build_dense_targets(ann, 96, out_stride=4)
    assert t["height"].shape == (1, 12, 24)
    np.testing.assert_allclose(t["height"], 0.0, atol=1e-4)
