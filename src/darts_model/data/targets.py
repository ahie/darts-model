"""Derive dense targets from a rendered frame, for both training stages.

The renderer emits per-pixel class and dart-instance ids plus a depth buffer;
everything else here is arithmetic on top of those and the frame's annotation.
Because they are derived rather than rendered, a sign flip or a transposed axis
produces a perfectly smooth loss curve while teaching the backbone the wrong
geometry -- so each function states the check that would catch it.
"""
from __future__ import annotations

import math

import numpy as np

from darts_model.board_geometry import RING_RADII_MM, SEGMENT_ORDER

#: Lower wins when several classes fall in one output cell. Identical to
#: kPriority in renderer/src/renderer_lib.cpp so the two stages of downsampling
#: agree; see the note there on why this is priority rather than averaging or
#: nearest.
CLASS_PRIORITY = (7, 6, 1, 4, 0, 3, 5)

# Must match dart::SegClass in renderer/src/constants.h
BACKGROUND, BOARDFACE, WIRE, NUMERALS, TIP, METAL, FLIGHT = range(7)
NUM_CLASSES = 7
DART_CLASSES = (TIP, METAL, FLIGHT)

#: The radii the renderer places the ``double_<n>`` and ``triple_<n>``
#: keypoints at.
R_DOUBLE_OUTER_MM = RING_RADII_MM["double_outer"]
R_TRIPLE_INNER_MM = RING_RADII_MM["triple_inner"]
#: The board face's height in world units: BOARD_SURFACE_Z in
#: renderer/src/constants.h. The annotation carries the camera but not the
#: board's placement.
BOARD_Z_BU = 0.19
BU_TO_MM = 100.0


def canonical_keypoints() -> dict[str, tuple[float, float]]:
    """Board-plane position in mm for each keypoint the renderer emits.

    Mirrors getBoardKeypoints3D: theta = i * 2pi/20, x = R sin, y = R cos.
    Unrotated deliberately -- the board's rotation is in-plane, so the fitted
    homography absorbs it.
    """
    out = {"center": (0.0, 0.0)}
    for i, seg in enumerate(SEGMENT_ORDER):
        th = i * 2.0 * math.pi / 20.0
        out[f"double_{seg}"] = (R_DOUBLE_OUTER_MM * math.sin(th),
                                R_DOUBLE_OUTER_MM * math.cos(th))
        out[f"triple_{seg}"] = (R_TRIPLE_INNER_MM * math.sin(th),
                                R_TRIPLE_INNER_MM * math.cos(th))
    return out


_CANON = canonical_keypoints()


def fit_homography(src, dst) -> np.ndarray:
    a, b = [], []
    for (xs, ys), (xd, yd) in zip(src, dst):
        a.append([xs, ys, 1, 0, 0, 0, -xd * xs, -xd * ys]); b.append(xd)
        a.append([0, 0, 0, xs, ys, 1, -yd * xs, -yd * ys]); b.append(yd)
    h = np.linalg.lstsq(np.asarray(a), np.asarray(b), rcond=None)[0]
    return np.append(h, 1.0).reshape(3, 3)


def board_homography(annotation) -> np.ndarray | None:
    """Image -> board-plane mm, from the labelled keypoints."""
    src, dst = [], []
    for kp in annotation["board_keypoints"]:
        if kp["name"] in _CANON:
            src.append((kp["x"], kp["y"]))
            dst.append(_CANON[kp["name"]])
    if len(src) < 4:
        return None
    return fit_homography(src, dst)


def board_uv_at(H: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Board-plane coordinates in mm at the given pixel centres, as (2, N).

    Evaluated at supplied points rather than over a grid: the targets are only
    ever consumed at the head's stride, and building them at full resolution
    then discarding fifteen of every sixteen values costs 56% of the data
    pipeline's time.
    """
    pts = np.stack([px, py, np.ones_like(px)])
    world = H @ pts
    world /= world[2]
    return world[:2]


def height_above_board_at(depth_vals: np.ndarray, px: np.ndarray, py: np.ndarray,
                          annotation, width: int, height: int) -> np.ndarray:
    """Height above the board plane in mm at the given pixels, as (N,).

    Rebuilds the renderer's projection exactly (cameraProjection in
    renderer/src/board_transforms.h). The focal length is quoted against the
    horizontal aperture, so it fixes the horizontal field of view, and the
    vertical one glm::perspective takes follows from the aspect ratio; for a
    square frame the two coincide. glm is not compiled with
    GLM_FORCE_DEPTH_ZERO_TO_ONE, so the depth convention is OpenGL's [-1, 1];
    assuming Vulkan's [0, 1] recovers a flat board about a metre from where it
    belongs. The board coming out at ~0 is the check.
    """
    cam = annotation["camera_params"]
    near, far = float(cam["near"]), float(cam["far"])
    focal = float(cam["focal_length"])
    aspect = width / height
    fov_x = 2.0 * math.atan(float(cam["h_aperture_mm"]) / (2.0 * focal))
    fov_y = 2.0 * math.atan(math.tan(fov_x / 2.0) / aspect)
    t = 1.0 / math.tan(fov_y / 2.0)

    P = np.zeros((4, 4))                 # column-major, P[col][row], as in glm
    P[0][0] = t / aspect
    P[1][1] = -t                         # Vulkan Y-flip, as in recordFrame
    P[2][2] = -(far + near) / (far - near)
    P[2][3] = -1.0
    P[3][2] = -(2.0 * far * near) / (far - near)

    V = np.asarray(cam["view_matrix"], float).reshape(4, 4).T
    inv_pv = np.linalg.inv(P.T @ V)

    clip = np.stack([2.0 * px / width - 1.0,
                     2.0 * py / height - 1.0,
                     depth_vals.astype(np.float64),
                     np.ones_like(px)])
    world = inv_pv @ clip
    world /= world[3]
    return (world[2] - BOARD_Z_BU) * BU_TO_MM


_IS_DART = np.zeros(256, bool)
_IS_DART[list(DART_CLASSES)] = True


def dart_instance_map(cls: np.ndarray, inst: np.ndarray, out_stride: int,
                      return_purity: bool = False):
    """Per output cell, the dart with the most visible pixels in it; 0 for none.

    With ``return_purity``, also the winning dart's share of the cell's dart
    pixels (float32, 1.0 where only one dart is present and 0 off any dart):
    how far the label can be trusted where two darts cross.

    Takes the full-resolution class and instance ids. A cell is a dart cell if
    ANY of its pixels is a dart part (tip, barrel or flight). This is
    deliberately not the priority reduction: there the wire outranks the
    barrel and the flight, so a cell a dart mostly covers would become a wire
    cell as soon as a strand of wire showed beside it, and the detector's
    silhouette would be cut wherever a dart crosses the spider. Ties go to the
    lowest instance id.
    """
    f = max(1, int(out_stride))
    h, w = cls.shape[0] // f, cls.shape[1] // f
    cls, inst = cls[:h * f, :w * f], inst[:h * f, :w * f]
    # The instance channel is non-zero on dart pixels only, which makes it
    # the cheap first filter; the class check keeps that an assumption rather
    # than a dependency.
    flat = np.ascontiguousarray(inst).ravel()
    idx = np.flatnonzero(flat)
    ys, xs = np.divmod(idx, w * f)
    keep = _IS_DART[cls[ys, xs]]
    if not keep.any():
        empty = np.zeros((h, w), np.int64)
        return (empty, np.zeros((h, w), np.float32)) if return_purity else empty
    ys, xs = ys[keep], xs[keep]
    ids = flat[idx[keep]].astype(np.int64)
    n = int(ids.max()) + 1
    cell = (ys // f) * w + xs // f
    counts = np.bincount(cell * n + ids, minlength=h * w * n).reshape(h * w, n)
    # argmax returns the first maximum, so ties resolve to the lowest id, and
    # a cell with no dart pixel (all zeros, id 0 never counted) to 0.
    winner = counts.argmax(1).reshape(h, w)
    if not return_purity:
        return winner
    total = counts.sum(1)
    purity = (counts.max(1) / np.maximum(total, 1)).astype(np.float32)
    return winner, purity.reshape(h, w)


def build_dense_targets(annotation, image_size: int,
                        out_stride: int = 1) -> dict[str, np.ndarray] | None:
    """Dense targets at the head's grid, normalised to O(1).

    Reduces the segmentation ids first, then evaluates the continuous targets
    only at the subpixel each output cell selected. The result is identical to
    building them over the full grid and reducing afterwards, since the
    reduction takes each continuous value from the winning subpixel, and it
    skips the fifteen of every sixteen values that would be discarded.

    ``seg_class``, ``seg_instance`` and the continuous maps come from that
    priority reduction and feed pretraining. ``dart_instance`` is the
    detector's silhouette and is reduced dart-first instead; see
    :func:`dart_instance_map`.

    Returns None when the frame carries no segmentation ids or depth, so the
    caller can fail loudly rather than train on zeros.
    """
    seg = annotation.get("seg_ids")
    depth = annotation.get("depth")
    if seg is None or depth is None:
        return None
    H = board_homography(annotation)
    if H is None:
        return None

    cls_full, inst_full = seg[..., 0], seg[..., 1]
    h, w = cls_full.shape
    f = max(1, int(out_stride))
    oh, ow = h // f, w // f

    # Winning subpixel per output cell, by class priority. Lower wins; see the
    # renderer's own reduction for why this is not averaging or nearest.
    prio = np.asarray(CLASS_PRIORITY, np.int64)
    cb = cls_full[:oh * f, :ow * f].reshape(oh, f, ow, f).transpose(0, 2, 1, 3)
    cb = cb.reshape(oh, ow, f * f)
    win = prio[cb].argmin(-1)
    dy, dx = np.divmod(win, f)
    ys = (np.arange(oh)[:, None] * f + dy)
    xs = (np.arange(ow)[None, :] * f + dx)

    cls = cls_full[ys, xs]
    inst = inst_full[ys, xs]
    dart_inst, dart_purity = dart_instance_map(cls_full, inst_full, f,
                                               return_purity=True)
    px = (xs + 0.5).ravel().astype(np.float64)
    py = (ys + 0.5).ravel().astype(np.float64)

    uv = board_uv_at(H, px, py).reshape(2, oh, ow)
    # Depth comes from the subsample the renderer's reduction picked, so it is
    # unprojected at that subsample's centre rather than the pixel's.
    hx, hy = px, py
    sub = annotation.get("seg_subpixel")
    if sub is not None:
        ss = int(annotation["supersample"])
        sdy, sdx = np.divmod(sub[ys, xs].ravel().astype(np.int64), ss)
        hx = xs.ravel() + (sdx + 0.5) / ss
        hy = ys.ravel() + (sdy + 0.5) / ss
    height = height_above_board_at(depth[ys, xs].ravel(), hx, hy,
                                   annotation, w, h).reshape(oh, ow)

    # Offsets follow the priority-reduced class map, which is what the
    # pretraining loss selects its offset cells by: a cell the wire wins is a
    # wire cell there, with no offset to supervise.
    off = np.zeros((2, oh, ow), np.float32)
    dart_mask = np.zeros((oh, ow), bool)
    for i, d in enumerate(annotation["darts"]):
        m = (inst == i + 1) & np.isin(cls, DART_CLASSES)
        if not m.any():
            continue
        # Targets the annotated tip, never the tip-label centroid: the point
        # mesh's centroid sits 1.6-10.5px behind the tip along the dart axis.
        off[0][m] = d["x"] - (xs + 0.5)[m]
        off[1][m] = d["y"] - (ys + 0.5)[m]
        dart_mask |= m

    drawn = cls != BACKGROUND
    on_board = cls == BOARDFACE
    # Depth clears to 1.0 where nothing is drawn, which unprojects to the far
    # plane, hundreds of metres away. Masked out of the loss either way, but
    # zeroed so that values that large never reach a consumer that forgets the
    # mask.
    height = np.where(drawn, height, 0.0)
    uv = np.where(on_board[None], uv, 0.0)

    return {
        "seg_class": cls.astype(np.int64),
        "seg_instance": inst.astype(np.int64),
        "dart_instance": dart_inst,
        "dart_instance_purity": dart_purity,
        # Divided by image size, board radius and 100mm, so every regression
        # term is O(1) and the loss weights mean what they look like rather
        # than silently encoding unit choices.
        "offset": (off / image_size).astype(np.float32),
        "uv": (uv / R_DOUBLE_OUTER_MM).astype(np.float32),
        "height": (height / BU_TO_MM)[None].astype(np.float32),
        "dart_mask": dart_mask,
        "board_mask": on_board,
        "drawn_mask": drawn,
    }
