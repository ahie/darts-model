"""Dartboard facts the renderer's annotations are expressed in.

Radii are in millimetres on the board plane. Segment and keypoint order run
clockwise from the top (12 o'clock), as ``renderer/src/constants.h`` and
``renderer/src/annotation.cpp`` define them.
"""

from __future__ import annotations

RING_RADII_MM: dict[str, float] = {
    "inner_bull": 6.35,
    "outer_bull": 15.9,
    # The spec's 107mm is the treble's OUTER wire, measured the same way as
    # 170mm for the double, so the treble bed is 99-107mm.
    "triple_inner": 99.0,
    "triple_outer": 107.0,
    "double_inner": 162.0,
    "double_outer": 170.0,
}
"""Radial boundaries of each ring in millimetres. Must match RING_RADII_BU in
``renderer/src/constants.h`` (1 BU = 100 mm)."""

SEGMENT_ORDER: tuple[int, ...] = (
    20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
    3, 19, 7, 16, 8, 11, 14, 9, 12, 5,
)
"""Clockwise segment order starting from the top (12 o'clock)."""

_CROSSING_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (s, SEGMENT_ORDER[(i + 1) % 20]) for i, s in enumerate(SEGMENT_ORDER)
)
"""Each radial wire, named by the two segments it separates."""

BOARD_KEYPOINT_NAMES: tuple[str, ...] = (
    ("center",)
    + tuple(f"double_{s}" for s in SEGMENT_ORDER)
    + tuple(f"triple_{s}" for s in SEGMENT_ORDER)
    + tuple(f"double_outer_{a}_{b}" for a, b in _CROSSING_PAIRS)
    + tuple(f"double_inner_{a}_{b}" for a, b in _CROSSING_PAIRS)
)
"""81 canonical keypoint names, as the renderer emits them.

    0       center
    1..20   double_<seg>          outer double ring, centred in its segment
    21..40  triple_<seg>          inner triple ring, centred in its segment
    41..60  double_outer_<a>_<b>  outer double ring, ON the wire between a and b
    61..80  double_inner_<a>_<b>  inner double ring, ON the wire between a and b

The last forty are the corners of the double beds: twenty radial wires, each
crossing the two circles that bound the double ring. They sit on a real visual
feature -- a wire meeting a ring -- rather than at a segment's angular centre,
where nothing in the image marks the spot.

The corners come after indices 0..40 so that the first 41 keep a fixed
meaning.
"""

DOUBLE_CROSSING_INDICES: tuple[int, ...] = tuple(range(41, 81))
"""The 40 double-bed corners, as one contiguous block."""
