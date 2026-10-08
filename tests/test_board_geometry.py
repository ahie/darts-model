"""The keypoint naming the datasets select board keypoints by."""
from __future__ import annotations

from darts_model.board_geometry import (
    BOARD_KEYPOINT_NAMES,
    DOUBLE_CROSSING_INDICES,
    SEGMENT_ORDER,
)


def test_segment_order_is_a_permutation_of_1_to_20() -> None:
    assert sorted(SEGMENT_ORDER) == list(range(1, 21))
    assert SEGMENT_ORDER[0] == 20


def test_keypoint_names_match_the_renderer_layout() -> None:
    """Order as make_keypoint_names in renderer/src/constants.h: centre,
    doubles, triples, then the outer and inner double-bed corners."""
    names = BOARD_KEYPOINT_NAMES
    assert len(names) == 81 and len(set(names)) == 81
    assert names[0] == "center"
    assert names[1] == "double_20" and names[21] == "triple_20"
    assert names[41] == "double_outer_20_1"
    assert names[60] == "double_outer_5_20"
    assert names[61] == "double_inner_20_1"


def test_double_crossings_are_the_forty_corners() -> None:
    corners = [BOARD_KEYPOINT_NAMES[i] for i in DOUBLE_CROSSING_INDICES]
    assert len(corners) == 40
    assert all(n.startswith(("double_outer_", "double_inner_")) for n in corners)
