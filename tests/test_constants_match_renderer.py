"""Facts defined in both the renderer and the Python package must agree.

Each test parses the C++ or GLSL source and compares it with the Python copy,
so a change on one side fails here instead of silently skewing labels or
targets.
"""
from __future__ import annotations

import re
import struct
from pathlib import Path

import pytest

from darts_model.board_geometry import RING_RADII_MM, SEGMENT_ORDER
from darts_model.data import targets
from darts_model.model.pretrain_head import CLASS_FREQ

ROOT = Path(__file__).resolve().parents[1]
CONSTANTS_H = (ROOT / "renderer/src/constants.h").read_text()
RENDERER_LIB = (ROOT / "renderer/src/renderer_lib.cpp").read_text()
BOARD_FACE_GLSL = (ROOT / "renderer/shaders/board_face.glsl").read_text()
ASSETS = ROOT / "renderer/assets"


def _cpp_double(name: str) -> float:
    m = re.search(rf"\b{name}\s*=\s*([0-9.eE+-]+)", CONSTANTS_H)
    assert m, f"{name} not found in constants.h"
    return float(m.group(1))


def _cpp_int(name: str) -> int:
    return int(_cpp_double(name))


def test_ring_radii() -> None:
    cpp = {
        "inner_bull": "INNER_BULL_R_MM", "outer_bull": "OUTER_BULL_R_MM",
        "triple_inner": "TRIPLE_INNER_R_MM", "triple_outer": "TRIPLE_OUTER_R_MM",
        "double_inner": "DOUBLE_INNER_R_MM", "double_outer": "DOUBLE_OUTER_R_MM",
    }
    for py_name, cpp_name in cpp.items():
        assert RING_RADII_MM[py_name] == pytest.approx(_cpp_double(cpp_name)), py_name


def test_board_face_shader_radii() -> None:
    """board_face.glsl paints the beds from its own literals (board units)."""
    lits = sorted({float(x) for x in re.findall(r"r\s*[<>]=?\s*([0-9.]+)",
                                                  BOARD_FACE_GLSL)})
    bu = sorted({round(v / 100.0, 6) for k, v in RING_RADII_MM.items()})
    assert lits == pytest.approx(bu)


def test_segment_order() -> None:
    m = re.search(r"SEGMENT_ORDER\s*=\s*\{([^}]*)\}", CONSTANTS_H)
    assert tuple(int(x) for x in m.group(1).split(",") if x.strip()) == tuple(SEGMENT_ORDER)


def test_segmentation_classes() -> None:
    body = re.search(r"enum class SegClass[^{]*\{([^}]*)\}", CONSTANTS_H).group(1)
    cpp = {name: int(v) for name, v in re.findall(r"(\w+)\s*=\s*(\d+)", body)}
    py = {"Background": targets.BACKGROUND, "BoardFace": targets.BOARDFACE,
          "Wire": targets.WIRE, "Numerals": targets.NUMERALS,
          "DartPoint": targets.TIP, "DartMetal": targets.METAL,
          "DartFlight": targets.FLIGHT, "Count": targets.NUM_CLASSES}
    assert cpp == py
    assert len(CLASS_FREQ) == targets.NUM_CLASSES


def test_class_priority() -> None:
    body = re.search(r"kPriority\[[^\]]*\]\s*=\s*\{([^}]*)\}", RENDERER_LIB).group(1)
    body = re.sub(r"//[^\n]*", "", body)
    cpp = tuple(int(x) for x in body.split(",") if x.strip())
    assert cpp == targets.CLASS_PRIORITY


def test_board_surface_z() -> None:
    assert targets.BOARD_Z_BU == pytest.approx(_cpp_double("BOARD_SURFACE_Z"), abs=1e-6)
    assert targets.BU_TO_MM == pytest.approx(1.0 / _cpp_double("BU_PER_MM"), rel=1e-6)


def test_decal_atlas_grid() -> None:
    cols, rows = _cpp_int("kDecalAtlasGlyphCols"), _cpp_int("kDecalAtlasFontRows")
    with open(ASSETS / "decal_atlas.png", "rb") as f:
        head = f.read(24)
    w, h = struct.unpack(">II", head[16:24])
    assert w % cols == 0 and h % rows == 0 and w // cols == h // rows


def test_numeral_variants() -> None:
    files = sorted(ASSETS.glob("numbers_variant_*.glb"))
    assert [p.name for p in files] == [
        f"numbers_variant_{i}.glb" for i in range(_cpp_int("NUM_FONT_VARIANTS"))]
