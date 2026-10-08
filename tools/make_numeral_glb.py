"""Generate a numbers_variant_N.glb straight from a TTF, without Blender.

The numeral GLBs are simple files: twenty nodes, one mesh each, no materials,
since the renderer supplies colour and draw mode itself.

Node placement is identical across every existing variant -- it is fixed by
the board geometry, not by the typeface -- so it is copied verbatim from a
reference variant rather than recomputed. Only the glyph geometry differs.

Usage:
    python tools/make_numeral_glb.py path/to/font.ttf \
        --reference renderer/assets/numbers_variant_0.glb \
        --out renderer/assets/numbers_variant_7.glb

A new variant also needs NUM_FONT_VARIANTS raised in renderer/src/constants.h.
"""

from __future__ import annotations

import argparse
import json
import struct

import numpy as np

try:
    import mapbox_earcut
    from fontTools.pens.basePen import BasePen
    from fontTools.ttLib import TTFont
except ImportError as e:
    raise SystemExit(f"{e.name} is missing; install the asset tools with "
                     "pip install -e \".[assets]\"") from e

#: Half the extrusion depth, board units. Every existing variant spans exactly
#: -0.02..0.02, so this matches rather than invents a value.
HALF_DEPTH = 0.02

#: Cap height in board units. The existing variants run 0.089 to 0.139
#: depending on the typeface's own proportions; this sits near the median, so
#: a new font reads at a comparable size without being an outlier.
CAP_HEIGHT_BU = 0.105

#: Points per curve segment when flattening outlines. At the size a numeral is
#: seen, finer adds triangles without adding visible shape.
CURVE_STEPS = 8


class FlattenPen(BasePen):
    """Collect closed contours as polylines in font units."""

    def __init__(self, glyphSet):
        super().__init__(glyphSet)
        self.contours: list[list[tuple[float, float]]] = []
        self._cur: list[tuple[float, float]] = []

    def _moveTo(self, pt):
        self._flush()
        self._cur = [pt]

    def _lineTo(self, pt):
        self._cur.append(pt)

    def _curveToOne(self, p1, p2, p3):
        p0 = self._cur[-1]
        for i in range(1, CURVE_STEPS + 1):
            t = i / CURVE_STEPS
            u = 1.0 - t
            self._cur.append((
                u**3 * p0[0] + 3*u*u*t * p1[0] + 3*u*t*t * p2[0] + t**3 * p3[0],
                u**3 * p0[1] + 3*u*u*t * p1[1] + 3*u*t*t * p2[1] + t**3 * p3[1],
            ))

    def _qCurveToOne(self, p1, p2):
        p0 = self._cur[-1]
        for i in range(1, CURVE_STEPS + 1):
            t = i / CURVE_STEPS
            u = 1.0 - t
            self._cur.append((
                u*u * p0[0] + 2*u*t * p1[0] + t*t * p2[0],
                u*u * p0[1] + 2*u*t * p1[1] + t*t * p2[1],
            ))

    def _closePath(self):
        self._flush()

    def _endPath(self):
        self._flush()

    def _flush(self):
        if len(self._cur) >= 3:
            c = self._cur
            # Drop a duplicated closing point; the ring is implicitly closed.
            if abs(c[0][0] - c[-1][0]) < 1e-9 and abs(c[0][1] - c[-1][1]) < 1e-9:
                c = c[:-1]
            if len(c) >= 3:
                self.contours.append(c)
        self._cur = []


def signed_area(ring) -> float:
    a = 0.0
    for i in range(len(ring)):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % len(ring)]
        a += x0 * y1 - x1 * y0
    return a * 0.5


def point_in_ring(pt, ring) -> bool:
    x, y = pt
    inside = False
    for i in range(len(ring)):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % len(ring)]
        if (y0 > y) != (y1 > y):
            xint = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xint:
                inside = not inside
    return inside


def text_contours(font: TTFont, text: str):
    """Contours for `text`, laid out by advance width, in font units."""
    glyphSet = font.getGlyphSet()
    cmap = font.getBestCmap()
    hmtx = font["hmtx"]
    out: list[list[tuple[float, float]]] = []
    pen_x = 0.0
    for ch in text:
        name = cmap[ord(ch)]
        pen = FlattenPen(glyphSet)
        glyphSet[name].draw(pen)
        for c in pen.contours:
            out.append([(x + pen_x, y) for x, y in c])
        pen_x += hmtx[name][0]
    return out, pen_x


def build_mesh(contours, scale, offset):
    """Extrude contours into an interleaved mesh."""
    rings = []
    for c in contours:
        pts = np.array([(x * scale + offset[0], y * scale + offset[1]) for x, y in c])
        rings.append(pts)

    # Outer rings vs holes by containment: an odd nesting depth is a hole.
    is_hole = [False] * len(rings)
    for i, r in enumerate(rings):
        depth = sum(1 for k, o in enumerate(rings)
                    if k != i and point_in_ring(r[0], o))
        is_hole[i] = depth % 2 == 1

    groups: list[list[int]] = []
    for i in range(len(rings)):
        if not is_hole[i]:
            groups.append([i] + [k for k in range(len(rings))
                                 if is_hole[k] and point_in_ring(rings[k][0], rings[i])])

    positions, normals, uvs, tangents, indices = [], [], [], [], []

    def emit(p, n, uv):
        positions.append(p)
        normals.append(n)
        uvs.append(uv)
        # Flat faces with a planar unwrap, so the tangent is constant.
        tangents.append((1.0, 0.0, 0.0, 1.0))
        return len(positions) - 1

    for grp in groups:
        verts, ring_sizes = [], []
        for gi in grp:
            r = rings[gi]
            # earcut wants the outer ring wound opposite to its holes.
            want_ccw = (gi == grp[0])
            if (signed_area(r.tolist()) > 0) != want_ccw:
                r = r[::-1]
            rings[gi] = r
            verts.append(r)
            ring_sizes.append(len(r))
        flat = np.concatenate(verts).astype(np.float64)
        ends = np.cumsum(ring_sizes).astype(np.uint32)
        tri = mapbox_earcut.triangulate_float64(flat, ends)

        # Front and back caps, each with its own vertices so the normals stay flat.
        base_f = len(positions)
        for x, y in flat:
            emit((x, y, HALF_DEPTH), (0.0, 0.0, 1.0), (x, y))
        base_b = len(positions)
        for x, y in flat:
            emit((x, y, -HALF_DEPTH), (0.0, 0.0, -1.0), (x, y))
        for i in range(0, len(tri), 3):
            a, b, c = int(tri[i]), int(tri[i + 1]), int(tri[i + 2])
            indices += [base_f + a, base_f + b, base_f + c]
            indices += [base_b + c, base_b + b, base_b + a]   # reversed winding

        # Side walls, one quad per edge, normals from the edge direction.
        for r in verts:
            n = len(r)
            for i in range(n):
                x0, y0 = r[i]
                x1, y1 = r[(i + 1) % n]
                ex, ey = x1 - x0, y1 - y0
                ln = float(np.hypot(ex, ey)) or 1.0
                nx, ny = ey / ln, -ex / ln
                v0 = emit((x0, y0, HALF_DEPTH), (nx, ny, 0.0), (x0, y0))
                v1 = emit((x1, y1, HALF_DEPTH), (nx, ny, 0.0), (x1, y1))
                v2 = emit((x1, y1, -HALF_DEPTH), (nx, ny, 0.0), (x1, y1))
                v3 = emit((x0, y0, -HALF_DEPTH), (nx, ny, 0.0), (x0, y0))
                indices += [v0, v1, v2, v0, v2, v3]

    return (np.array(positions, np.float32), np.array(normals, np.float32),
            np.array(uvs, np.float32), np.array(tangents, np.float32),
            np.array(indices, np.uint32))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("font")
    ap.add_argument("--reference", default="renderer/assets/numbers_variant_0.glb")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.reference, "rb") as f:
        struct.unpack("<III", f.read(12))
        jlen, _ = struct.unpack("<II", f.read(8))
        ref = json.loads(f.read(jlen))

    font = TTFont(args.font)
    upem = font["head"].unitsPerEm
    cap = font["OS/2"].sCapHeight or int(upem * 0.7)
    scale = CAP_HEIGHT_BU / cap

    meshes, accessors, views, nodes = [], [], [], []
    blob = bytearray()

    def add_view(data: bytes, target: int | None = None) -> int:
        while len(blob) % 4:
            blob.append(0)
        off = len(blob)
        blob.extend(data)
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            v["target"] = target
        views.append(v)
        return len(views) - 1

    def add_accessor(arr, ctype, atype, target) -> int:
        vi = add_view(arr.tobytes(), target)
        a = {"bufferView": vi, "componentType": ctype, "count": len(arr), "type": atype}
        if atype != "SCALAR":
            a["min"] = arr.min(axis=0).tolist()
            a["max"] = arr.max(axis=0).tolist()
        else:
            a["min"] = [int(arr.min())]
            a["max"] = [int(arr.max())]
        accessors.append(a)
        return len(accessors) - 1

    for node in ref["nodes"]:
        number = node["name"].split("_")[1]
        contours, advance = text_contours(font, number)
        # Centre on the advance box horizontally, matching the existing
        # variants, and on the ink vertically.
        ys = [y for c in contours for _, y in c]
        offset = (-advance * scale / 2.0,
                  -(min(ys) + max(ys)) / 2.0 * scale)
        pos, nrm, uv, tan, idx = build_mesh(contours, scale, offset)

        prim = {"attributes": {
            "POSITION": add_accessor(pos, 5126, "VEC3", 34962),
            "NORMAL": add_accessor(nrm, 5126, "VEC3", 34962),
            "TEXCOORD_0": add_accessor(uv, 5126, "VEC2", 34962),
            "TANGENT": add_accessor(tan, 5126, "VEC4", 34962),
        }, "indices": add_accessor(idx, 5125, "SCALAR", 34963)}
        meshes.append({"name": f"Text_{number}", "primitives": [prim]})
        nodes.append({"mesh": len(meshes) - 1, "name": node["name"],
                      "translation": node["translation"],
                      **({"rotation": node["rotation"]} if "rotation" in node else {})})

    gltf = {
        "asset": {"version": "2.0", "generator": "make_numeral_glb.py"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }

    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    while len(blob) % 4:
        blob.append(0)
    with open(args.out, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack("<II", len(blob), 0x004E4942)); f.write(bytes(blob))

    tris = sum(a["count"] for i, a in enumerate(accessors) if a["type"] == "SCALAR") // 3
    print(f"wrote {args.out}")
    print(f"  {len(nodes)} numerals, {tris} triangles, cap height {CAP_HEIGHT_BU} BU")


if __name__ == "__main__":
    main()
