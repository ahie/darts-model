"""Generate a glyph atlas for procedurally assembled board printing.

Real boards carry printed logos and wordmarks. A model trained on faces with no
printing reads them as darts -- a red elongated mark on sisal is, to the
network, a dart -- so the renderer prints marks on the board as negatives.

A fixed set of finished wordmarks would be memorised: at 500 epochs of 10k
samples with two to four marks a frame, each of 256 is presented about 59,000
times. The network would learn those specific marks rather than the class they
belong to.

So the atlas holds individual GLYPHS and the shader assembles a mark from a
random sequence at draw time. The marks themselves never repeat, so there is
nothing specific to learn -- while the number ring, being separate geometry,
stays informative.

Layout: one row per font, one column per character. A mark picks a row and
walks columns, which keeps the typography within a single mark consistent, the
way real printing is.

Usage:
    python tools/generate_decal_atlas.py --out renderer/assets/decal_atlas.png
"""

from __future__ import annotations

import argparse
import glob
import random

from PIL import Image, ImageDraw, ImageFont

#: Column order. Its length must match kDecalAtlasGlyphCols in the renderer's
#: constants.h.
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

#: A font missing any of these renders tofu, which looks nothing like board
#: printing and would teach a distractor class that does not exist.
REQUIRED = set(CHARS)


def pick_fonts(limit: int) -> list[str]:
    """System fonts with full coverage that actually draw the glyphs.

    A cmap entry is not proof of a glyph: some faces map the range but draw
    .notdef, which renders as an empty box. Comparing a real string against
    private-use codepoints that are certainly absent catches the substitution.
    """
    from fontTools.ttLib import TTFont

    paths: list[str] = []
    for pat in (
        "/System/Library/Fonts/Supplemental/*.ttf",
        "/System/Library/Fonts/*.ttf",
        "/Library/Fonts/*.ttf",
    ):
        paths.extend(glob.glob(pat))
    paths.sort()  # deterministic before the seeded shuffle
    random.shuffle(paths)

    good: list[str] = []
    for path in paths:
        if len(good) >= limit:
            break
        try:
            tt = TTFont(path, fontNumber=0, lazy=True)
            cmap: set[int] = set()
            for table in tt["cmap"].tables:
                cmap.update(table.cmap.keys())
            tt.close()
            if not all(ord(c) in cmap for c in REQUIRED):
                continue
            probe = ImageFont.truetype(path, 48)
            if bytes(probe.getmask("ABC")) == bytes(probe.getmask("")):
                continue
            good.append(path)
        except Exception:
            continue
    return good


def render_glyph(ch: str, path: str, cell: int) -> Image.Image:
    """One glyph, centred and scaled to a consistent cap height.

    Scaled per glyph rather than per font: the shader lays glyphs out on a
    uniform grid, so a face with unusual metrics would otherwise produce marks
    whose letters jump in size mid-word.
    """
    blank = Image.new("L", (cell, cell), 0)
    target = int(cell * 0.72)
    size = target
    f = None
    for _ in range(10):
        try:
            f = ImageFont.truetype(path, size)
        except Exception:
            return blank
        bb = ImageDraw.Draw(blank).textbbox((0, 0), ch, font=f)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        if h <= 0 or w <= 0:
            return blank
        if h <= target and w <= cell * 0.92:
            break
        size = max(6, int(size * min(target / h, cell * 0.92 / w) * 0.96))
    if f is None:
        return blank
    img = Image.new("L", (cell, cell), 0)
    d = ImageDraw.Draw(img)
    bb = d.textbbox((0, 0), ch, font=f)
    d.text(((cell - (bb[2] - bb[0])) / 2 - bb[0],
            (cell - (bb[3] - bb[1])) / 2 - bb[1]), ch, font=f, fill=255)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="renderer/assets/decal_atlas.png")
    ap.add_argument("--fonts", type=int, default=32,
                    help="atlas rows; must equal kDecalAtlasFontRows in "
                         "renderer/src/constants.h")
    ap.add_argument("--cell", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    fonts = pick_fonts(args.fonts)
    # The renderer checks the atlas dimensions against kDecalAtlasFontRows and
    # aborts on a mismatch, so a short atlas is an error here, not later.
    if len(fonts) < args.fonts:
        raise SystemExit(
            f"found {len(fonts)} usable fonts, need {args.fonts}; nothing "
            f"written. Install more TTF faces with full A-Z 0-9 coverage.")
    cols, rows = len(CHARS), len(fonts)
    print(f"{rows} fonts x {cols} glyphs at {args.cell}px "
          f"-> {cols * args.cell}x{rows * args.cell}")

    # Alpha-only coverage: the shader supplies the ink colour, so one atlas
    # serves every tint instead of baking a palette in.
    atlas = Image.new("LA", (cols * args.cell, rows * args.cell), (255, 0))
    for r, path in enumerate(fonts):
        for c, ch in enumerate(CHARS):
            g = render_glyph(ch, path, args.cell)
            tile = Image.merge("LA", (Image.new("L", g.size, 255), g))
            atlas.paste(tile, (c * args.cell, r * args.cell))

    atlas.convert("RGBA").save(args.out)
    print(f"wrote {args.out}")
    for p in fonts:
        print(f"  {p.split('/')[-1]}")
    print("  rows must match kDecalAtlasFontRows, cols kDecalAtlasGlyphCols")


if __name__ == "__main__":
    main()
