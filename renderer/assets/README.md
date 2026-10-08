# Renderer assets

The renderer generates the board, the wire frame (spider) and the darts from
the constants in `src/constants.h`. The files here are the only things it
loads, and all of them are required: the renderer aborts if one is missing.

| File | What | Made by |
|---|---|---|
| `numbers_variant_0.glb` … `_5.glb` | Number ring, one typeface per file (20 meshes, `Text.*`) | Exported from Blender; the source scene is not included |
| `numbers_variant_6.glb` | Number ring (20 meshes, `Text_*`) | `tools/make_numeral_glb.py` from a TTF |
| `decal_atlas.png` | 32 fonts × 36 glyphs (A–Z, 0–9), 128 px cells | `tools/generate_decal_atlas.py` (macOS system fonts) |

There are 7 numeral variants. `NUM_FONT_VARIANTS` in `src/constants.h` must
equal the number of `numbers_variant_*.glb` files, and a new variant is added
as `numbers_variant_7.glb` with the constant raised to match.

`kDecalAtlasGlyphCols` / `kDecalAtlasFontRows` in `src/constants.h` are the
atlas's only layout description: the renderer derives the cell size from the
image and checks its dimensions against them at load.
`tools/generate_decal_atlas.py` refuses to write an atlas with fewer rows than
`--fonts` (default 32).

The decal atlas supplies printed marks (logos, lettering) on the board's outer
ring. Each mark is a new random glyph sequence, so no specific mark repeats
often enough to be memorised. Without marks, models read real board printing
as darts.

These are the exact files the released models were trained with.
