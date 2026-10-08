"""Export a dense-readout checkpoint to a .tflite for the Android app.

The CoreML exporter's sibling, and deliberately thin: it imports
`ExportWrapper` from `darts_model.export.coreml` rather than restating it, so
the ImageNet normalisation and the in-graph readout have exactly one
definition and the two platforms cannot score the same throw differently.

## What is different from the CoreML path

**Output names may not survive.** CoreML carries the eight names and the Swift
side looks them up by name, precisely because an ordering is not promised and a
permuted field would produce plausible detections in the wrong places. A TFLite
signature may hand the outputs back positionally instead -- and five of the
eight share the shape (1, K, 2), so shape cannot tell `dart_tip` from
`dart_flight`. Guessing there would be silent and wrong in the worst way:
scores in the right beds for the wrong darts.

So the mapping is not assumed. It is resolved by running the torch reference on
the same frame and matching each converted output to the tensor it actually
equals, and the result is written next to the model as `<out>.outputs.json` for
the Android reader to load rather than hardcode.

**torch.export, not torch.jit.trace.** litert-torch builds on the former,
which is stricter about shapes. The wrapper specialises to a fixed grid for a
fixed input size, which is what both paths want anyway.

The whole export happens in a scratch directory next to `--out`: the .tflite
and its sidecar are moved into place together, and only once the mapping is
resolved and the converted model agrees with torch. A failed export leaves
nothing behind.

    darts-export-litert CHECKPOINT --config configs/detector.yaml \
        --out darts.tflite --sample frame.jpg
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from darts_model.export.common import (
    MIN_KEPT_CELLS,
    OUTPUTS,
    ExportError,
    output_names,
    check_against_reference,
    effective_topk,
    topk_for_stride,
    load_checkpoint,
    readout_contract,
    require_export_heads,
    set_distance,
    verification_frame,
    write_json,
)
from darts_model.model.detector import slot_floor
# The one wrapper, shared with the CoreML exporter. `coreml` imports
# coremltools only inside its functions, so this costs nothing here.
from darts_model.export.coreml import ExportWrapper

#: How much better the winning candidate must be than the runner-up before a
#: mapping is written. A permuted plane is not a crash -- it is scores in the
#: right beds for the wrong darts -- so an ambiguous resolution is refused
#: rather than guessed.
MIN_MARGIN = 10.0


def _distance(c: np.ndarray, r: np.ndarray, keep: np.ndarray,
              candidate_keep: np.ndarray | None) -> float:
    """Relative distance between a converted output and a reference plane.

    Per-cell planes are compared SET-wise: topk returns near-equal scores in a
    different order per backend, so rank i in one is not rank i in the other.
    Each cell the reference kept is matched to its nearest row among the cells
    the converted model scores comparably; the low-scoring rest are
    background cells whose planes hold arbitrary values that could sit close
    to anything. The keypoint outputs, whose order the per-channel argmax
    fixes, are compared elementwise.
    """
    per_cell = r.ndim >= 2 and r.shape[1] == keep.size
    if per_cell:
        err = set_distance(c, r, keep, candidate_keep)
        rows = r[0][keep]
        scale = float(np.abs(rows).max()) if rows.size else 0.0
    else:
        err = float(np.abs(c - r).max())
        scale = float(np.abs(r).max())
    return err / max(scale, 1e-9)


#: Slack below the lowest kept reference score within which a converted cell
#: still counts as a candidate match.
SCORE_SLACK = 0.05


def _candidate_cells(converted: list[np.ndarray], score: np.ndarray,
                     keep: np.ndarray) -> np.ndarray | None:
    """Rows of the converted outputs worth matching against.

    The score plane is the only (1, K) output, so it is identified by shape
    alone; its high-scoring rows are the candidates. None, meaning every row,
    when it cannot be identified.
    """
    same = [np.asarray(c) for c in converted
            if np.asarray(c).shape == score.shape]
    if len(same) != 1 or not keep.any():
        return None
    floor = float(score[0][keep].min()) - SCORE_SLACK
    return same[0][0] >= floor


def resolve_outputs(
    converted: list[np.ndarray],
    reference: list[np.ndarray],
    keep: np.ndarray,
    names: list[str] = OUTPUTS,
) -> tuple[dict[str, int], list[tuple[int, str, float, float]]]:
    """Work out which converted output is which named plane.

    Matched by value, not by shape and not by position. Five of the eight
    planes are (1, K, 2), so shape narrows it to five candidates and no
    further, and position is whatever torch.export and the converter happened
    to agree on -- which is the thing being checked rather than assumed.

    Greedy on the closest pair first, so the most certain assignment is made
    before the ambiguous ones and cannot be stolen.

    Returns the mapping, and per converted index how much better the winner was
    than the runner-up. The margin is the point: a near-tie means the resolution
    was luck, and a permuted plane produces plausible detections in the wrong
    places rather than an error anybody would see.
    """
    ref_score = np.asarray(reference[names.index("dart_score")])
    candidate_keep = _candidate_cells(converted, ref_score, keep)
    scores: dict[tuple[int, int], float] = {}
    for ci, c in enumerate(converted):
        c = np.asarray(c)
        for ri, r in enumerate(reference):
            r = np.asarray(r)
            if c.shape != r.shape:
                continue
            scores[(ci, ri)] = _distance(c, r, keep, candidate_keep)

    assigned: dict[int, int] = {}
    taken: set[int] = set()
    for (ci, ri), _ in sorted(scores.items(), key=lambda kv: kv[1]):
        if ci in assigned or ri in taken:
            continue
        assigned[ci] = ri
        taken.add(ri)

    mapping: dict[str, int] = {}
    report: list[tuple[int, str, float, float]] = []
    for ci in range(len(converted)):
        ri = assigned.get(ci)
        if ri is None:
            report.append((ci, "?", float("nan"), 0.0))
            continue
        best = scores[(ci, ri)]
        rivals = [v for (c, r), v in scores.items() if c == ci and r != ri]
        runner_up = min(rivals) if rivals else float("inf")
        # A ratio rather than an absolute error: an fp32 conversion lands near
        # zero anyway, and what makes the assignment safe is that every wrong
        # answer is much worse. Infinite when a shape has only one candidate
        # or the winner is exact and every rival is not; an exact tie is 1.
        if best > 0:
            margin = runner_up / best
        else:
            margin = float("inf") if runner_up > 0 else 1.0
        mapping[names[ri]] = ci
        report.append((ci, names[ri], best, margin))
    return mapping, report


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export a detector checkpoint to a .tflite plus its "
                    ".outputs.json sidecar.")
    ap.add_argument("checkpoint")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="darts.tflite")
    ap.add_argument("--size", type=int, default=None,
                    help="input side; defaults to the config's image_size")
    ap.add_argument("--sample", default=None,
                    help="an image showing darts in a board, used to resolve "
                         "the output mapping and verify the conversion. "
                         "Without it one frame is rendered, which needs the "
                         "renderer module.")
    args = ap.parse_args()
    try:
        return export(args)
    except ExportError as exc:
        print(f"!! {exc}", file=sys.stderr)
        return 1


def export(args) -> int:
    loaded = load_checkpoint(args.checkpoint, args.config)
    cfg = loaded.cfg
    require_export_heads(cfg)
    size = args.size or int(loaded.raw["data"]["image_size"])
    topk = effective_topk(size, cfg.out_stride)
    if topk < topk_for_stride(cfg.out_stride):
        print(f"  {size}px gives {topk} cells: K clamped from "
              f"{topk_for_stride(cfg.out_stride)}")
    gate = float(cfg.fg_threshold)

    model = ExportWrapper(loaded.lit.model, topk=topk).eval()
    names = output_names(cfg)
    example = verification_frame(args.sample, size, loaded.raw)
    with torch.no_grad():
        reference = [t.numpy() for t in model(example)]
    for name, t in zip(names, reference):
        print(f"  {name:<15} {tuple(t.shape)}")
    ref = dict(zip(names, reference))
    keep = ref["dart_score"][0] >= gate
    if int(keep.sum()) < MIN_KEPT_CELLS:
        raise ExportError(
            f"the model keeps {int(keep.sum())} cells on the verification "
            f"frame, too few to tell the per-cell planes apart; pass --sample "
            f"with darts in the board")

    try:
        import litert_torch
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        raise ExportError('the LiteRT packages are not installed: '
                          'pip install -e ".[litert]"') from None
    print("\nconverting ...")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    side = out.with_suffix(".outputs.json")
    scratch = Path(tempfile.mkdtemp(prefix=".export-", dir=out.parent))
    try:
        tmp = scratch / out.name
        litert_torch.convert(model, (example,)).export(str(tmp))

        interp = Interpreter(model_path=str(tmp))
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        interp.set_tensor(inp["index"], example.numpy().astype(inp["dtype"]))
        interp.invoke()
        details = interp.get_output_details()
        converted = [np.asarray(interp.get_tensor(d["index"]))
                     for d in details]

        print(f"\nresolving the output mapping over {int(keep.sum())} kept "
              f"cells ...")
        mapping, report = resolve_outputs(converted, reference, keep, names)
        for ci, name, err, margin in report:
            shape = tuple(int(x) for x in details[ci]["shape"])
            margin_s = ("only candidate" if margin == float("inf")
                        else f"{margin:.0f}x")
            print(f"  [{ci}] {str(shape):<18} -> {name:<15} err {err:.2e}   "
                  f"next best {margin_s}")

        unresolved = [n for n in names if n not in mapping]
        if unresolved:
            print(f"!! could not resolve {unresolved}; nothing written")
            return 1
        weak = [(ci, n, m) for ci, n, _, m in report if m < MIN_MARGIN]
        if weak:
            for ci, n, m in weak:
                print(f"!! [{ci}] -> {n} was only {m:.1f}x better than the "
                      f"next candidate; {MIN_MARGIN:.0f}x is the floor")
            print("   an ambiguous mapping would send the Android reader to "
                  "the wrong plane; nothing written")
            return 1

        print("\nchecking the converted model against torch ...")
        got = {name: converted[i] for name, i in mapping.items()}
        failures = check_against_reference(
            ref, got, size, gate, px_limit=1.0,
            slot_floor=slot_floor(cfg),
            tip_floor=float(getattr(cfg, "tip_snap_score", 0.3)))
        if failures:
            print(f"\n{failures} check(s) failed; nothing written")
            return 1

        write_json(side, {"abi": 1, "inputSize": size, "outputs": mapping,
                          **readout_contract(cfg, size, topk)})
        os.replace(tmp, out)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"\nwrote {out} and {side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
