"""What the exporters and the visualiser share.

Checkpoint loading, the verification frame, the output contract written into
every exported model, and the set-wise comparison between a converted model
and the torch reference. One definition of each, so the CoreML and LiteRT
paths cannot drift apart.

Nothing here imports coremltools or the LiteRT packages.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml

#: Output names, in the order ExportWrapper returns them. Both apps look these
#: up BY NAME (CoreML) or through the resolved mapping (LiteRT): an ordering is
#: not promised, and a silently permuted field still produces plausible
#: detections in the wrong places.
OUTPUTS = [
    "dart_score",      # (1, K)     descending, sigmoid of the fg logit
    "dart_centre",     # (1, K, 2)  absolute, normalised image coords
    "dart_direction",  # (1, K, 2)  unit, tip -> flight
    "dart_extent",     # (1, K, 2)  half-length, half-width, normalised
    "dart_tip",        # (1, K, 2)  absolute -- the landing point
    "dart_flight",     # (1, K, 2)  absolute
    "kp_xy",           # (1, 40, 2) argmax cell + sub-cell offset, normalised
    "kp_conf",         # (1, 40)    sigmoid of the peak logit
]

#: The per-dart readout's outputs, after OUTPUTS, when the model has one
#: (``instance_head`` or ``query_head``). The per-cell outputs stay: they cost
#: nothing, and they are what the comparison against torch can check cell by
#: cell.
SLOT_OUTPUTS = [
    "slot_score",      # (1, S)     a dart when >= the contract's
                       #            darts.min_score
    "slot_tip",        # (1, S, 2)  absolute -- the landing point
    "slot_flight",     # (1, S, 2)  absolute
]

#: With the stage-3 token readout (``token_readout``), after SLOT_OUTPUTS: the
#: landing point read as the tip pointer's single top option instead of its
#: blend, so an app can compare the two.
HARD_SLOT_OUTPUTS = [
    "slot_tip_hard",   # (1, S, 2)  absolute
]


#: The tip heatmap's peaks, last, when the model has the head
#: (``predict_tip_heatmap``): what a reader without in-graph slots snaps to.
TIP_OUTPUTS = [
    "tip_xy",          # (1, P, 2)  peak plus sub-cell offset, normalised
    "tip_score",       # (1, P)     sigmoid at the peak, descending; 0 = none
]


def output_names(cfg) -> list[str]:
    """The exported outputs for this config, in ExportWrapper's order."""
    from darts_model.model.detector import slot_count
    return (OUTPUTS + (SLOT_OUTPUTS if slot_count(cfg) else [])
            + (HARD_SLOT_OUTPUTS if getattr(cfg, "token_readout", False)
               else [])
            + (TIP_OUTPUTS if getattr(cfg, "predict_tip_heatmap", False)
               else []))


#: Cells kept per frame at output stride 8; see :func:`topk_for_stride`.
#:
#: A dart covers ~83 cells at stride 8, so three darts is ~250 and nine would
#: still fit. Truncation is not silent: the scores come back descending, so
#: `dart_score[K-1]` still clearing the threshold means the frame hit the cap.
#: Clamped to the grid size for inputs too small to have this many cells; the
#: effective value is written into the exported model's metadata.
TOPK = 1024


def topk_for_stride(out_stride: int) -> int:
    """Cells kept per frame at ``out_stride``: TOPK scaled with cell area.

    A dart covers four times as many cells at half the stride, so K scales
    by the same factor to keep the same number of darts' worth of cells.
    """
    return max(1, round(TOPK * (8 / out_stride) ** 2))

#: Kept cells a verification frame must produce. Below this the per-cell
#: planes cannot be told apart or meaningfully compared.
MIN_KEPT_CELLS = 3


class ExportError(RuntimeError):
    """An export precondition failed; the message says what to do."""


# --------------------------------------------------------------------------
# config and checkpoint
# --------------------------------------------------------------------------

def require_export_heads(cfg) -> None:
    """The exported readout needs both optional heads.

    `dart_tip`/`dart_flight` come from the ends head and `kp_xy`/`kp_conf`
    from the keypoint head; a model without them has nothing to put there.
    """
    missing = [k for k in ("predict_ends", "predict_keypoints")
               if not getattr(cfg, k, False)]
    if missing:
        raise ExportError(
            f"the config sets {', '.join(f'{k}: false' for k in missing)}; the "
            f"exported readout requires predict_ends and predict_keypoints "
            f"(configs/detector.yaml, head section)")
    if getattr(cfg, "claim_by_embedding", False):
        # The embedding head itself exports fine and is simply not read; it is
        # the readout that would differ from what the app implements.
        raise ExportError(
            "the config sets claim_by_embedding: true, but the exported "
            "outputs carry no embedding and the apps group cells by centre "
            "bin. Export with claim_by_embedding: false")


def effective_topk(size: int, out_stride: int, topk: int | None = None) -> int:
    """Validate an input size and return the K the exported graph uses.

    The size must be a multiple of the output stride, or the cell-centre
    convention `(i + 0.5) / w` no longer maps cells onto image pixels. K is
    clamped to the number of cells.
    """
    if size <= 0 or size % out_stride:
        raise ExportError(
            f"--size {size} is not a positive multiple of the output stride "
            f"{out_stride}")
    cells = (size // out_stride) ** 2
    return min(topk if topk is not None else topk_for_stride(out_stride), cells)


def load_config(config_path: str):
    """(raw yaml, DenseDartConfig) with the weight-init paths cleared.

    The checkpoint carries every weight, so loading the pretrained backbone or
    `init_weights` first is wasted work and names files that may exist only on
    the machine that trained it.
    """
    from darts_model.cli.train import build_config

    raw = yaml.safe_load(Path(config_path).read_text())
    cfg = build_config(raw)
    cfg.backbone_weights = ""
    cfg.init_weights = ""
    return raw, cfg


@dataclass
class LoadedCheckpoint:
    lit: torch.nn.Module
    cfg: object
    raw: dict
    epoch: int | None
    global_step: int | None


def load_checkpoint(checkpoint: str, config_path: str,
                    log=print) -> LoadedCheckpoint:
    """Build the detector from its config and load a trained checkpoint.

    Fails on ANY missing `model.` key: that is the network every consumer runs,
    and a missing tensor there means randomly initialised weights shipped or
    drawn as if trained. Keys outside `model.` (GPUAugment's ImageNet
    constants) are defaults rather than trained values and are only reported.
    Unexpected keys are reported, not fatal.
    """
    from darts_model.model.detector import DenseDartLitModule

    raw, cfg = load_config(config_path)
    lit = DenseDartLitModule(cfg)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" not in ck:
        raise ExportError(f"{checkpoint} has no state_dict; is it a Lightning "
                          f"checkpoint?")
    result = lit.load_state_dict(ck["state_dict"], strict=False)
    absent = [k for k in result.missing_keys if k.startswith("model.")]
    if absent:
        raise ExportError(
            f"{len(absent)} model weights are missing from {checkpoint}, e.g. "
            f"{absent[:4]}. The config does not describe the network this "
            f"checkpoint was trained with; using it would run randomly "
            f"initialised weights.")
    outside = [k for k in result.missing_keys if not k.startswith("model.")]
    if outside:
        log(f"  {len(outside)} non-model keys defaulted (constants, not "
            f"weights): {outside[:4]}")
    if result.unexpected_keys:
        log(f"  {len(result.unexpected_keys)} unexpected keys (ignored), e.g. "
            f"{result.unexpected_keys[:3]}")
    log(f"loaded {checkpoint}  (epoch {ck.get('epoch')}, "
        f"step {ck.get('global_step')})")
    return LoadedCheckpoint(lit, cfg, raw, ck.get("epoch"),
                            ck.get("global_step"))


# --------------------------------------------------------------------------
# the verification frame
# --------------------------------------------------------------------------

def centre_crop_resize(im, size: int):
    """Centre-crop to a square, then resize: what Vision's `.centerCrop` does
    in the iOS app. Verifying through a different crop would compare two
    different pictures."""
    from PIL import Image

    w, h = im.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    im = im.crop((left, top, left + side, top + side))
    return im.resize((size, size), Image.BILINEAR)


def to_input(rgb: np.ndarray) -> torch.Tensor:
    """(H, W, 3) 0-255 -> (1, 3, H, W) float32 0-255, the exported input."""
    arr = np.ascontiguousarray(np.asarray(rgb)[..., :3]).astype(np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1)[None].contiguous()


def load_sample(path: str, size: int) -> torch.Tensor:
    from PIL import Image

    return to_input(np.asarray(
        centre_crop_resize(Image.open(path).convert("RGB"), size)))


def render_sample(size: int, raw: dict, seed: int = 1234,
                  attempts: int = 10) -> torch.Tensor:
    """Render one frame with three darts in it, at the export size.

    Raises ExportError if the renderer module is not importable, so the
    exporter stops before writing anything.
    """
    try:
        from darts_model.renderer import import_renderer
        dartboard_renderer = import_renderer()
    except ImportError as exc:
        raise ExportError(
            "no --sample given and the renderer is not importable, so there "
            "is no frame to verify the export against. Pass --sample with a "
            "photo or rendered frame that shows darts in the board, or build "
            "the renderer and put it on PYTHONPATH.\n" + str(exc)) from exc

    data = raw.get("data") or {}
    kwargs = dict(width=size, height=size, seed=seed,
                  dart_count_weights=[0.0, 0.0, 0.0, 1.0])
    bg = data.get("bg_image_dir") or ""
    if bg and Path(bg).is_dir():
        kwargs["bg_image_dir"] = bg
    if data.get("asset_dir"):
        kwargs["asset_dir"] = data["asset_dir"]
    renderer = dartboard_renderer.Renderer(**kwargs)
    for _ in range(attempts):
        image, annotation = renderer.render_frame()
        if annotation.get("darts"):
            return to_input(image)
    raise ExportError(f"the renderer produced no dart in {attempts} frames; "
                      f"pass --sample")


def verification_frame(sample: str | None, size: int, raw: dict,
                       log=print) -> torch.Tensor:
    """The frame both backends are run on: --sample, else a rendered one.

    Never noise. On random input the model emits a degenerate field with no
    detections, so a comparison would measure rounding on values the model
    never produces, and the LiteRT output mapping could not be resolved.
    """
    if sample:
        log(f"verifying against {sample}")
        return load_sample(sample, size)
    log("no --sample: rendering a verification frame")
    return render_sample(size, raw)


# --------------------------------------------------------------------------
# the output contract
# --------------------------------------------------------------------------

def output_shapes(topk: int, num_keypoints: int, slots: int = 0,
                  tip_peaks: int = 0, hard_slots: bool = False) -> dict[str, list[int]]:
    k, nk = topk, num_keypoints
    shapes = {
        "dart_score": [1, k], "dart_centre": [1, k, 2],
        "dart_direction": [1, k, 2], "dart_extent": [1, k, 2],
        "dart_tip": [1, k, 2], "dart_flight": [1, k, 2],
        "kp_xy": [1, nk, 2], "kp_conf": [1, nk],
    }
    if slots:
        shapes.update({"slot_score": [1, slots], "slot_tip": [1, slots, 2],
                       "slot_flight": [1, slots, 2]})
        if hard_slots:
            shapes["slot_tip_hard"] = [1, slots, 2]
    if tip_peaks:
        shapes.update({"tip_xy": [1, tip_peaks, 2], "tip_score": [1, tip_peaks]})
    return shapes


def readout_contract(cfg, size: int, topk: int) -> dict:
    """Everything an app needs to turn the outputs into detections.

    Written into the CoreML metadata and the LiteRT sidecar alike, so neither
    app hardcodes a number that a retrained model could change.
    """
    from darts_model.board_geometry import (
        BOARD_KEYPOINT_NAMES,
        DOUBLE_CROSSING_INDICES,
    )

    kp_names = [BOARD_KEYPOINT_NAMES[i] for i in DOUBLE_CROSSING_INDICES]
    if len(kp_names) != cfg.num_keypoints:
        raise ExportError(
            f"num_keypoints is {cfg.num_keypoints}, but the keypoint order has "
            f"{len(kp_names)} names")
    from darts_model.model.detector import slot_count, slot_floor

    contract = {
        "input_size": size,
        "input": {
            "name": "image", "shape": [1, 3, size, size], "channels": "RGB",
            "range": "0-255",
        },
        "normalisation": (
            "ImageNet mean/std is applied inside the graph; pass raw RGB "
            "0-255 and do not normalise in the app"),
        "coordinates": "normalised image coordinates, x right, y down, [0, 1]",
        "output_stride": int(cfg.out_stride),
        "grid": [size // cfg.out_stride, size // cfg.out_stride],
        "topk": topk,
        "readout": {
            "vote_bin_px": float(cfg.vote_bin_px),
            "fg_threshold": float(cfg.fg_threshold),
            "peak_min_votes": int(cfg.peak_min_votes),
            "vote_bins": max(int(round(size / cfg.vote_bin_px)), 1),
        },
        "kp_names": kp_names,
        "output_shapes": output_shapes(
            topk, cfg.num_keypoints, slot_count(cfg),
            cfg.tip_peak_count if getattr(cfg, "predict_tip_heatmap", False)
            else 0, bool(getattr(cfg, "token_readout", False))),
    }
    if getattr(cfg, "predict_tip_heatmap", False):
        contract["tips"] = {
            "snap_score": float(cfg.tip_snap_score),
            "snap_px": float(cfg.tip_snap_px),
            "snap_axis_px": float(cfg.tip_snap_axis_px),
            "slots_snapped": bool(cfg.tip_snap and slot_count(cfg)
                                  and cfg.tip_assign == "nearest"),
            "assign": getattr(cfg, "tip_assign", "nearest"),
            "assign_gate_px": float(getattr(cfg, "tip_assign_gate_px", 40.0)),
            # The query readout chose among its estimate and the peaks
            # itself; slot_tip is that choice and needs no snap.
            "slots_pointed": bool(getattr(cfg, "tip_pointer", False)),
            "note": ("assign 'hungarian': give the frame's darts the tip_xy "
                     "scoring >= snap_score one-to-one, minimising total "
                     "distance, none further than assign_gate_px; a dart "
                     "with none keeps its own landing point. assign "
                     "'nearest': each dart takes the nearest tip_xy scoring "
                     ">= snap_score within snap_px of it and snap_axis_px of "
                     "its axis. slot_tip already has this applied when "
                     "slots_snapped"),
        }
    if slot_count(cfg):
        # The darts are read out in the graph; the cells and the Hough
        # parameters above remain for debugging and older readers.
        contract["darts"] = {
            "slots": slot_count(cfg),
            "min_score": slot_floor(cfg),
            "note": ("slot i is a dart when slot_score[i] >= min_score; its "
                     "landing point is slot_tip[i]. The cell outputs need no "
                     "Hough voting for this."),
        }
        if getattr(cfg, "token_readout", False):
            contract["darts"]["hard_note"] = (
                "slot_tip_hard[i] is the same dart's landing point read as the "
                "tip pointer's single top option instead of its blend")
    return contract


# --------------------------------------------------------------------------
# comparing a converted model against torch
# --------------------------------------------------------------------------

#: The per-cell planes, which topk may return in a different order per backend.
CELL_PLANES = ("dart_centre", "dart_direction", "dart_extent", "dart_tip",
               "dart_flight")


def _rows(plane: np.ndarray) -> np.ndarray:
    """(1, K[, C]) -> (K, C)."""
    a = np.asarray(plane, dtype=np.float64)[0]
    return a.reshape(a.shape[0], -1)


def set_distance(candidate: np.ndarray, reference: np.ndarray,
                 keep: np.ndarray,
                 candidate_keep: np.ndarray | None = None) -> float:
    """Worst distance from a kept reference row to its nearest candidate row.

    Order-free: a cell kept by the reference may sit at any rank in the
    candidate. `candidate_keep` limits the candidate rows (to the cells the
    candidate itself scores highly); by default every row is eligible.
    """
    ref = _rows(reference)[keep]
    cand = _rows(candidate)
    if candidate_keep is not None:
        cand = cand[candidate_keep]
    if cand.shape[0] == 0:
        return float("inf")
    if ref.shape[0] == 0:
        return 0.0
    d = np.linalg.norm(ref[:, None, :] - cand[None, :, :], axis=-1)
    return float(d.min(axis=1).max())


@dataclass
class CellMatch:
    """Per-plane worst error over the kept cells, after set-wise matching."""
    kept_ref: int
    kept_got: int
    worst: dict[str, float]


def match_cells(ref: dict[str, np.ndarray], got: dict[str, np.ndarray],
                gate: float) -> CellMatch:
    """Pair every kept reference cell with the same cell in the converted
    output, and report how far each plane moved.

    topk reorders near-equal scores between backends, so rank i against rank
    i compares different cells. Matching on the centre alone does not work
    either: every cell on one dart votes nearly the same centre. So cells are
    matched on the concatenation of every per-cell value -- score, centre,
    direction, extent, tip, flight -- where the same cell agrees to rounding
    and any other cell differs in something.
    """
    keep = np.asarray(ref["dart_score"])[0] >= gate
    names = ("dart_score",) + CELL_PLANES

    def joint(d):
        return np.concatenate([_rows(d[n]) for n in names], axis=1)

    a, b = joint(ref)[keep], joint(got)
    worst = {n: 0.0 for n in names}
    if a.shape[0]:
        j = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1).argmin(1)
        start = 0
        for n in names:
            w = _rows(ref[n]).shape[1]
            diff = a[:, start:start + w] - b[j, start:start + w]
            worst[n] = float(np.linalg.norm(diff, axis=1).max())
            start += w
    return CellMatch(int(keep.sum()),
                     int((np.asarray(got["dart_score"])[0] >= gate).sum()),
                     worst)


def check_slots(ref: dict[str, np.ndarray], got: dict[str, np.ndarray],
                size: int, floor: float, px_limit: float, log=print) -> int:
    """Compare the per-dart outputs; returns the number of failed checks.

    Set-wise on the landing point: two slots of near-equal vote density may
    come out in either order. The same darts must clear ``floor`` on both
    sides, and each one's landing and flight points must agree within
    ``px_limit``.
    """
    failures = 0
    rs, gs = np.asarray(ref["slot_score"])[0], np.asarray(got["slot_score"])[0]
    rk, gk = rs >= floor, gs >= floor
    log(f"  darts over {floor}: torch {int(rk.sum())}, converted {int(gk.sum())}")
    if rk.sum() != gk.sum():
        log("  !! a different number of darts was read out")
        return failures + 1
    rt = np.asarray(ref["slot_tip"])[0][rk]
    gt = np.asarray(got["slot_tip"])[0][gk]
    if not rt.shape[0]:
        return failures
    j = np.linalg.norm(rt[:, None] - gt[None], axis=-1).argmin(1)
    for name in ("slot_tip", "slot_flight"):
        a = np.asarray(ref[name])[0][rk]
        b = np.asarray(got[name])[0][gk][j]
        worst = float(np.linalg.norm(a - b, axis=-1).max()) * size
        log(f"  {name:<15} worst {worst:.4f}px")
        if worst > px_limit:
            log(f"  !! {name} moved more than {px_limit}px")
            failures += 1
    return failures


def check_tips(ref: dict[str, np.ndarray], got: dict[str, np.ndarray],
               size: int, floor: float, px_limit: float, log=print) -> int:
    """Compare the tip peaks set-wise; returns the number of failed checks.

    The peaks clearing ``floor`` must be as many on both sides and each within
    ``px_limit`` of its counterpart.
    """
    rs, gs = np.asarray(ref["tip_score"])[0], np.asarray(got["tip_score"])[0]
    rk, gk = rs >= floor, gs >= floor
    log(f"  tip peaks over {floor}: torch {int(rk.sum())}, converted "
        f"{int(gk.sum())}")
    if rk.sum() != gk.sum():
        log("  !! a different set of tip peaks cleared the snap threshold")
        return 1
    a = np.asarray(ref["tip_xy"])[0][rk]
    b = np.asarray(got["tip_xy"])[0][gk]
    if not a.shape[0]:
        return 0
    worst = float(np.linalg.norm(a[:, None] - b[None], axis=-1).min(1).max()) * size
    log(f"  tip_xy          worst {worst:.4f}px")
    if worst > px_limit:
        log(f"  !! tip_xy moved more than {px_limit}px")
        return 1
    return 0


def check_against_reference(ref: dict[str, np.ndarray],
                            got: dict[str, np.ndarray], size: int,
                            gate: float, px_limit: float = 1.0,
                            log=print, slot_floor: float = 0.0,
                            tip_floor: float = 0.3) -> int:
    """Print the comparison and return the number of failed checks.

    Keypoints are compared elementwise, since the per-channel argmax fixes
    their order; cells set-wise, through `match_cells`; the per-dart slots,
    when present, through `check_slots` against ``slot_floor``. Positional planes fail
    above `px_limit` pixels, the direction above `px_limit / 100`; the score
    is reported only.
    """
    failures = 0
    kp_dxy = np.abs(np.asarray(got["kp_xy"]) - ref["kp_xy"])
    kp_dc = np.abs(np.asarray(got["kp_conf"]) - ref["kp_conf"])
    px = float(kp_dxy.max()) * size
    log(f"  kp_xy    worst {px:.3f}px   median "
        f"{float(np.median(kp_dxy)) * size:.3f}px")
    log(f"  kp_conf  worst {kp_dc.max():.4f}   median "
        f"{np.median(kp_dc):.4f}   over 0.01: "
        f"{int((kp_dc > 0.01).sum())}/{kp_dc.size}")
    # A corner can land in the right cell and still come back with a
    # confidence 0.6 lower, which is enough to be rejected by the app's gate or
    # mis-weighted by the homography fit.
    if kp_dc.max() > 0.05:
        log(f"  !! kp_conf moved by up to {kp_dc.max():.3f} while positions "
            f"moved {px:.2f}px: precision loss in the keypoint head")
        failures += 1

    m = match_cells(ref, got, gate)
    log(f"  cells over {gate}: torch {m.kept_ref}, converted {m.kept_got}")
    # Cells right at the gate may cross it by rounding; a large difference is
    # a real change in what the readout sees.
    if abs(m.kept_ref - m.kept_got) > max(2, m.kept_ref // 50):
        log("  !! a different set of cells cleared the gate")
        failures += 1
    limits = {"dart_score": (float("inf"), 1.0, ""),
              "dart_direction": (px_limit / 100.0, 1.0, "")}
    for name, worst in m.worst.items():
        limit, scale, unit = limits.get(name, (px_limit, float(size), "px"))
        shown = worst * scale
        log(f"  {name:<15} worst {shown:.4f}{unit}")
        if shown > limit:
            log(f"  !! {name} moved more than {limit}{unit}")
            failures += 1
    if "slot_score" in ref:
        failures += check_slots(ref, got, size, slot_floor, px_limit, log)
    if "tip_score" in ref:
        failures += check_tips(ref, got, size, tip_floor, px_limit, log)
    if np.asarray(ref["dart_score"])[0][-1] >= gate:
        log(f"  !! every one of the {ref['dart_score'].shape[1]} slots cleared "
            f"the gate, so votes were dropped on this frame")
    return failures


# --------------------------------------------------------------------------
# fp16 placement
# --------------------------------------------------------------------------

def keypoint_module_names(model: torch.nn.Module) -> list[str]:
    """Top-level submodules of the keypoint head (`kp_trunk`, `kp_head`,
    `kp_offset`) and of the tip heatmap head, which is built the same way
    and is as exposed to fp16 weights; read from the network rather than
    assumed."""
    return [n for n, _ in model.named_children()
            if n.startswith(("kp_", "tip_"))]


def _normalise_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (name or "").lower())


def uses_keypoint_weights(op_name: str, input_names: Iterable[str],
                          modules: Iterable[str]) -> bool:
    """Does this op belong to the keypoint head?

    Decided by the names of the op and of its inputs. The converter derives a
    weight constant's name from the module path of the parameter
    (`model.kp_trunk.0.0.weight` becomes something like
    `model_kp_trunk_0_0_weight`), so every conv and every GroupNorm affine in
    the head consumes a constant carrying one of `modules` as a whole
    underscore-delimited token, whatever the op itself was named.
    """
    tokens = [_normalise_name(m).strip("_") for m in modules]
    pats = [re.compile(r"(?:^|_)" + re.escape(t) + r"(?:_|$)")
            for t in tokens if t]
    for name in (op_name, *input_names):
        n = _normalise_name(name)
        if any(p.search(n) for p in pats):
            return True
    return False


def _producers(op) -> list:
    out = []
    for v in (getattr(op, "inputs", None) or {}).values():
        for var in (v if isinstance(v, (list, tuple)) else (v,)):
            producer = getattr(var, "op", None)
            if producer is not None:
                out.append(producer)
    return out


class KeypointFp32Selector:
    """`op_selector` for `ct.transform.FP16ComputePrecision`: True casts an op
    to fp16; the keypoint head's ops, and every op downstream of a ``topk``,
    return False and stay fp32.

    Downstream of ``topk`` is the readout: the kept cells' positions plus
    their offsets, and with a per-dart head the seed picking, membership and
    weighted means over normalised coordinates. fp16 holds a normalised
    coordinate in steps of ~0.5px and turns near-equal vote densities into
    ties that pick different seeds -- measured: one fp16 conversion moved a
    slot's landing point by 44px where fp32 agrees to 0.001px. These ops run
    over a thousand cells, not the image, so fp32 costs nothing measurable.

    Records what it kept, so the exporter can refuse a conversion in which
    nothing matched -- the failure mode of any name-based selection.
    """

    def __init__(self, modules: Iterable[str]) -> None:
        self.modules = list(modules)
        self.kept: list[str] = []
        self.readout_kept: list[str] = []
        self._after_topk: dict[int, bool] = {}

    def after_topk(self, op) -> bool:
        """Is ``op`` a ``topk`` or fed, at any depth, by one? Iterative, as
        the graph is deeper than Python's recursion limit."""
        memo = self._after_topk
        stack = [(op, False)]
        while stack:
            node, expanded = stack.pop()
            key = id(node)
            if key in memo:
                continue
            if getattr(node, "op_type", None) == "topk":
                memo[key] = True
                continue
            parents = _producers(node)
            if expanded:
                memo[key] = any(memo.get(id(p), False) for p in parents)
                continue
            stack.append((node, True))
            stack.extend((p, False) for p in parents if id(p) not in memo)
        return memo[id(op)]

    def __call__(self, op) -> bool:
        names = []
        for v in (getattr(op, "inputs", None) or {}).values():
            for var in (v if isinstance(v, (list, tuple)) else (v,)):
                names.append(getattr(var, "name", "") or "")
                producer = getattr(var, "op", None)
                if producer is not None:
                    names.append(getattr(producer, "name", "") or "")
        if uses_keypoint_weights(getattr(op, "name", "") or "", names,
                                 self.modules):
            self.kept.append(op.name)
            return False
        if self.after_topk(op):
            self.readout_kept.append(op.name)
            return False
        return True


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")
