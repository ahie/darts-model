"""Draw what the dense readout actually predicts, next to the truth.

Usage::

    darts-visualize <checkpoint> --config configs/detector.yaml \\
        [--n 8] [--out preview/dense]

Renders fresh frames, runs the model, and draws the predicted oriented box with
its tip and flight against the ground truth on the same image.

Draws three things per dart, because they are three different claims:

  the oriented box        centre, direction and both extents
  the landing point       where the dart entered the board -- what SCORING uses
  the flight tip          the other end

The landing point and the flight tip are predicted directly, not read off the
box, and the renderer reports both as ground truth. The box-derived tip is
drawn too, as a faint cross, because the gap between it and the landing point
is why they are separate outputs: for a dart pointing near the camera the
entry falls INSIDE the silhouette, up to 27px from any end of the box, since
the dart's own barrel and flight cover where it went in.

A box that is the right size but rotated 180 degrees is also a real failure
mode here, which is why tip and flight get different markers -- two identical
ones would hide it completely, and a tip-pixel average certainly does.

The 40 board corners are drawn too, small and faint. At a median error near
0.25px the predicted and true marks sit on top of each other and there is
nothing to see, which is the point: what matters in that head is the OUTLIERS,
since RANSAC discards them downstream and a few bad corners cost nothing. So a
corner is only called out -- ringed, and joined to its true position -- once it
misses by more than a few pixels.

Detections come through the real Hough readout, not from reading the dense
field at the ground-truth locations, so what is drawn is what inference would
produce.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from darts_model.data.config import build_data_config
from darts_model.data.detect_dataset import DenseDetectDataset
from darts_model.export.common import ExportError, load_checkpoint
from darts_model.model.detector import decode_boxes
from darts_model.model.hough import detect

PRED = "#ff3b30"
TRUE = "#34c759"
#: A corner missing by more than this is ringed. Below it the two marks
#: overlap and drawing a line between them is noise.
KP_CALLOUT_PX = 3.0


def corners(cx, cy, dx, dy, hl, hw):
    """Oriented box -> its four corners, tip end first."""
    px, py = -dy, dx                       # across the axis
    return np.array([
        (cx - dx * hl - px * hw, cy - dy * hl - py * hw),
        (cx + dx * hl - px * hw, cy + dy * hl - py * hw),
        (cx + dx * hl + px * hw, cy + dy * hl + py * hw),
        (cx - dx * hl + px * hw, cy - dy * hl + py * hw),
    ])


def draw_box(ax, cx, cy, dx, dy, hl, hw, colour, label, scale,
             tip=None, flight=None):
    """Box plus its two ends.

    `tip` and `flight` are the points to mark. Passed in rather than derived,
    because they are not in general the box's ends: the fallback of
    centre -/+ direction * half_length is only right for a dart seen side-on.
    """
    cx, cy, hl, hw = cx * scale, cy * scale, hl * scale, hw * scale
    c = corners(cx, cy, dx, dy, hl, hw)
    ax.plot(*np.append(c, c[:1], axis=0).T, color=colour, lw=1.4, alpha=0.9)

    box_tip = (cx - dx * hl, cy - dy * hl)
    t = (tip[0] * scale, tip[1] * scale) if tip is not None else box_tip
    f = ((flight[0] * scale, flight[1] * scale) if flight is not None
         else (cx + dx * hl, cy + dy * hl))

    # Faint cross where the BOX says the tip is. When it separates from the
    # circle, that is the end-on case the direct prediction exists for.
    if tip is not None:
        ax.plot(*box_tip, marker="x", ms=6, color=colour, mew=1.2, alpha=0.45)

    ax.plot(*t, marker="o", ms=8, mfc="none", mec=colour, mew=2.2)
    ax.plot(*f, marker="s", ms=6, mfc="none", mec=colour, mew=1.6)
    ax.plot([t[0], f[0]], [t[1], f[1]], color=colour, lw=0.9, ls=":",
            alpha=0.8)
    if label:
        ax.text(t[0] + 7, t[1] - 7, label, color=colour, fontsize=7,
                weight="bold")


def predicted_keypoints(out, scale):
    """Board corners, read out exactly as the metric and inference do.

    Argmax per channel plus the predicted sub-cell offset -- not a soft-argmax
    over the whole map, which would let a broad, badly-peaked heatmap average
    its way to the right answer and draw a precision the head does not have.
    """
    logits = out["kp_logits"].float()
    b, k, h, w = logits.shape
    flat = logits.flatten(2).argmax(dim=2)
    iy, ix = flat // w, flat % w
    off = out["kp_offset"].float()
    bi = torch.arange(b, device=logits.device).view(b, 1).expand(b, k)
    px = (ix.float() + 0.5 + off[bi, 0, iy, ix]) / w
    py = (iy.float() + 0.5 + off[bi, 1, iy, ix]) / h
    return torch.stack([px, py], dim=-1)[0].cpu().numpy() * scale


def draw_keypoints(ax, pred, true, mask):
    """Both sets, with only the misses called out."""
    errs = []
    for i in range(len(true)):
        if not mask[i]:
            continue
        t, p = true[i], pred[i]
        e = float(np.hypot(p[0] - t[0], p[1] - t[1]))
        errs.append(e)
        ax.plot(*t, marker=".", ms=3.5, color=TRUE, alpha=0.75)
        ax.plot(*p, marker=".", ms=3.5, color=PRED, alpha=0.75)
        if e > KP_CALLOUT_PX:
            ax.plot([t[0], p[0]], [t[1], p[1]], color=PRED, lw=1.0,
                    alpha=0.9)
            ax.plot(*p, marker="o", ms=11, mfc="none", mec=PRED, mew=1.4)
            ax.text(p[0] + 8, p[1] + 10, f"{e:.0f}px", color=PRED,
                    fontsize=6.5)
    return np.array(errs)


def _pyplot():
    try:
        import matplotlib
    except ImportError:
        raise SystemExit("darts-visualize needs matplotlib: "
                         "pip install -e \".[viz]\"") from None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Draw predictions against ground truth on rendered frames.")
    ap.add_argument("checkpoint")
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="preview/dense")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0,
                    help="added to the config's experiment seed; 0 draws from "
                         "the run's validation stream")
    ap.add_argument("--no-keypoints", action="store_true",
                    help="skip the 40 board corners")
    args = ap.parse_args()

    plt = _pyplot()
    try:
        loaded = load_checkpoint(args.checkpoint, args.config)
    except ExportError as exc:
        raise SystemExit(f"!! {exc}") from None
    lit, cfg, raw = loaded.lit, loaded.cfg, loaded.raw
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lit = lit.to(dev).eval()
    print(f"running on {dev}")

    data_cfg = build_data_config(raw)
    # Augmentation off: this is about what the head predicts, and noise and
    # erasing on top would make a miss ambiguous between the two.
    # The validation split under the experiment seed: --seed 0 draws from the
    # run's own validation stream (its first worker's), another value from an
    # unseen one.
    seed = (raw.get("experiment") or {}).get("seed", 42) + args.seed
    ds = DenseDetectDataset(data_cfg, out_stride=cfg.out_stride,
                            epoch_length=args.n, augment=False, prefetch=2,
                            seed=seed, split="val")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = float(data_cfg.image_size)

    n_gt = n_found = 0
    errs = []
    kp_errs = []
    for i, sample in zip(range(args.n), iter(ds)):
        # uint8, 0..255. The drawing needs [0, 1]: divide, since clipping
        # would take every pixel above 1 to pure white.
        img = sample["image"].float()
        shown = (img / 255.0).clamp(0.0, 1.0)
        with torch.no_grad():
            # The same normalisation the model trained on, without the noise.
            x = lit.gpu_augment(sample["image"][None].to(dev), training=False)
            out = lit.model(x)
            h, w = out["fg_logits"].shape[-2:]
            dec = decode_boxes(out, (h, w))
            got = detect({k: v[0] for k, v in dec.items()},
                         out["fg_logits"][0].sigmoid(), cfg)
            fg_prob = out["fg_logits"][0].sigmoid()

        # Every detection is built from the foreground field, so report it: a
        # plausible detection count can sit on top of one firing nearly
        # everywhere or almost nowhere.
        n_fg = int((fg_prob > cfg.fg_threshold).sum())
        fg_stat = (float(fg_prob.max()), float(fg_prob.mean()))

        fig, ax = plt.subplots(figsize=(9, 9), dpi=120)
        ax.imshow(shown.permute(1, 2, 0).numpy())
        ax.set_xlim(0, scale)
        ax.set_ylim(scale, 0)
        ax.axis("off")

        # Corners first, so the dart annotations draw over them rather than
        # under: the darts are the subject and 40 faint marks should not sit
        # on top of a tip.
        frame_kp = np.array([])
        if not args.no_keypoints and "kp_logits" in out \
                and "keypoints" in sample:
            frame_kp = draw_keypoints(
                ax, predicted_keypoints(out, scale),
                sample["keypoints"].numpy() * scale,
                sample["keypoint_mask"].numpy())
            kp_errs.append(frame_kp)

        gt = sample["dart_box"][sample["dart_mask"]]
        # The renderer's own two points, not the box's ends.
        gt_ends = (sample["dart_ends"][sample["dart_mask"]]
                   if "dart_ends" in sample else None)
        n_gt += len(gt)
        for j, g in enumerate(gt):
            e = gt_ends[j] if gt_ends is not None else None
            draw_box(ax, *[float(v) for v in g], TRUE, "", scale,
                     tip=(float(e[0]), float(e[1])) if e is not None else None,
                     flight=(float(e[2]), float(e[3])) if e is not None
                     else None)
        for p in got:
            c, dv = p["centre"], p["direction"]
            # detect() returns the PREDICTED landing point and flight tip when
            # the head has them, and the box's ends only as a fallback.
            direct = bool(p.get("tip_is_predicted"))
            draw_box(ax, float(c[0]), float(c[1]), float(dv[0]), float(dv[1]),
                     float(p["half_length"]), float(p["half_width"]), PRED,
                     f"{p['votes']}v", scale,
                     tip=(float(p["tip"][0]), float(p["tip"][1]))
                     if direct else None,
                     flight=(float(p["flight"][0]), float(p["flight"][1]))
                     if direct else None)
        n_found += len(got)

        # Nearest-tip error per ground-truth dart, printed on the figure so a
        # visually plausible box that is quietly 20px out cannot pass unnoticed.
        frame_errs = []
        for j, g in enumerate(gt):
            if gt_ends is not None:
                # The landing point the renderer reports, which is what
                # scoring is computed from.
                gx, gy = float(gt_ends[j][0]), float(gt_ends[j][1])
            else:
                gx = float(g[0]) - float(g[2]) * float(g[4])
                gy = float(g[1]) - float(g[3]) * float(g[4])
            best = None
            for p in got:
                t = p["tip"]
                dpx = float(np.hypot(float(t[0]) - gx,
                                     float(t[1]) - gy)) * scale
                best = dpx if best is None else min(best, dpx)
            if best is not None:
                frame_errs.append(best)
        errs.extend(frame_errs)

        # Guarded on this frame's own errors, not on the ground-truth count: a
        # frame with darts but no detections has nothing to average.
        tail = (f"  tip err {np.mean(frame_errs):.1f}px" if frame_errs else
                "  (no detections)")
        ax.set_title(
            f"frame {i}   gt {len(gt)}   detected {len(got)}{tail}\n"
            f"foreground: {n_fg} cells over {cfg.fg_threshold}, "
            f"max {fg_stat[0]:.2f}, mean {fg_stat[1]:.3f}\n"
            f"red = predicted, green = truth;  o = landing point, "
            f"[] = flight tip, x = box-derived tip, . = board corner"
            + (f"\ncorners: {frame_kp.mean():.2f}px mean, "
               f"{int((frame_kp > KP_CALLOUT_PX).sum())}/{frame_kp.size} "
               f"over {KP_CALLOUT_PX:.0f}px" if frame_kp.size else ""),
            fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"dense_{i:02d}.png", bbox_inches="tight")
        plt.close(fig)
        kp_tail = ""
        if frame_kp.size:
            kp_tail = (f", kp {frame_kp.mean():.2f}px "
                       f"(max {frame_kp.max():.1f}, "
                       f"{int((frame_kp > KP_CALLOUT_PX).sum())} over "
                       f"{KP_CALLOUT_PX:.0f}px)")
        print(f"  frame {i}: gt {len(gt)}, detected {len(got)}, "
              f"fg cells {n_fg}, fg max {fg_stat[0]:.2f}, "
              f"mean {fg_stat[1]:.3f}{kp_tail}", flush=True)

    print(f"\n{args.n} frames -> {out_dir}")
    print(f"ground-truth darts {n_gt}, detections {n_found}")
    print("tip error is against the renderer's LANDING POINT, not the box end")
    if kp_errs:
        k = np.concatenate([e for e in kp_errs if e.size])
        over = int((k > KP_CALLOUT_PX).sum())
        print(f"board corners: {k.size} predictions, mean {k.mean():.3f}px  "
              f"p90 {np.quantile(k, 0.9):.3f}px  max {k.max():.2f}px")
        # The number that decides whether RANSAC has an easy job downstream.
        print(f"  over {KP_CALLOUT_PX:.0f}px: {over} "
              f"({100.0 * over / k.size:.2f}%)  -- "
              f"within 3px: {100.0 * (k <= 3.0).mean():.2f}%")
    if errs:
        e = np.array(errs)
        print(f"nearest-tip error: mean {e.mean():.2f}px  "
              f"p90 {np.quantile(e, 0.9):.2f}px  max {e.max():.2f}px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
