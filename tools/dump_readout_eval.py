"""Render a held-out set and dump what a dart readout needs, for evaluating
readouts offline.

Renders ``--frames`` frames with a seed training never used, runs a detector
checkpoint on them, and saves per frame: the ground-truth landing and flight
points, whether each dart's tip is visible, the Hough readout's detections
(landing estimate and axis) and the tip heatmap's peaks. Any assignment of
peaks to detections can then be fitted and compared without a GPU or a
renderer. With a stage-3 token readout, also each query's confidence and its
landing point three ways: the soft blend it was trained with, the hard
reading of its single highest-weighted option, and the blend over tokens
alone, without the estimate's share.

The renderer is deterministic in its seed, so the same ``--seed`` and
``--frames`` reproduce the same frames for any checkpoint.

A tip counts as visible when a pixel of that dart's tip class lies within
``--visible-px`` of its landing point in the full-resolution segmentation.

Run from the repository root on a machine with the renderer:

    python tools/dump_readout_eval.py outputs/detector_tips4/checkpoints/last.ckpt \\
        --config configs/detector_tips4.yaml --frames 2000 --out readout_eval.pt
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import yaml

from darts_model.data.config import build_data_config
from darts_model.data.detect_dataset import DenseDetectDataset
from darts_model.data.targets import TIP
from darts_model.export.common import load_checkpoint
from darts_model.model.detector import decode_boxes
from darts_model.model.hough import detect
from darts_model.model.tips import tip_peaks
from darts_model.model.token_readout import gather_tokens


def tip_visible(annotation: dict, i: int, radius: float) -> bool:
    seg = annotation["seg_ids"]
    h, w = seg.shape[:2]
    x, y = annotation["darts"][i]["x"], annotation["darts"][i]["y"]
    r = int(np.ceil(radius))
    x0, x1 = max(int(x) - r, 0), min(int(x) + r + 1, w)
    y0, y1 = max(int(y) - r, 0), min(int(y) + r + 1, h)
    if x0 >= x1 or y0 >= y1:
        return False
    cls = seg[y0:y1, x0:x1, 0]
    inst = seg[y0:y1, x0:x1, 1]
    ys, xs = np.mgrid[y0:y1, x0:x1]
    near = (xs + 0.5 - x) ** 2 + (ys + 0.5 - y) ** 2 <= radius ** 2
    return bool(((cls == TIP) & (inst == i + 1) & near).any())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--config", required=True)
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=9001,
                    help="renderer seed; anything but the training seed")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--visible-px", type=float, default=3.0)
    ap.add_argument("--out", default="readout_eval.pt")
    args = ap.parse_args()

    raw = yaml.safe_load(open(args.config))
    data = build_data_config(raw)
    loaded = load_checkpoint(args.checkpoint, args.config)
    lit, cfg = loaded.lit.eval().cuda(), loaded.cfg
    ds = DenseDetectDataset(data, out_stride=cfg.out_stride,
                            epoch_length=args.frames, augment=False,
                            seed=args.seed, split="val")
    renderer = ds._ensure_renderer(0, 0)
    size = float(data.image_size)

    records, pending = [], []
    t0 = time.time()

    def flush():
        batch = {k: torch.stack([s[k] for s, _ in pending]).cuda()
                 for k in pending[0][0]}
        batch = lit.on_after_batch_transfer(batch, 0)
        with torch.no_grad():
            out = lit.model(batch["image"].float())
        h, w = out["fg_logits"].shape[-2:]
        dec = decode_boxes(out, (h, w))
        fg = out["fg_logits"].float().sigmoid()
        if "tip_logits" in out:
            xy, score = tip_peaks(out["tip_logits"], out["tip_cell_offset"],
                                  cfg.tip_peak_count)
        else:
            # A model without a tip head: no peaks, Hough detections only.
            xy = torch.zeros(len(pending), 0, 2)
            score = torch.zeros(len(pending), 0)
        slots = None
        if getattr(lit.model, "token_readout", None) is not None:
            with torch.no_grad():
                tok = gather_tokens(out, cfg.readout_tokens)
                r = lit.model.token_readout(tok)
            w = r["tip_logits"].softmax(-1)                         # (B,Q,1+M)
            values = torch.cat((r["tip_estimate"].unsqueeze(2),
                                tok["tip_point"].unsqueeze(1).expand(
                                    -1, w.shape[1], -1, -1)), 2)    # (B,Q,1+M,2)
            pick = w.argmax(-1)
            hard = torch.gather(values, 2, pick[..., None, None].expand(
                -1, -1, 1, 2))[:, :, 0]
            wt = w[..., 1:] / w[..., 1:].sum(-1, keepdim=True).clamp(min=1e-9)
            tokens_only = (wt.unsqueeze(-1) * values[:, :, 1:]).sum(2)
            slots = {"score": r["score"], "soft": r["tip"], "hard": hard,
                     "tokens_only": tokens_only, "estimate": r["tip_estimate"],
                     "picked_estimate": pick == 0, "top_weight": w.amax(-1)}
        for b, (s, vis) in enumerate(pending):
            dets = detect({k: v[b] for k, v in dec.items()}, fg[b], cfg)
            mask = s["dart_mask"]
            ends = s["dart_ends"][mask].numpy()
            records.append({
                "land": ends[:, 0:2] * size, "flight": ends[:, 2:4] * size,
                "visible": np.array(vis, bool),
                "det_tip": np.array([d["tip"].cpu().numpy() for d in dets]).reshape(-1, 2) * size,
                "det_dir": np.array([d["direction"].cpu().numpy() for d in dets]).reshape(-1, 2),
                "det_votes": np.array([d["votes"] for d in dets]),
                "peak_xy": xy[b].cpu().numpy() * size,
                "peak_score": score[b].cpu().numpy(),
            })
            if slots is not None:
                records[-1]["slots"] = {
                    k: (v[b].cpu().numpy() * size
                        if k in ("soft", "hard", "tokens_only", "estimate")
                        else v[b].cpu().numpy())
                    for k, v in slots.items()}
        pending.clear()

    for i in range(args.frames):
        image, ann = renderer.render_frame()
        sample = ds._sample(image, ann)
        # In the sample's slot order, which skips darts without a usable box.
        vis = [tip_visible(ann, j, args.visible_px)
               for j in range(len(sample["dart_mask"]))
               if bool(sample["dart_mask"][j])]
        pending.append((sample, vis))
        if len(pending) == args.batch:
            flush()
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{args.frames} frames, {time.time() - t0:.0f}s",
                  flush=True)
    if pending:
        flush()
    torch.save({"size": size, "frames": records,
                "checkpoint": args.checkpoint, "config": args.config,
                "tip_stride": cfg.tip_stride, "seed": args.seed}, args.out)
    n = sum(len(r["land"]) for r in records)
    hidden = sum(int((~r["visible"]).sum()) for r in records)
    print(f"wrote {args.out}: {len(records)} frames, {n} darts, {hidden} with "
          f"the tip hidden")


if __name__ == "__main__":
    main()
