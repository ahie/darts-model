"""Copy a pretraining run's BEST checkpoint to a stable filename.

The pretrain keeps its best two checkpoints as `epochNNN.ckpt`, named by
epoch but not by metric, so the best cannot be identified by name. `last.ckpt`
is the final epoch, which is usually close to the best after a full cosine but
is not the same thing.

The callback state saved in `last.ckpt` records it exactly. Read `best_model_path` out of the
checkpoint rather than guessing, and copy that file to the name the detector
config names (`backbone_weights` in configs/detector.yaml).

Usage:
    darts-promote-backbone \\
        outputs/pretrain outputs/backbone.ckpt
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Copy a pretraining run's best checkpoint to a stable name.")
    ap.add_argument("run_dir", help="outputs/<experiment name>")
    ap.add_argument("out", help="destination filename")
    ap.add_argument("--allow-last", action="store_true",
                    help="fall back to last.ckpt if the callback state has no "
                         "best_model_path (an interrupted run)")
    args = ap.parse_args(argv)

    ckpt_dir = Path(args.run_dir) / "checkpoints"
    last = ckpt_dir / "last.ckpt"
    if not last.exists():
        sys.exit(f"no {last}")

    ck = torch.load(last, map_location="cpu", weights_only=False)
    best_path, best_score = None, None
    for key, state in (ck.get("callbacks") or {}).items():
        if "ModelCheckpoint" not in str(key):
            continue
        if state.get("best_model_path"):
            best_path = state["best_model_path"]
            best_score = state.get("best_model_score")
            break

    if best_path:
        src = Path(best_path)
        # The run happened on another machine, or under a different absolute
        # path; only the basename is reliable.
        if not src.exists():
            src = ckpt_dir / src.name
    elif args.allow_last:
        src, best_score = last, None
        print("no best_model_path in the callback state; using last.ckpt")
    else:
        sys.exit("callback state has no best_model_path (pass --allow-last "
                 "to accept the final epoch instead)")

    if not src.exists():
        sys.exit(f"best checkpoint {src} does not exist")

    score = f"{float(best_score):.4f}" if best_score is not None else "n/a"
    print(f"best: {src.name}  val/loss {score}")

    # Sanity: the detector loads `model.backbone.` out of this, so refuse a
    # checkpoint that has no backbone in it rather than let the detector start
    # from a silently random one.
    sd = ck.get("state_dict") if src == last else torch.load(
        src, map_location="cpu", weights_only=False).get("state_dict", {})
    n = sum(1 for k in sd if k.startswith(("backbone.", "model.backbone.")))
    if n == 0:
        sys.exit(f"{src.name} carries no backbone tensors; refusing to promote")
    print(f"      {n} backbone tensors")

    shutil.copy2(src, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
