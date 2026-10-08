"""Strip a detector checkpoint to its model weights, for publishing.

Keeps the ``model.`` tensors, the epoch and the global step, and drops the
optimizer, scheduler and callback state, which is about two thirds of a
training checkpoint. The result loads wherever a checkpoint is taken: the
exporters, ``darts-visualize`` and a config's ``init_weights``. It cannot
resume training.

    python tools/strip_checkpoint.py outputs/detector/checkpoints/epoch768-4.055.ckpt \\
        detector.ckpt

With ``--backbone`` it instead keeps only the pretrained backbone of a
``darts-pretrain`` checkpoint (such as ``outputs/backbone.ckpt``), in the
form a detector config's ``backbone_weights`` loads.
"""
from __future__ import annotations

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--backbone", action="store_true")
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.backbone:
        sd = ck.get("backbone_state_dict") or {
            k.split("backbone.", 1)[1]: v for k, v in ck["state_dict"].items()
            if k.startswith("backbone.")}
        if not sd:
            raise SystemExit(f"{args.checkpoint} holds no backbone weights")
        torch.save({"backbone_state_dict": sd, "epoch": ck.get("epoch")},
                   args.out)
        print(f"{args.out}: backbone, {len(sd)} tensors, epoch {ck.get('epoch')}")
        return
    sd = {k: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
    if not sd:
        raise SystemExit(f"{args.checkpoint} has no model. tensors")
    torch.save({"state_dict": sd, "epoch": ck.get("epoch"),
                "global_step": ck.get("global_step")}, args.out)
    n = sum(v.numel() for v in sd.values())
    print(f"{args.out}: {len(sd)} tensors, {n / 1e6:.2f}M values, "
          f"epoch {ck.get('epoch')}")


if __name__ == "__main__":
    main()
