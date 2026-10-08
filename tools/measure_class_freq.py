"""Measure the segmentation class frequencies behind the pretraining loss weights.

Renders frames with the pretraining data settings, reduces their class maps to
the pretraining head's stride exactly as the targets are built, and prints the
per-class share in SegClass order -- the value for ``CLASS_FREQ`` in
``src/darts_model/model/pretrain_head.py``.

Run on a machine with a ray-tracing GPU, from the repository root, with the
renderer module on PYTHONPATH:

    python tools/measure_class_freq.py --config configs/pretrain.yaml --frames 200
"""
from __future__ import annotations

import argparse

import numpy as np
import yaml

from darts_model.data.targets import NUM_CLASSES, build_dense_targets
from darts_model.model.backbone import FineBackbone, FineBackboneConfig
from darts_model.renderer import check_bg_image_dir, import_renderer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/pretrain.yaml")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    d = cfg["data"]
    stride = FineBackbone(FineBackboneConfig(**cfg["backbone"])).stage_reductions[0]
    bg = d.get("bg_image_dir", "")
    check_bg_image_dir(bg)

    kwargs = dict(width=d["image_size"], height=d["image_size"], seed=args.seed,
                  bg_image_dir=bg,
                  skill_placement=d.get("skill_placement", True),
                  dart_count_weights=list(d.get("dart_count_weights", [])))
    if d.get("asset_dir"):
        kwargs["asset_dir"] = d["asset_dir"]
    renderer = import_renderer().Renderer(**kwargs)

    counts = np.zeros(NUM_CLASSES, np.int64)
    for i in range(args.frames):
        _, ann = renderer.render_frame()
        t = build_dense_targets(ann, d["image_size"], out_stride=stride)
        if t is None:
            raise SystemExit("renderer returned no seg_ids/depth")
        counts += np.bincount(t["seg_class"].ravel(), minlength=NUM_CLASSES)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.frames} frames", flush=True)

    freq = counts / counts.sum()
    print(f"stride {stride}, {args.frames} frames")
    print("CLASS_FREQ = (" + ", ".join(f"{f:.5f}" for f in freq) + ")")


if __name__ == "__main__":
    main()
