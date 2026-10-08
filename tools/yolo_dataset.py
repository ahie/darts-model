"""Write rendered frames as an Ultralytics YOLO pose dataset, for a baseline.

Frames come from the renderer configured exactly as detector training
configures it (``DenseDetectDataset``: dart counts, grouping, backgrounds), so
a YOLO model trained on them sees the same distribution as the detector.

One class, ``dart``. Each label is the axis-aligned bounds of the dart's
oriented box together with its landing and flight points, and two keypoints:
the landing point first -- the one scoring uses -- then the flight end, whose
visibility flag is 1 when the renderer reports the flight hidden behind the
dart. No photometric augmentation is applied here; YOLO applies its own.

    python tools/yolo_dataset.py --config configs/detector.yaml \\
        --out yolo_darts --train 20000 --val 1000

With ``--heldout N --seed S`` it instead renders the frames
``tools/dump_readout_eval.py`` renders for the same seed and count, and writes
them with their ground truth (``heldout.json``) for scoring a YOLO model on
the same frames as the other readouts.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from darts_model.data.config import build_data_config
from darts_model.data.detect_dataset import DenseDetectDataset


def keypoint(x: float, y: float, vis: int, size: int) -> list:
    """A normalised keypoint, or ``0 0 0`` (unlabelled) when it lies outside
    the frame: Ultralytics rejects the whole image for an out-of-range
    coordinate."""
    if not (0.0 <= x <= size and 0.0 <= y <= size):
        return [0.0, 0.0, 0]
    return [x / size, y / size, vis]


def label_lines(annotation: dict, size: int) -> list[str]:
    lines = []
    for d in annotation["darts"]:
        xs = list(d["box_x"]) + [d["x"], d["flight_x"]]
        ys = list(d["box_y"]) + [d["y"], d["flight_y"]]
        x0, x1 = max(min(xs), 0.0), min(max(xs), float(size))
        y0, y1 = max(min(ys), 0.0), min(max(ys), float(size))
        if x1 <= x0 or y1 <= y0:
            continue
        cx, cy = (x0 + x1) / 2 / size, (y0 + y1) / 2 / size
        w, h = (x1 - x0) / size, (y1 - y0) / size
        flight_vis = 2 if d.get("flight_in_front", True) else 1
        kp = keypoint(d["x"], d["y"], 2, size) + keypoint(
            d["flight_x"], d["flight_y"], flight_vis, size)
        lines.append("0 " + " ".join(f"{v:.6f}" for v in (cx, cy, w, h))
                     + " " + " ".join(f"{v:.6f}" if isinstance(v, float)
                                      else str(v) for v in kp))
    return lines


def render(data, split: str, seed: int, n: int):
    ds = DenseDetectDataset(data, epoch_length=n, augment=False, seed=seed,
                            split=split)
    renderer = ds._ensure_renderer(0, 0)
    for _ in range(n):
        yield ds, renderer.render_frame()


def write_split(data, root: Path, split: str, seed: int, n: int) -> None:
    images = root / "images" / split
    labels = root / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    size = data.image_size
    t0 = time.time()
    for i, (_, (image, ann)) in enumerate(render(data, split, seed, n)):
        Image.fromarray(np.ascontiguousarray(image[..., :3])).save(
            images / f"{i:06d}.jpg", quality=95)
        (labels / f"{i:06d}.txt").write_text("\n".join(label_lines(ann, size)))
        if (i + 1) % 1000 == 0:
            print(f"{split}: {i + 1}/{n}, {time.time() - t0:.0f}s", flush=True)


def write_heldout(data, root: Path, seed: int, n: int) -> None:
    """The frames dump_readout_eval.py renders for this seed, with truth in
    the sample's slot order (darts without a usable box are skipped there)."""
    images = root / "heldout"
    images.mkdir(parents=True, exist_ok=True)
    size = float(data.image_size)
    truth = []
    for i, (ds, (image, ann)) in enumerate(render(data, "val", seed, n)):
        sample = ds._sample(image, ann)
        mask = sample["dart_mask"]
        ends = sample["dart_ends"][mask].numpy() * size
        Image.fromarray(np.ascontiguousarray(image[..., :3])).save(
            images / f"{i:06d}.jpg", quality=95)
        truth.append({"image": f"heldout/{i:06d}.jpg",
                      "land": ends[:, 0:2].tolist(),
                      "flight": ends[:, 2:4].tolist()})
    (root / "heldout.json").write_text(json.dumps(truth))
    print(f"heldout: {n} frames")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--heldout", type=int, default=0)
    args = ap.parse_args()
    data = build_data_config(yaml.safe_load(open(args.config)))
    root = Path(args.out)
    if args.heldout:
        write_heldout(data, root, args.seed, args.heldout)
        return
    write_split(data, root, "train", args.seed, args.train)
    write_split(data, root, "val", args.seed, args.val)
    (root / "darts.yaml").write_text(yaml.safe_dump({
        "path": str(root.resolve()), "train": "images/train",
        "val": "images/val", "kpt_shape": [2, 3], "flip_idx": [0, 1],
        "names": {0: "dart"}}))
    print(f"wrote {root / 'darts.yaml'}")


if __name__ == "__main__":
    main()
