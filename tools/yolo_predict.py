"""Run a YOLO pose model over the held-out frames ``tools/yolo_dataset.py``
wrote, and save each frame's detections for scoring.

Needs the ``ultralytics`` package, which this repository does not depend on
and which is licensed AGPL-3.0; install it separately, in its own environment. Saves, per frame, every detection
down to ``--conf``: its confidence, then the landing and flight keypoints in
pixels. The confidence threshold that counts a detection as a dart is chosen
when scoring, not here.

    python tools/yolo_predict.py runs/pose/train/weights/best.pt \\
        --data yolo_darts --out yolo_heldout_preds.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("weights")
    ap.add_argument("--data", required=True, help="the dataset root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.05)
    args = ap.parse_args()
    root = Path(args.data)
    truth = json.loads((root / "heldout.json").read_text())
    model = YOLO(args.weights)
    preds = []
    for i, frame in enumerate(truth):
        r = model.predict(str(root / frame["image"]), imgsz=args.imgsz,
                          conf=args.conf, verbose=False)[0]
        conf = r.boxes.conf.cpu().numpy().tolist() if r.boxes is not None else []
        kp = r.keypoints.xy.cpu().numpy().tolist() if r.keypoints is not None else []
        preds.append([[c, k[0][0], k[0][1], k[1][0], k[1][1]]
                      for c, k in zip(conf, kp)])
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(truth)}", flush=True)
    Path(args.out).write_text(json.dumps(preds))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
