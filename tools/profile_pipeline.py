"""Measure rendering and training throughput separately.

Training renders its data on the fly, normally on the training GPU. This
times the two halves apart, in frames per second, so it can be seen how much
of the GPU rendering takes and how many render GPUs keep one trainer fed
(``data.render_gpus``):

  render   N worker processes, each owning a renderer and building the
           detector's targets per frame, exactly as the dataloader workers do.
  train    the detector's training step (forward, backward, optimizer) on
           synthetic batches at the config's batch size and precision, with
           the backbone unfrozen. No rendering at all.

Run on the machine to be measured, from the repository root, with the
renderer module importable and nothing else using the GPUs:

    python tools/profile_pipeline.py --config configs/detector.yaml \\
        --render-workers 4 --render-gpu 0 --train-gpu 0
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np
import torch
import yaml


def _render_worker(args) -> tuple[int, float, float]:
    """Render ``frames`` frames and build their targets; (frames, seconds
    rendering, seconds building targets)."""
    (worker_id, frames, warmup, size, stride, bg, uuid, weights) = args
    from darts_model.data.detect_dataset import DenseDetectDataset
    from darts_model.data.targets import build_dense_targets
    from darts_model.renderer import import_renderer

    r = import_renderer().Renderer(width=size, height=size, seed=1000 + worker_id,
                                   bg_image_dir=bg, gpu_uuid=uuid,
                                   dart_count_weights=weights)
    t_render = t_targets = 0.0
    for i in range(warmup + frames):
        t0 = time.perf_counter()
        _, ann = r.render_frame()
        t1 = time.perf_counter()
        build_dense_targets(ann, size, out_stride=stride)
        DenseDetectDataset.boxes_from(ann, size)
        t2 = time.perf_counter()
        if i >= warmup:
            t_render += t1 - t0
            t_targets += t2 - t1
    return frames, t_render, t_targets


def profile_render(cfg_raw, workers: int, frames: int, gpu: int) -> None:
    from darts_model.cli.train import build_config
    from darts_model.data.config import build_data_config
    from darts_model.renderer import cuda_device_uuids

    cfg, data = build_config(cfg_raw), build_data_config(cfg_raw)
    uuids = cuda_device_uuids()
    uuid = uuids[gpu] if uuids else ""
    jobs = [(w, frames, 10, data.image_size, cfg.out_stride, data.bg_image_dir,
             uuid, list(data.dart_count_weights)) for w in range(workers)]
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(workers) as pool:
        results = pool.map(_render_worker, jobs)
    wall = time.perf_counter() - t0
    n = sum(r[0] for r in results)
    render_s = sum(r[1] for r in results) / n * 1e3
    target_s = sum(r[2] for r in results) / n * 1e3
    # Wall time includes renderer start-up; the per-frame figures do not.
    per_worker = [r[0] / (r[1] + r[2]) for r in results]
    print(f"render: {workers} workers on CUDA {gpu}: {sum(per_worker):.1f} frames/s "
          f"steady state ({render_s:.1f} ms render + {target_s:.1f} ms targets per "
          f"frame per worker); {n / wall:.1f} frames/s including start-up")


def _fake_batch(cfg, size: int, batch: int, device) -> dict:
    h = size // cfg.out_stride
    inst = torch.zeros(batch, h, h, dtype=torch.long)
    box = torch.zeros(batch, 3, 6)
    ends = torch.zeros(batch, 3, 4)
    for b in range(batch):
        for d in range(3):
            y, x = np.random.randint(h // 4, 3 * h // 4, 2)
            inst[b, y:y + h // 20, x:x + 2] = d + 1
            box[b, d] = torch.tensor([(x + 1) / h, (y + h / 40) / h, 0.0, 1.0, 0.05, 0.01])
            ends[b, d] = torch.tensor([x / h, y / h, x / h, (y + h / 20) / h])
    out = {
        "image": torch.randint(0, 256, (batch, 3, size, size), dtype=torch.uint8),
        "instance": inst, "foreground": (inst > 0).float(),
        "dart_box": box, "dart_ends": ends,
        "dart_mask": torch.ones(batch, 3, dtype=torch.bool),
        "dart_box_mask": torch.ones(batch, 3, dtype=torch.bool),
        "keypoints": torch.rand(batch, 40, 2),
        "keypoint_mask": torch.ones(batch, 40, dtype=torch.bool),
    }
    return {k: v.to(device) for k, v in out.items()}


def profile_train(cfg_raw, steps: int, gpu: int) -> None:
    from darts_model.cli.train import build_config
    from darts_model.data.config import build_data_config
    from darts_model.data.gpu_augment import gpu_augment_from_config
    from darts_model.model.detector import DenseDartLitModule

    cfg, data = build_config(cfg_raw), build_data_config(cfg_raw)
    cfg.backbone_weights = ""
    cfg.init_weights = ""
    t = cfg_raw.get("training") or {}
    batch = int(t.get("batch_size", 8))
    bf16 = "bf16" in str(t.get("precision", "bf16-mixed"))
    device = torch.device(f"cuda:{gpu}")
    lit = DenseDartLitModule(cfg, gpu_augment=gpu_augment_from_config(data.augmentation)).to(device)
    lit.model.backbone.unfreeze()
    lit.train()
    opt = torch.optim.AdamW(lit.parameters(), lr=1e-4)
    fake = _fake_batch(cfg, data.image_size, batch, device)

    def step():
        b = lit.on_after_batch_transfer({k: v.clone() for k, v in fake.items()}, 0)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            loss = lit._step(b, "train")
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(5):
        step()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    mem = torch.cuda.max_memory_allocated(device) / 2 ** 30
    print(f"train: CUDA {gpu}, batch {batch}, stride {cfg.out_stride}: "
          f"{steps * batch / dt:.1f} frames/s ({dt / steps * 1e3:.0f} ms/step), "
          f"peak {mem:.1f} GiB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/detector.yaml")
    ap.add_argument("--only", choices=("render", "train"), default=None)
    ap.add_argument("--render-workers", type=int, default=4)
    ap.add_argument("--render-frames", type=int, default=100,
                    help="timed frames per render worker")
    ap.add_argument("--render-gpu", type=int, default=0, help="CUDA index")
    ap.add_argument("--train-steps", type=int, default=30)
    ap.add_argument("--train-gpu", type=int, default=0, help="CUDA index")
    args = ap.parse_args()
    raw = yaml.safe_load(open(args.config))
    if args.only in (None, "render"):
        profile_render(raw, args.render_workers, args.render_frames, args.render_gpu)
    if args.only in (None, "train"):
        profile_train(raw, args.train_steps, args.train_gpu)


if __name__ == "__main__":
    main()
