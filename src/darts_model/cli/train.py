"""Train the detector (stage 2 of 2).

    darts-train --config configs/detector.yaml

Needs the backbone from stage 1 (``darts-pretrain``, then
``darts-promote-backbone``) at the path the config's ``backbone_weights`` names.

Unknown or misplaced config keys are an error: a misspelt key would otherwise fall back to
its default, and a long run would answer a question nobody asked.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from darts_model.cli.checkpoint import SaveLatest
from darts_model.data.config import build_data_config
from darts_model.data.detect_dataset import DenseDetectDataset
from darts_model.data.gpu_augment import gpu_augment_from_config
from darts_model.model.detector import DenseDartConfig, DenseDartLitModule

MODEL_SECTIONS = ("backbone", "head", "loss", "readout", "training")
#: Keys of the ``training`` section read here for the Trainer and loaders
#: rather than by DenseDartConfig. Valid in ``training`` only.
TRAINER_KEYS = {
    "batch_size", "accumulate_grad_batches", "precision", "num_workers",
    "accelerator", "devices",
}
#: Keys of the ``experiment`` section.
EXPERIMENT_KEYS = {
    "name", "output_dir", "seed", "log_every_n_steps", "checkpoint_monitor",
}


def build_config(raw: dict) -> DenseDartConfig:
    valid = {f.name for f in dataclasses.fields(DenseDartConfig)}
    flat: dict = {}
    problems = []
    for section in MODEL_SECTIONS:
        body = raw.get(section) or {}
        allowed = valid | (TRAINER_KEYS if section == "training" else set())
        for key in sorted(set(body) - allowed):
            where = ("training" if key in TRAINER_KEYS
                     else "experiment" if key in EXPERIMENT_KEYS else None)
            problems.append(f"{section}.{key}" + (
                f" (belongs in {where}:)" if where else ""))
        for key in sorted(set(body) & set(flat)):
            problems.append(f"{section}.{key} (set in two sections)")
        flat.update(body)
    unknown_exp = sorted(set(raw.get("experiment") or {}) - EXPERIMENT_KEYS)
    problems += [f"experiment.{k}" for k in unknown_exp]
    if problems:
        raise SystemExit(
            "unknown or misplaced config keys: " + ", ".join(problems)
            + "\nThese would have been silently ignored. Fix the spelling, or "
              "move them to the section named.")
    flat = {k: v for k, v in flat.items() if k in valid}
    for key in ("backbone_widths", "backbone_depths"):
        if isinstance(flat.get(key), list):
            flat[key] = tuple(flat[key])
    return DenseDartConfig(**flat)


def _single_device(devices) -> bool:
    if isinstance(devices, (list, tuple)):
        return len(devices) == 1
    return str(devices).strip() == "1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text())
    cfg = build_config(raw)
    data_cfg = build_data_config(raw)
    if cfg.backbone_weights and not Path(cfg.backbone_weights).is_file():
        raise SystemExit(
            f"backbone_weights: {cfg.backbone_weights} not found. Run "
            "darts-pretrain, then darts-promote-backbone.")
    train_raw = raw.get("training") or {}
    exp = raw.get("experiment") or {}
    name = exp.get("name", "detector")
    out_dir = exp.get("output_dir", "outputs")
    seed = exp.get("seed", 42)
    L.seed_everything(seed, workers=True)

    # Ranked on the landing error with misses charged the match gate: the
    # landing point is what scoring reads, and charging a miss the gate means
    # an epoch that stops detecting hard darts cannot rank better for it.
    # Noisy epoch to epoch, so the best three are kept.
    monitor = exp.get("checkpoint_monitor", DenseDartLitModule.MONITOR)
    if monitor not in DenseDartLitModule.RANKABLE_METRICS:
        raise SystemExit(
            f"checkpoint_monitor {monitor!r} is not logged every validation "
            "epoch, so checkpoints could be ranked on a stale value. Use one "
            "of: " + ", ".join(DenseDartLitModule.RANKABLE_METRICS))
    devices = train_raw.get("devices", 1)
    if not _single_device(devices):
        raise NotImplementedError(
            f"devices: {devices!r} -- training is single-device. The "
            "non-finite batch skip would deadlock DDP, and the validation "
            "metrics are not reduced across ranks.")
    print(f"checkpoints ranked on {monitor} (top 3)", flush=True)

    # The dataset is iterable and has no length, so the schedule cannot be
    # derived from the dataloader -- it is computed here and passed in.
    _bs = train_raw.get("batch_size", 8)
    _accum = train_raw.get("accumulate_grad_batches", 1)
    cfg.steps_per_epoch = max(
        math.ceil(data_cfg.epoch_length / _bs / _accum), 1)
    module = DenseDartLitModule(
        cfg, gpu_augment=gpu_augment_from_config(data_cfg.augmentation))
    print(f"DenseDart: {cfg.steps_per_epoch} optimizer steps/epoch "
          f"({data_cfg.epoch_length} frames / {_bs} / {_accum})",
          flush=True)
    print(f"DenseDart: stride {cfg.out_stride}, head {cfg.head_width}x"
          f"{cfg.head_depth}, "
          f"{sum(p.numel() for p in module.model.parameters()) / 1e6:.2f} M params",
          flush=True)

    common = dict(out_stride=cfg.out_stride, seed=seed)
    train_ds = DenseDetectDataset(
        data_cfg, epoch_length=data_cfg.epoch_length, augment=True, **common)
    val_ds = DenseDetectDataset(
        data_cfg, epoch_length=data_cfg.val_epoch_length, augment=False,
        **common)
    nw = train_raw.get("num_workers", 4)
    loader = dict(batch_size=train_raw.get("batch_size", 8), num_workers=nw,
                  persistent_workers=nw > 0, pin_memory=True)

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        gradient_clip_val=cfg.gradient_clip_val,
        precision=train_raw.get("precision", "bf16-mixed"),
        accumulate_grad_batches=train_raw.get("accumulate_grad_batches", 1),
        accelerator=train_raw.get("accelerator", "auto"),
        devices=devices,
        logger=TensorBoardLogger(out_dir, name=name),
        callbacks=[
            ModelCheckpoint(
                dirpath=f"{out_dir}/{name}/checkpoints",
                filename="epoch{epoch:03d}-{" + monitor + ":.3f}",
                monitor=monitor, mode="min", save_top_k=3,
                auto_insert_metric_name=False),
            SaveLatest(f"{out_dir}/{name}/checkpoints"),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        log_every_n_steps=exp.get("log_every_n_steps", 50),
    )
    trainer.fit(module,
                DataLoader(train_ds, **loader),
                DataLoader(val_ds, **loader),
                ckpt_path=args.resume)


if __name__ == "__main__":
    main()
