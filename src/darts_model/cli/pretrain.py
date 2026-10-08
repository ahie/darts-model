"""Pretrain a backbone on dense targets from the renderer.

The deliverable is the backbone. The heads are scaffolding whose job is to make
invariance unlearnable -- they demand per-pixel answers a smooth representation
cannot give -- and they are discarded afterwards.

The success criterion is transfer, not anything logged here. Pretraining IoU and
offset error only say the heads are learning at all; whether one backbone is
better than another is settled by fine-tuning the detector from each and
comparing tip error on real images.

Usage::

    darts-pretrain --config configs/pretrain.yaml
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from darts_model.data.config import (AugmentationConfig, check_removed_data_keys,
                                     build_augmentation_config)
from darts_model.data.gpu_augment import gpu_augment_from_config
from darts_model.data.pretrain_dataset import DensePretrainDataset
from darts_model.cli.checkpoint import SaveLatest
from darts_model.data.targets import FLIGHT, METAL, TIP, WIRE
from darts_model.model.pretrain_head import (DensePretrainConfig, DensePretrainHead,
                                             DensePretrainLoss)
from darts_model.model.backbone import FineBackbone, FineBackboneConfig


class DensePretrainModule(L.LightningModule):
    def __init__(self, backbone_cfg: dict, head_cfg: dict, optim_cfg: dict,
                 aug_cfg: dict | None = None) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Sensor noise and normalisation on device, built exactly as the
        # detector builds them, so the two stages see identical input
        # distributions. aug_cfg is a plain dict so it round-trips through the
        # checkpoint's hyperparameters.
        self.gpu_augment = gpu_augment_from_config(
            build_augmentation_config(aug_cfg) if aug_cfg is not None else None)
        self.backbone = FineBackbone(FineBackboneConfig(**backbone_cfg))
        hcfg = DensePretrainConfig(**head_cfg)
        self.head = DensePretrainHead(self.backbone.stage_channels, hcfg)
        self.loss_fn = DensePretrainLoss(hcfg)
        self.optim_cfg = optim_cfg
        self.stride = self.backbone.stage_reductions[0]

    def _step(self, batch, stage: str):
        # Targets arrive already reduced to the head's grid; the dataset does it
        # on CPU so only the stride-4 maps cross the bus.
        t = {k: batch[k] for k in ("class", "offset", "uv", "height",
                                   "dart_mask", "board_mask", "drawn_mask")}
        _, stages = self.backbone(batch["image"])
        pred = self.head(stages)
        total, parts = self.loss_fn(pred, t)

        self.log(f"{stage}/loss", total, prog_bar=True, sync_dist=True)
        for k, v in parts.items():
            self.log(f"{stage}/{k}", v, sync_dist=True)
        if stage == "train":
            # How many cells each part actually supervises. The tip term holds
            # 57% of the offset objective on whatever handful of cells a batch
            # happens to contain, so this is the sample size behind that term.
            for name, c in (("tip", TIP), ("barrel", METAL), ("flight", FLIGHT)):
                self.log(f"cells/{name}", (t["class"] == c).sum().float(),
                         sync_dist=True)
        if stage == "val":
            with torch.no_grad():
                p = pred["class_logits"].argmax(1)
                for name, c in (("tip", TIP), ("wire", WIRE)):
                    inter = ((p == c) & (t["class"] == c)).sum()
                    union = ((p == c) | (t["class"] == c)).sum().clamp_min(1)
                    self.log(f"val/iou_{name}", inter / union, sync_dist=True)
        return total

    def on_after_backward(self) -> None:
        """Per-term gradient norms, read off each head separately.

        The heads are disjoint 1x1 convolutions, so a head's parameters receive
        gradient from its own loss term and nothing else -- which makes this an
        exact per-term measurement rather than a proxy, and free.

        L1's gradient is sign(p-t), so per-part averaging gives each part a
        total gradient set by its weight and independent of its cell count --
        tip cells, at ~0.036% of the grid, carry 57% of the offset gradient
        spread over a few dozen positions. grad/offset against grad/class tells
        a drowned offset term from that concentration; cells/tip says how thin
        the sample is on any given batch.
        """
        for name, mod in (("class", self.head.head_class),
                          ("offset", self.head.head_offset),
                          ("height", self.head.head_height),
                          ("uv", self.head.head_uv)):
            g = [p.grad.flatten() for p in mod.parameters() if p.grad is not None]
            if g:
                self.log(f"grad/{name}", torch.cat(g).norm(), sync_dist=True)
        bb = [p.grad.flatten() for p in self.backbone.parameters()
              if p.grad is not None]
        if bb:
            self.log("grad/backbone", torch.cat(bb).norm(), sync_dist=True)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        # Noise only while training; validation gets normalisation alone, so the
        # metric measures the model rather than the augmentation.
        # self.trainer raises when unattached rather than returning None, so
        # this reads the private handle -- the hook has to work outside a
        # Trainer for the pipeline to be testable at all.
        tr = getattr(self, "_trainer", None)
        batch["image"] = self.gpu_augment(batch["image"],
                                          training=bool(tr and tr.training))
        return batch

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        # No weight decay on 1-D parameters (norm weights, biases, layer-scale
        # gamma): decaying them only fights the normalisation, as in ConvNeXt.
        wd = self.optim_cfg.get("weight_decay", 0.05)
        decay = [p for p in self.parameters() if p.ndim > 1]
        no_decay = [p for p in self.parameters() if p.ndim <= 1]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd},
                                 {"params": no_decay, "weight_decay": 0.0}],
                                lr=self.optim_cfg.get("lr", 3e-4))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.optim_cfg.get("max_epochs", 100),
            eta_min=self.optim_cfg.get("min_lr", 1e-6))
        return {"optimizer": opt, "lr_scheduler": sched}

    def on_save_checkpoint(self, checkpoint) -> None:
        # The backbone is the product. Saved separately so transferring it does
        # not require importing the heads or knowing this module exists.
        checkpoint["backbone_state_dict"] = self.backbone.state_dict()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/pretrain.yaml")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    d, t, e = cfg["data"], cfg["training"], cfg["experiment"]
    check_removed_data_keys(d)
    # Validated through the same dataclass as the detector's block, so an
    # unknown or misspelt key fails here rather than silently defaulting.
    aug = build_augmentation_config(cfg.get("augmentation"))
    L.seed_everything(e["seed"], workers=True)

    module = DensePretrainModule(cfg.get("backbone", {}), cfg.get("head", {}),
                                 {**t, "max_epochs": t["max_epochs"]},
                                 dataclasses.asdict(aug))

    def loader(split: str, length: int,
               augmentation: AugmentationConfig | None) -> DataLoader:
        ds = DensePretrainDataset(
            asset_dir=d.get("asset_dir", ""), image_size=d["image_size"],
            epoch_length=length, seed=e["seed"], split=split,
            bg_image_dir=d.get("bg_image_dir", ""),
            render_gpus=tuple(d.get("render_gpus") or ()),
            dart_count_weights=tuple(d.get("dart_count_weights",
                                           (0.05, 0.25, 0.30, 0.40))),
            skill_placement=d.get("skill_placement", True),
            # The head predicts at the backbone's first stage; the targets
            # must be built on the same grid.
            out_stride=module.stride,
            augmentation=augmentation)
        return DataLoader(ds, batch_size=t["batch_size"],
                          num_workers=t["num_workers"],
                          pin_memory=True, persistent_workers=t["num_workers"] > 0)

    train_dl = loader("train", d["epoch_length"], aug)
    # Validation stays clean: augmentation is there to make the backbone
    # robust, not to make the metric noisier.
    val_dl = loader("val", d["val_epoch_length"], None)

    out = Path(e["output_dir"]) / e["name"]
    trainer = L.Trainer(
        max_epochs=t["max_epochs"],
        precision=t.get("precision", "bf16-mixed"),
        accumulate_grad_batches=t.get("accumulate_grad_batches", 1),
        gradient_clip_val=t.get("gradient_clip_val", 1.0),
        # Explicit, because Lightning's default takes every visible GPU and
        # would turn a single-GPU config into DDP on a multi-GPU host.
        accelerator=t.get("accelerator", "auto"),
        devices=t.get("devices", 1),
        log_every_n_steps=e.get("log_every_n_steps", 50),
        logger=TensorBoardLogger(str(out.parent), name=e["name"]),
        callbacks=[ModelCheckpoint(dirpath=str(out / "checkpoints"),
                                   filename="epoch{epoch:03d}",
                                   monitor="val/loss", mode="min",
                                   save_top_k=2, auto_insert_metric_name=False),
                   SaveLatest(out / "checkpoints")],
    )
    trainer.fit(module, train_dl, val_dl, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
