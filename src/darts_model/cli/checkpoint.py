"""Checkpointing shared by the training commands."""
from __future__ import annotations

import os
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import Checkpoint


class SaveLatest(Checkpoint):
    """Write ``last.ckpt`` after every validation, whatever it scored.

    Lightning's ``ModelCheckpoint(save_last=True)`` copies ``last.ckpt`` only
    when its top-k set changes, so late in a run -- when improvements are rare
    -- it can lag the real last epoch by many epochs, and resuming from it
    repeats them. This is the resume point; the top-k checkpoints are for
    picking a model.

    The file is written to a temporary name and renamed into place, so an
    interruption mid-write leaves the previous ``last.ckpt`` intact.

    A ``Checkpoint`` subclass so Lightning orders it among the checkpoint
    callbacks, after any ``ModelCheckpoint`` listed before it: the saved
    callback state then already records this epoch's ``best_model_path``.
    """

    def __init__(self, dirpath: str | os.PathLike) -> None:
        super().__init__()
        self.path = Path(dirpath) / "last.ckpt"

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        trainer.save_checkpoint(tmp)
        if trainer.is_global_zero:
            os.replace(tmp, self.path)
