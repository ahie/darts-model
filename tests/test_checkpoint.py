"""last.ckpt follows the run, not the best score."""
from __future__ import annotations

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from darts_model.cli.checkpoint import SaveLatest


class _Worsening(L.LightningModule):
    """Validation score gets worse every epoch, so only epoch 0 is ever best."""

    def __init__(self) -> None:
        super().__init__()
        self.l = torch.nn.Linear(1, 1)

    def training_step(self, batch, _):
        return self.l(batch[0]).mean()

    def validation_step(self, batch, _):
        self.log("score", float(self.current_epoch))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), 0.0)


def test_last_ckpt_is_the_final_epoch_and_records_the_best(tmp_path) -> None:
    dl = DataLoader(TensorDataset(torch.zeros(4, 1)), batch_size=2)
    best = ModelCheckpoint(dirpath=tmp_path, filename="epoch{epoch:03d}",
                           monitor="score", mode="min", save_top_k=1,
                           auto_insert_metric_name=False)
    L.Trainer(max_epochs=4, logger=False, enable_progress_bar=False,
              enable_model_summary=False,
              callbacks=[best, SaveLatest(tmp_path)]).fit(_Worsening(), dl, dl)

    ck = torch.load(tmp_path / "last.ckpt", map_location="cpu", weights_only=False)
    assert ck["epoch"] == 3
    state = next(v for k, v in ck["callbacks"].items() if k.startswith("ModelCheckpoint"))
    assert state["best_model_path"].endswith("epoch000.ckpt")
    assert not (tmp_path / "last.ckpt.tmp").exists()
