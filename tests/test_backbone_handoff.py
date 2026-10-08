"""Stage 1 hands its backbone to stage 2 through a checkpoint file.

The detector loads that file strictly, so these pin the two ends to each
other: a pretraining checkpoint must load into the detector completely, and a
file without backbone weights must be refused rather than leave the detector
at random init.
"""
from __future__ import annotations

import pytest
import torch

from darts_model.cli.pretrain import DensePretrainModule
from darts_model.model.detector import DenseDartConfig, DenseDartNet

WIDTHS = (16, 24, 32, 48)
DEPTHS = (1, 1, 1, 1)


def _detector(path: str) -> DenseDartNet:
    return DenseDartNet(DenseDartConfig(
        backbone_widths=WIDTHS, backbone_depths=DEPTHS, backbone_dilate_last=False,
        head_width=16, head_depth=1, backbone_weights=path))


def test_a_pretraining_checkpoint_loads_into_the_detector(tmp_path) -> None:
    pre = DensePretrainModule(
        {"widths": WIDTHS, "depths": DEPTHS, "dilate_last": False}, {}, {})
    ck: dict = {"state_dict": pre.state_dict()}
    pre.on_save_checkpoint(ck)
    path = tmp_path / "backbone.ckpt"
    torch.save(ck, path)

    net = _detector(str(path))
    for k, v in pre.backbone.state_dict().items():
        assert torch.equal(net.backbone.state_dict()[k], v), k


def test_a_checkpoint_without_backbone_weights_is_refused(tmp_path) -> None:
    path = tmp_path / "empty.ckpt"
    torch.save({"state_dict": {"head.weight": torch.zeros(1)}}, path)
    with pytest.raises(RuntimeError, match="no backbone weights"):
        _detector(str(path))
