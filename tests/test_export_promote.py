"""darts-promote-backbone picks the checkpoint the callback ranked best."""
from __future__ import annotations

import pytest
import torch

from darts_model.cli.promote_backbone import main


def _run_dir(tmp_path, best_name="epochepoch=147.ckpt", best_sd=None,
             record_best=True):
    ckpts = tmp_path / "run" / "checkpoints"
    ckpts.mkdir(parents=True)
    callbacks = {}
    if record_best:
        callbacks["ModelCheckpoint{'monitor': 'val/loss', 'mode': 'min'}"] = {
            # Recorded on the training machine, under another absolute path.
            "best_model_path": f"/elsewhere/outputs/pretrain/checkpoints/{best_name}",
            "best_model_score": torch.tensor(0.1234),
        }
    torch.save({"state_dict": {"backbone.stem.weight": torch.zeros(1),
                               "head.weight": torch.ones(1)},
                "callbacks": callbacks}, ckpts / "last.ckpt")
    torch.save({"state_dict": best_sd if best_sd is not None else
                {"backbone.stem.weight": torch.full((1,), 7.0)}},
               ckpts / best_name)
    return tmp_path / "run"


def test_the_best_checkpoint_is_copied(tmp_path, capsys) -> None:
    run = _run_dir(tmp_path)
    out = tmp_path / "backbone.ckpt"
    main([str(run), str(out)])
    got = torch.load(out, weights_only=False)["state_dict"]
    assert float(got["backbone.stem.weight"]) == 7.0, "copied last, not best"
    assert "0.1234" in capsys.readouterr().out


def test_a_best_without_backbone_is_refused(tmp_path) -> None:
    run = _run_dir(tmp_path, best_sd={"head.weight": torch.zeros(1)})
    out = tmp_path / "backbone.ckpt"
    with pytest.raises(SystemExit, match="no backbone"):
        main([str(run), str(out)])
    assert not out.exists()


def test_no_best_needs_allow_last(tmp_path) -> None:
    run = _run_dir(tmp_path, record_best=False)
    out = tmp_path / "backbone.ckpt"
    with pytest.raises(SystemExit, match="allow-last"):
        main([str(run), str(out)])
    main([str(run), str(out), "--allow-last"])
    got = torch.load(out, weights_only=False)["state_dict"]
    assert float(got["backbone.stem.weight"]) == 0.0


def test_a_missing_run_is_an_error(tmp_path) -> None:
    with pytest.raises(SystemExit, match="last.ckpt"):
        main([str(tmp_path / "nope"), str(tmp_path / "x.ckpt")])
