"""Per-worker renderer settings: seeds, GPU, epoch shares, backgrounds."""
from __future__ import annotations

import itertools
import pickle

import pytest
import torch.distributed as dist

from darts_model.data.config import DataConfig
from darts_model.data.detect_dataset import DenseDetectDataset
from darts_model.data.pretrain_dataset import DensePretrainDataset
from darts_model.renderer import (
    check_bg_image_dir,
    distributed_rank_world,
    frames_for_worker,
    render_gpu_uuid,
    renderer_seed,
    split_evenly,
)


def test_seeds_are_distinct_across_split_rank_and_worker() -> None:
    combos = list(itertools.product([0, 1, 42, 1000], ["train", "val"],
                                    range(8), range(8)))
    seeds = [renderer_seed(*c) for c in combos]
    assert len(set(seeds)) == len(seeds)
    assert all(0 <= s < 2 ** 32 for s in seeds)


def test_seeds_are_reproducible() -> None:
    assert renderer_seed(42, "train", 1, 2) == renderer_seed(42, "train", 1, 2)


def test_bad_split_raises() -> None:
    with pytest.raises(ValueError):
        renderer_seed(0, "test", 0, 0)


@pytest.mark.parametrize("n,parts", [(10, 3), (2, 4), (7, 7), (0, 2), (10000, 6)])
def test_split_evenly_is_exact(n: int, parts: int) -> None:
    shares = [split_evenly(n, parts, i) for i in range(parts)]
    assert sum(shares) == n
    assert max(shares) - min(shares) <= 1


@pytest.mark.parametrize("n,world,workers", [(10000, 1, 4), (10001, 3, 4),
                                             (5, 2, 4), (1000, 8, 6)])
def test_epoch_share_is_equal_per_rank_and_independent_of_gpu_count(
        n: int, world: int, workers: int) -> None:
    per_rank = [sum(frames_for_worker(n, world, workers, w) for w in range(workers))
                for _ in range(world)]
    assert len(set(per_rank)) == 1
    assert n <= sum(per_rank) < n + world


def test_rank_from_environment(monkeypatch) -> None:
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "NODE_RANK"):
        monkeypatch.delenv(k, raising=False)
    assert distributed_rank_world() == (0, 1)
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "2")
    assert distributed_rank_world() == (2, 4)
    monkeypatch.setenv("RANK", "3")
    assert distributed_rank_world() == (3, 4)
    monkeypatch.delenv("RANK")
    monkeypatch.setenv("NODE_RANK", "1")
    with pytest.raises(RuntimeError, match="RANK"):
        distributed_rank_world()


def test_render_gpu_defaults_to_the_training_gpu(monkeypatch) -> None:
    uuids = ["u0", "u1", "u2"]
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert render_gpu_uuid((), uuids, worker_id=3, n_workers=4) == "u0"
    monkeypatch.setenv("LOCAL_RANK", "2")
    assert render_gpu_uuid((), uuids, worker_id=0, n_workers=4) == "u2"
    assert render_gpu_uuid((), [], worker_id=0, n_workers=4) == ""


def test_render_gpus_spread_workers_over_the_listed_gpus(monkeypatch) -> None:
    uuids = ["u0", "u1", "u2", "u3"]
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    got = [render_gpu_uuid((2, 3), uuids, worker_id=w, n_workers=4) for w in range(4)]
    assert got == ["u2", "u3", "u2", "u3"]
    # A second rank continues the rotation rather than repeating rank 0's.
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert render_gpu_uuid((1, 2, 3), uuids, worker_id=0, n_workers=4) == "u2"
    with pytest.raises(ValueError, match="not a visible CUDA device"):
        render_gpu_uuid((7,), uuids, worker_id=0, n_workers=1)


def test_gpu_index_is_replaced_by_render_gpus() -> None:
    from darts_model.data.config import build_data_config
    with pytest.raises(ValueError, match="render_gpus"):
        build_data_config({"data": {"gpu_index": 1}})
    assert build_data_config({"data": {"render_gpus": [1, 2]}}).render_gpus == (1, 2)


def test_bg_image_dir(tmp_path) -> None:
    check_bg_image_dir("")
    with pytest.raises(FileNotFoundError, match="download_places365"):
        check_bg_image_dir(str(tmp_path / "missing"))
    (tmp_path / "notes.txt").write_text("x")
    with pytest.raises(FileNotFoundError, match="no .*images"):
        check_bg_image_dir(str(tmp_path))
    (tmp_path / "0000000.JPG").write_bytes(b"x")
    check_bg_image_dir(str(tmp_path))


def test_datasets_fail_fast_on_missing_backgrounds(tmp_path) -> None:
    missing = str(tmp_path / "bg_images")
    with pytest.raises(FileNotFoundError):
        DenseDetectDataset(DataConfig(bg_image_dir=missing))
    with pytest.raises(FileNotFoundError):
        DensePretrainDataset(bg_image_dir=missing)


def test_datasets_reject_too_many_darts() -> None:
    with pytest.raises(ValueError, match="MAX_DARTS"):
        DenseDetectDataset(DataConfig(dart_count_weights=(1, 1, 1, 1, 1)))
    with pytest.raises(ValueError, match="MAX_DARTS"):
        DensePretrainDataset(dart_count_weights=(1, 1, 1, 1, 1))


def test_rank_is_captured_when_pickled_into_a_worker(monkeypatch) -> None:
    """A spawned worker has no process group to ask, so the launching process
    records its rank in the pickle."""
    ds = DensePretrainDataset()
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 3)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    clone = pickle.loads(pickle.dumps(ds))
    assert clone._dist == (3, 4)
    assert ds._dist is None
