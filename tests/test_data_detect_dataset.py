"""The detector's dataset: what a sample holds, and how frames reach it.

A stub ``dartboard_renderer`` stands in for the Vulkan module. Its frames come
from :func:`test_data_synthetic.synthetic_frame`, and each one carries its
renderer's seed and frame number in the first image pixels so the stream can
be traced through a DataLoader.
"""
from __future__ import annotations

import sys
import textwrap
import threading

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from darts_model.data.config import MAX_DARTS, DataConfig
from darts_model.data.detect_dataset import DenseDetectDataset, _RenderAhead
from darts_model.data.pretrain_dataset import DensePretrainDataset
from darts_model.data.targets import METAL, WIRE
from test_data_synthetic import dart, synthetic_frame

SIZE = 32

STUB = textwrap.dedent('''
    import numpy as np
    from test_data_synthetic import dart, synthetic_frame

    class Renderer:
        def __init__(self, seed, width, height, **kw):
            self.seed, self.size, self.n = seed, width, 0
            self.has_bg_textures = True

        def render_frame(self):
            self.n += 1
            cls = np.full((self.size, self.size), 1, np.uint8)
            inst = np.zeros_like(cls)
            darts = []
            if self.n % 2:
                darts = [dart(8.0, 8.0)]
                cls[8:16, 6:10], inst[8:16, 6:10] = 6, 1
            img, ann = synthetic_frame(self.size, darts=darts, seg_class=cls,
                                       seg_instance=inst)
            img[0, 0] = list(int(self.seed).to_bytes(4, "little"))[:3]
            img[0, 1] = [int(self.seed) >> 24, self.n % 256, self.n // 256]
            return img, ann
''')


def _identity(image: torch.Tensor) -> tuple[int, int]:
    """(seed, frame number) from a CHW uint8 image the stub rendered."""
    p = image[:, 0, :2].T.tolist()
    seed = p[0][0] | p[0][1] << 8 | p[0][2] << 16 | p[1][0] << 24
    return seed, p[1][1] | p[1][2] << 8


@pytest.fixture
def stub_renderer(tmp_path, monkeypatch):
    (tmp_path / "dartboard_renderer.py").write_text(STUB)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "dartboard_renderer", raising=False)
    yield
    sys.modules.pop("dartboard_renderer", None)


def _dataset(**kw) -> DenseDetectDataset:
    kw.setdefault("augment", False)
    return DenseDetectDataset(DataConfig(image_size=SIZE), out_stride=8, **kw)


def test_zero_dart_frame_is_a_sample() -> None:
    """An empty board supervises the foreground toward background and the
    keypoints as usual; the box terms are fully masked."""
    ds = _dataset()
    s = ds._sample(*synthetic_frame(SIZE))
    assert s["foreground"].shape == (SIZE // 8, SIZE // 8)
    assert s["foreground"].sum() == 0 and s["instance"].sum() == 0
    assert s["dart_mask"].shape == (MAX_DARTS,) and not s["dart_mask"].any()
    assert not s["dart_box_mask"].any()
    assert s["dart_box"].abs().sum() == 0 and s["dart_ends"].abs().sum() == 0
    assert s["keypoint_mask"].all()


def test_wire_across_a_dart_stays_foreground() -> None:
    cls = np.full((SIZE, SIZE), 1, np.uint8)
    inst = np.zeros((SIZE, SIZE), np.uint8)
    cls[8:16, 8:16], inst[8:16, 8:16] = METAL, 1
    cls[8:16, 11], inst[8:16, 11] = WIRE, 0
    ds = _dataset()
    s = ds._sample(*synthetic_frame(SIZE, darts=[dart(12, 8)], seg_class=cls,
                                    seg_instance=inst))
    assert s["foreground"][1, 1] == 1 and s["instance"][1, 1] == 1
    assert s["foreground"].sum() == 1
    assert s["dart_mask"].tolist() == [True, False, False]


def test_end_on_dart_keeps_its_ends_but_not_its_box_axis() -> None:
    """An end-on dart's box is the renderer's minimum-area fallback: its
    direction and extents are masked, its landing and flight points are not."""
    ds = _dataset()
    s = ds._sample(*synthetic_frame(SIZE, darts=[dart(12, 8),
                                                 dart(20, 8, end_on=True)]))
    assert s["dart_mask"].tolist() == [True, True, False]
    assert s["dart_box_mask"].tolist() == [True, False, False]
    assert s["dart_ends"][1].tolist() == pytest.approx(
        [20 / SIZE, 8 / SIZE, 20 / SIZE, 28 / SIZE])


def test_render_thread_failure_reaches_the_consumer() -> None:
    class Broken:
        def render_frame(self):
            raise ValueError("device lost")

    ahead = _RenderAhead(Broken(), depth=2, timeout_s=5.0)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="render thread failed") as e:
            ahead.get()
        assert isinstance(e.value.__cause__, ValueError)


def test_hung_render_times_out_with_a_clear_error() -> None:
    release = threading.Event()

    class Hung:
        def render_frame(self):
            release.wait()
            return None, None

    ahead = _RenderAhead(Hung(), depth=1, timeout_s=0.2)
    try:
        with pytest.raises(RuntimeError, match="still inside render_frame"):
            ahead.get()
    finally:
        release.set()


def test_one_render_thread_across_epochs(stub_renderer) -> None:
    ds = _dataset(epoch_length=3)
    before = {t for t in threading.enumerate() if t.name == "render-ahead"}
    ids = []
    for _ in range(3):
        ids += [_identity(s["image"]) for s in ds]
    threads = {t for t in threading.enumerate() if t.name == "render-ahead"} - before
    assert len(threads) == 1
    assert len(ids) == 9 and len(set(ids)) == 9


def test_train_and_val_render_different_frames_in_process(stub_renderer) -> None:
    """The case the process-id seed got wrong: num_workers=0, where train and
    val share a process."""
    seed = lambda ds: _identity(next(iter(ds))["image"])[0]  # noqa: E731
    train = _dataset(epoch_length=1, seed=7, split="train")
    val = _dataset(epoch_length=1, seed=7, split="val")
    assert seed(train) != seed(val)
    assert seed(_dataset(epoch_length=1, seed=7, split="train")) == seed(train)


@pytest.mark.parametrize("make", ["detect", "pretrain"])
def test_dataloader_epochs_are_exact_and_workers_distinct(stub_renderer, make) -> None:
    """Two spawned workers, persistent as the CLIs build them: every epoch is
    exactly epoch_length frames, the workers render distinct streams, and no
    frame repeats across epochs."""
    if make == "detect":
        ds = _dataset(epoch_length=7, seed=3, split="train")
    else:
        ds = DensePretrainDataset(image_size=SIZE, epoch_length=7, seed=3,
                                  out_stride=8)
    dl = DataLoader(ds, batch_size=2, num_workers=2, persistent_workers=True,
                    multiprocessing_context="spawn")
    seen = []
    for _ in range(2):
        epoch = [_identity(img) for b in dl for img in b["image"]]
        assert len(epoch) == 7
        seen += epoch
    assert len({s for s, _ in seen}) == 2
    assert len(set(seen)) == len(seen)


def test_zero_dart_frames_batch_with_others(stub_renderer) -> None:
    """The stub alternates one dart and none; both collate into one batch."""
    dl = DataLoader(_dataset(epoch_length=4), batch_size=4)
    b = next(iter(dl))
    assert b["dart_mask"][:, 0].tolist() == [True, False, True, False]
    assert (b["foreground"].flatten(1).sum(1) > 0).tolist() == [True, False, True, False]


def test_end_on_darts_keep_their_ends_but_not_their_box_orientation() -> None:
    from test_data_synthetic import dart, synthetic_frame
    _, ann = synthetic_frame(64, darts=[dart(20, 20), dart(40, 40)])
    ann["darts"][1]["box_end_on"] = True
    box, ends, valid, box_valid = DenseDetectDataset.boxes_from(ann, 64)
    assert valid[:2].tolist() == [True, True]
    assert box_valid[:2].tolist() == [True, False]
