"""Access to the compiled renderer, and the per-worker settings it is built with.

Both training datasets own one renderer per dataloader worker. What differs
between workers -- the seed, the GPU, and the share of the epoch -- is decided
here, once, so the two stages cannot drift apart.
"""
from __future__ import annotations

import math
import os

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

#: Split ids folded into the renderer seed, so train and validation streams
#: never coincide.
SPLITS: dict[str, int] = {"train": 0, "val": 1}

#: Extensions the renderer loads as backgrounds (``isImageExt`` in
#: renderer/src/renderer_lib.cpp).
BG_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def import_renderer():
    """Import ``dartboard_renderer``, the pybind11 module built by CMake."""
    try:
        import dartboard_renderer
    except ImportError as e:
        raise ImportError(
            "dartboard_renderer is not importable. Build it with\n"
            "    cmake -B build renderer -DCMAKE_BUILD_TYPE=Release\n"
            "    cmake --build build --parallel\n"
            "and put the directory holding the built module on PYTHONPATH\n"
            "(build/ on Linux and macOS, build/Release on Windows)."
        ) from e
    return dartboard_renderer


def renderer_seed(seed: int, split: str, rank: int, worker_id: int) -> int:
    """The renderer's uint32 seed for one worker of one rank of one split.

    Hashed through ``numpy.random.SeedSequence`` rather than packed
    arithmetically: any affine packing (``seed + 1000 * worker``) makes
    distinct ``(seed, worker)`` pairs collide, e.g. seed 1000 worker 0 against
    seed 0 worker 1. The chance of two of a run's streams sharing a hashed
    seed is about n^2 / 2^33 -- under 1e-7 for a hundred workers.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}, got {split!r}")
    for name, v in (("seed", seed), ("rank", rank), ("worker_id", worker_id)):
        if int(v) < 0:
            raise ValueError(f"{name} must be non-negative, got {v}")
    ss = np.random.SeedSequence([int(seed), SPLITS[split], int(rank), int(worker_id)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def cuda_device_uuids() -> list[str]:
    """The UUID of every CUDA device this process can see, in CUDA order.

    Empty without CUDA. Read in the launching process: a dataloader worker
    forked from it cannot initialise CUDA itself.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        return [str(torch.cuda.get_device_properties(i).uuid)
                for i in range(torch.cuda.device_count())]
    except (ImportError, RuntimeError, AttributeError):
        return []


def render_gpu_uuid(render_gpus: tuple[int, ...], uuids: list[str],
                    worker_id: int, n_workers: int) -> str:
    """UUID of the GPU this dataloader worker renders on; empty to let the
    renderer choose.

    ``render_gpus`` lists CUDA device indices -- the numbering training uses.
    Empty renders on the process's own training GPU (``LOCAL_RANK``, else
    device 0). Otherwise each rank's workers are spread over the listed GPUs
    in turn, so rendering can run on GPUs that do not train.

    The renderer is told the UUID rather than an index because Vulkan
    enumerates devices in its own order, which can include GPUs CUDA does not
    see; NVIDIA's Vulkan device UUID equals the CUDA one.
    """
    if not uuids:
        return ""
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if render_gpus:
        idx = render_gpus[(local * n_workers + worker_id) % len(render_gpus)]
    else:
        idx = local
    if not 0 <= idx < len(uuids):
        raise ValueError(f"render GPU {idx} is not a visible CUDA device "
                         f"(have {len(uuids)})")
    return uuids[idx]


def distributed_rank_world() -> tuple[int, int]:
    """``(global rank, world size)`` of this process; ``(0, 1)`` when not
    distributed.

    Read from ``torch.distributed`` when a process group exists (the training
    process, and dataloader workers forked from it), and otherwise from the
    launcher's environment: ``RANK``/``WORLD_SIZE`` (torchrun), or
    ``LOCAL_RANK`` on a single node (Lightning's own launcher, which sets no
    ``RANK``).
    """
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    env = os.environ
    world = int(env.get("WORLD_SIZE", "1"))
    if world <= 1:
        return 0, 1
    if "RANK" in env:
        return int(env["RANK"]), world
    if "LOCAL_RANK" in env and int(env.get("NODE_RANK", "0")) == 0:
        return int(env["LOCAL_RANK"]), world
    raise RuntimeError(
        f"WORLD_SIZE={world} but this process's global rank is unknown: set "
        "RANK, or launch through torchrun or Lightning.")


def split_evenly(total: int, parts: int, index: int) -> int:
    """Share ``index`` of ``total`` split over ``parts``: the first
    ``total % parts`` shares take one extra, so the shares sum to ``total``."""
    base, extra = divmod(int(total), int(parts))
    return base + (1 if index < extra else 0)


def frames_for_worker(epoch_length: int, world_size: int, num_workers: int,
                      worker_id: int) -> int:
    """Frames one dataloader worker renders per epoch.

    ``epoch_length`` counts frames across all GPUs, so an epoch is the same
    amount of data whatever the device count. Each rank renders
    ``ceil(epoch_length / world_size)``: rounded up rather than split exactly
    because DDP needs every rank to run the same number of steps, and a rank
    one frame short can finish its epoch a batch early and leave the others
    waiting on a gradient all-reduce. The epoch therefore overshoots
    ``epoch_length`` by fewer than ``world_size`` frames. Within a rank the
    share is split exactly over the workers.
    """
    per_rank = math.ceil(int(epoch_length) / max(1, int(world_size)))
    return split_evenly(per_rank, num_workers, worker_id)


def check_bg_image_dir(bg_image_dir: str) -> None:
    """Fail unless ``bg_image_dir`` is empty or a directory holding an image.

    The renderer only warns about a missing directory and renders on without
    backgrounds, which would train a model that has never seen a photograph
    behind the board. Checked in the dataset's constructor so it fails in the
    launching process, before any worker starts. Relative paths resolve
    against the working directory, as they do for the renderer.
    """
    if not bg_image_dir:
        return
    hint = ("Download backgrounds with\n"
            f"    python scripts/download_places365.py --output-dir {bg_image_dir}\n"
            "or set bg_image_dir: \"\" to train without them.")
    if not os.path.isdir(bg_image_dir):
        raise FileNotFoundError(
            f"bg_image_dir {bg_image_dir!r} is not a directory "
            f"(cwd {os.getcwd()}). {hint}")
    with os.scandir(bg_image_dir) as it:
        for entry in it:
            if entry.name.lower().endswith(BG_IMAGE_EXTS) and entry.is_file():
                return
    raise FileNotFoundError(
        f"bg_image_dir {bg_image_dir!r} holds no {'/'.join(BG_IMAGE_EXTS)} "
        f"images. {hint}")


class RenderedDataset(IterableDataset):
    """An iterable dataset with one renderer per dataloader worker.

    Subclasses call :meth:`_make_renderer` with the renderer's own options and
    :meth:`_worker_plan` for the frame count.

    The renderer is created lazily inside the worker and lives as long as the
    worker, so its frame stream continues across epochs. That needs
    ``persistent_workers=True`` whenever ``num_workers > 0``: a worker rebuilt
    each epoch starts a fresh renderer from the same seed and repeats the
    previous epoch's frames.
    """

    #: Attributes holding live renderer state, dropped when the dataset is
    #: pickled into a worker.
    _live_state: tuple[str, ...] = ("_renderer",)

    def __init__(self, *, epoch_length: int, seed: int, split: str,
                 render_gpus: tuple[int, ...], bg_image_dir: str) -> None:
        super().__init__()
        if split not in SPLITS:
            raise ValueError(f"split must be one of {sorted(SPLITS)}, got {split!r}")
        if int(seed) < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")
        if int(epoch_length) < 1:
            raise ValueError(f"epoch_length must be positive, got {epoch_length}")
        check_bg_image_dir(bg_image_dir)
        self.epoch_length = int(epoch_length)
        self.seed = int(seed)
        self.split = split
        self.render_gpus = tuple(int(g) for g in render_gpus)
        #: Captured here, in the launching process, for the workers to use.
        self._cuda_uuids = cuda_device_uuids()
        for g in self.render_gpus:
            if self._cuda_uuids and not 0 <= g < len(self._cuda_uuids):
                raise ValueError(f"render_gpus names CUDA device {g}, but "
                                 f"{len(self._cuda_uuids)} are visible")
        self.bg_image_dir = bg_image_dir or ""
        #: (rank, world size), captured in the launching process when the
        #: dataset is pickled into a spawned worker, where no process group
        #: exists to ask.
        self._dist: tuple[int, int] | None = None
        self._renderer = None

    def __getstate__(self):
        state = self.__dict__.copy()
        for name in self._live_state:
            state[name] = None
        if state.get("_dist") is None:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                state["_dist"] = (dist.get_rank(), dist.get_world_size())
        return state

    def _worker_plan(self) -> tuple[int, int, int]:
        """``(worker id, global rank, frames this worker renders this epoch)``."""
        info = get_worker_info()
        wid, n_workers = (info.id, info.num_workers) if info else (0, 1)
        rank, world = self._dist or distributed_rank_world()
        return wid, rank, frames_for_worker(self.epoch_length, world,
                                            n_workers, wid)

    def _make_renderer(self, worker_id: int, rank: int, **options):
        """This worker's renderer, created on first use."""
        if self._renderer is not None:
            return self._renderer
        dartboard_renderer = import_renderer()
        info = get_worker_info()
        renderer = dartboard_renderer.Renderer(
            seed=renderer_seed(self.seed, self.split, rank, worker_id),
            gpu_uuid=render_gpu_uuid(self.render_gpus, self._cuda_uuids,
                                     worker_id, info.num_workers if info else 1),
            bg_image_dir=self.bg_image_dir,
            **options)
        if self.bg_image_dir and not getattr(renderer, "has_bg_textures", True):
            raise RuntimeError(
                f"the renderer loaded no backgrounds from {self.bg_image_dir!r}; "
                "the images there failed to decode.")
        self._renderer = renderer
        return renderer
