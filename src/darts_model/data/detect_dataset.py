"""Frames with a per-cell dart silhouette and each dart's oriented box.

The dense readout needs two things per frame: which cells lie on a dart, and
which dart each of those cells belongs to.  ``build_dense_targets`` produces
both as ``dart_instance`` -- 0 for background and 1..N for dart index plus one,
with any cell holding a dart pixel counted as dart -- at whatever stride the
head runs at, so this dataset is mostly the dense-pretraining one with the
boxes added.

Every frame is yielded, including those without a usable dart: the configured
share of empty boards is where the foreground head learns that a board alone
is not a dart, and the keypoints are supervised on every frame. Such a frame
has an all-zero ``foreground`` and ``instance`` and an all-False
``dart_mask``.

Augmentation is photometric only: the renderer already samples pose, so there
is nothing to mirror onto the targets, and the boxes and keypoints come from
the annotation at full resolution and are normalised once.

Rendering runs on one background thread per renderer, for the renderer's
lifetime (:class:`_RenderAhead`).  Measured at 1024 with the segmentation
pass on, a sample costs 15.79ms of render and 6.74ms of CPU, and doing them in
sequence leaves each side idle while the other works.  The model consumes 222
samples/s with the backbone frozen; a synchronous pipeline delivers 64.
"""
from __future__ import annotations

import math
import queue
import threading

import numpy as np
import torch

from darts_model.board_geometry import (
    BOARD_KEYPOINT_NAMES,
    DOUBLE_CROSSING_INDICES,
)
from darts_model.data.config import MAX_DARTS, validate_dart_count_weights
from darts_model.data.pretrain_dataset import photometric_pipeline
from darts_model.data.targets import build_dense_targets
from darts_model.renderer import RenderedDataset

__all__ = ["MAX_DARTS", "DOUBLE_CROSSING_NAMES", "DenseDetectDataset"]

DOUBLE_CROSSING_NAMES: tuple[str, ...] = tuple(
    BOARD_KEYPOINT_NAMES[i] for i in DOUBLE_CROSSING_INDICES
)
"""The 40 double-bed corners, in canonical order."""


class _Failure:
    """An exception raised on the render thread, carried to the consumer."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


class _RenderAhead:
    """Renders ahead of the consumer on one background thread.

    One instance per renderer, created with it and never replaced, so no two
    threads ever call the same renderer: a thread started per epoch would have
    to be joined before the next could start, and a render stuck in the driver
    would make that join either hang or be abandoned with the old thread still
    rendering. Frames left in the queue when an epoch ends open the next.

    The GIL is not a problem here: render_frame spends its time inside the
    Vulkan driver and releases the GIL, so this genuinely overlaps with the
    consumer's numpy and albumentations work rather than time-slicing
    against it.
    """

    def __init__(self, renderer, depth: int, timeout_s: float) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=max(1, depth))
        self._timeout_s = timeout_s
        self._thread = threading.Thread(target=self._run, args=(renderer,),
                                        name="render-ahead", daemon=True)
        self._thread.start()

    def _run(self, renderer) -> None:
        while True:
            try:
                item = renderer.render_frame()
            except Exception as exc:
                item = _Failure(exc)
            self._q.put(item)
            if isinstance(item, _Failure):
                return

    def get(self):
        """The next ``(image, annotation)``; raises if the renderer failed or
        produced nothing within the timeout."""
        try:
            item = self._q.get(timeout=self._timeout_s)
        except queue.Empty:
            state = ("is still inside render_frame" if self._thread.is_alive()
                     else "has exited")
            raise RuntimeError(
                f"no frame from the renderer in {self._timeout_s:.0f}s; the "
                f"render thread {state}. A hung GPU or driver stalls here."
            ) from None
        if isinstance(item, _Failure):
            # Left in place, so every later call fails the same way rather
            # than waiting out the timeout on a thread that has exited.
            self._q.put(item)
            raise RuntimeError("the render thread failed") from item.exc
        return item


class DenseDetectDataset(RenderedDataset):
    """Rendered frames, silhouette instances at ``out_stride``, boxes per dart.

    ``seed`` is the experiment seed; each worker's renderer seed is derived
    from it, ``split``, the global rank and the worker id
    (:func:`darts_model.renderer.renderer_seed`). ``split`` defaults to
    ``"train"`` when augmenting and ``"val"`` otherwise. ``epoch_length``
    counts frames across all GPUs
    (:func:`darts_model.renderer.frames_for_worker`).
    """

    _live_state = ("_renderer", "_ahead")

    def __init__(self, config, out_stride: int = 8,
                 epoch_length: int = 10000, augment: bool = True,
                 prefetch: int = 3, seed: int = 0, split: str | None = None,
                 render_timeout_s: float = 300.0) -> None:
        super().__init__(epoch_length=epoch_length, seed=seed,
                         split=split or ("train" if augment else "val"),
                         render_gpus=config.render_gpus,
                         bg_image_dir=config.bg_image_dir)
        #: Frames the render thread may run ahead by. Small on purpose: each
        #: held frame is a 1024x1024x3 image plus its segmentation, and the
        #: point is to cover one sample's CPU work, not to build a buffer.
        self.prefetch = prefetch
        #: Longest wait for one frame before the worker gives up. Generous,
        #: because the first frame includes pipeline and background setup.
        self.render_timeout_s = render_timeout_s
        self.config = config
        self.out_stride = out_stride
        self.image_size = config.image_size
        self.dart_count_weights = list(
            validate_dart_count_weights(config.dart_count_weights))
        self._aug = photometric_pipeline(config.augmentation) if augment else None
        self._ahead: _RenderAhead | None = None

    def _ensure_renderer(self, worker_id: int, rank: int):
        c = self.config
        options = dict(
            width=self.image_size, height=self.image_size,
            skill_placement=True,
            dart_count_weights=self.dart_count_weights,
            grouping_prob=c.grouping_prob,
            tight_prob=c.tight_prob,
        )
        if c.asset_dir:
            options["asset_dir"] = c.asset_dir
        return self._make_renderer(worker_id, rank, **options)

    @staticmethod
    def boxes_from(annotation, image_size: int):
        """Per-dart oriented box as (cx, cy, cos, sin, half_len, half_width).

        Returns ``(box, ends, valid, box_valid)``. ``ends`` is (MAX_DARTS, 4):
        (landing_x, landing_y, flight_x, flight_y), normalised. ``box_valid``
        excludes darts the renderer flags ``box_end_on``: seen end-on, their box
        is a minimum-area fallback whose direction and extents are not the
        dart's, so only its centre and the two ends are supervised.

        Six numbers for five degrees of freedom on purpose: regressing the angle
        directly has a wrap-around discontinuity at +/-pi, so a dart turning
        through it would take a large loss for a negligible change.  The
        direction runs tip -> flight, which fixes the 180-degree ambiguity a
        bare box would otherwise carry.
        """
        box = torch.zeros((MAX_DARTS, 6), dtype=torch.float32)
        # The two ends, as the renderer reports them rather than as the box
        # implies them. The landing point is where the dart's axis meets the
        # board face -- the point scoring is computed from -- and it is NOT
        # recoverable from the box in general: for a dart pointing near the
        # camera the entry projects into the INTERIOR of the silhouette, up to
        # 27px from any end of the box, because the dart's own barrel and
        # flight cover where it went in. Measured on 175 darts: recoverable for
        # 98%, and hopeless for the shortest-projecting 2%.
        ends = torch.zeros((MAX_DARTS, 4), dtype=torch.float32)
        valid = torch.zeros((MAX_DARTS,), dtype=torch.bool)
        box_valid = torch.zeros((MAX_DARTS,), dtype=torch.bool)
        for i, d in enumerate(annotation["darts"][:MAX_DARTS]):
            bx, by = d["box_x"], d["box_y"]
            # corners: tip-left, flight-left, flight-right, tip-right
            cx, cy = sum(bx) / 4.0, sum(by) / 4.0
            ax = 0.5 * ((bx[1] + bx[2]) - (bx[0] + bx[3]))
            ay = 0.5 * ((by[1] + by[2]) - (by[0] + by[3]))
            half_len = 0.5 * math.hypot(bx[1] - bx[0], by[1] - by[0])
            half_wid = 0.5 * math.hypot(bx[3] - bx[0], by[3] - by[0])
            n = math.hypot(ax, ay)
            if n < 1e-6 or half_len < 1e-6:
                continue
            box[i] = torch.tensor([
                cx / image_size, cy / image_size, ax / n, ay / n,
                half_len / image_size, half_wid / image_size,
            ], dtype=torch.float32)
            ends[i] = torch.tensor([
                d["x"] / image_size, d["y"] / image_size,
                d["flight_x"] / image_size, d["flight_y"] / image_size,
            ], dtype=torch.float32)
            valid[i] = True
            box_valid[i] = not d.get("box_end_on", False)
        return box, ends, valid, box_valid

    @staticmethod
    def keypoints_from(annotation, image_size: int):
        """The 40 double-bed corners, normalised, plus which are usable.

        Taken by NAME rather than by slicing indices 41..80 out of the list,
        which would go silently wrong if anything were inserted ahead of them:
        a keypoint tensor of the correct shape full of the wrong points trains
        perfectly happily.

        A keypoint off the edge of the image is masked rather than clamped: the
        board is often cropped, and a clamped keypoint is a confident label on
        the wrong pixel, which is worse than no label at all.
        """
        n = len(DOUBLE_CROSSING_NAMES)
        kp = torch.zeros((n, 2), dtype=torch.float32)
        mask = torch.zeros((n,), dtype=torch.bool)
        found = {k["name"]: k for k in annotation.get("board_keypoints", ())}
        for i, name in enumerate(DOUBLE_CROSSING_NAMES):
            k = found.get(name)
            if k is None:
                continue
            x, y = float(k["x"]), float(k["y"])
            if not (0.0 <= x < image_size and 0.0 <= y < image_size):
                continue
            kp[i] = torch.tensor([x / image_size, y / image_size])
            mask[i] = True
        return kp, mask

    def __iter__(self):
        wid, rank, n_frames = self._worker_plan()
        renderer = self._ensure_renderer(wid, rank)
        if self._ahead is None:
            self._ahead = _RenderAhead(renderer, self.prefetch,
                                       self.render_timeout_s)
        for _ in range(n_frames):
            yield self._sample(*self._ahead.get())

    def _sample(self, image_np: np.ndarray, annotation: dict) -> dict:
        targets = build_dense_targets(annotation, self.image_size,
                                      out_stride=self.out_stride)
        if targets is None:
            raise RuntimeError(
                "renderer returned no seg_ids/depth -- rebuild it, or the "
                "dense readout trains on nothing")
        box, ends, valid, box_valid = self.boxes_from(annotation, self.image_size)

        rgb = image_np[..., :3]
        if self._aug is not None:
            rgb = self._aug(image=np.ascontiguousarray(rgb))["image"]
        img = np.ascontiguousarray(rgb.transpose(2, 0, 1))

        # dart_instance is 0 for background, dart index + 1 otherwise. A dart
        # whose box could not be built is dropped from the mask too, so no
        # cell is ever supervised toward a box that does not exist.
        inst = torch.from_numpy(targets["dart_instance"]).long()
        keep = torch.zeros_like(inst, dtype=torch.bool)
        for i in range(MAX_DARTS):
            if valid[i]:
                keep |= inst == (i + 1)
        inst = torch.where(keep, inst, torch.zeros_like(inst))

        kp, kp_mask = self.keypoints_from(annotation, self.image_size)

        return {
            "keypoints": kp,                      # (40, 2) normalised
            "keypoint_mask": kp_mask,             # (40,)
            "image": torch.from_numpy(img),
            "instance": inst,                     # (h, w) 0 = background
            "foreground": (inst > 0).float(),     # (h, w)
            # The labelled dart's share of the cell's dart pixels.
            "instance_purity": torch.from_numpy(
                targets["dart_instance_purity"]),  # (h, w)
            "dart_box": box,                      # (MAX_DARTS, 6)
            "dart_ends": ends,                    # (MAX_DARTS, 4)
            "dart_mask": valid,                   # (MAX_DARTS,)
            "dart_box_mask": box_valid,           # (MAX_DARTS,) direction/size usable
        }
