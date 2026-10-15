"""Export a dense-readout checkpoint to a CoreML .mlpackage for the iOS app.

Two things here are load-bearing.

**MLProgram, not the legacy NeuralNetwork format.** The legacy format's op
coverage is far narrower; a model CoreML cannot build falls back to CPU
silently, at ~500ms per inference against ~120ms. `convert_to="mlprogram"` is
not optional.

**Normalisation lives in the GRAPH, not in the app and not in the image
input.** The model expects ImageNet-normalised input -- `GPUAugment` applies
`(x/255 - mean)/std` in `on_after_batch_transfer`, and the backbone was
pretrained through the same module. `ct.ImageType` cannot express it: it takes
one scalar `scale` and a per-channel bias, while ImageNet needs a per-channel
scale (std is 0.229/0.224/0.225). So `ExportWrapper` folds it in as
`(x - 255*mean) / (255*std)`, the image input stays a passthrough, and the app
hands over a raw CVPixelBuffer.

Keeping it in the graph means there is exactly one place it can live, so no
consumer can omit it or apply it twice, and
`tests/test_export_normalisation.py` pins it to `GPUAugment`.

The converted model is checked against torch on one frame (`--sample`, else a
freshly rendered one) before anything is written, and the readout contract --
input size, K, the Hough parameters, keypoint order -- goes into the package's
`user_defined_metadata`.

    darts-export-coreml CHECKPOINT --config configs/detector.yaml \\
        --out darts.mlpackage --sample frame.jpg
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

from darts_model.export.common import (
    MIN_KEPT_CELLS,
    OUTPUTS,
    SLOT_OUTPUTS,
    output_names,
    TOPK,
    ExportError,
    KeypointFp32Selector,
    check_against_reference,
    effective_topk,
    topk_for_stride,
    keypoint_module_names,
    load_checkpoint,
    readout_contract,
    require_export_heads,
    verification_frame,
)
from darts_model.model.detector import slot_floor

__all__ = ["OUTPUTS", "SLOT_OUTPUTS", "TOPK", "ExportWrapper",
           "keypoint_fp32_selector"]


def keypoint_fp32_selector(model: torch.nn.Module) -> KeypointFp32Selector:
    """`op_selector` keeping the keypoint head at fp32 and the rest at fp16.

    The keypoint head's weights must not be rounded to fp16. It does not
    degrade the corners gracefully -- it collapses a contiguous block of them.
    Measured at the app's 0.30 confidence gate:

        all fp16          corners 32-39 fall from ~0.95 to ~0.005, so 32/40
                          corners survive and the pose fit loses a whole ARC
                          of the board, which conditions a homography far worse
                          than eight scattered losses would
        kp weights fp32   40/40, and the detections match torch exactly

    Positions are unaffected either way (0.25px), which is what makes it nasty:
    the argmax lands in the right cell and only the magnitude is wrong, so
    nothing looks broken. The reduction is innocent -- a standalone fp16
    reduce_max over the same shape is exact to 1e-4 -- and so are the
    activations. It is the stored weights.

    Costs nothing measurable: the package is 9.5 MB either way, against 19 MB
    for a wholly fp32 model.

    Selection is by the module path carried in each weight constant's name
    (see `uses_keypoint_weights`), which reaches every conv and GroupNorm of
    the head regardless of how the traced op itself is named.
    """
    return KeypointFp32Selector(keypoint_module_names(model))


class ExportWrapper(torch.nn.Module):
    """Normalisation in, compact detections out -- the readout runs here.

    Handing the dense planes to the app would cost 3.5 MB a frame. The
    keypoint branch is the clearest case: `kp_logits` is 655,360 of the 868,352
    floats -- 75% of everything -- and the only use the app makes of it is 40
    argmaxes. Computing them here turns that plane into 40 coordinates.

    Written as ordinary torch rather than hand-authored MIL, so it is tested
    against `decode_boxes` + `hough.detect` on the same weights
    (tests/test_export_wrapper.py).

    What deliberately does NOT move: the Hough accumulator, its 3x3
    non-maximum suppression and the per-peak weighted average. Their output
    length depends on the data, a graph output shape cannot, and they run over
    the few hundred cells that survive the gate rather than the whole grid.
    """

    def __init__(self, model: torch.nn.Module, topk: int = TOPK,
                 float_keypoints: bool = False) -> None:
        super().__init__()
        require_export_heads(model.config)
        self.model = model
        self.topk = topk
        # See `_keypoints_float`. Off by default, so the CoreML export -- which
        # has no trouble with argmax and is already on devices -- is unchanged.
        self.float_keypoints = float_keypoints
        from darts_model.data.gpu_augment import IMAGENET_MEAN, IMAGENET_STD
        # Pre-multiplied by 255 so the graph takes 0-255 directly and the
        # divide disappears into the same elementwise op.
        self.register_buffer(
            "norm_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) * 255.0)
        self.register_buffer(
            "norm_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) * 255.0)

    def _keypoints_argmax(self, out, n: int, h: int, w: int):
        """The direct spelling: argmax, then gather the offset at that cell."""
        kpl = out["kp_logits"]                      # (1, nk, h, w)
        nk = int(kpl.shape[1])
        kv, ki = kpl.reshape(1, nk, n).max(dim=2)   # (1, nk)
        ko = torch.gather(out["kp_offset"].reshape(1, 2, n), 2,
                          ki.unsqueeze(1).expand(1, 2, nk))
        kx = ((ki % w).to(kv.dtype) + 0.5 + ko[:, 0]) / w
        ky = (torch.div(ki, w, rounding_mode="floor").to(kv.dtype)
              + 0.5 + ko[:, 1]) / h
        return torch.stack((kx, ky), dim=2), torch.sigmoid(kv)

    def _keypoints_float(self, out, n: int, h: int, w: int):
        """The same selection, without an argmax, an index or a gather.

        Why it exists: a WebGPU delegate will not take `ARG_MAX`, because the
        operation EMITS int64 and the delegate has no int64 -- casting
        afterwards does not help, the output type is the problem. Nor will it
        take the `GATHER_ND` the offset lookup lowers to. So the direct
        spelling strands the keypoint branch on the CPU, and `kp_logits` is
        655,360 of the 868,352 floats the model produces: three quarters of
        everything, crossing to the CPU every frame to be scanned there. On a
        desktop that costs a few milliseconds. On a four-core phone running the
        CPU side in wasm it is the dominant cost of the frame.

        What replaces it: a peak is where a value EQUALS the maximum, and a
        mask of that is one-hot over the spatial axis. Multiplying by the mask
        and summing therefore selects -- the same value a gather would have
        fetched, by arithmetic the delegate accepts. The index comes out the
        same way, as a float, by summing the mask against a ramp.

        Ties are handled rather than ignored: two cells at exactly the same
        maximum would both survive the mask, so every selection is divided by
        the number of hits. That averages them, where an argmax would take the
        first. On a trained heatmap an exact float tie at the peak essentially
        does not happen; dividing costs one operation and removes the
        possibility of a keypoint landing at the sum of two positions.

        Everything here is float32 and elementwise or a reduction, so nothing
        in it needs a type the delegate lacks.
        """
        kpl = out["kp_logits"].reshape(1, -1, n)    # (1, nk, n)
        nk = int(kpl.shape[1])
        peak = kpl.amax(dim=2, keepdim=True)        # (1, nk, 1)
        # Bool -> float, not int: a cast to an integer type is the thing being
        # avoided.
        hit = (kpl == peak).to(kpl.dtype)           # (1, nk, n), one-hot-ish
        hits = hit.sum(dim=2)                       # (1, nk), 1 unless tied

        ramp = torch.arange(n, dtype=kpl.dtype, device=kpl.device).reshape(1, 1, n)
        index = (hit * ramp).sum(dim=2) / hits      # (1, nk), the cell, as a float

        # The offset planes are shared by every keypoint, so they broadcast
        # against the mask.
        offset = out["kp_offset"].reshape(1, 2, n)
        ox = (hit * offset[:, 0:1]).sum(dim=2) / hits
        oy = (hit * offset[:, 1:2]).sum(dim=2) / hits

        # Row and column without a modulo or an integer divide: both are float
        # operations here, and `index - iy * w` is exactly `index % w` for the
        # non-negative integers this holds.
        iy = torch.floor(index / w)
        ix = index - iy * w

        kx = (ix + 0.5 + ox) / w
        ky = (iy + 0.5 + oy) / h
        return torch.stack((kx, ky), dim=2), torch.sigmoid(peak.squeeze(2))

    def forward(self, image: torch.Tensor):
        out = self.model((image - self.norm_mean) / self.norm_std)

        fg = out["fg_logits"]                       # (1, h, w)
        h, w = int(fg.shape[-2]), int(fg.shape[-1])
        n = h * w
        k = min(self.topk, n)

        # Descending, so the Swift side stops at the first cell under its
        # threshold instead of scanning the grid.
        top_logit, idx = torch.topk(fg.reshape(1, n), k, dim=1, sorted=True)

        # Cell centre of each kept cell, in normalised image coords. The same
        # `(i + 0.5) / n` convention as `cell_centres` in model/detector.py -- a
        # half-cell error here is a half-cell error in every detection.
        ix = (idx % w).to(top_logit.dtype)
        iy = torch.div(idx, w, rounding_mode="floor").to(top_logit.dtype)
        cell = torch.stack(((ix + 0.5) / w, (iy + 0.5) / h), dim=1)  # (1,2,k)

        def gather2(name: str) -> torch.Tensor:
            t = out[name].reshape(1, 2, n)
            return torch.gather(t, 2, idx.unsqueeze(1).expand(1, 2, k))

        # Normalised here, outside the gradient path, exactly as decode_boxes
        # does it: a unit vector averaged and then renormalised is a direction,
        # an average alone is not.
        direction = gather2("direction")
        direction = direction / direction.norm(
            dim=1, keepdim=True).clamp(min=1e-6)

        # (1, k, 2) rather than (1, 2, k): the reader walks cells, so the two
        # components of one cell should be adjacent in memory.
        dart_score = torch.sigmoid(top_logit)
        dart_centre = (gather2("centre_offset") + cell).transpose(1, 2)
        dart_direction = direction.transpose(1, 2)
        dart_extent = gather2("log_extent").exp().transpose(1, 2)
        dart_tip = (gather2("tip_offset") + cell).transpose(1, 2)
        dart_flight = (gather2("flight_offset") + cell).transpose(1, 2)

        # --- keypoints: 655,360 floats -> 120
        kp_xy, kp_conf = (self._keypoints_float(out, n, h, w)
                          if self.float_keypoints
                          else self._keypoints_argmax(out, n, h, w))

        outs = (dart_score, dart_centre, dart_direction, dart_extent,
                dart_tip, dart_flight, kp_xy, kp_conf)
        cfg = self.model.config
        peaks = None
        if getattr(self.model, "tip_trunk", None) is not None:
            from darts_model.model.tips import tip_peaks
            peaks = tip_peaks(out["tip_logits"], out["tip_cell_offset"],
                              cfg.tip_peak_count)
        if (getattr(self.model, "instance_head", None) is not None
                or getattr(self.model, "query_head", None) is not None
                or getattr(self.model, "token_readout", None) is not None):
            # The darts themselves: seeds picked, cells grouped and read in
            # the graph, so the app reads S slots instead of voting.
            darts = self.model.read_darts(out)
            tip = darts["tip"]
            if peaks is not None and cfg.tip_snap and cfg.tip_assign == "nearest":
                from darts_model.model.tips import snap_to_tips
                tip, _ = snap_to_tips(tip, darts["direction"], *peaks, cfg,
                                      float(image.shape[-1]))
            outs = outs + (darts["score"], tip, darts["flight"])
            if "tip_hard" in darts:
                outs = outs + (darts["tip_hard"],)
        if peaks is not None:
            outs = outs + peaks
        return outs


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export a detector checkpoint to a CoreML .mlpackage.")
    ap.add_argument("checkpoint")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="darts.mlpackage")
    ap.add_argument("--size", type=int, default=None,
                    help="input side; defaults to the config's image_size")
    ap.add_argument("--precision", default="fp16", choices=("fp16", "fp32"),
                    help="fp16 keeps the keypoint head at fp32; see "
                         "keypoint_fp32_selector for why that is not optional")
    ap.add_argument("--sample", default=None,
                    help="an image showing darts in a board, to verify the "
                         "converted model against torch. Without it one frame "
                         "is rendered, which needs the renderer module.")
    args = ap.parse_args()
    try:
        return export(args)
    except ExportError as exc:
        print(f"!! {exc}", file=sys.stderr)
        return 1


def export(args) -> int:
    loaded = load_checkpoint(args.checkpoint, args.config)
    cfg = loaded.cfg
    require_export_heads(cfg)
    size = args.size or int(loaded.raw["data"]["image_size"])
    topk = effective_topk(size, cfg.out_stride)
    if topk < topk_for_stride(cfg.out_stride):
        print(f"  {size}px gives {topk} cells: K clamped from "
              f"{topk_for_stride(cfg.out_stride)}")

    model = ExportWrapper(loaded.lit.model, topk=topk).eval()
    names = output_names(cfg)

    # 0-255, which is what the WRAPPER takes -- it normalises internally, so
    # the reference and the converted model are fed what the app feeds them.
    example = verification_frame(args.sample, size, loaded.raw)
    with torch.no_grad():
        reference = model(example)
    ref = {n: t.numpy() for n, t in zip(names, reference)}
    for name, t in ref.items():
        print(f"  {name:<15} {tuple(t.shape)}")
    gate = float(cfg.fg_threshold)
    kept = int((ref["dart_score"][0] >= gate).sum())
    if kept < MIN_KEPT_CELLS:
        raise ExportError(
            f"the model keeps {kept} cells on the verification frame, too few "
            f"to verify the conversion; pass --sample with darts in the board")

    with torch.no_grad():
        traced = torch.jit.trace(model, example, strict=False)
        traced = torch.jit.freeze(traced.eval())

    try:
        import coremltools as ct
    except ImportError:
        raise ExportError('coremltools is not installed: pip install -e ".[coreml]"') from None

    selector = keypoint_fp32_selector(loaded.lit.model)
    print("converting ...")
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.ImageType(name="image", shape=(1, 3, size, size),
                             color_layout=ct.colorlayout.RGB,
                             # Passthrough: ExportWrapper does the
                             # normalisation, because ImageType cannot express
                             # a per-channel scale.
                             scale=1.0, bias=[0.0, 0.0, 0.0])],
        # float32 OUTPUTS, while the compute stays float16.
        #
        # With no dtype CoreML hands back Float16 arrays, and Swift reading one
        # as Float does not return an error -- it traps: "The specified scalar
        # type Float does not match the multiarray's data type Float16". The
        # cast sits at the very end of the graph, so the body still runs in
        # fp16 on the ANE.
        outputs=[ct.TensorType(name=n, dtype=np.float32) for n in names],
        # iOS 15, which is as far back as the app goes. A program built for a
        # newer target carries only that target's block, and an older runtime
        # fails to load it ("specialization ios17 not in blocks").
        #
        # Nothing in the graph requires a newer opset. The exported ops are
        # conv, reduce_mean, gelu, gather_along_axis, upsample_bilinear, topk,
        # reduce_max and reduce_argmax, and every one exists in iOS15's. Keep
        # `grid_sample` out of the exported path: it lowers to `resample`,
        # which would force iOS16.
        #
        # An old phone is exactly the device someone stands in the corner as a
        # second camera, so this is worth more than a version number.
        minimum_deployment_target=ct.target.iOS15,
        # fp16 here means "fp16 except the keypoint head and the readout".
        # Plain fp16 is not offered, because it is measurably wrong.
        compute_precision=(
            ct.precision.FLOAT32 if args.precision == "fp32"
            else ct.transform.FP16ComputePrecision(op_selector=selector)),
    )
    if args.precision == "fp16":
        # A conv per trunk layer, plus the heatmap and offset convs, at least.
        need = int(cfg.kp_head_depth) + 2
        print(f"  kept at fp32: {len(selector.kept)} keypoint-head ops, "
              f"{len(selector.readout_kept)} readout ops")
        if len(selector.kept) < need:
            raise ExportError(
                f"only {len(selector.kept)} ops were recognised as the keypoint "
                f"head (expected at least {need}), so its weights were cast to "
                f"fp16. The converter's constant naming has changed; update "
                f"uses_keypoint_weights in export/common.py, or export with "
                f"--precision fp32.")

    contract = readout_contract(cfg, size, topk)
    mlmodel.short_description = (
        "Dartboard readout: per-dart landing and flight points in S slots, "
        "plus the top-K cells and 40 double-bed corners; see "
        "user_defined_metadata." if output_names(cfg) != OUTPUTS else
        "Dense dartboard readout: top-K cells with oriented box, landing point "
        "and flight tip, plus 40 double-bed corners. Darts need Hough voting "
        "over the cells; see user_defined_metadata and the README.")
    mlmodel.user_defined_metadata.update(
        {k: v if isinstance(v, str) else json.dumps(v)
         for k, v in contract.items()})

    # --- verify against the torch model rather than trusting the conversion
    spec_outputs = [o.name for o in mlmodel.get_spec().description.output]
    missing = [n for n in names if n not in spec_outputs]
    if missing:
        raise ExportError(f"{missing} missing from the converted model; the "
                          f"Swift side looks outputs up by name")

    print("\nchecking the converted model against torch ...")
    try:
        from PIL import Image
        arr = example[0].permute(1, 2, 0).numpy().astype(np.uint8)
        got = {n: np.asarray(v) for n, v
               in mlmodel.predict({"image": Image.fromarray(arr)}).items()}
    except Exception as exc:                        # noqa: BLE001
        # CoreML prediction runs on macOS only.
        got = None
        print(f"  !! prediction unavailable here ({exc}); the package is "
              f"written UNVERIFIED")
    if got is not None:
        failures = check_against_reference(
            ref, got, size, gate, px_limit=2.0,
            slot_floor=slot_floor(cfg),
            tip_floor=float(getattr(cfg, "tip_snap_score", 0.3)))
        if failures:
            print(f"\n{failures} check(s) failed; nothing written. Re-run "
                  f"with --precision fp32 to separate precision from wiring.")
            return 1

    mlmodel.save(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
