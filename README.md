# darts-model

Dart and dartboard detection for phone cameras, trained entirely on synthetic
images from a procedural Vulkan renderer.

The repository has two parts:

- **`renderer/`**: a headless Vulkan renderer (C++20) that generates
  randomised dartboard scenes with pixel-exact labels. The board, wire frame
  and darts are built procedurally. Shadows, ambient occlusion and reflections
  are ray traced, and the scene is lit and backed by real photographs. It is
  usable from Python (`dartboard_renderer`) or as a standalone dataset
  generator (`dartboard_gen`).
- **`src/darts_model/`**: a two-stage PyTorch Lightning training pipeline that
  renders its data on the fly, plus exporters to CoreML (iOS) and LiteRT
  (Android).

## Requirements

- A GPU with Vulkan 1.2 and ray-query support (`VK_KHR_ray_query`,
  `VK_KHR_acceleration_structure`), which is any recent NVIDIA or AMD desktop
  card. Without ray tracing the renderer falls back to a raster shader whose
  lighting does not match; it warns on startup, and its output should not be
  used for training.
- The Vulkan SDK (for `glslc`), CMake ≥ 3.21, a C++20 compiler and
  libjpeg-turbo. The other C++ dependencies are fetched by CMake.
- Python ≥ 3.10.

On Windows, install libjpeg-turbo with vcpkg and pass its toolchain file to
CMake (`-DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake`).

## Setup

```sh
# Renderer: the executable, the Python module and a CPU-only placement test
cmake -B build renderer -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel --config Release
ctest --test-dir build -C Release

# Python package
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,viz]"
export PYTHONPATH=$PWD/build            # build\Release on Windows

# Background photographs (Places365-Standard, ~24 GB download)
python scripts/download_places365.py --output-dir bg_images
```

Compiled shaders are read from the build directory and assets from
`renderer/assets/`. Both paths are fixed when CMake configures the build,
so run from a build tree.

## Training

Training runs in two stages, and each renders its own data as it goes.

```sh
# 1. Pretrain the backbone on dense per-pixel targets (~11 h)
darts-pretrain --config configs/pretrain.yaml
darts-promote-backbone outputs/pretrain outputs/backbone.ckpt

# 2. Train the detector on that backbone (~3 days)
darts-train --config configs/detector.yaml
```

These two stages produce the base model, read out by Hough voting. The
other detector configs are experiments; see [Experimental configurations](#experimental-configurations).

The times are for one RTX 5090. Stage 2 refuses to start if the
backbone file is missing, and unknown config keys are errors. Logs go to
TensorBoard under `outputs/`.

- Every frame is rendered fresh; nothing is cached or replayed.
  `epoch_length` is the number of frames per epoch across all GPUs.
- Renderer seeds derive from `experiment.seed`, the split, the rank and the
  dataloader worker, so a run is reproducible from its config.
- A non-empty `bg_image_dir` must exist and contain images, or both stages
  stop before training starts. `""` renders without photographs.
- The detector trains on a single GPU; `devices` other than 1 is refused.

### Checkpoint selection

Detector checkpoints are ranked on `val/landing_px_error_penalised`: the mean,
over every ground-truth dart in the validation epoch, of the pixel distance
from its landing point to the detection matched to it. Matching is one-to-one
(Hungarian) within `match_gate_px` (40 px at 1024), and a dart with no match
is charged the full 40 px, so an epoch cannot rank better by failing to
detect the hard darts. The best three and `last.ckpt` are kept in
`outputs/detector/checkpoints/`.

`darts-visualize CHECKPOINT --config configs/detector.yaml` draws predictions
against ground truth on freshly rendered frames (needs the `viz` extra).

## Model

The base model is the one `configs/pretrain.yaml` and
`configs/detector.yaml` produce, read out by Hough voting.

- **Backbone**: a ConvNeXt-style CNN (3.5M parameters, strides 4/8/16/32),
  pretrained on the renderer's dense targets: per-pixel part class, offsets
  to the dart tip, board UV and height.
- **Detector**: a PAN neck to stride 8. For every cell it predicts a dart
  foreground score and that dart's oriented box, landing point and flight
  end. Darts are read out by Hough voting over the foreground cells. A
  separate head predicts the 40 corners of the double ring, from which the
  board homography and the scores follow.

## Experimental configurations

The other configs in `configs/` add alternative heads and readouts to the
base detector, most of them as fine-tunes of a trained detector. They sit
behind config flags that default to off.

Results are landing errors in pixels at 1024 px, with a missed dart counted
as 40 px. "Validation" is the best epoch on the training run's own rendered
validation set; "held-out" is a separate set of 2000 frames rendered with an
unused seed and dumped by `tools/dump_readout_eval.py`, scored on 1000 of
them, at the intended framing (below).

| Config | Adds | Synthetic result | Status |
|---|---|---|---|
| `detector.yaml` | — (Hough readout) | validation 4.06; held-out 4.28 | base model |
| `detector_stride4.yaml` | output stride 4 instead of 8 | tracked stride 8 epoch for epoch, no gain (5.05 at epoch 411) | stopped |
| `detector_embed.yaml` | a per-cell instance embedding (pull/push loss) | did not reduce cells predicting a neighbouring dart's landing point | stopped |
| `detector_per_dart.yaml` | a per-dart attention head over cells grouped by embedding and seeded in-graph | validation 3.59 | stopped |
| `detector_queries*.yaml` | learned queries that point at cells, Hungarian-matched in training | validation 4.50 at best | not competitive |
| `detector_queries_rich_tips*.yaml`, `detector_queries_rich_pointer.yaml` | a tip heatmap on the query readout, snapped or pointed to | validation 4.07 at best | stopped |
| `detector_tips.yaml` | a stride-8 tip heatmap with Hough | — | not trained |
| `detector_tips4.yaml` | a stride-4 tip heatmap whose peaks go to the Hough detections one-to-one (Hungarian, 40 px gate) | validation 2.89; held-out 2.77 | trained |
| `detector_readout*.yaml` | learned queries over stride-4 tokens carrying the tip heatmap, on the frozen `detector_tips4` model | validation 3.13; held-out 3.65 soft, 3.56 hard | trained |

### Held-out results at the intended framing

The model is meant for frames in which the board just about fills the
frame, with clearance for the number ring. The renderer samples a wider
range -- the full board, number ring included, spans 62–105% of the frame
width, the range measured on 49 real captures -- so the held-out frames are
restricted to those where it spans at least 88% (measured along the board
image's major axis, which perspective does not shorten): 421 of the 1000
frames, 877 darts. Results over all framings follow at the end.

A dart's case is set by the distance to its nearest neighbour's landing
point: 333 darts lie under 30 px from one, 178 at 30–60 px and 242 at
120 px or more; 53 have the tip hidden. Only 8 have a coinciding tip (under
4 px), too few for a column.

Landing error in pixels, penalised (a missed dart counts 40 px) except for
the median, which is over the matched darts only. Pixels are image pixels;
a larger board makes the same error on the board more pixels, so framings
are compared in millimetres on the board, as the notes below do.

| Readout | median | all | <30 px | 30–60 px | isolated ≥120 px | tip hidden |
|---|---|---|---|---|---|---|
| Hough (`detector.yaml`) | 2.63 | 4.28 | 5.39 | 4.84 | 2.55 | 10.48 |
| Hough + tip peaks, Hungarian (`detector_tips4.yaml`) | **1.14** | **2.77** | **3.76** | **2.66** | **1.60** | 10.14 |
| Token readout, soft (`detector_readout*.yaml`) | 1.56 | 3.65 | 4.44 | 4.03 | 2.36 | 11.20 |
| Token readout, hard (`detector_readout*.yaml`) | 1.41 | 3.56 | 4.15 | 4.09 | 2.25 | 13.01 |
| YOLO26s-pose baseline (see below) | 2.15 | 4.55 | 6.99 | 4.01 | 2.15 | **7.97** |

Detection. A predicted dart is correct when it is matched, one-to-one, to a
ground-truth dart within 40 px; FP counts the rest, over the 421 frames.
The penalised error above charges misses but not false positives.

| Readout | recall | precision | FP | missed: <30 px | missed: tip hidden |
|---|---|---|---|---|---|
| Hough (`detector.yaml`) | 98.7% | 95.5% | 41 | 2.1% | 5.7% |
| Hough + tip peaks, Hungarian (`detector_tips4.yaml`) | **99.1%** | 93.8% | 57 | 1.5% | 3.8% |
| Token readout, soft (`detector_readout*.yaml`) | **99.1%** | 98.4% | 14 | **0.6%** | **1.9%** |
| Token readout, hard (`detector_readout*.yaml`) | 99.0% | 98.3% | 15 | **0.6%** | 3.8% |
| YOLO26s-pose baseline (see below) | 95.9% | **99.8%** | **2** | 9.3% | 3.8% |

Scoring. The share of ground-truth darts whose predicted landing point
falls in the right scoring zone, ring and segment, mapped to the board
through the ground-truth homography fitted to the frame's double-ring
corners, so every readout, including the baseline, is scored with the same
board. A missed dart counts as wrong. Through the same homographies the
ground-truth landing points reproduce the renderer's own zones for 99.8% of
darts; the rest lie within rounding of a wire. Grouped turns aim at
trebles and the bull, so many darts sit within a few pixels of a wire,
where a pixel or two decides the zone. The last column re-scores each
readout with its own errors -- each matched dart's error vector, applied to
five darts placed uniformly over the scoring area of the same frame -- to
show the same detectors under ordinary scatter.

| Readout | all | <30 px | isolated ≥120 px | tip hidden | uniform placement |
|---|---|---|---|---|---|
| Hough (`detector.yaml`) | 86.2% | 78.7% | 93.0% | **75.5%** | 89.4% |
| Hough + tip peaks, Hungarian (`detector_tips4.yaml`) | **90.9%** | **84.7%** | **96.3%** | 71.7% | **93.0%** |
| Token readout, soft (`detector_readout*.yaml`) | 87.6% | 80.2% | 95.9% | 64.2% | 91.0% |
| Token readout, hard (`detector_readout*.yaml`) | 88.4% | 82.0% | 95.0% | 62.3% | 91.3% |
| YOLO26s-pose baseline (see below) | 84.0% | 74.2% | 92.6% | 73.6% | 87.8% |

Strengths and weaknesses:

- **Hough.** The simplest readout, stable in fp16. The least precise:
  every landing point is an average of per-cell regressions, about 2.6 px
  median. Every patch of foreground votes becomes a detection, so it has
  more spurious darts than the token readout (41 against 14–15). Over all
  framings it is the best on the 22 coinciding tips, which it neither has
  to separate nor assign.
- **Hough + tip peaks.** The stride-4 tip heatmap has a peak for about nine
  darts in ten, and taking it cuts the median error of the same model's
  own Hough readout from 2.90 to 1.14 px (isolated darts: 1.76 to 0.79);
  `detector_tips4.yaml` fine-tunes the whole base network, so that Hough
  readout differs slightly from the base model's. Best overall, in tight
  clusters, on separated darts and in scoring. Its median error on the
  board falls from 1.00 mm on boards spanning under 80% of the frame width
  to 0.67 mm here, as the base model's does (2.07 to 1.59 mm). The
  assignment is one-to-one within a fixed 40 px gate, so coinciding tips
  share one peak; over all framings it is worse than plain Hough on them. The most spurious darts (57), and the
  assignment runs outside the graph.
- **Token readout.** Hough and assignment happen in the graph, with
  learned queries that are Hungarian-matched in training, so spurious darts
  are penalised during training: 14–15, the fewest of this package's
  readouts.
  The fewest misses in tight clusters (0.6%) and, soft, of hidden tips.
  Less precise than the tip peaks, mostly from choosing between a token and
  the dense estimate. It is the one readout that gets worse on the largest
  boards: its median error on the board falls from 0.98 mm under 80% of the
  frame width to 0.73 mm at 80–88%, then rises to 0.81 mm here (hard). A
  fixed attention window over larger darts would explain it; that is not
  verified. Across 108 slots on 49 real photographs its fp16 output
  differed from fp32 by 0.06 px typically and about 90 px at worst, as
  near-equal options flip under rounding, so it exports at fp32, which does
  not run on the Neural Engine.
  Learned corrections on top of the pointed-at estimates contributed about
  0.2 px.
- **YOLO26s-pose baseline.** An off-the-shelf comparison, not part of this
  package: Ultralytics' COCO-pretrained `yolo26s-pose` (10.6M parameters)
  fine-tuned for 100 epochs with its default settings on 20,000 fixed
  frames from this renderer -- about what could be annotated by hand -- as
  one class with two keypoints, landing point and flight end
  (`tools/yolo_dataset.py`, `tools/yolo_predict.py`), and read at its
  default confidence of 0.25. Its learned one-detection-per-dart head gives
  almost no spurious darts, and it has the lowest landing error on hidden
  tips. Its axis-aligned boxes cannot separate overlapping darts: it misses
  9.3% of darts in tight clusters, against 0.6–2.1% here. Its median error
  on the board hardly changes with framing (1.31 mm under 80% of the frame
  width, 1.25 mm here), where the dense readouts gain from the larger
  image. It has no board keypoints.

Over all 1000 held-out frames, every framing included (scoring: as
rendered, and under uniform placement):

| Readout | penalised | median | recall | FP | scoring | uniform placement |
|---|---|---|---|---|---|---|
| Hough (`detector.yaml`) | 4.44 | 2.52 | 98.0% | 68 | 84.3% | 88.9% |
| Hough + tip peaks, Hungarian (`detector_tips4.yaml`) | **3.12** | **1.16** | 98.2% | 95 | **89.5%** | **93.0%** |
| Token readout, soft (`detector_readout*.yaml`) | 3.35 | 1.34 | **98.7%** | 27 | 88.4% | 92.4% |
| Token readout, hard (`detector_readout*.yaml`) | 3.32 | 1.24 | **98.7%** | 26 | 88.9% | 92.0% |
| YOLO26s-pose baseline | 4.17 | 1.92 | 95.9% | **4** | 85.6% | 91.5% |

## Trained weights

Weights for the base model and the two variants whose held-out results are
reported above, plus the pretrained backbone, are attached to the
[`weights-v1` release](https://github.com/ahie/darts-model/releases/tag/weights-v1):

| File | Config | Readout | Epoch | Size |
|---|---|---|---|---|
| `backbone.ckpt` | `configs/pretrain.yaml` | — | 272 | 14 MB |
| `detector.ckpt` | `configs/detector.yaml` | Hough | 768 | 20 MB |
| `detector_tips4.ckpt` | `configs/detector_tips4.yaml` | Hough + tip peaks | 59 | 20 MB |
| `detector_readout.ckpt` | `configs/detector_readout_long.yaml` | token readout | 179 | 22 MB |

```sh
mkdir -p weights && cd weights
for f in backbone detector detector_tips4 detector_readout; do
  curl -LO https://github.com/ahie/darts-model/releases/download/weights-v1/$f.ckpt
done
shasum -a 256 -c ../weights.sha256
```

The detector files hold the model weights only, without optimizer state, so
they load wherever a checkpoint is taken -- the exporters,
`darts-visualize` and a config's `init_weights` -- but cannot resume a run.
Each must be paired with the config in its row: a config that describes a
different network refuses to load it. `detector_readout.ckpt` contains the
whole `detector_tips4` network, frozen, with the token readout on top.
`backbone.ckpt` is what `backbone_weights` loads, so `configs/detector.yaml`
can be trained without first running the pretraining stage. The configs
point at the paths their own runs wrote (`backbone_weights:
"outputs/backbone.ckpt"` in `configs/detector.yaml`, `init_weights` in the
fine-tuning configs); to train from the downloaded files, point those at
`weights/` instead.

## Export

```sh
pip install -e ".[coreml]"
darts-export-coreml CHECKPOINT --config configs/detector.yaml \
    --out darts.mlpackage --sample frame.jpg

pip install -e ".[litert]"
darts-export-litert CHECKPOINT --config configs/detector.yaml \
    --out darts.tflite --sample frame.jpg
```

The config must be the one the checkpoint was trained with (see
[Trained weights](#trained-weights)). Export the token readout with
`--precision fp32`: under fp16 its slots can flip between near-equal
options, and the exporter's comparison against torch fails.

Both exporters run the converted model and the torch model on one frame and
compare them before writing anything. `--sample` is that frame: an image
showing darts in a board, centre-cropped and resized to the input size, as
an app should feed the model. Without `--sample` a frame is rendered, which needs the
renderer module on `PYTHONPATH`; if neither is available the export stops.
The LiteRT exporter writes `darts.tflite` together with
`darts.outputs.json`, which an app needs to find the outputs (see below),
or neither.

`--size` overrides the config's `image_size`; it must be a multiple of the
output stride (8). `--precision fp32` (CoreML) disables fp16. The default
fp16 keeps the keypoint head at fp32, because rounding its weights collapses
the confidence of a block of corners.

## Using the exported model

**Input**: one S×S RGB image, S = `input_size` (1024 by default), values
0–255. ImageNet normalisation is inside the graph; do not normalise in the
app. CoreML takes an image (`image`); LiteRT a float32 tensor (1, 3, S, S).

**Outputs**. All coordinates are normalised to the input square, x right,
y down; multiply by S for pixels. K = `topk` (1024, fewer for inputs under
256 px).

| Name | Shape | Contents |
|---|---|---|
| `dart_score` | (1, K) | foreground probability of the top-K cells, descending |
| `dart_centre` | (1, K, 2) | the box centre each cell predicts |
| `dart_direction` | (1, K, 2) | unit vector, landing point → flight |
| `dart_extent` | (1, K, 2) | box half-length, half-width |
| `dart_tip` | (1, K, 2) | predicted landing point (what scoring uses) |
| `dart_flight` | (1, K, 2) | predicted flight end |
| `kp_xy` | (1, 40, 2) | board corner positions |
| `kp_conf` | (1, 40) | board corner confidences |

These eight are common to every variant. The variants add outputs after
them, and read their darts differently:

| Variant | Further outputs | Darts |
|---|---|---|
| Hough (`detector.yaml`) | — | the [dart readout](#dart-readout) below |
| Hough + tip peaks (`detector_tips4.yaml`) | `tip_xy` (1, P, 2), `tip_score` (1, P) | the dart readout, then [tip assignment](#tip-assignment) |
| Token readout (`detector_readout_long.yaml`) | `slot_score` (1, Q), `slot_tip` (1, Q, 2), `slot_flight` (1, Q, 2), `slot_tip_hard` (1, Q, 2), `tip_xy`, `tip_score` | the [slots](#slots), no readout in the app |

P = 16 tip peaks, Q = 3 slots; both are in `output_shapes`.

CoreML outputs are looked up by name. LiteRT outputs are positional and
their order is not fixed, so read `outputs` (name → output index) from the
sidecar.

**Metadata**. The CoreML package's `user_defined_metadata` and the LiteRT
sidecar carry the same fields: `input_size`, `topk`, `output_stride`, `grid`,
`readout` (`vote_bin_px`, `fg_threshold`, `peak_min_votes`, `vote_bins`),
`kp_names`, `output_shapes`, `normalisation` and `coordinates`. In CoreML
each non-string value is a JSON string. Read these rather than hardcoding
them; they come from the config the model was trained with.

<a id="dart-readout"></a>**Dart readout** (`src/darts_model/model/hough.py`, over the K cells instead
of the full grid). With τ = `fg_threshold`, P = `peak_min_votes` and
B = `vote_bins` (= round(S / `vote_bin_px`)):

1. *Voters*: the cells with `dart_score` ≥ τ. Scores are descending, so stop
   at the first one below. If `dart_score[K-1]` ≥ τ the frame filled every
   slot and some voters were cut.
2. *Vote*: each voter adds its score to bin
   (clamp(⌊y·B⌋, 0, B−1), clamp(⌊x·B⌋, 0, B−1)) of a B×B float32
   accumulator, at its `dart_centre` (x, y).
3. *Peaks*: bins with a value ≥ P·τ that are the maximum of their 3×3
   neighbourhood, ties broken by raster order: a peak must be strictly
   greater than the neighbours before it in row-major order (the row above,
   and the bin to its left) and ≥ the others. Neighbours outside the grid
   are ignored.
4. *Claim*: visit peaks by descending value, equal values in row-major
   order. A peak's voters are the not-yet-claimed voters whose bin is within
   one bin of it in both x and y. If there are fewer than P, skip the peak
   and claim nothing; otherwise claim them all.
5. *Detection*: over the claimed voters, with weights w = `dart_score`:
   centre, extent, landing point and flight are the w-weighted means of
   `dart_centre`, `dart_extent`, `dart_tip` and `dart_flight`; the direction
   is the w-weighted sum of `dart_direction`, renormalised to unit length;
   the score is Σw / n and the vote count n. The box's own ends are
   centre ∓ direction × half-length, but the landing point is `dart_tip`:
   for a dart pointing toward the camera it lies inside the silhouette.
6. Sort detections by vote count, descending.

<a id="tip-assignment"></a>**Tip assignment** (Hough + tip peaks; the metadata's
`tips` field, `assign` = `hungarian`). `tip_xy` are the tip heatmap's peaks
with their sub-cell offsets, `tip_score` their probabilities, descending, 0
for an unused entry. Take the peaks scoring ≥ `snap_score`, and assign them
to the dart readout's detections one-to-one, minimising the total distance
from each detection's landing point, with no pair further apart than
`assign_gate_px` (in pixels at `input_size`) -- e.g. the Hungarian algorithm
on a cost matrix whose gated-out entries are prohibitive. A detection that
receives a peak takes it as its landing point; one that receives none keeps
its own. Two darts whose tips coincide share a single peak, so only one of
them gets it.

<a id="slots"></a>**Slots** (token readout; the metadata's `darts` field).
Slot i is a dart when `slot_score[i]` ≥ `min_score`; its landing point is
`slot_tip[i]` and its flight end `slot_flight[i]`. `slot_tip_hard[i]` is the
same dart's landing point taken from the readout's single highest-weighted
option instead of the blend it was trained with (held-out: 3.56 against
3.65, see above). No Hough voting or assignment is needed; the per-cell
outputs and the tip peaks remain for debugging. At most Q darts are read
per frame.

**Board corners**: `kp_xy[i]` is corner `kp_names[i]`, the 40 double-ring
corners in `BOARD_KEYPOINT_NAMES[41:81]` (`src/darts_model/board_geometry.py`):
first where each of the 20 radial wires crosses the outer double circle,
then the inner, each clockwise from the wire between 20 and 1
(`double_outer_20_1`, `double_outer_1_18`, …, `double_inner_5_20`). Gate on
`kp_conf` and fit the board homography robustly (e.g. RANSAC) from the
corners that pass.

## Standalone dataset generation

```sh
./build/dartboard_gen --num_frames 1000 --output_dir data --seed 42 \
    --bg_image_dir bg_images
```

Without `--bg_image_dir` the backgrounds are procedural noise rather than
photographs. `./build/dartboard_gen --help` lists the other options (size,
GPU, thread count, asset directory).

The output is `rgb/NNNNNN.jpg`, `annotations/NNNNNN.json` and a
`manifest.json`. Images are sRGB. Pixel coordinates in the annotation are at
the image resolution, origin top left:

- `board_keypoints`: all 81 named board keypoints (`name`, `x`, `y`), in the
  order of `BOARD_KEYPOINT_NAMES`.
- `darts`: per dart, the landing point (`x`, `y`), the flight end
  (`flight_x`, `flight_y`, `flight_in_front`), the oriented box's four
  corners (`box_x`, `box_y`, `box_end_on`) and the `score_zone`.
- `camera_params`: `focal_length`, `h_aperture_mm`, `resolution`, `near`,
  `far` and the column-major `view_matrix`.
- `metadata`: dart count, board rotation and lighting.

## Tests

```sh
pytest
ctest --test-dir build -C Release
```

## License

The code is licensed under the [Apache License 2.0](LICENSE); see also
[NOTICE](NOTICE). The renderer's bundled and fetched third-party libraries
carry their own permissive licences (MIT, BSD or public domain).

The [trained weights](#trained-weights) are released under the same licence.
They were trained on renders composited over Places365 photographs, whose
terms of use limit the images to non-commercial research and education. The
images are not redistributed here; whether those terms reach weights trained
on them is unsettled, as it is for most models pretrained on public image
datasets.

`tools/yolo_predict.py` is an optional baseline tool that imports
Ultralytics, which is AGPL-3.0 and installed separately; nothing in the
package depends on it.
