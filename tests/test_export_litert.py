"""`resolve_outputs`: which converted LiteRT output is which named plane.

Pure numpy, so it runs without the LiteRT packages. Five of the eight planes
share the shape (1, K, 2); a wrong assignment is not an error anybody would
see, so these pin that the resolution works by value, survives the rank
reordering topk produces across backends, and reports a near-tie as one.
"""
from __future__ import annotations

import numpy as np

from darts_model.export.common import OUTPUTS
from darts_model.export.litert import MIN_MARGIN, resolve_outputs


def _reference(seed=0, k=128, kept=30):
    """A plausible frame: three darts, each with its cells agreeing on one
    centre/tip/flight up to a little scatter, and background cells with
    arbitrary values."""
    rng = np.random.default_rng(seed)
    score = np.sort(rng.uniform(0.0, 0.4, k))[::-1].copy()
    score[:kept] = np.linspace(0.97, 0.55, kept)
    dart = rng.integers(0, 3, kept)
    centres = rng.uniform(0.3, 0.7, (3, 2))
    dirs = rng.normal(size=(3, 2))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    half = 0.06

    def per_cell(fg_value, spread):
        a = rng.uniform(0, 1, (k, 2))
        a[:kept] = fg_value[dart] + rng.normal(0, spread, (kept, 2))
        return a[None]

    planes = {
        "dart_score": score[None],
        "dart_centre": per_cell(centres, 0.002),
        "dart_direction": per_cell(dirs, 0.01),
        "dart_extent": per_cell(np.tile([half, 0.015], (3, 1)), 0.001),
        "dart_tip": per_cell(centres - dirs * half, 0.002),
        "dart_flight": per_cell(centres + dirs * half, 0.002),
        "kp_xy": rng.uniform(0, 1, (1, 40, 2)),
        "kp_conf": rng.uniform(0, 1, (1, 40)),
    }
    reference = [planes[n] for n in OUTPUTS]
    return reference, score >= 0.5


def test_a_shuffled_output_order_is_recovered() -> None:
    reference, keep = _reference()
    order = [5, 2, 7, 0, 4, 1, 6, 3]           # converted[i] = reference[order[i]]
    converted = [reference[j] for j in order]
    mapping, report = resolve_outputs(converted, reference, keep)
    assert mapping == {OUTPUTS[j]: i for i, j in enumerate(order)}
    assert all(m >= MIN_MARGIN for _, _, _, m in report)


def test_reordered_ranks_still_resolve() -> None:
    """Near-equal scores come back in another order on another backend. Every
    per-cell plane is permuted the same way, and the mapping must still be
    found, with a margin, rather than every plane looking equally wrong."""
    reference, keep = _reference(seed=1)
    rng = np.random.default_rng(7)
    perm = np.arange(keep.size)
    perm[:int(keep.sum())] = rng.permutation(int(keep.sum()))
    shuffled = [p[:, perm] if p.shape[1] == keep.size else p
                for p in reference]
    mapping, report = resolve_outputs(shuffled, reference, keep)
    assert mapping == {n: i for i, n in enumerate(OUTPUTS)}
    for _, _, err, margin in report:
        assert err < 1e-9
        assert margin >= MIN_MARGIN


def test_small_conversion_error_keeps_a_wide_margin() -> None:
    reference, keep = _reference(seed=2)
    rng = np.random.default_rng(3)
    converted = [p + rng.normal(0, 1e-5, p.shape) for p in reference]
    mapping, report = resolve_outputs(converted, reference, keep)
    assert mapping == {n: i for i, n in enumerate(OUTPUTS)}
    assert all(m >= MIN_MARGIN for _, _, _, m in report)


def test_indistinguishable_planes_report_a_low_margin() -> None:
    """Tip and flight identical on this frame: the resolution is a guess, and
    the margin must say so."""
    reference, keep = _reference(seed=3)
    ti, fi = OUTPUTS.index("dart_tip"), OUTPUTS.index("dart_flight")
    reference[fi] = reference[ti].copy()
    _, report = resolve_outputs(list(reference), reference, keep)
    margins = {name: m for _, name, _, m in report}
    assert min(margins["dart_tip"], margins["dart_flight"]) < MIN_MARGIN
