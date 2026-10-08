"""Voting readout: does a field of per-cell boxes resolve into darts?

Tested with synthetic fields rather than a trained model, because the readout
must be correct independently of whether a network has learned anything. If
these pass and a real run still fails, the fault is in the predictions, not in
the readout.
"""
from dataclasses import dataclass

import pytest
import torch


from darts_model.model.hough import detect, local_maxima, vote


@dataclass
class _Cfg:
    out_stride: int = 8
    vote_bin_px: float = 4.0
    fg_threshold: float = 0.5
    peak_min_votes: int = 8


def _field(h, w, darts, noise_cells=0, seed=0):
    """A dense field where the cells of each dart all vote for its box."""
    torch.manual_seed(seed)
    fg = torch.zeros(h, w)
    centre = torch.zeros(2, h, w)
    direction = torch.zeros(2, h, w)
    direction[0] = 1.0
    extent = torch.zeros(2, h, w)
    for (cx, cy, dx, dy, hl, hw_, cells) in darts:
        n = 0
        for yy in range(h):
            for xx in range(w):
                if n >= cells:
                    break
                # put the dart's cells in a band near its centre
                if abs((xx + 0.5) / w - cx) < 0.06 and abs((yy + 0.5) / h - cy) < 0.03:
                    fg[yy, xx] = 0.9
                    centre[0, yy, xx] = cx
                    centre[1, yy, xx] = cy
                    direction[0, yy, xx] = dx
                    direction[1, yy, xx] = dy
                    extent[0, yy, xx] = hl
                    extent[1, yy, xx] = hw_
                    n += 1
    for _ in range(noise_cells):
        yy = torch.randint(0, h, (1,)).item()
        xx = torch.randint(0, w, (1,)).item()
        if fg[yy, xx] > 0:
            continue
        fg[yy, xx] = 0.9
        centre[0, yy, xx] = torch.rand(1).item()
        centre[1, yy, xx] = torch.rand(1).item()
        extent[:, yy, xx] = torch.tensor([0.07, 0.017])
    return ({"centre": centre, "direction": direction, "extent": extent}, fg)


# ------------------------------------------------------------- the primitives

def test_vote_accumulates_into_the_right_bin():
    c = torch.tensor([[0.25, 0.75], [0.25, 0.75], [0.9, 0.1]])
    acc = vote(c, torch.ones(3), bins=4)
    assert acc[3, 1] == 2.0      # [y, x]
    assert acc[0, 3] == 1.0
    assert acc.sum() == 3.0


def test_local_maxima_is_a_3x3_comparison():
    acc = torch.zeros(9, 9)
    acc[4, 4] = 10.0
    acc[4, 5] = 6.0              # adjacent, must be suppressed
    acc[8, 0] = 9.0              # far, must survive
    m = local_maxima(acc, min_value=5.0)
    assert m[4, 4] and m[8, 0]
    assert not m[4, 5]


def test_local_maxima_respects_the_floor():
    acc = torch.zeros(5, 5)
    acc[2, 2] = 3.0
    assert not local_maxima(acc, min_value=5.0).any()


# ---------------------------------------------------------------- detection

def test_one_dart_gives_one_detection():
    dec, fg = _field(64, 64, [(0.5, 0.5, 1.0, 0.0, 0.08, 0.018, 40)])
    got = detect(dec, fg, _Cfg())
    assert len(got) == 1
    assert got[0]["centre"][0] == pytest.approx(0.5, abs=0.01)
    assert got[0]["votes"] >= 8


def test_many_cells_collapse_to_one_detection():
    """Duplicate suppression is arithmetic here: agreeing votes make one peak
    however many cells there are. Nothing has to learn it."""
    few = detect(*_field(64, 64, [(0.5, 0.5, 1.0, 0.0, 0.08, 0.018, 20)]),
                 cfg=_Cfg())
    many = detect(*_field(64, 64, [(0.5, 0.5, 1.0, 0.0, 0.08, 0.018, 90)]),
                  cfg=_Cfg())
    assert len(few) == len(many) == 1


def test_two_separated_darts_give_two_detections():
    dec, fg = _field(64, 64, [(0.30, 0.30, 1.0, 0.0, 0.08, 0.018, 40),
                              (0.70, 0.70, 0.0, 1.0, 0.08, 0.018, 40)])
    got = detect(dec, fg, _Cfg())
    assert len(got) == 2
    cx = sorted(float(g["centre"][0]) for g in got)
    assert cx[0] == pytest.approx(0.30, abs=0.02)
    assert cx[1] == pytest.approx(0.70, abs=0.02)


def test_tip_and_flight_come_out_of_the_box():
    """The tip is a derived quantity -- one end of the centreline -- not a
    separately predicted point."""
    dec, fg = _field(64, 64, [(0.5, 0.5, 1.0, 0.0, 0.08, 0.018, 40)])
    d = detect(dec, fg, _Cfg())[0]
    assert d["tip"][0] == pytest.approx(0.42, abs=0.01)
    assert d["flight"][0] == pytest.approx(0.58, abs=0.01)


def test_hedging_cells_do_not_create_a_false_detection():
    """The property voting exists for. Cells in an overlap that split the
    difference disagree with each other, so they scatter instead of
    accumulating -- no rule is needed to resolve them."""
    dec, fg = _field(64, 64, [(0.25, 0.5, 1.0, 0.0, 0.08, 0.018, 40),
                              (0.75, 0.5, 1.0, 0.0, 0.08, 0.018, 40)],
                     noise_cells=60, seed=3)
    got = detect(dec, fg, _Cfg())
    assert len(got) == 2, [round(float(g["centre"][0]), 3) for g in got]


def test_a_weakly_supported_dart_is_dropped_by_the_vote_floor():
    dec, fg = _field(64, 64, [(0.5, 0.5, 1.0, 0.0, 0.08, 0.018, 3)])
    assert detect(dec, fg, _Cfg(peak_min_votes=8)) == []


def test_empty_field_detects_nothing():
    dec, fg = _field(64, 64, [])
    assert detect(dec, fg, _Cfg()) == []


def test_direction_is_averaged_as_a_unit_vector():
    dec, fg = _field(64, 64, [(0.5, 0.5, 0.6, 0.8, 0.08, 0.018, 40)])
    d = detect(dec, fg, _Cfg())[0]
    assert float(d["direction"].norm()) == pytest.approx(1.0, abs=1e-5)
    assert float(d["direction"][0]) == pytest.approx(0.6, abs=0.02)


def test_tied_neighbours_make_one_peak():
    """Two adjacent bins with exactly equal votes are one plateau, not two
    peaks: ties go to the first bin in raster order."""
    acc = torch.zeros(9, 9)
    acc[4, 4] = acc[4, 5] = 10.0
    acc[5, 4] = acc[5, 5] = 10.0        # a 2x2 plateau
    m = local_maxima(acc, min_value=5.0)
    assert int(m.sum()) == 1
    assert m[4, 4]


def test_no_two_peaks_are_ever_adjacent():
    torch.manual_seed(0)
    # Coarse values so ties are common.
    acc = torch.randint(0, 4, (32, 32)).float()
    m = local_maxima(acc, min_value=1.0)
    ys, xs = torch.nonzero(m, as_tuple=True)
    for y, x in zip(ys.tolist(), xs.tolist()):
        window = m[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2]
        assert int(window.sum()) == 1, (y, x)


def test_a_dart_whose_votes_tie_across_a_bin_edge_is_one_detection():
    """Half the cells vote into one bin and half into the next, equally
    weighted: the exact tie that used to yield two peaks, i.e. two detections
    of one dart."""
    h = w = 64
    fg = torch.zeros(h, w)
    centre = torch.zeros(2, h, w)
    direction = torch.zeros(2, h, w)
    direction[0] = 1.0
    extent = torch.zeros(2, h, w)
    extent[0], extent[1] = 0.08, 0.018
    # 64 * 8 = 512 px, 4 px bins -> 128 bins; bin 64 starts at x = 0.5.
    fg[30:32, 20:30] = 0.9
    centre[0, 30] = 0.5 - 1e-4
    centre[0, 31] = 0.5 + 1e-4
    centre[1] = 0.5 + 1e-4
    got = detect({"centre": centre, "direction": direction, "extent": extent},
                 fg, _Cfg())
    assert len(got) == 1
    assert got[0]["votes"] == 20


def test_a_voter_is_counted_by_one_peak_only():
    """Two peaks two bins apart share the column of bins between them. Those
    voters belong to the stronger peak; the weaker keeps only its own, and if
    that leaves it under the vote floor it is not a detection at all."""
    h = w = 64
    bins = 128                          # 64 cells * stride 8 / 4 px
    fg = torch.zeros(h, w)
    centre = torch.zeros(2, h, w)
    direction = torch.zeros(2, h, w)
    direction[0] = 1.0
    extent = torch.zeros(2, h, w)
    extent[0], extent[1] = 0.08, 0.018

    def put(row, n, bx):
        fg[row, :n] = 0.9
        centre[0, row, :n] = (bx + 0.5) / bins
        centre[1, row, :n] = (60 + 0.5) / bins

    put(0, 12, 60)     # strong peak, bin 60
    put(1, 6, 61)      # shared column between the two peaks
    put(2, 10, 62)     # weak peak, bin 62: 10 own votes, 16 with the shared
    dec = {"centre": centre, "direction": direction, "extent": extent}

    got = detect(dec, fg, _Cfg(peak_min_votes=8))
    assert sorted(g["votes"] for g in got) == [10, 18]
    assert sum(g["votes"] for g in got) == 28      # every voter counted once

    # Under a floor of 12 the weak peak's 10 own votes are not enough; the
    # shared voters cannot prop it up.
    got = detect(dec, fg, _Cfg(peak_min_votes=12))
    assert [g["votes"] for g in got] == [18]


def test_box_ends_are_reported_alongside_the_predicted_ends():
    dec, fg = _field(64, 64, [(0.5, 0.5, 1.0, 0.0, 0.08, 0.018, 40)])
    h, w = fg.shape
    dec["tip_point"] = torch.tensor([0.40, 0.52]).view(2, 1, 1).expand(2, h, w)
    dec["flight_point"] = torch.tensor([0.60, 0.5]).view(2, 1, 1).expand(2, h, w)
    d = detect(dec, fg, _Cfg())[0]
    assert d["tip_is_predicted"]
    assert torch.allclose(d["tip"], torch.tensor([0.40, 0.52]), atol=1e-5)
    assert torch.allclose(d["box_tip"], torch.tensor([0.42, 0.5]), atol=1e-3)
    assert torch.allclose(d["box_flight"], torch.tensor([0.58, 0.5]), atol=1e-3)
