import pytest
import torch
from torch import nn

from darts_model.export.common import ExportError
from darts_model.export.groupnorm import (
    GroupNormExplicit,
    rewrite_group_norms,
)


def _activation(n=2, c=12, h=5, w=7):
    """Non-constant, and not zero-mean: a constant frame makes every group's
    standard deviation zero and the comparison vacuous."""
    torch.manual_seed(0)
    return torch.randn(n, c, h, w) * 3.0 + 1.5


@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("groups", [1, 3, 12])
def test_matches_group_norm(affine, groups):
    source = nn.GroupNorm(groups, 12, affine=affine)
    if affine:
        with torch.no_grad():
            source.weight.uniform_(0.5, 1.5)
            source.bias.uniform_(-1.0, 1.0)
    x = _activation()
    with torch.no_grad():
        assert torch.allclose(GroupNormExplicit(source)(x), source(x), atol=1e-5)


def test_affine_is_stored_broadcast_shaped():
    """The whole point of the rewrite: a (1, C, 1, 1) constant needs no
    reshape, and the reshape is what the converter turned into a gather."""
    explicit = GroupNormExplicit(nn.GroupNorm(3, 12))
    assert explicit.w.shape == (1, 12, 1, 1)
    assert explicit.b.shape == (1, 12, 1, 1)


def test_output_stays_four_dimensional():
    """A 5-D intermediate is refused by the WebGPU delegate outright, which
    cost more than the gather did. The group must fold into the batch."""
    captured = []
    original = torch.Tensor.reshape

    def spy(self, *shape):
        out = original(self, *shape)
        captured.append(out.dim())
        return out

    x = _activation()
    with torch.no_grad():
        torch.Tensor.reshape = spy
        try:
            GroupNormExplicit(nn.GroupNorm(3, 12))(x)
        finally:
            torch.Tensor.reshape = original
    assert captured and max(captured) <= 4


def test_rejects_channels_not_divisible():
    source = nn.GroupNorm(1, 12)
    source.num_groups = 5
    with pytest.raises(ExportError, match="divisible"):
        GroupNormExplicit(source)


class Nested(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(12, 12, 1), nn.GroupNorm(3, 12))
        self.inner = nn.ModuleList([nn.GroupNorm(4, 12), nn.Conv2d(12, 12, 1)])
        self.norm = nn.GroupNorm(2, 12)

    def forward(self, x):
        return self.norm(self.inner[0](self.head(x)))


def test_rewrites_every_nesting_level():
    model = Nested().eval()
    x = _activation()
    swapped = rewrite_group_norms(model, lambda: model(x), log=lambda _: None)
    assert swapped == 3
    assert not [m for m in model.modules() if isinstance(m, nn.GroupNorm)]


def test_returns_zero_when_there_is_nothing_to_rewrite():
    model = nn.Sequential(nn.Conv2d(12, 12, 1)).eval()
    x = _activation()
    assert rewrite_group_norms(model, lambda: model(x), log=lambda _: None) == 0


def test_accepts_a_dict_probe():
    model = Nested().eval()
    x = _activation()
    assert rewrite_group_norms(
        model, lambda: {"a": model(x), "b": model(x) * 2},
        log=lambda _: None) == 3


def test_raises_when_the_rewrite_changes_the_function(monkeypatch):
    """The guard exists for a rewrite that is subtly wrong -- the unbiased
    variance, say, which differs by m/(m-1) and passes any smoke test. If this
    test fails, nothing is checking that."""
    class Wrong(GroupNormExplicit):
        def forward(self, x):
            n, c, h, w = x.shape
            grouped = x.reshape(n * self.groups, c // self.groups, h, w)
            mean = grouped.mean(dim=(1, 2, 3), keepdim=True)
            var = grouped.var(dim=(1, 2, 3), keepdim=True, unbiased=True)
            normed = ((grouped - mean) / torch.sqrt(var + self.eps)).reshape(n, c, h, w)
            return normed * self.w + self.b if self.affine else normed

    monkeypatch.setattr("darts_model.export.groupnorm.GroupNormExplicit", Wrong)
    model = Nested().eval()
    x = _activation()
    with pytest.raises(ExportError, match="not the same function"):
        rewrite_group_norms(model, lambda: model(x), log=lambda _: None)


def test_rejects_a_probe_that_does_not_return_the_same_tensors():
    model = Nested().eval()
    x = _activation()
    calls = []

    def probe():
        calls.append(None)
        return model(x) if len(calls) == 1 else (model(x), model(x))

    with pytest.raises(ExportError, match="must return the same"):
        rewrite_group_norms(model, probe, log=lambda _: None)


def test_rejects_a_probe_whose_shape_changes():
    model = Nested().eval()
    calls = []

    def probe():
        calls.append(None)
        return model(_activation(h=5 if len(calls) == 1 else 6))

    with pytest.raises(ExportError, match="before the rewrite"):
        rewrite_group_norms(model, probe, log=lambda _: None)
