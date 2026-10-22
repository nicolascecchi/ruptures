"""Tests for the compiled Segment Neighborhood backend (ruptures._core)."""

from unittest import mock

import pytest

from ruptures.datasets import pw_constant
from ruptures.detection import Dynp

_core = pytest.importorskip("ruptures._core", reason="the compiled core is not built")


@pytest.fixture(scope="module")
def signal_1D():
    signal, _ = pw_constant(n_samples=200, n_bkps=3, noise_std=1, seed=42)
    return signal


def pure_python_bkps(signal, n_bkps, min_size=1, jump=1):
    """Breakpoints of ``Dynp(model="l2", min_size, jump)``, Python only.

    ``Dynp`` delegates built-in costs to the compiled backend, so the
    module global is disabled for the duration of the call to exercise
    the reference implementation instead.
    """
    with mock.patch("ruptures.detection.dynp._core", None):
        dynp = Dynp(model="l2", min_size=min_size, jump=jump)
        return dynp.fit_predict(signal, n_bkps)


def test_core_importable():
    from ruptures import _core

    assert hasattr(_core.algorithms, "SegmentNeighborhood")


def test_dynp_delegates_to_compiled_backend(signal_1D):
    """A valid built-in-cost configuration runs the compiled backend."""
    calls = []
    compiled_bkps = Dynp._compiled_bkps

    def spy(self, model, n_bkps):
        calls.append((model is not None, n_bkps))
        return compiled_bkps(self, model, n_bkps)

    with mock.patch.object(Dynp, "_compiled_bkps", spy):
        Dynp(model="l2", min_size=1, jump=1).fit_predict(signal_1D, 3)
    assert calls == [(True, 3)]


def test_dynp_does_not_delegate_invalid_grid():
    """An invalid grid stays on the Python path."""
    assert not Dynp(model="l2", jump=-1)._use_compiled


@pytest.mark.parametrize(
    "n_bkps,min_size,jump", [(1, 1, 1), (2, 1, 1), (3, 2, 5), (2, 7, 3)]
)
def test_matches_pure_python_dynp_configs(signal_1D, n_bkps, min_size, jump):
    """Compiled and pure-Python implementations return the same breakpoints.

    This pins the grid semantics (candidate changes are multiples of
    ``jump`` plus the final sample), the ``min_size`` enforcement and the
    first-wins tie-breaking that mirrors Python's ``min()``.
    """
    got = Dynp(model="l2", min_size=min_size, jump=jump).fit_predict(
        signal_1D, n_bkps
    )
    assert got == pure_python_bkps(signal_1D, n_bkps, min_size=min_size, jump=jump)


def test_zero_bkps_stays_on_python_path(signal_1D):
    """``n_bkps=0`` never dispatches; the Python recursion returns a single
    segment closed by the last sample."""
    assert Dynp(model="l2", min_size=1, jump=1).fit_predict(signal_1D, 0) == [200]