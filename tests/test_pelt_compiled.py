"""Tests for the compiled PELT backend (ruptures._core)."""

from unittest import mock

import numpy as np
import pytest

from ruptures.costs import CostL2
from ruptures.datasets import pw_constant
from ruptures.detection import Pelt

pytest.importorskip("ruptures._core", reason="the compiled core is not built")


@pytest.fixture(scope="module")
def signal_bkps_1D():
    signal, bkps = pw_constant(n_samples=500, n_bkps=4, noise_std=1, seed=42)
    return signal[:, 0], bkps


@pytest.fixture(scope="module")
def signal_bkps_5D():
    signal, bkps = pw_constant(
        n_samples=500, n_features=5, n_bkps=4, noise_std=1, seed=42
    )
    return signal, bkps


def pure_python_bkps(signal, pen):
    """Breakpoints of ``Pelt(model="l2", min_size=1, jump=1)``, Python only.

    ``Pelt`` delegates that configuration to the compiled backend, so
    the module global is disabled for the duration of the call to
    exercise the reference implementation instead.
    """
    with mock.patch("ruptures.detection.pelt._core", None):
        return Pelt(model="l2", min_size=1, jump=1).fit_predict(signal, pen)


def test_core_importable():
    from ruptures import _core

    assert hasattr(_core.models, "ConstantNormal")
    assert hasattr(_core.algorithms, "PELT")


def test_pelt_delegates_to_compiled_backend(monkeypatch, signal_bkps_1D):
    """The (min_size=1, jump=1) L2 configuration runs the compiled backend."""
    signal, _ = signal_bkps_1D
    pens = []
    compiled_bkps = Pelt._compiled_bkps

    def spy(self, pen):
        pens.append(pen)
        return compiled_bkps(self, pen)

    monkeypatch.setattr(Pelt, "_compiled_bkps", spy)
    Pelt(model="l2", min_size=1, jump=1).fit_predict(signal, 50.0)
    assert pens == [50.0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "l2"},  # the defaults are min_size=2, jump=5
        {"model": "l2", "min_size": 2, "jump": 1},
        {"model": "l2", "min_size": 1, "jump": 2},
        {"model": "l1", "min_size": 1, "jump": 1},
    ],
)
def test_pelt_does_not_delegate(kwargs):
    assert not Pelt(**kwargs)._use_compiled


def test_pelt_does_not_delegate_for_a_cost_subclass():
    """A user cost named "l2" is not assumed to be the built-in L2 cost."""

    class MyCost(CostL2):
        model = "l2"

    assert not Pelt(custom_cost=MyCost(), min_size=1, jump=1)._use_compiled


@pytest.mark.parametrize("pen", [-1.0, 0.0])
def test_non_positive_penalty_stays_on_python_path(pen, signal_bkps_1D):
    """The compiled classical pruning rule is only used for pen > 0."""
    signal, _ = signal_bkps_1D
    with mock.patch.object(Pelt, "_compiled_bkps") as spy:
        Pelt(model="l2", min_size=1, jump=1).fit_predict(signal, pen)
    spy.assert_not_called()


def test_constant_normal_cost_matches_costl2(signal_bkps_5D):
    signal, _ = signal_bkps_5D
    from ruptures import _core

    model = _core.models.ConstantNormal(np.ascontiguousarray(signal, dtype=np.float64))
    cost = CostL2().fit(signal)
    for start, end in [(0, 1), (0, 10), (17, 250), (100, 499), (0, 500), (243, 500)]:
        assert model.compute_segment_cost(start, end) == pytest.approx(
            cost.error(start, end), rel=1e-6, abs=1e-8
        )


@pytest.mark.parametrize("pen", [10.0, 50.0, 100.0, 200.0])
def test_matches_pure_python_pelt_1D(signal_bkps_1D, pen):
    signal, _ = signal_bkps_1D
    got = Pelt(model="l2", min_size=1, jump=1).fit_predict(signal, pen)
    assert got == pure_python_bkps(signal, pen)


@pytest.mark.parametrize("pen", [10.0, 50.0, 100.0, 200.0])
def test_matches_pure_python_pelt_5D(signal_bkps_5D, pen):
    signal, _ = signal_bkps_5D
    got = Pelt(model="l2", min_size=1, jump=1).fit_predict(signal, pen)
    assert got == pure_python_bkps(signal, pen)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_pure_python_pelt_various_signals(seed):
    signal, _ = pw_constant(
        n_samples=200, n_features=3, n_bkps=3, noise_std=1, seed=seed
    )
    for pen in [1.0, 5.0, 20.0, 100.0]:
        got = Pelt(model="l2", min_size=1, jump=1).fit_predict(signal, pen)
        assert got == pure_python_bkps(signal, pen)


def test_recovers_true_bkps_no_noise():
    signal, bkps = pw_constant(n_samples=300, n_bkps=3, noise_std=0, seed=7)
    got = Pelt(model="l2", min_size=1, jump=1).fit_predict(signal, pen=1.0)
    assert got == list(bkps)


def test_last_bkp_is_n_samples(signal_bkps_1D):
    signal, _ = signal_bkps_1D
    bkps = Pelt(model="l2", min_size=1, jump=1).fit_predict(signal, 50.0)
    assert bkps[-1] == signal.shape[0]
    assert bkps == sorted(bkps)


def test_large_penalty_single_segment(signal_bkps_1D):
    signal, _ = signal_bkps_1D
    assert Pelt(model="l2", min_size=1, jump=1).fit_predict(signal, 1e12) == [
        signal.shape[0]
    ]


def test_reusable_after_fit(signal_bkps_5D):
    signal, _ = signal_bkps_5D
    algo = Pelt(model="l2", min_size=1, jump=1).fit(signal)
    first = algo.predict(50.0)
    second = algo.predict(50.0)
    assert first == second


def test_accepts_non_contiguous_and_non_float(signal_bkps_5D):
    signal, _ = signal_bkps_5D
    view = np.asarray(signal, dtype=np.int64)[::1, ::2]
    got = Pelt(model="l2", min_size=1, jump=1).fit_predict(view, 50.0)
    assert got == pure_python_bkps(view.astype(float), 50.0)
