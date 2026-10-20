"""Tests for the compiled PELT backend (ruptures._core)."""

from unittest import mock

import numpy as np
import pytest

from ruptures.costs import CostL2
from ruptures.datasets import pw_constant
from ruptures.detection import Pelt

_core = pytest.importorskip("ruptures._core", reason="the compiled core is not built")


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


def pure_python_bkps(signal, pen, model="l2", params=None, min_size=1, jump=1):
    """Breakpoints of ``Pelt(model, params, min_size, jump)``, Python only.

    ``Pelt`` delegates built-in costs to the compiled backend, so the
    module global is disabled for the duration of the call to exercise
    the reference implementation instead.
    """
    with mock.patch("ruptures.detection.pelt._core", None):
        pelt = Pelt(model=model, params=params, min_size=min_size, jump=jump)
        return pelt.fit_predict(signal, pen)


def test_core_importable():
    from ruptures import _core

    assert hasattr(_core.models, "ConstantNormal")
    assert hasattr(_core.algorithms, "PELT")


@pytest.mark.parametrize("min_size,jump", [(1, 1), (2, 1), (1, 2), (3, 5), (2, 5)])
def test_pelt_delegates_to_compiled_backend(
    monkeypatch, signal_bkps_1D, min_size, jump
):
    """Every valid (min_size, jump) L2 configuration runs the compiled
    backend."""
    signal, _ = signal_bkps_1D
    pens = []
    compiled_bkps = Pelt._compiled_bkps

    def spy(self, model, pen):
        pens.append(pen)
        return compiled_bkps(self, model, pen)

    monkeypatch.setattr(Pelt, "_compiled_bkps", spy)
    Pelt(model="l2", min_size=min_size, jump=jump).fit_predict(signal, 50.0)
    assert pens == [50.0]


@pytest.mark.parametrize(
    "kwargs",
    [{"model": "l2", "jump": -1}],  # an invalid grid stays on the Python path
)
def test_pelt_does_not_delegate(kwargs):
    assert not Pelt(**kwargs)._use_compiled


def test_pelt_does_not_delegate_for_a_cost_subclass():
    """A user cost named "l2" is not assumed to be the built-in L2 cost."""

    class MyCost(CostL2):
        model = "l2"

    assert not Pelt(custom_cost=MyCost(), min_size=1, jump=1)._use_compiled


@pytest.mark.parametrize(
    "model,pen", [("l2", -1.0), ("l2", 0.0), ("l1", 0.0)]
)
def test_non_positive_penalty_stays_on_python_path(model, pen, signal_bkps_1D):
    """The compiled classical pruning rule is only used for pen > 0.

    l1 has no pen=-1.0 case: a negative penalty empties the admissible
    set of the Python PELT recursion, which crashes ``_seg`` for every
    cost with ``min_size >= 2`` at ``jump=1`` (pre-existing behaviour,
    unrelated to the dispatch)."""
    signal, _ = signal_bkps_1D
    with mock.patch.object(Pelt, "_compiled_bkps") as spy:
        Pelt(model=model, min_size=1, jump=1).fit_predict(signal, pen)
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


# (min_size, jump) grid: covers m > j, m a multiple of j, m == j and m == 1.
MIN_SIZE_JUMP = [(1, 1), (2, 1), (1, 2), (2, 5), (3, 3), (7, 3), (5, 8), (12, 5)]


@pytest.mark.parametrize("min_size,jump", MIN_SIZE_JUMP)
@pytest.mark.parametrize("pen", [10.0, 50.0, 200.0])
def test_matches_pure_python_pelt_configs_1D(signal_bkps_1D, min_size, jump, pen):
    signal, _ = signal_bkps_1D
    got = Pelt(model="l2", min_size=min_size, jump=jump).fit_predict(signal, pen)
    assert got == pure_python_bkps(signal, pen, min_size=min_size, jump=jump)


@pytest.mark.parametrize("min_size,jump", [(2, 5), (7, 3)])
@pytest.mark.parametrize("pen", [50.0, 200.0])
def test_matches_pure_python_pelt_configs_5D(signal_bkps_5D, min_size, jump, pen):
    signal, _ = signal_bkps_5D
    got = Pelt(model="l2", min_size=min_size, jump=jump).fit_predict(signal, pen)
    assert got == pure_python_bkps(signal, pen, min_size=min_size, jump=jump)


@pytest.mark.parametrize("n_samples", [16, 17, 26])
@pytest.mark.parametrize("min_size,jump", [(3, 5), (12, 5)])
def test_matches_pure_python_off_grid_starts(n_samples, min_size, jump):
    """n_samples just past a grid point exercises duplicate admissible starts.

    For such signals ``Pelt._seg`` appends the same admissible start twice
    (Python: `jump * floor((n_samples - min_size) / jump)` for the last
    change point equals an earlier one); the compiled implementation must
    replicate that too.
    """
    signal = np.random.RandomState(n_samples).standard_normal(n_samples)
    got = Pelt(model="l2", min_size=min_size, jump=jump).fit_predict(signal, 3.0)
    assert got == pure_python_bkps(signal, 3.0, min_size=min_size, jump=jump)


@pytest.mark.parametrize("min_size,jump", [(1, 1), (2, 5), (3, 3), (7, 3)])
def test_delegated_bkps_are_on_the_jump_grid(signal_bkps_1D, min_size, jump):
    """Compiled breakpoints respect the jump grid and the minimum length."""
    signal, _ = signal_bkps_1D
    n = signal.shape[0]
    bkps = Pelt(model="l2", min_size=min_size, jump=jump).fit_predict(signal, 50.0)
    assert bkps == sorted(set(bkps))
    assert bkps[-1] == n
    assert all(b % jump == 0 for b in bkps[:-1])
    assert all(b - s >= min_size for s, b in zip([0] + bkps[:-1], bkps))


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


@pytest.mark.parametrize("min_size,jump", [(1, 1), (2, 5)])
def test_large_penalty_single_segment(min_size, jump, signal_bkps_1D):
    signal, _ = signal_bkps_1D
    assert Pelt(model="l2", min_size=min_size, jump=jump).fit_predict(signal, 1e12) == [
        signal.shape[0]
    ]


@pytest.mark.parametrize("min_size,jump", [(1, 1), (2, 5)])
def test_reusable_after_fit(min_size, jump, signal_bkps_5D):
    signal, _ = signal_bkps_5D
    algo = Pelt(model="l2", min_size=min_size, jump=jump).fit(signal)
    first = algo.predict(50.0)
    second = algo.predict(50.0)
    assert first == second


@pytest.mark.parametrize("model", ["l2", "l1", "rbf"])
@pytest.mark.parametrize("min_size,jump", [(1, 1), (2, 3)])
def test_accepts_non_contiguous_and_non_float(min_size, jump, model, signal_bkps_5D):
    signal, _ = signal_bkps_5D
    view = np.asarray(signal, dtype=np.int64)[::1, ::2]
    got = Pelt(model=model, min_size=min_size, jump=jump).fit_predict(view, 50.0)
    assert got == pure_python_bkps(
        view.astype(float), 50.0, model=model, min_size=min_size, jump=jump
    )


def test_binding_rejects_invalid_parameters(signal_bkps_1D):
    signal, _ = signal_bkps_1D
    from ruptures import _core

    model = _core.models.ConstantNormal(np.ascontiguousarray(signal, dtype=np.float64))
    with pytest.raises(ValueError):
        _core.algorithms.PELT(model, min_size=0)
    with pytest.raises(ValueError):
        _core.algorithms.PELT(model, jump=0)


# ---------------------------------------------------------------------------
# Built-in costs dispatched to the compiled backend
# (see _COMPILED_COST_BUILDERS in ruptures.detection.pelt)
# ---------------------------------------------------------------------------

DISPATCHED_MODELS = [
    "l1", "l2", "linear", "clinear", "normal", "rank", "rbf", "cosine", "ar",
    "mahalanobis",
]


@pytest.mark.parametrize("model", DISPATCHED_MODELS)
def test_pelt_delegates_every_builtin_model(model):
    """Every built-in model string with a compiled counterpart dispatches."""
    assert Pelt(model=model)._use_compiled


def test_normal_delegates_only_with_small_diag():
    """model="normal" is delegated only for the default add_small_diag=True
    behaviour: the compiled backend bakes in the 1e-6 diagonal bias
    (v1.1.5 semantics, PR#198); add_small_diag=False stays on Python."""
    pelt = Pelt(model="normal").fit(np.zeros(10).reshape(-1, 1))
    assert pelt._use_compiled
    assert pelt._compiled_model() is not None

    pelt = Pelt(model="normal", params={"add_small_diag": False})
    pelt.fit(np.zeros(10).reshape(-1, 1))
    assert pelt._compiled_model() is None


def _assert_cost_matches_python(signal, model, params, segments):
    """The compiled model adapted from a fitted cost matches its Python
    error() values on the given segments."""
    pelt = Pelt(model=model, params=params, min_size=1, jump=1).fit(signal)
    compiled = pelt._compiled_model()
    for start, end in segments:
        if end - start < pelt.cost.min_size:
            continue
        assert compiled.compute_segment_cost(start, end) == pytest.approx(
            pelt.cost.error(start, end), rel=1e-6, abs=1e-8
        )


SEGMENTS_500 = [(0, 1), (0, 10), (17, 250), (100, 499), (0, 500), (243, 500)]


@pytest.mark.parametrize(
    "model,params",
    [
        ("l1", {}),
        ("rbf", {}),  # gamma is resolved by the median heuristic in fit()
        ("rbf", {"gamma": 0.02}),
        ("cosine", {}),
        ("clinear", {}),
        ("ar", {"order": 1}),
        ("ar", {"order": 4}),
        ("normal", {}),
    ],
)
def test_builtin_cost_matches_python_1D(signal_bkps_1D, model, params):
    signal, _ = signal_bkps_1D
    _assert_cost_matches_python(signal, model, params, SEGMENTS_500)


@pytest.mark.parametrize("model", ["mahalanobis", "rank", "normal"])
def test_builtin_cost_matches_python_5D(signal_bkps_5D, model):
    signal, _ = signal_bkps_5D
    _assert_cost_matches_python(signal, model, {}, SEGMENTS_500)


def test_rank_cost_matches_python_on_tied_values():
    """scipy rankdata averages tied ranks, so a constant column has exactly
    0 cost everywhere; the compiled rank cost must do the same, otherwise
    PELT splits tied segments into spurious change points."""
    rng = np.random.default_rng(0)
    signal = np.round(rng.normal(size=(200, 2)))  # many tied values
    segments = [(0, 1), (0, 10), (17, 150), (100, 200), (0, 200), (60, 133)]
    _assert_cost_matches_python(signal, "rank", {}, segments)
    assert Pelt(model="rank").fit_predict(np.zeros(200).reshape(-1, 1), 1.0) == [
        200
    ]


@pytest.fixture(scope="module")
def linear_signal():
    """2-column signal whose observed variable is regressed on a covariate,
    with a slope change half-way."""
    covariate = np.random.default_rng(11).normal(size=(500, 1))
    gain = np.r_[np.full(250, 2.0), np.full(250, -1.0)]
    observed = gain * covariate
    observed += np.random.default_rng(12).normal(scale=0.1, size=(500, 1))
    return np.column_stack([observed, covariate])


def test_builtin_cost_matches_python_linear(linear_signal):
    _assert_cost_matches_python(
        linear_signal, "linear", {}, [(0, 2), (0, 10), (17, 250), (0, 500)]
    )


# penalty levels chosen so the segmentations are non-trivial on the fixtures
@pytest.mark.parametrize("min_size,jump", MIN_SIZE_JUMP)
@pytest.mark.parametrize("pen", [10.0, 50.0, 200.0])
def test_matches_pure_python_pelt_l1_configs(signal_bkps_1D, min_size, jump, pen):
    signal, _ = signal_bkps_1D
    got = Pelt(model="l1", min_size=min_size, jump=jump).fit_predict(signal, pen)
    assert got == pure_python_bkps(
        signal, pen, model="l1", min_size=min_size, jump=jump
    )


@pytest.mark.parametrize("min_size,jump", [(1, 1), (2, 5), (7, 3), (12, 5)])
@pytest.mark.parametrize(
    "model,params,pen",
    [
        ("rbf", {}, 2.0),
        ("cosine", {}, 2.0),
        ("clinear", {}, 50.0),
        ("ar", {"order": 4}, 100.0),
        ("normal", {}, 1.0),
    ],
)
def test_matches_pure_python_pelt_models_1D(
    signal_bkps_1D, model, params, pen, min_size, jump
):
    """Compiled and pure-Python PELT return the same breakpoints for the
    other built-in costs."""
    signal, _ = signal_bkps_1D
    got = Pelt(
        model=model, params=params, min_size=min_size, jump=jump
    ).fit_predict(signal, pen)
    assert got == pure_python_bkps(
        signal, pen, model=model, params=params, min_size=min_size, jump=jump
    )


@pytest.mark.parametrize("min_size,jump", [(2, 5), (7, 3)])
@pytest.mark.parametrize("model", ["mahalanobis", "rank"])
def test_matches_pure_python_pelt_models_5D(signal_bkps_5D, model, min_size, jump):
    signal, _ = signal_bkps_5D
    got = Pelt(model=model, min_size=min_size, jump=jump).fit_predict(signal, 5.0)
    assert got == pure_python_bkps(
        signal, 5.0, model=model, min_size=min_size, jump=jump
    )


@pytest.mark.parametrize("min_size,jump", [(2, 5), (7, 3)])
def test_matches_pure_python_pelt_normal_5D(signal_bkps_5D, min_size, jump):
    """5-D normal segments aggressively (every split gains likelihood), so
    only a large penalty keeps the segmentation non-trivial."""
    signal, _ = signal_bkps_5D
    got = Pelt(model="normal", min_size=min_size, jump=jump).fit_predict(
        signal, 100.0
    )
    assert got == pure_python_bkps(
        signal, 100.0, model="normal", min_size=min_size, jump=jump
    )


def test_matches_pure_python_pelt_linear(linear_signal):
    got = Pelt(model="linear", min_size=5, jump=3).fit_predict(linear_signal, 1.0)
    assert got == pure_python_bkps(
        linear_signal, 1.0, model="linear", min_size=5, jump=3
    )


@pytest.mark.parametrize(
    "model,params,pen", [("l1", {}, 15.0), ("rbf", {}, 2.0), ("ar", {"order": 1}, 100.0)]
)
def test_reusable_after_fit_built_in_models(signal_bkps_1D, model, params, pen):
    signal, _ = signal_bkps_1D
    algo = Pelt(model=model, params=params).fit(signal)
    assert algo.predict(pen) == algo.predict(pen)


@pytest.fixture(scope="module")
def variance_break_signal():
    """Two Gaussian stretches whose variance changes half-way, with no mean
    change: only a variance-sensitive cost such as normal detects it."""
    rng = np.random.default_rng(3)
    return np.r_[
        rng.normal(scale=0.2, size=250), rng.normal(scale=2.0, size=250)
    ].reshape(-1, 1)


@pytest.mark.parametrize(
    "min_size,jump,pen",
    [(2, 5, 1.0), (2, 5, 5.0), (7, 3, 5.0), (12, 5, 2.0)],
)
def test_matches_pure_python_pelt_normal_varbreak(
    min_size, jump, pen, variance_break_signal
):
    got = Pelt(model="normal", min_size=min_size, jump=jump).fit_predict(
        variance_break_signal, pen
    )
    assert got == pure_python_bkps(
        variance_break_signal, pen, model="normal", min_size=min_size, jump=jump
    )


@pytest.mark.parametrize("min_size,jump", [(2, 5), (12, 5)])
def test_normal_recovers_planted_variance_break(min_size, jump, variance_break_signal):
    """At pen=10 the planted change is recovered exactly (smaller penalties
    keep extra likelihood-chatter splits, which the equality test above
    covers instead)."""
    assert Pelt(model="normal", min_size=min_size, jump=jump).fit_predict(
        variance_break_signal, 10.0
    ) == [250, 500]


# ---------------------------------------------------------------------------
# Compiled-only distribution models (Poisson, exponential, binomial).
# They have no Python counterpart; costs are checked against their closed
# forms and against the degenerate-segment limit (cost 0).
# ---------------------------------------------------------------------------


def test_compiled_poisson_cost():
    """Poisson cost of [1, 2, 3]: S - S log(S / L), with S = 6, L = 3."""
    model = _core.models.ConstantPoisson(np.array([[1.0], [2.0], [3.0]]))
    assert model.compute_segment_cost(0, 3) == pytest.approx(6 - 6 * np.log(2))


def test_compiled_exponential_cost():
    """Exponential cost of [1, 2, 3]: L + L log(S / L)."""
    model = _core.models.ConstantExponential(np.array([[1.0], [2.0], [3.0]]))
    assert model.compute_segment_cost(0, 3) == pytest.approx(3 + 3 * np.log(2))


def test_compiled_binomial_cost():
    """Binomial cost of [2, 4, 6] with m = 10: NLL at p = 0.4."""
    model = _core.models.ConstantBinomial(np.array([[2.0], [4.0], [6.0]]), 10)
    expected = -(12 * np.log(0.4) + 18 * np.log(0.6))
    assert model.compute_segment_cost(0, 3) == pytest.approx(expected)


def test_compiled_poisson_all_zero_segment_has_zero_cost():
    """An all-zero segment has the limit cost 0, not NaN."""
    zeros = np.zeros((10, 2))
    model = _core.models.ConstantPoisson(zeros)
    assert model.compute_segment_cost(0, 5) == 0.0
    assert model.compute_segment_cost(5, 10) == 0.0


def test_compiled_poisson_mixed_dimensions():
    """A degenerate dimension adds 0: the mixed cost equals the count one."""
    counts = np.arange(1, 11, dtype=float).reshape(-1, 1)
    signal = np.hstack([np.zeros((10, 1)), counts])
    mixed = _core.models.ConstantPoisson(signal).compute_segment_cost(0, 5)
    only_counts = _core.models.ConstantPoisson(counts).compute_segment_cost(0, 5)
    assert mixed == pytest.approx(only_counts)


def test_compiled_binomial_degenerate_segments_have_zero_cost():
    """All failures (p = 0) and all successes (p = 1) cost 0."""
    m = 6
    failures = np.zeros((10, 2))
    successes = np.full((10, 2), float(m))
    assert _core.models.ConstantBinomial(failures, m).compute_segment_cost(0, 5) == 0.0
    assert _core.models.ConstantBinomial(successes, m).compute_segment_cost(0, 5) == 0.0


def test_compiled_binomial_mixed_dimensions():
    """A degenerate dimension adds 0: the mixed cost equals the count one."""
    m = 6
    counts = np.array([[1.0], [2.0], [3.0], [2.0], [4.0]] + [[3.0]] * 5)
    signal = np.hstack([np.zeros((10, 1)), counts])
    mixed = _core.models.ConstantBinomial(signal, m).compute_segment_cost(0, 5)
    only_counts = _core.models.ConstantBinomial(counts, m).compute_segment_cost(0, 5)
    assert mixed == pytest.approx(only_counts)
