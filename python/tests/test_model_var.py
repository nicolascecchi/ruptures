"""Tests for the compiled VAR cost model (ruptures._core.models.CostVar)."""

import numpy as np
import pytest

pytest.importorskip("ruptures._core", reason="the compiled core is not built")


def compiled_core():
    from ruptures import _core

    return _core


def lag_matrix(signal, order, intercept):
    """Lag design matrix built like the C++ side: row t holds
    [y_{t-1}, ..., y_{t-order}] (+ a trailing 1) and the first order rows,
    which have no lags yet, are zero."""
    n_samples, n_dims = signal.shape
    n_cols = order * n_dims + int(intercept)
    covar = np.zeros((n_samples, n_cols))
    for t in range(order, n_samples):
        for j in range(order):
            covar[t, j * n_dims : (j + 1) * n_dims] = signal[t - 1 - j]
        if intercept:
            covar[t, -1] = 1.0
    return covar


def reference_var_cost(signal, start, end, order, intercept):
    """RSS of the least-squares fit of signal[start:end] on its lags."""
    y = signal[start:end]
    z = lag_matrix(signal, order, intercept)[start:end]
    coef, _, _, _ = np.linalg.lstsq(z, y, rcond=None)
    return float(np.sum((y - z @ coef) ** 2))


@pytest.fixture(scope="module")
def signal():
    return np.random.default_rng(42).normal(size=(60, 3))


@pytest.mark.parametrize(
    "order,intercept", [(1, True), (2, True), (2, False), (3, True), (3, False)]
)
def test_cost_var_matches_reference_ols(signal, order, intercept):
    """Segments starting after the lag head and at the signal start (whose
    rows have zero covariates) both match a plain numpy OLS."""
    model = compiled_core().models.CostVar(signal, order=order, intercept=intercept)
    for start, end in [(0, 25), (10, 45), (30, 60), (0, 60)]:
        np.testing.assert_allclose(
            model.compute_segment_cost(start, end),
            reference_var_cost(signal, start, end, order, intercept),
            atol=1e-8,
        )


def test_cost_var_rank_deficient_segment_matches_ols_minimal_rss():
    """A segment shorter than the number of regressors is rank deficient; the
    returned cost is still the minimal least-squares residual sum of squares
    (all least-squares solutions share it)."""
    sig = np.random.default_rng(7).normal(size=(30, 3))
    order, intercept = 2, True  # 7 regressors, segments of length 4 or 5
    model = compiled_core().models.CostVar(sig, order=order, intercept=intercept)
    np.testing.assert_allclose(
        model.compute_segment_cost(25, 29),
        reference_var_cost(sig, 25, 29, order, intercept),
        atol=1e-8,
    )


def test_cost_var_univariate_signal():
    sig = np.random.default_rng(13).normal(size=(40, 1))
    model = compiled_core().models.CostVar(sig, order=2)
    np.testing.assert_allclose(
        model.compute_segment_cost(5, 40),
        reference_var_cost(sig, 5, 40, 2, True),
        atol=1e-8,
    )


def test_cost_var_positive_order_required(signal):
    with pytest.raises(ValueError):
        compiled_core().models.CostVar(signal, order=0)