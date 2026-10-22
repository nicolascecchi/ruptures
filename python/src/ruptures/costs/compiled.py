r"""Bindings to the compiled distribution cost models.

The Poisson, exponential and binomial costs are implemented in C++ only
(``ruptures._core.models``). The classes in this module are thin adapters:
they validate and store the signal, build the compiled model in ``fit`` and
forward every segment-cost evaluation to C++. No cost formula is implemented
in Python. They raise ``ImportError`` when the compiled backend is not built.
"""

import warnings

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from ruptures.base import BaseCost
from ruptures.exceptions import NotEnoughPoints

try:
    from ruptures import _core
except ImportError:  # compiled backend not built
    _core = None


class _CompiledCost:
    """Shared behavior for costs computed by the compiled backend."""

    def __init__(self) -> None:
        """Initialize the object."""
        if _core is None:
            raise ImportError(
                "This cost requires the compiled backend ruptures._core. "
                "Build it with 'pip install -e .' after initializing the git "
                "submodules ('git submodule update --init')."
            )
        self.signal = None
        self.min_size = 1
        self._cpp_model = None

    def fit(self, signal: NDArray[np.number]) -> Self:
        """Set parameters of the instance and build the compiled model.

        Args:
            signal (array): array of shape (n_samples,) or
                (n_samples, n_features).

        Returns:
            self
        """
        signal = np.ascontiguousarray(signal, dtype=np.float64)
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)

        self._validate(signal)
        self.signal = signal
        self._cpp_model = self._make_model(signal)
        return self

    def error(self, start: int, end: int) -> float:
        """Return the cost on the segment [start:end], computed in C++.

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            segment cost

        Raises:
            NotEnoughPoints: when the segment is too short (less than
                `min_size` samples).
        """
        if end - start < self.min_size:
            raise NotEnoughPoints
        return float(self._cpp_model.compute_segment_cost(start, end))

    def _validate(self, signal: NDArray[np.float64]) -> None:
        raise NotImplementedError

    def _make_model(self, signal: NDArray[np.float64]):
        raise NotImplementedError


class CostPoisson(_CompiledCost, BaseCost):
    r"""Poisson change, computed by the compiled backend.

    Model for detecting changes in the event rate of a Poisson process. The
    signal is assumed to be non-negative count data. The cost itself is
    implemented in C++ (``ruptures._core.models.ConstantPoisson``), so the
    compiled backend must be built.
    """

    model = "poisson"

    def _validate(self, signal: NDArray[np.float64]) -> None:
        if not np.isfinite(signal).all():
            raise ValueError("CostPoisson: the signal must contain finite values.")
        if np.any(signal < 0):
            raise ValueError("CostPoisson: the signal must be non-negative.")
        if np.any(signal != np.round(signal)):
            warnings.warn(
                "CostPoisson: non-integer values detected. The Poisson model "
                "assumes count data.",
                UserWarning,
                stacklevel=3,
            )

    def _make_model(self, signal: NDArray[np.float64]):
        return _core.models.ConstantPoisson(signal)


class CostExponential(_CompiledCost, BaseCost):
    r"""Exponential change, computed by the compiled backend.

    Model for detecting changes in the rate of an exponential distribution.
    The signal is assumed to be strictly positive. The cost itself is
    implemented in C++ (``ruptures._core.models.ConstantExponential``), so the
    compiled backend must be built.
    """

    model = "exponential"

    def _validate(self, signal: NDArray[np.float64]) -> None:
        if not np.isfinite(signal).all():
            raise ValueError("CostExponential: the signal must contain finite values.")
        if np.any(signal <= 0):
            raise ValueError("CostExponential: the signal must be positive.")

    def _make_model(self, signal: NDArray[np.float64]):
        return _core.models.ConstantExponential(signal)


class CostBinomial(_CompiledCost, BaseCost):
    r"""Binomial change, computed by the compiled backend.

    Model for detecting changes in the success probability of a binomial
    distribution, with a known number of trials ``m`` and the signal valued in
    ``[0, m]``. The cost itself is implemented in C++
    (``ruptures._core.models.ConstantBinomial``), so the compiled backend must
    be built.
    """

    model = "binomial"

    def __init__(self, m: int) -> None:
        """Initialize the object.

        Args:
            m (int): number of trials of the binomial distribution. Unlike the
                other cost parameters, it is required: the model is
                meaningless without it.
        """
        if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m < 1:
            raise ValueError("CostBinomial: m must be a positive integer.")
        super().__init__()
        self.m = int(m)

    def _validate(self, signal: NDArray[np.float64]) -> None:
        if not np.isfinite(signal).all():
            raise ValueError("CostBinomial: the signal must contain finite values.")
        if np.any(signal < 0) or np.any(signal > self.m):
            raise ValueError(f"CostBinomial: the signal must lie in [0, {self.m}].")
        if np.any(signal != np.round(signal)):
            warnings.warn(
                "CostBinomial: non-integer values detected. The binomial model "
                "assumes count data.",
                UserWarning,
                stacklevel=3,
            )

    def _make_model(self, signal: NDArray[np.float64]):
        return _core.models.ConstantBinomial(signal, self.m)
