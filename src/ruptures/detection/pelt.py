r"""Pelt."""

from math import floor
from typing import Any, Optional
from typing_extensions import Self

import numpy as np
from numpy.typing import NDArray
from ruptures.costs import CostL2, cost_factory
from ruptures.base import BaseCost, BaseEstimator
from ruptures.exceptions import BadSegmentationParameters
from ruptures.utils import sanity_check

try:
    from ruptures import _core
except ImportError:  # compiled backend not built; the Python path is used.
    _core = None


class Pelt(BaseEstimator):
    """Penalized change point detection.

    For a given model and penalty level, computes the segmentation which
    minimizes the constrained sum of approximation errors.

    For a positive penalty and the built-in L2 cost, the segmentation is
    delegated to the compiled C++ backend ``ruptures._core`` (imported
    from the ``biblio-these`` project), which returns the same
    breakpoints much faster for every valid ``min_size`` and ``jump``.
    Every other configuration, or an unbuilt backend, uses the Python
    implementation below.
    """

    def __init__(
        self,
        model="l2",
        custom_cost: Optional[BaseCost] = None,
        min_size: int = 2,
        jump: int = 5,
        params: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize a Pelt instance.

        Args:
            model (str, optional): segment model, ["l1", "l2", "rbf"]. Not used if ``'custom_cost'`` is not None.
            custom_cost (BaseCost, optional): custom cost function. Defaults to None.
            min_size (int, optional): minimum segment length.
            jump (int, optional): subsample (one every *jump* points).
            params (dict, optional): a dictionary of parameters for the cost instance.
        """
        if custom_cost is not None and isinstance(custom_cost, BaseCost):
            self.cost = custom_cost
        else:
            if params is None:
                self.cost = cost_factory(model=model)
            else:
                self.cost = cost_factory(model=model, **params)
        self.min_size = max(min_size, self.cost.min_size)
        self.jump = jump
        self.n_samples = None

    @property
    def _use_compiled(self) -> bool:
        """Whether this configuration is handled by the compiled backend.

        The compiled implementation scans the change points on the
        ``jump`` grid and enforces the minimum segment length
        ``min_size``, exactly like the Python implementation, so every
        valid ``min_size``/``jump`` combination is delegated. It only
        knows the constant-normal cost, so a built-in ``CostL2`` cost is
        required: subclasses or other cost classes use the Python
        implementation, even when they happen to be named ``"l2"``.
        """
        return (
            _core is not None
            and type(self.cost) is CostL2
            and self.min_size >= 1
            and self.jump >= 1
        )

    def _seg(self, pen: float) -> dict[tuple[int, int], float]:
        """Computes the segmentation for a given penalty using PELT (or a list
        of penalties).

        Args:
            penalty (float): penalty value

        Returns:
            dict: partition dict {(start, end): cost value,...}
        """
        # initialization
        # partitions[t] contains the optimal partition of signal[0:t]
        partitions = dict()  # this dict will be recursively filled
        partitions[0] = {(0, 0): 0}
        admissible = []

        # Recursion
        ind = [k for k in range(0, self.n_samples, self.jump) if k >= self.min_size]
        ind += [self.n_samples]
        for bkp in ind:
            # adding a point to the admissible set from the previous loop.
            new_adm_pt = floor((bkp - self.min_size) / self.jump)
            new_adm_pt *= self.jump
            admissible.append(new_adm_pt)

            subproblems = list()
            for t in admissible:
                # left partition
                try:
                    tmp_partition = partitions[t].copy()
                except KeyError:  # no partition of 0:t exists
                    continue
                # we update with the right partition
                tmp_partition.update({(t, bkp): self.cost.error(t, bkp) + pen})
                subproblems.append(tmp_partition)

            # finding the optimal partition
            partitions[bkp] = min(subproblems, key=lambda d: sum(d.values()))
            # trimming the admissible set
            admissible = [
                t
                for t, partition in zip(admissible, subproblems)
                if sum(partition.values()) <= sum(partitions[bkp].values()) + pen
            ]

        best_partition = partitions[self.n_samples]
        del best_partition[(0, 0)]
        return best_partition

    def _compiled_bkps(self, pen: float) -> list[int]:
        """Return the optimal breakpoints, computed by the compiled backend.

        Args:
            pen (float): penalty value (>0)

        Returns:
            list: sorted list of breakpoints
        """
        # CostL2.fit() stores the signal as a 2D array but keeps its dtype, so
        # it is normalised here into what the compiled model expects.
        signal = np.ascontiguousarray(self.cost.signal, dtype=np.float64)
        model = _core.models.ConstantNormal(signal)
        algorithm = _core.algorithms.PELT(model, min_size=self.min_size, jump=self.jump)
        algorithm.predict(float(pen))
        return np.asarray(algorithm.get_changepoints(), dtype=int).ravel().tolist()

    def fit(self, signal: NDArray[np.number]) -> Self:
        """Set params.

        Args:
            signal (array): signal to segment. Shape (n_samples, n_features) or (n_samples,).

        Returns:
            self
        """
        # update params
        self.cost.fit(signal)
        if signal.ndim == 1:
            (n_samples,) = signal.shape
        else:
            n_samples, _ = signal.shape
        self.n_samples = n_samples
        return self

    def predict(self, pen: float) -> list[int]:
        """Return the optimal breakpoints.

        Must be called after the fit method. The breakpoints are associated with the signal passed
        to [`fit()`][ruptures.detection.pelt.Pelt.fit].

        Args:
            pen (float): penalty value (>0)

        Raises:
            BadSegmentationParameters: in case of impossible segmentation
                configuration

        Returns:
            list: sorted list of breakpoints
        """
        # raise an exception in case of impossible segmentation configuration
        if not sanity_check(
            n_samples=self.cost.signal.shape[0],
            n_bkps=0,
            jump=self.jump,
            min_size=self.min_size,
        ):
            raise BadSegmentationParameters

        if self._use_compiled and pen > 0:
            return self._compiled_bkps(pen)

        partition = self._seg(pen)
        bkps = sorted(e for s, e in partition.keys())
        return bkps

    def fit_predict(self, signal: NDArray[np.number], pen: float) -> list[int]:
        """Fit to the signal and return the optimal breakpoints.

        Helper method to call fit and predict once

        Args:
            signal (array): signal. Shape (n_samples, n_features) or (n_samples,).
            pen (float): penalty value (>0)

        Returns:
            list: sorted list of breakpoints
        """
        self.fit(signal)
        return self.predict(pen)
