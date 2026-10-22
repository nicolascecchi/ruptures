r"""Dynamic programming."""

from functools import lru_cache
from typing import Any, Optional
from typing_extensions import Self

import numpy as np
from numpy.typing import NDArray
from ruptures.utils import sanity_check
from ruptures.costs import cost_factory
from ruptures.base import BaseCost, BaseEstimator
from ruptures.exceptions import BadSegmentationParameters
# The builders are shared with Pelt so the exact-type dispatch table has a
# single source of truth.
from ruptures.detection.pelt import _COMPILED_COST_BUILDERS

try:
    from ruptures import _core
except ImportError:  # compiled backend not built; the Python path is used.
    _core = None


class Dynp(BaseEstimator):
    """Find optimal change points using dynamic programming.

    Given a segment model, it computes the best partition for which the
    sum of errors is minimum.

    For a fixed number of breakpoints and a built-in cost with a compiled
    counterpart, the segmentation is delegated to the compiled C++
    backend ``ruptures._core``, which returns the same breakpoints much faster for every
    valid ``min_size`` and ``jump``. Every other configuration (custom
    cost, subclass, unbuilt backend) uses the Python implementation
    below.
    """

    def __init__(
        self,
        model: str = "l2",
        custom_cost: Optional[BaseCost] = None,
        min_size: int = 2,
        jump: int = 5,
        params: Optional[dict[str, Any]] = None,
    ) -> None:
        """Creates a Dynp instance.

        Args:
            model (str, optional): segment model, ["l1", "l2", "linear", "clinear",
                "normal", "rank", "rbf", "cosine", "ar", "mahalanobis", "poisson",
                "exponential", "binomial"]. Models that have a compiled
                counterpart are dispatched to the compiled backend when it is
                built. "poisson", "exponential" and "binomial" only exist in the
                compiled backend, which must therefore be built; "binomial"
                additionally requires the number of trials through
                ``params={"m": ...}``. Not used if ``'custom_cost'`` is not None.
            custom_cost (BaseCost, optional): custom cost function. Defaults to None.
            min_size (int, optional): minimum segment length.
            jump (int, optional): subsample (one every *jump* points).
            params (dict, optional): a dictionary of parameters for the cost instance.
        """
        if custom_cost is not None and isinstance(custom_cost, BaseCost):
            self.cost = custom_cost
        else:
            self.model_name = model
            if params is None:
                self.cost = cost_factory(model=model)
            else:
                self.cost = cost_factory(model=model, **params)
        self.min_size = max(min_size, self.cost.min_size)
        self.jump = jump
        self.n_samples = None

    def _compiled_model(self):
        """Adapt the fitted cost into its compiled counterpart, or None.

        Returns None when the Python implementation must be used (no
        compiled backend, non-builtin cost, invalid segmentation
        parameters, or a builder gate failing).
        """
        if not self._use_compiled:
            return None
        return _COMPILED_COST_BUILDERS[type(self.cost)](self.cost)

    @property
    def _use_compiled(self) -> bool:
        """Whether this configuration is handled by the compiled backend.

        The compiled implementation scans the change points on the
        ``jump`` grid and enforces the minimum segment length
        ``min_size``, exactly like the Python implementation, so every
        valid ``min_size``/``jump`` combination is delegated. Only exact
        built-in cost types are delegated: subclasses or custom cost
        classes use the Python implementation, even when they happen to
        be named like a built-in one.
        """
        return (
            _core is not None
            and type(self.cost) in _COMPILED_COST_BUILDERS
            and self.min_size >= 1
            and self.jump >= 1
        )

    def _compiled_bkps(self, model, n_bkps: int) -> list[int]:
        """Return the optimal breakpoints, computed by the compiled backend.

        Args:
            model: compiled model adapted from the fitted cost (see ``_compiled_model``).
            n_bkps (int): number of breakpoints.

        Returns:
            list: sorted list of breakpoints
        """
        # The compiled implementation scans the change points on the same
        # jump grid and enforces the same min_size, hence the identical
        # breakpoints for every valid configuration.
        algorithm = _core.algorithms.SegmentNeighborhood(
            model, min_size=self.min_size, jump=self.jump
        )
        algorithm.predict(int(n_bkps))
        return np.asarray(algorithm.get_changepoints(), dtype=int).ravel().tolist()

    @lru_cache(maxsize=None)
    def seg(self, start: int, end: int, n_bkps: int) -> dict[tuple[int, int], float]:
        """Recurrence to find the optimal partition of signal[start:end].

        This method is to be memoized and then used.

        Args:
            start (int): start of the segment (inclusive)
            end (int): end of the segment (exclusive)
            n_bkps (int): number of breakpoints

        Returns:
            dict: {(start, end): cost value, ...}
        """
        jump, min_size = self.jump, self.min_size

        if n_bkps == 0:
            cost = self.cost.error(start, end)
            return {(start, end): cost}
        elif n_bkps > 0:
            # Let's fill the list of admissible last breakpoints
            multiple_of_jump = (k for k in range(start, end) if k % jump == 0)
            admissible_bkps = list()
            for bkp in multiple_of_jump:
                n_samples = bkp - start
                # first check if left subproblem is possible
                if sanity_check(
                    n_samples=n_samples,
                    n_bkps=n_bkps - 1,
                    jump=jump,
                    min_size=min_size,
                ):
                    # second check if the right subproblem has enough points
                    if end - bkp >= min_size:
                        admissible_bkps.append(bkp)

            assert (
                len(admissible_bkps) > 0
            ), "No admissible last breakpoints found.\
             start, end: ({},{}), n_bkps: {}.".format(
                start, end, n_bkps
            )

            # Compute the subproblems
            sub_problems = list()
            for bkp in admissible_bkps:
                left_partition = self.seg(start, bkp, n_bkps - 1)
                right_partition = self.seg(bkp, end, 0)
                tmp_partition = dict(left_partition)
                tmp_partition[(bkp, end)] = right_partition[(bkp, end)]
                sub_problems.append(tmp_partition)

            # Find the optimal partition
            return min(sub_problems, key=lambda d: sum(d.values()))

    def fit(self, signal: NDArray[np.number]) -> Self:
        """Create the cache associated with the signal.

        Dynamic programming is a recurrence; intermediate results are cached to speed up
        computations. This method sets up the cache.

        Args:
            signal (array): signal. Shape (n_samples, n_features) or (n_samples,).

        Returns:
            self
        """
        # clear cache
        self.seg.cache_clear()
        # update some params
        self.cost.fit(signal)
        self.n_samples = signal.shape[0]
        return self

    def predict(self, n_bkps: int) -> list[int]:
        """Return the optimal breakpoints.

        Must be called after the fit method. The breakpoints are associated with the signal passed
        to [`fit()`][ruptures.detection.dynp.Dynp.fit].

        Args:
            n_bkps (int): number of breakpoints.

        Raises:
            BadSegmentationParameters: in case of impossible segmentation
                configuration

        Returns:
            list: sorted list of breakpoints
        """
        # raise an exception in case of impossible segmentation configuration
        if not sanity_check(
            n_samples=self.cost.signal.shape[0],
            n_bkps=n_bkps,
            jump=self.jump,
            min_size=self.min_size,
        ):
            raise BadSegmentationParameters

        # the model is built once: building can pay a non-trivial price
        # (e.g. an O(nobs^2) Gram matrix for rbf/cosine).
        model = self._compiled_model() if self._use_compiled and n_bkps >= 1 else None
        if model is not None:
            return self._compiled_bkps(model, n_bkps)

        partition = self.seg(0, self.n_samples, n_bkps)
        bkps = sorted(e for s, e in partition.keys())
        return bkps

    def fit_predict(self, signal: NDArray[np.number], n_bkps: int) -> list[int]:
        """Fit to the signal and return the optimal breakpoints.

        Helper method to call fit and predict once

        Args:
            signal (array): signal. Shape (n_samples, n_features) or (n_samples,).
            n_bkps (int): number of breakpoints.

        Returns:
            list: sorted list of breakpoints
        """
        self.fit(signal)
        return self.predict(n_bkps)
