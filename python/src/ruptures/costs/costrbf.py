r"""Kernelized mean change."""

from typing import Optional
from typing_extensions import Self
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform

from ruptures.exceptions import NotEnoughPoints
from ruptures.base import BaseCost


class CostRbf(BaseCost):
    r"""Kernel cost function (rbf kernel)."""

    model = "rbf"

    def __init__(self, gamma: Optional[float] = None) -> None:
        """Initialize the object."""
        self.min_size = 1
        self.gamma = gamma
        self._gram = None
        self._sqdist = None

    def _gamma_from_sqdist(self, K: NDArray[np.number]) -> float:
        """Median heuristic: gamma = 1 / median(pairwise squared distances)."""
        K_median = np.median(K)
        return 1.0 if K_median == 0 else 1 / K_median

    @property
    def gram(self) -> NDArray[np.number]:
        """Generate the gram matrix (lazy loading).

        Only access this function after a `.fit()` (otherwise
        `self.signal` is not defined).
        """
        if self._gram is None:
            if self._sqdist is not None:
                # already computed by fit(); consumed, since K is modified below
                K, self._sqdist = self._sqdist, None
            else:
                K = pdist(self.signal, metric="sqeuclidean")
            if self.gamma is None:
                self.gamma = self._gamma_from_sqdist(K)  # median heuristics
            K *= self.gamma
            np.clip(K, 1e-2, 1e2, K)  # clipping to avoid exponential under/overflow
            self._gram = np.exp(squareform(-K))
        return self._gram

    def fit(self, signal: NDArray[np.number]) -> Self:
        """Sets parameters of the instance.

        Args:
            signal (array): signal. Shape (n_samples,) or (n_samples, n_features)

        Returns:
            self
        """
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal

        # If gamma is none, set it using the median heuristic. Only the
        # pairwise squared distances are needed: the full Gram is left to the
        # lazy `.gram` property, so a compiled cost built from `gamma` never
        # pays for the expansion.
        self._sqdist = None
        if self.gamma is None:
            self._sqdist = pdist(self.signal, metric="sqeuclidean")
            self.gamma = self._gamma_from_sqdist(self._sqdist)

        return self

    def error(self, start: int, end: int) -> float:
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            segment cost

        Raises:
            NotEnoughPoints: when the segment is too short (less than `min_size` samples).
        """
        if end - start < self.min_size:
            raise NotEnoughPoints
        sub_gram = self.gram[start:end, start:end]
        val = np.diagonal(sub_gram).sum()
        val -= sub_gram.sum() / (end - start)
        return val
