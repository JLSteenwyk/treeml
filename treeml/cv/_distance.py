from typing import List, Optional

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.model_selection import BaseCrossValidator

from phykit.services.tree.vcv_utils import build_vcv_matrix


class PhyloDistanceCV(BaseCrossValidator):
    """Cross-validation splitter based on phylogenetic distance."""

    def __init__(
        self,
        tree,
        species_names: List[str],
        n_splits: int = 5,
        min_dist: Optional[float] = None,
    ):
        self.tree = tree
        self.species_names = species_names
        self.n_splits = n_splits
        self.min_dist = min_dist
        self._groups = self._compute_groups()

    def _compute_groups(self) -> np.ndarray:
        vcv = build_vcv_matrix(self.tree, self.species_names)
        n = len(self.species_names)

        diag = np.diag(vcv)
        dist_matrix = diag[:, None] + diag[None, :] - 2 * vcv

        condensed = []
        for i in range(n):
            for j in range(i + 1, n):
                condensed.append(dist_matrix[i, j])
        condensed = np.array(condensed)

        Z = linkage(condensed, method="average")

        if self.min_dist is not None:
            groups = fcluster(Z, t=self.min_dist, criterion="distance")
        else:
            lo, hi = 0.0, float(condensed.max())
            mid = hi
            for _ in range(50):
                mid = (lo + hi) / 2.0
                groups = fcluster(Z, t=mid, criterion="distance")
                n_groups = len(set(groups))
                if n_groups > self.n_splits:
                    lo = mid
                elif n_groups < self.n_splits:
                    hi = mid
                else:
                    break
            groups = fcluster(Z, t=mid, criterion="distance")

        return groups

    def split(self, X=None, y=None, groups=None):
        unique_groups = sorted(set(self._groups))

        fold_for_group = {}
        for i, g in enumerate(unique_groups):
            fold_for_group[g] = i % self.n_splits

        indices = np.arange(len(self._groups))

        for fold in range(self.n_splits):
            test_mask = np.array([fold_for_group[g] == fold for g in self._groups])
            if not test_mask.any():
                continue
            train_idx = indices[~test_mask]
            test_idx = indices[test_mask]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
