from typing import List

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class PhyloCladeCV(BaseCrossValidator):
    """Cross-validation splitter that holds out monophyletic clades."""

    def __init__(
        self,
        tree,
        species_names: List[str],
        n_splits: int = 5,
        min_clade_size: int = 2,
    ):
        self.tree = tree
        self.species_names = species_names
        self.n_splits = n_splits
        self.min_clade_size = min_clade_size
        self._folds = self._compute_folds()

    def _compute_folds(self) -> List[List[int]]:
        name_to_idx = {name: i for i, name in enumerate(self.species_names)}
        n = len(self.species_names)

        clades = []
        for clade in self.tree.find_clades(order="level"):
            tips = [t.name for t in clade.get_terminals()]
            tip_indices = [name_to_idx[t] for t in tips if t in name_to_idx]
            if self.min_clade_size <= len(tip_indices) < n:
                clades.append(set(tip_indices))

        target_size = n / self.n_splits
        clades.sort(key=lambda c: abs(len(c) - target_size))

        selected = []
        assigned = set()
        for clade in clades:
            if len(selected) >= self.n_splits:
                break
            if not (clade & assigned):
                selected.append(sorted(clade))
                assigned.update(clade)

        remaining = sorted(set(range(n)) - assigned)
        if remaining:
            if selected:
                smallest_idx = min(range(len(selected)), key=lambda i: len(selected[i]))
                selected[smallest_idx].extend(remaining)
            else:
                selected.append(remaining)

        while len(selected) < self.n_splits:
            largest_idx = max(range(len(selected)), key=lambda i: len(selected[i]))
            fold = selected[largest_idx]
            if len(fold) < 2:
                break
            mid = len(fold) // 2
            selected[largest_idx] = fold[:mid]
            selected.append(fold[mid:])

        return selected

    def split(self, X=None, y=None, groups=None):
        all_indices = set(range(len(self.species_names)))
        for fold in self._folds:
            test_idx = np.array(fold)
            train_idx = np.array(sorted(all_indices - set(fold)))
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self._folds)
