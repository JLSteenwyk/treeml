import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator

from treeml._eigenvectors import extract_phylo_eigenvectors
from treeml._whitening import phylo_whiten, phylo_unwhiten


class PhyloBaseEstimator(BaseEstimator):
    """Base class for phylogenetic ML estimators."""

    def __init__(
        self,
        include_eigenvectors: bool = True,
        eigenvector_variance: float = 0.90,
        **kwargs,
    ):
        self.include_eigenvectors = include_eigenvectors
        self.eigenvector_variance = eigenvector_variance

    @staticmethod
    def _validate_tree(tree):
        """Warn about tree issues that affect phylogenetic correction."""
        root = tree.root
        if len(root.clades) > 2:
            warnings.warn(
                "Tree root has more than 2 children, which suggests an "
                "unrooted tree. Phylogenetic correction depends on root "
                "placement — consider rooting the tree first.",
                UserWarning,
                stacklevel=4,
            )
        terminals = tree.get_terminals()
        missing = [t.name for t in terminals if t.branch_length is None]
        if missing:
            warnings.warn(
                f"Tree has {len(missing)} terminal(s) with missing branch "
                f"lengths (e.g., {missing[0]!r}). This will produce "
                f"incorrect phylogenetic corrections.",
                UserWarning,
                stacklevel=4,
            )

    def _build_vcv(
        self,
        tree,
        ordered_names: List[str],
        gene_trees: Optional[List] = None,
    ) -> np.ndarray:
        self._validate_tree(tree)
        self.tree_ = tree
        self.species_names_ = list(ordered_names)
        self.gene_trees_ = gene_trees

        if gene_trees is not None:
            from phykit.services.tree.vcv_utils import build_discordance_vcv
            vcv, metadata = build_discordance_vcv(
                tree, gene_trees, ordered_names
            )
            self.discordance_metadata_ = metadata
            return vcv
        from phykit.services.tree.vcv_utils import build_vcv_matrix
        return build_vcv_matrix(tree, ordered_names)

    def _augment_features(
        self,
        X: np.ndarray,
        tree,
        ordered_names: List[str],
        gene_trees: Optional[List] = None,
    ) -> Tuple[np.ndarray, Dict]:
        if not self.include_eigenvectors:
            return X, {"n_components": 0}

        vcv = self._build_vcv(tree, ordered_names, gene_trees=gene_trees)
        E, info = extract_phylo_eigenvectors(
            tree, ordered_names,
            variance_threshold=self.eigenvector_variance,
            vcv=vcv,
        )
        X_aug = np.column_stack([X, E])
        return X_aug, info

    def _augment_features_predict(
        self,
        X_new: np.ndarray,
        tree,
        species_names: Optional[List[str]],
        n_eigenvector_cols: int,
        gene_trees: Optional[List] = None,
    ) -> Tuple[np.ndarray, bool]:
        if n_eigenvector_cols == 0:
            return X_new, tree is not None

        if tree is not None and species_names is not None:
            vcv = self._build_vcv(
                tree, species_names, gene_trees=gene_trees
            )
            E, _ = extract_phylo_eigenvectors(
                tree, species_names,
                variance_threshold=self.eigenvector_variance,
                vcv=vcv,
            )
            if E.shape[1] < n_eigenvector_cols:
                pad = np.zeros((E.shape[0], n_eigenvector_cols - E.shape[1]))
                E = np.column_stack([E, pad])
            elif E.shape[1] > n_eigenvector_cols:
                E = E[:, :n_eigenvector_cols]
            X_aug = np.column_stack([X_new, E])
            return X_aug, True
        else:
            warnings.warn(
                "No tree provided for prediction. "
                "Predictions made without phylogenetic correction.",
                UserWarning,
                stacklevel=2,
            )
            zeros = np.zeros((X_new.shape[0], n_eigenvector_cols))
            X_aug = np.column_stack([X_new, zeros])
            return X_aug, False
