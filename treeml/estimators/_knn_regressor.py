from typing import List, Optional

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from treeml.estimators._base import PhyloBaseEstimator
from treeml._whitening import phylo_whiten, phylo_unwhiten


class PhyloKNNRegressor(PhyloBaseEstimator):
    """K-Nearest Neighbors regressor with phylogenetic correction."""

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        include_eigenvectors: bool = True,
        eigenvector_variance: float = 0.90,
        **knn_kwargs,
    ):
        super().__init__(
            include_eigenvectors=include_eigenvectors,
            eigenvector_variance=eigenvector_variance,
        )
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.knn_kwargs = knn_kwargs

    def fit(self, X, y, tree=None, species_names=None, gene_trees=None):
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)

        if tree is None or species_names is None:
            raise ValueError("tree and species_names are required for fit().")

        vcv = self._build_vcv(tree, species_names, gene_trees=gene_trees)
        y_white, L = phylo_whiten(y, tree, species_names, vcv=vcv)
        self.L_ = L

        X_aug, eigvec_info = self._augment_features(
            X, tree, species_names, gene_trees=gene_trees
        )
        self.n_eigenvector_cols_ = eigvec_info["n_components"]
        self.n_features_original_ = X.shape[1]

        # Clamp n_neighbors to at most n_samples - 1
        n_samples = X_aug.shape[0]
        effective_k = min(self.n_neighbors, n_samples - 1)

        self.inner_model_ = KNeighborsRegressor(
            n_neighbors=effective_k,
            weights=self.weights,
            **self.knn_kwargs,
        )
        self.inner_model_.fit(X_aug, y_white)

        return self

    def predict(self, X, tree=None, species_names=None, gene_trees=None):
        X = np.asarray(X)

        X_aug, phylo_corrected = self._augment_features_predict(
            X, tree, species_names, self.n_eigenvector_cols_,
            gene_trees=gene_trees,
        )

        y_pred_white = self.inner_model_.predict(X_aug)

        if phylo_corrected and tree is not None and species_names is not None:
            vcv = self._build_vcv(tree, species_names, gene_trees=gene_trees)
            L_pred = np.linalg.cholesky(vcv)
            return phylo_unwhiten(y_pred_white, L_pred)
        else:
            return y_pred_white
