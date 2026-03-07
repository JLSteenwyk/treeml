from typing import List, Optional

import numpy as np
from sklearn.linear_model import Ridge

from treeml.estimators._base import PhyloBaseEstimator
from treeml._whitening import phylo_whiten, phylo_unwhiten


class PhyloRidge(PhyloBaseEstimator):
    """Ridge regressor with phylogenetic correction."""

    def __init__(
        self,
        alpha: float = 1.0,
        include_eigenvectors: bool = True,
        eigenvector_variance: float = 0.90,
        whiten_features: bool = True,
        whiten_target: bool = False,
        **ridge_kwargs,
    ):
        super().__init__(
            include_eigenvectors=include_eigenvectors,
            eigenvector_variance=eigenvector_variance,
            whiten_features=whiten_features,
        )
        self.alpha = alpha
        self.whiten_target = whiten_target
        self.ridge_kwargs = ridge_kwargs

    def fit(self, X, y, tree=None, species_names=None, gene_trees=None):
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)

        if tree is None or species_names is None:
            raise ValueError("tree and species_names are required for fit().")

        vcv = self._build_vcv(tree, species_names, gene_trees=gene_trees)

        if self.whiten_target:
            y_fit, L = phylo_whiten(y, tree, species_names, vcv=vcv)
            self.L_ = L
        else:
            y_fit = y
            self.L_ = None

        X_aug, eigvec_info = self._augment_features(
            X, tree, species_names, gene_trees=gene_trees
        )
        self.n_eigenvector_cols_ = eigvec_info["n_components"]
        self.n_features_original_ = X.shape[1]

        self.inner_model_ = Ridge(
            alpha=self.alpha,
            **self.ridge_kwargs,
        )
        self.inner_model_.fit(X_aug, y_fit)

        return self

    def predict(self, X, tree=None, species_names=None, gene_trees=None):
        X = np.asarray(X)

        X_aug, phylo_corrected = self._augment_features_predict(
            X, tree, species_names, self.n_eigenvector_cols_,
            gene_trees=gene_trees,
        )

        y_pred = self.inner_model_.predict(X_aug)

        if self.whiten_target and self.L_ is not None and phylo_corrected and tree is not None and species_names is not None:
            vcv = self._build_vcv(tree, species_names, gene_trees=gene_trees)
            L_pred = np.linalg.cholesky(vcv)
            return phylo_unwhiten(y_pred, L_pred)
        else:
            return y_pred
