from typing import List, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from treeml.estimators._base import PhyloBaseEstimator
from treeml._whitening import phylo_whiten, phylo_unwhiten


class PhyloGradientBoostingRegressor(PhyloBaseEstimator):
    """Gradient Boosting regressor with phylogenetic correction."""

    def __init__(
        self,
        n_estimators: int = 100,
        include_eigenvectors: bool = True,
        eigenvector_variance: float = 0.90,
        random_state=None,
        **gb_kwargs,
    ):
        super().__init__(
            include_eigenvectors=include_eigenvectors,
            eigenvector_variance=eigenvector_variance,
        )
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.gb_kwargs = gb_kwargs

    def fit(self, X, y, tree=None, species_names=None):
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)

        if tree is None or species_names is None:
            raise ValueError("tree and species_names are required for fit().")

        y_white, L = phylo_whiten(y, tree, species_names)
        self.L_ = L

        X_aug, eigvec_info = self._augment_features(X, tree, species_names)
        self.n_eigenvector_cols_ = eigvec_info["n_components"]
        self.n_features_original_ = X.shape[1]

        self.inner_model_ = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            **self.gb_kwargs,
        )
        self.inner_model_.fit(X_aug, y_white)

        return self

    def predict(self, X, tree=None, species_names=None):
        X = np.asarray(X)

        X_aug, phylo_corrected = self._augment_features_predict(
            X, tree, species_names, self.n_eigenvector_cols_
        )

        y_pred_white = self.inner_model_.predict(X_aug)

        if phylo_corrected and tree is not None and species_names is not None:
            from phykit.services.tree.vcv_utils import build_vcv_matrix
            C = build_vcv_matrix(tree, species_names)
            L_pred = np.linalg.cholesky(C)
            return phylo_unwhiten(y_pred_white, L_pred)
        else:
            return y_pred_white
