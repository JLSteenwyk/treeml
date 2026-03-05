from typing import List, Optional

import numpy as np
from sklearn.svm import SVR

from treeml.estimators._base import PhyloBaseEstimator
from treeml._whitening import phylo_whiten, phylo_unwhiten


class PhyloSVMRegressor(PhyloBaseEstimator):
    """Support Vector Machine regressor with phylogenetic correction."""

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 0.1,
        include_eigenvectors: bool = True,
        eigenvector_variance: float = 0.90,
        **svm_kwargs,
    ):
        super().__init__(
            include_eigenvectors=include_eigenvectors,
            eigenvector_variance=eigenvector_variance,
        )
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.svm_kwargs = svm_kwargs

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

        self.inner_model_ = SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
            **self.svm_kwargs,
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
