from typing import List, Optional

import numpy as np
from sklearn.svm import SVC

from treeml.estimators._base import PhyloBaseEstimator


class PhyloSVMClassifier(PhyloBaseEstimator):
    """Support Vector Machine classifier with phylogenetic correction."""

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        include_eigenvectors: bool = True,
        eigenvector_variance: float = 0.90,
        random_state=None,
        **svm_kwargs,
    ):
        super().__init__(
            include_eigenvectors=include_eigenvectors,
            eigenvector_variance=eigenvector_variance,
        )
        self.kernel = kernel
        self.C = C
        self.random_state = random_state
        self.svm_kwargs = svm_kwargs

    def fit(self, X, y, tree=None, species_names=None):
        X = np.asarray(X)
        y = np.asarray(y)

        if tree is None or species_names is None:
            raise ValueError("tree and species_names are required for fit().")

        X_aug, eigvec_info = self._augment_features(X, tree, species_names)
        self.n_eigenvector_cols_ = eigvec_info["n_components"]
        self.n_features_original_ = X.shape[1]

        self.inner_model_ = SVC(
            kernel=self.kernel,
            C=self.C,
            probability=True,
            random_state=self.random_state,
            **self.svm_kwargs,
        )
        self.inner_model_.fit(X_aug, y)

        return self

    def predict(self, X, tree=None, species_names=None):
        X = np.asarray(X)
        X_aug, _ = self._augment_features_predict(
            X, tree, species_names, self.n_eigenvector_cols_
        )
        return self.inner_model_.predict(X_aug)

    def predict_proba(self, X, tree=None, species_names=None):
        X = np.asarray(X)
        X_aug, _ = self._augment_features_predict(
            X, tree, species_names, self.n_eigenvector_cols_
        )
        return self.inner_model_.predict_proba(X_aug)
