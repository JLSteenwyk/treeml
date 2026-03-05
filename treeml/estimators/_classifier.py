from typing import List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from treeml.estimators._base import PhyloBaseEstimator


class PhyloRandomForestClassifier(PhyloBaseEstimator):
    """Random Forest classifier with phylogenetic correction."""

    def __init__(
        self,
        n_estimators: int = 100,
        include_eigenvectors: bool = True,
        eigenvector_variance: float = 0.90,
        random_state=None,
        **rf_kwargs,
    ):
        super().__init__(
            include_eigenvectors=include_eigenvectors,
            eigenvector_variance=eigenvector_variance,
        )
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.rf_kwargs = rf_kwargs

    def fit(self, X, y, tree=None, species_names=None):
        X = np.asarray(X)
        y = np.asarray(y)

        if tree is None or species_names is None:
            raise ValueError("tree and species_names are required for fit().")

        X_aug, eigvec_info = self._augment_features(X, tree, species_names)
        self.n_eigenvector_cols_ = eigvec_info["n_components"]
        self.n_features_original_ = X.shape[1]

        self.inner_model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            **self.rf_kwargs,
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
