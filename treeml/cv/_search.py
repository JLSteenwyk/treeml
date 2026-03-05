from typing import List, Optional

import copy

import numpy as np
from sklearn.base import BaseEstimator, is_classifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def _auto_scoring(estimator, y):
    """Auto-detect scoring metric from estimator type, falling back to y."""
    if is_classifier(estimator):
        return "accuracy"
    if hasattr(estimator, "predict_proba"):
        return "accuracy"
    unique_y = np.unique(y)
    is_classification = len(unique_y) <= 20 and np.all(unique_y == unique_y.astype(int))
    return "accuracy" if is_classification else "r2"


class _PhyloEstimatorAdapter(BaseEstimator):
    """Wraps a treeml estimator so it looks like a standard sklearn estimator.

    Binds tree/species_names/gene_trees into fit/predict so that sklearn's
    GridSearchCV can call fit(X, y) and predict(X) without those kwargs.

    When used inside GridSearchCV, X has an extra last column containing
    integer row indices that map back into the full species_names list.
    This allows the adapter to determine which species correspond to the
    subset of rows that sklearn passes during cross-validation.

    get_params/set_params deliberately delegate to the inner estimator
    (not the adapter's own constructor params) so that param_grid keys
    map directly to the inner estimator's hyperparameters. The custom
    __sklearn_clone__ handles proper cloning despite this.
    """

    def __init__(
        self,
        estimator,
        tree,
        species_names: List[str],
        gene_trees: Optional[List] = None,
        _has_index_col: bool = False,
    ):
        self.estimator = estimator
        self.tree = tree
        self.species_names = species_names
        self.gene_trees = gene_trees
        self._has_index_col = _has_index_col

    def _split_X(self, X):
        """Split X into real features, subset species names, and gene_trees."""
        X = np.asarray(X)
        if self._has_index_col:
            indices = X[:, -1].astype(int)
            X_real = X[:, :-1]
            subset_names = [self.species_names[i] for i in indices]
            gt = self.gene_trees if len(subset_names) >= 3 else None
            return X_real, subset_names, gt
        return X, list(self.species_names), self.gene_trees

    def fit(self, X, y, **kwargs):
        X_real, subset_names, gt = self._split_X(X)
        self.estimator.fit(
            X_real, y,
            tree=self.tree,
            species_names=subset_names,
            gene_trees=gt,
            **kwargs,
        )
        return self

    def predict(self, X):
        X_real, subset_names, gt = self._split_X(X)
        return self.estimator.predict(
            X_real,
            tree=self.tree,
            species_names=subset_names,
            gene_trees=gt,
        )

    def predict_proba(self, X):
        X_real, subset_names, gt = self._split_X(X)
        return self.estimator.predict_proba(
            X_real,
            tree=self.tree,
            species_names=subset_names,
            gene_trees=gt,
        )

    def __sklearn_clone__(self):
        """Custom clone that preserves tree/species_names bindings."""
        cloned_estimator = copy.deepcopy(self.estimator)
        return _PhyloEstimatorAdapter(
            estimator=cloned_estimator,
            tree=self.tree,
            species_names=self.species_names,
            gene_trees=self.gene_trees,
            _has_index_col=self._has_index_col,
        )

    def get_params(self, deep=True):
        return self.estimator.get_params(deep=deep)

    def set_params(self, **params):
        self.estimator.set_params(**params)
        return self

    @property
    def classes_(self):
        return self.estimator.inner_model_.classes_

    def _more_tags(self):
        return {"no_validation": True}


class _BasePhyloSearchCV:
    """Shared base for PhyloGridSearchCV and PhyloRandomizedSearchCV."""

    def __init__(
        self,
        estimator,
        tree,
        species_names: List[str],
        gene_trees: Optional[List] = None,
        cv=None,
        scoring=None,
        n_jobs=None,
        refit=True,
        **kwargs,
    ):
        self.estimator = estimator
        self.tree = tree
        self.species_names = species_names
        self.gene_trees = gene_trees
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.kwargs = kwargs

    def _resolve_cv(self):
        from treeml.cv._distance import PhyloDistanceCV
        if self.cv is None:
            return PhyloDistanceCV(
                tree=self.tree,
                species_names=self.species_names,
                n_splits=5,
            )
        if isinstance(self.cv, int):
            return PhyloDistanceCV(
                tree=self.tree,
                species_names=self.species_names,
                n_splits=self.cv,
            )
        return self.cv

    @staticmethod
    def _add_index_col(X):
        """Append an index column to X so row identity survives CV slicing."""
        X = np.asarray(X, dtype=float)
        idx = np.arange(X.shape[0], dtype=float).reshape(-1, 1)
        return np.hstack([X, idx])

    def _check_fitted(self):
        if not hasattr(self, "_inner_search"):
            from sklearn.exceptions import NotFittedError
            raise NotFittedError(
                "This search instance is not fitted yet. Call 'fit' first."
            )

    def predict(self, X):
        self._check_fitted()
        X_idx = self._add_index_col(X)
        return self._inner_search.best_estimator_.predict(X_idx)

    def predict_proba(self, X):
        self._check_fitted()
        X_idx = self._add_index_col(X)
        return self._inner_search.best_estimator_.predict_proba(X_idx)

    @property
    def best_params_(self):
        self._check_fitted()
        return self._inner_search.best_params_

    @property
    def best_score_(self):
        self._check_fitted()
        return self._inner_search.best_score_

    @property
    def best_estimator_(self):
        self._check_fitted()
        return self._inner_search.best_estimator_.estimator

    @property
    def cv_results_(self):
        self._check_fitted()
        return self._inner_search.cv_results_


class PhyloGridSearchCV(_BasePhyloSearchCV):
    """Grid search with phylogenetic cross-validation.

    Wraps sklearn's GridSearchCV, automatically forwarding tree/species_names
    to the estimator's fit/predict during cross-validation.
    """

    def __init__(
        self,
        estimator,
        param_grid,
        tree,
        species_names: List[str],
        gene_trees: Optional[List] = None,
        cv=None,
        scoring=None,
        n_jobs=None,
        refit=True,
        **kwargs,
    ):
        super().__init__(
            estimator=estimator,
            tree=tree,
            species_names=species_names,
            gene_trees=gene_trees,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            **kwargs,
        )
        self.param_grid = param_grid

    def fit(self, X, y):
        X_idx = self._add_index_col(X)
        adapter = _PhyloEstimatorAdapter(
            self.estimator, self.tree, self.species_names, self.gene_trees,
            _has_index_col=True,
        )
        scoring = self.scoring if self.scoring is not None else _auto_scoring(self.estimator, y)
        cv = self._resolve_cv()
        self._inner_search = GridSearchCV(
            estimator=adapter,
            param_grid=self.param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=self.n_jobs,
            refit=self.refit,
            **self.kwargs,
        )
        self._inner_search.fit(X_idx, y)
        return self


class PhyloRandomizedSearchCV(_BasePhyloSearchCV):
    """Randomized search with phylogenetic cross-validation.

    Wraps sklearn's RandomizedSearchCV, automatically forwarding
    tree/species_names to the estimator's fit/predict during cross-validation.
    """

    def __init__(
        self,
        estimator,
        param_distributions,
        tree,
        species_names: List[str],
        gene_trees: Optional[List] = None,
        n_iter: int = 10,
        cv=None,
        scoring=None,
        n_jobs=None,
        refit=True,
        random_state=None,
        **kwargs,
    ):
        super().__init__(
            estimator=estimator,
            tree=tree,
            species_names=species_names,
            gene_trees=gene_trees,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            **kwargs,
        )
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        X_idx = self._add_index_col(X)
        adapter = _PhyloEstimatorAdapter(
            self.estimator, self.tree, self.species_names, self.gene_trees,
            _has_index_col=True,
        )
        scoring = self.scoring if self.scoring is not None else _auto_scoring(self.estimator, y)
        cv = self._resolve_cv()
        self._inner_search = RandomizedSearchCV(
            estimator=adapter,
            param_distributions=self.param_distributions,
            n_iter=self.n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=self.n_jobs,
            refit=self.refit,
            random_state=self.random_state,
            **self.kwargs,
        )
        self._inner_search.fit(X_idx, y)
        return self
