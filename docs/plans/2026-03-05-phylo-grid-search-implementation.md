# PhyloGridSearchCV Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add phylogenetic-aware hyperparameter tuning (grid search and randomized search) that forwards tree/species_names/gene_trees through to estimator fit/predict during cross-validation.

**Architecture:** Adapter pattern — `_PhyloEstimatorAdapter` wraps a treeml estimator so it looks like standard sklearn (tree bound into fit/predict). `PhyloGridSearchCV` and `PhyloRandomizedSearchCV` wrap sklearn's search classes, default CV to PhyloDistanceCV, and unwrap `best_estimator_` back to the real treeml model.

**Tech Stack:** scikit-learn (GridSearchCV, RandomizedSearchCV, BaseEstimator), numpy

---

### Task 1: Create _PhyloEstimatorAdapter

**Files:**
- Create: `treeml/cv/_search.py`
- Test: `tests/unit/test_grid_search.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_grid_search.py`:

```python
import numpy as np
import pytest
from Bio import Phylo
from io import StringIO


def _make_test_data():
    nwk = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0,(E:1.5,F:0.5):1.0);"
    tree = Phylo.read(StringIO(nwk), "newick")
    names = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(42)
    X = rng.standard_normal((6, 3))
    y = rng.standard_normal(6)
    y_class = np.array([0, 1, 0, 1, 0, 1])
    return X, y, y_class, tree, names


def test_adapter_fit_predict():
    from treeml import PhyloRandomForestRegressor
    from treeml.cv._search import _PhyloEstimatorAdapter
    X, y, _, tree, names = _make_test_data()
    model = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    adapter = _PhyloEstimatorAdapter(model, tree, names)
    adapter.fit(X, y)
    preds = adapter.predict(X)
    assert preds.shape == (6,)


def test_adapter_get_set_params():
    from treeml import PhyloRandomForestRegressor
    from treeml.cv._search import _PhyloEstimatorAdapter
    X, _, _, tree, names = _make_test_data()
    model = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    adapter = _PhyloEstimatorAdapter(model, tree, names)
    params = adapter.get_params()
    assert params["n_estimators"] == 10
    adapter.set_params(n_estimators=50)
    assert adapter.get_params()["n_estimators"] == 50


def test_adapter_predict_proba():
    from treeml import PhyloRandomForestClassifier
    from treeml.cv._search import _PhyloEstimatorAdapter
    X, _, y_class, tree, names = _make_test_data()
    model = PhyloRandomForestClassifier(n_estimators=10, random_state=42)
    adapter = _PhyloEstimatorAdapter(model, tree, names)
    adapter.fit(X, y_class)
    proba = adapter.predict_proba(X)
    assert proba.shape[0] == 6
    assert proba.shape[1] == 2


def test_adapter_with_gene_trees():
    from treeml import PhyloRandomForestRegressor
    from treeml.cv._search import _PhyloEstimatorAdapter
    X, y, _, tree, names = _make_test_data()
    gt_nwk = "((A:0.8,C:0.8):1.2,(B:1.0,D:1.0):1.0,(E:1.0,F:1.0):1.0);"
    gt = [Phylo.read(StringIO(gt_nwk), "newick")]
    model = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    adapter = _PhyloEstimatorAdapter(model, tree, names, gene_trees=gt)
    adapter.fit(X, y)
    preds = adapter.predict(X)
    assert preds.shape == (6,)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_grid_search.py::test_adapter_fit_predict -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'treeml.cv._search'`

**Step 3: Write minimal implementation**

Create `treeml/cv/_search.py`:

```python
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class _PhyloEstimatorAdapter(BaseEstimator):
    """Wraps a treeml estimator so it looks like a standard sklearn estimator.

    Binds tree/species_names/gene_trees into fit/predict so that sklearn's
    GridSearchCV can call fit(X, y) and predict(X) without those kwargs.
    """

    def __init__(
        self,
        estimator,
        tree,
        species_names: List[str],
        gene_trees: Optional[List] = None,
    ):
        self.estimator = estimator
        self.tree = tree
        self.species_names = species_names
        self.gene_trees = gene_trees

    def fit(self, X, y, **kwargs):
        self.estimator.fit(
            X, y,
            tree=self.tree,
            species_names=self.species_names,
            gene_trees=self.gene_trees,
            **kwargs,
        )
        return self

    def predict(self, X):
        return self.estimator.predict(
            X,
            tree=self.tree,
            species_names=self.species_names,
            gene_trees=self.gene_trees,
        )

    def predict_proba(self, X):
        return self.estimator.predict_proba(
            X,
            tree=self.tree,
            species_names=self.species_names,
            gene_trees=self.gene_trees,
        )

    def get_params(self, deep=True):
        return self.estimator.get_params(deep=deep)

    def set_params(self, **params):
        self.estimator.set_params(**params)
        return self

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import accuracy_score, r2_score
        preds = self.predict(X)
        if hasattr(self.estimator, "predict_proba"):
            return accuracy_score(y, preds)
        return r2_score(y, preds)

    @property
    def classes_(self):
        """Forward classes_ for classifiers (needed by sklearn scoring)."""
        return self.estimator.inner_model_.classes_

    def _more_tags(self):
        """Tell sklearn this can handle its test suite."""
        return {"no_validation": True}
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_grid_search.py -v -k adapter`
Expected: All 4 adapter tests PASS

**Step 5: Commit**

```bash
git add treeml/cv/_search.py tests/unit/test_grid_search.py
git commit -m "feat: add _PhyloEstimatorAdapter for sklearn compatibility"
```

---

### Task 2: Implement PhyloGridSearchCV

**Files:**
- Modify: `treeml/cv/_search.py`
- Test: `tests/unit/test_grid_search.py`

**Step 1: Write the failing tests**

Add to `tests/unit/test_grid_search.py`:

```python
from treeml import PhyloGridSearchCV


def test_grid_search_fit():
    from treeml import PhyloRandomForestRegressor
    X, y, _, tree, names = _make_test_data()
    search = PhyloGridSearchCV(
        estimator=PhyloRandomForestRegressor(random_state=42),
        param_grid={"n_estimators": [10, 20]},
        tree=tree,
        species_names=names,
        cv=3,
    )
    search.fit(X, y)
    assert hasattr(search, "best_params_")
    assert "n_estimators" in search.best_params_


def test_grid_search_best_score():
    from treeml import PhyloRandomForestRegressor
    X, y, _, tree, names = _make_test_data()
    search = PhyloGridSearchCV(
        estimator=PhyloRandomForestRegressor(random_state=42),
        param_grid={"n_estimators": [10, 20]},
        tree=tree,
        species_names=names,
        cv=3,
    )
    search.fit(X, y)
    assert isinstance(search.best_score_, float)


def test_grid_search_best_estimator_is_unwrapped():
    from treeml import PhyloRandomForestRegressor
    X, y, _, tree, names = _make_test_data()
    search = PhyloGridSearchCV(
        estimator=PhyloRandomForestRegressor(random_state=42),
        param_grid={"n_estimators": [10, 20]},
        tree=tree,
        species_names=names,
        cv=3,
    )
    search.fit(X, y)
    assert isinstance(search.best_estimator_, PhyloRandomForestRegressor)


def test_grid_search_cv_results():
    from treeml import PhyloRandomForestRegressor
    X, y, _, tree, names = _make_test_data()
    search = PhyloGridSearchCV(
        estimator=PhyloRandomForestRegressor(random_state=42),
        param_grid={"n_estimators": [10, 20]},
        tree=tree,
        species_names=names,
        cv=3,
    )
    search.fit(X, y)
    assert "mean_test_score" in search.cv_results_
    assert "params" in search.cv_results_


def test_grid_search_predict():
    from treeml import PhyloRandomForestRegressor
    X, y, _, tree, names = _make_test_data()
    search = PhyloGridSearchCV(
        estimator=PhyloRandomForestRegressor(random_state=42),
        param_grid={"n_estimators": [10, 20]},
        tree=tree,
        species_names=names,
        cv=3,
    )
    search.fit(X, y)
    preds = search.predict(X)
    assert preds.shape == (6,)


def test_grid_search_default_cv():
    from treeml import PhyloRandomForestRegressor
    from treeml.cv._distance import PhyloDistanceCV
    X, y, _, tree, names = _make_test_data()
    search = PhyloGridSearchCV(
        estimator=PhyloRandomForestRegressor(random_state=42),
        param_grid={"n_estimators": [10]},
        tree=tree,
        species_names=names,
    )
    search.fit(X, y)
    # Should work without specifying cv (defaults to PhyloDistanceCV)
    assert hasattr(search, "best_params_")


def test_grid_search_custom_cv():
    from treeml import PhyloRandomForestRegressor, PhyloCladeCV
    X, y, _, tree, names = _make_test_data()
    cv = PhyloCladeCV(tree=tree, species_names=names, n_splits=3)
    search = PhyloGridSearchCV(
        estimator=PhyloRandomForestRegressor(random_state=42),
        param_grid={"n_estimators": [10]},
        tree=tree,
        species_names=names,
        cv=cv,
    )
    search.fit(X, y)
    assert hasattr(search, "best_params_")


def test_grid_search_classifier():
    from treeml import PhyloRandomForestClassifier
    X, _, y_class, tree, names = _make_test_data()
    search = PhyloGridSearchCV(
        estimator=PhyloRandomForestClassifier(random_state=42),
        param_grid={"n_estimators": [10, 20]},
        tree=tree,
        species_names=names,
        cv=3,
    )
    search.fit(X, y_class)
    preds = search.predict(X)
    assert set(preds).issubset({0, 1})


def test_grid_search_with_gene_trees():
    from treeml import PhyloRandomForestRegressor
    X, y, _, tree, names = _make_test_data()
    gt_nwk = "((A:0.8,C:0.8):1.2,(B:1.0,D:1.0):1.0,(E:1.0,F:1.0):1.0);"
    gt = [Phylo.read(StringIO(gt_nwk), "newick")]
    search = PhyloGridSearchCV(
        estimator=PhyloRandomForestRegressor(random_state=42),
        param_grid={"n_estimators": [10]},
        tree=tree,
        species_names=names,
        gene_trees=gt,
        cv=3,
    )
    search.fit(X, y)
    assert hasattr(search, "best_params_")


def test_unfitted_predict_raises():
    from treeml import PhyloRandomForestRegressor
    X, _, _, tree, names = _make_test_data()
    search = PhyloGridSearchCV(
        estimator=PhyloRandomForestRegressor(random_state=42),
        param_grid={"n_estimators": [10]},
        tree=tree,
        species_names=names,
    )
    with pytest.raises(Exception):
        search.predict(X)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_grid_search.py::test_grid_search_fit -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Add to `treeml/cv/_search.py` after the `_PhyloEstimatorAdapter` class:

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def _auto_scoring(y):
    """Auto-detect scoring metric from target values."""
    unique_y = np.unique(y)
    is_classification = len(unique_y) <= 20 and np.all(unique_y == unique_y.astype(int))
    return "accuracy" if is_classification else "r2"


class PhyloGridSearchCV:
    """Grid search with phylogenetic cross-validation.

    Wraps sklearn's GridSearchCV, automatically forwarding tree/species_names
    to the estimator's fit/predict during cross-validation.

    Parameters
    ----------
    estimator : PhyloBaseEstimator
        A treeml estimator.
    param_grid : dict or list of dicts
        Hyperparameter grid (same as sklearn GridSearchCV).
    tree : Bio.Phylo tree
        Phylogenetic tree.
    species_names : list of str
        Species names matching rows of X.
    gene_trees : list of trees, optional
        Gene trees for discordance-aware VCV.
    cv : int or CV splitter, optional
        Cross-validation strategy. If None, defaults to
        PhyloDistanceCV(tree, species_names, n_splits=5).
        If int, uses PhyloDistanceCV with that many splits.
    scoring : str, optional
        Scoring metric. Auto-detects from y if None.
    n_jobs : int, optional
        Number of parallel jobs (passed to sklearn).
    refit : bool, default=True
        Refit best model on full dataset.
    **kwargs
        Additional arguments passed to sklearn GridSearchCV.
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
        self.estimator = estimator
        self.param_grid = param_grid
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

    def fit(self, X, y):
        adapter = _PhyloEstimatorAdapter(
            self.estimator, self.tree, self.species_names, self.gene_trees
        )

        scoring = self.scoring if self.scoring is not None else _auto_scoring(y)
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
        self._inner_search.fit(X, y)
        return self

    def predict(self, X):
        return self._inner_search.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self._inner_search.best_estimator_.predict_proba(X)

    @property
    def best_params_(self):
        return self._inner_search.best_params_

    @property
    def best_score_(self):
        return self._inner_search.best_score_

    @property
    def best_estimator_(self):
        """Return the unwrapped treeml estimator, not the adapter."""
        return self._inner_search.best_estimator_.estimator

    @property
    def cv_results_(self):
        return self._inner_search.cv_results_
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_grid_search.py -v -k "grid_search"`
Expected: All 10 grid search tests PASS

**Step 5: Commit**

```bash
git add treeml/cv/_search.py tests/unit/test_grid_search.py
git commit -m "feat: add PhyloGridSearchCV with adapter pattern"
```

---

### Task 3: Implement PhyloRandomizedSearchCV

**Files:**
- Modify: `treeml/cv/_search.py`
- Test: `tests/unit/test_grid_search.py`

**Step 1: Write the failing tests**

Add to `tests/unit/test_grid_search.py`:

```python
from treeml import PhyloRandomizedSearchCV


def test_randomized_search_fit():
    from treeml import PhyloRandomForestRegressor
    X, y, _, tree, names = _make_test_data()
    search = PhyloRandomizedSearchCV(
        estimator=PhyloRandomForestRegressor(random_state=42),
        param_distributions={"n_estimators": [10, 20, 50]},
        n_iter=2,
        tree=tree,
        species_names=names,
        cv=3,
        random_state=42,
    )
    search.fit(X, y)
    assert hasattr(search, "best_params_")
    assert "n_estimators" in search.best_params_


def test_randomized_search_best_estimator_unwrapped():
    from treeml import PhyloRandomForestRegressor
    X, y, _, tree, names = _make_test_data()
    search = PhyloRandomizedSearchCV(
        estimator=PhyloRandomForestRegressor(random_state=42),
        param_distributions={"n_estimators": [10, 20]},
        n_iter=2,
        tree=tree,
        species_names=names,
        cv=3,
        random_state=42,
    )
    search.fit(X, y)
    assert isinstance(search.best_estimator_, PhyloRandomForestRegressor)


def test_randomized_search_predict():
    from treeml import PhyloRandomForestRegressor
    X, y, _, tree, names = _make_test_data()
    search = PhyloRandomizedSearchCV(
        estimator=PhyloRandomForestRegressor(random_state=42),
        param_distributions={"n_estimators": [10, 20]},
        n_iter=2,
        tree=tree,
        species_names=names,
        cv=3,
        random_state=42,
    )
    search.fit(X, y)
    preds = search.predict(X)
    assert preds.shape == (6,)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_grid_search.py::test_randomized_search_fit -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Add to `treeml/cv/_search.py` after `PhyloGridSearchCV`:

```python
class PhyloRandomizedSearchCV:
    """Randomized search with phylogenetic cross-validation.

    Wraps sklearn's RandomizedSearchCV, automatically forwarding
    tree/species_names to the estimator's fit/predict during cross-validation.

    Parameters
    ----------
    estimator : PhyloBaseEstimator
        A treeml estimator.
    param_distributions : dict
        Hyperparameter distributions (same as sklearn RandomizedSearchCV).
    tree : Bio.Phylo tree
        Phylogenetic tree.
    species_names : list of str
        Species names matching rows of X.
    gene_trees : list of trees, optional
        Gene trees for discordance-aware VCV.
    n_iter : int, default=10
        Number of parameter settings sampled.
    cv : int or CV splitter, optional
        Cross-validation strategy. If None, defaults to
        PhyloDistanceCV(tree, species_names, n_splits=5).
        If int, uses PhyloDistanceCV with that many splits.
    scoring : str, optional
        Scoring metric. Auto-detects from y if None.
    n_jobs : int, optional
        Number of parallel jobs.
    refit : bool, default=True
        Refit best model on full dataset.
    random_state : int, optional
        Random state for reproducibility.
    **kwargs
        Additional arguments passed to sklearn RandomizedSearchCV.
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
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.tree = tree
        self.species_names = species_names
        self.gene_trees = gene_trees
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.random_state = random_state
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

    def fit(self, X, y):
        adapter = _PhyloEstimatorAdapter(
            self.estimator, self.tree, self.species_names, self.gene_trees
        )

        scoring = self.scoring if self.scoring is not None else _auto_scoring(y)
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
        self._inner_search.fit(X, y)
        return self

    def predict(self, X):
        return self._inner_search.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self._inner_search.best_estimator_.predict_proba(X)

    @property
    def best_params_(self):
        return self._inner_search.best_params_

    @property
    def best_score_(self):
        return self._inner_search.best_score_

    @property
    def best_estimator_(self):
        """Return the unwrapped treeml estimator, not the adapter."""
        return self._inner_search.best_estimator_.estimator

    @property
    def cv_results_(self):
        return self._inner_search.cv_results_
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_grid_search.py -v`
Expected: All 17 tests PASS (4 adapter + 10 grid search + 3 randomized search)

**Step 5: Commit**

```bash
git add treeml/cv/_search.py tests/unit/test_grid_search.py
git commit -m "feat: add PhyloRandomizedSearchCV"
```

---

### Task 4: Export from treeml and update __init__.py

**Files:**
- Modify: `treeml/__init__.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_grid_search.py`:

```python
def test_importable_from_treeml():
    from treeml import PhyloGridSearchCV, PhyloRandomizedSearchCV
    assert PhyloGridSearchCV is not None
    assert PhyloRandomizedSearchCV is not None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_grid_search.py::test_importable_from_treeml -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

In `treeml/__init__.py`, add after the existing cv imports:

```python
from treeml.cv._search import PhyloGridSearchCV, PhyloRandomizedSearchCV
```

And add to `__all__`:

```python
"PhyloGridSearchCV",
"PhyloRandomizedSearchCV",
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_grid_search.py::test_importable_from_treeml -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add treeml/__init__.py tests/unit/test_grid_search.py
git commit -m "feat: export PhyloGridSearchCV and PhyloRandomizedSearchCV"
```

---

### Task 5: Add integration test

**Files:**
- Modify: `tests/integration/test_end_to_end.py`

**Step 1: Write the test**

Add to `tests/integration/test_end_to_end.py`:

```python
@pytest.mark.integration
class TestPhyloGridSearchEndToEnd:
    def test_grid_search_regression(self):
        from treeml import PhyloGridSearchCV
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        search = PhyloGridSearchCV(
            estimator=PhyloRandomForestRegressor(random_state=42),
            param_grid={"n_estimators": [10, 50]},
            tree=tree,
            species_names=names,
            cv=3,
        )
        search.fit(X, y)
        assert isinstance(search.best_estimator_, PhyloRandomForestRegressor)
        preds = search.predict(X)
        assert preds.shape == y.shape

    def test_randomized_search_regression(self):
        from treeml import PhyloRandomizedSearchCV
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        search = PhyloRandomizedSearchCV(
            estimator=PhyloRidge(),
            param_distributions={"alpha": [0.1, 1.0, 10.0]},
            n_iter=2,
            tree=tree,
            species_names=names,
            cv=3,
            random_state=42,
        )
        search.fit(X, y)
        assert hasattr(search, "best_params_")
        preds = search.predict(X)
        assert preds.shape == y.shape
```

**Step 2: Run tests**

Run: `python -m pytest tests/integration/test_end_to_end.py::TestPhyloGridSearchEndToEnd -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_end_to_end.py
git commit -m "test: add grid search integration tests"
```

---

### Task 6: Update docs

**Files:**
- Modify: `docs/usage/index.rst`

**Step 1: Update usage docs**

Add a "Hyperparameter Tuning" section after "Model Comparison" in `docs/usage/index.rst`:

```rst
Hyperparameter Tuning
~~~~~~~~~~~~~~~~~~~~~

- ``PhyloGridSearchCV`` -- Grid search with phylogenetic cross-validation
- ``PhyloRandomizedSearchCV`` -- Randomized search with phylogenetic cross-validation

Example usage:

.. code-block:: python

   from treeml import PhyloRandomForestRegressor, PhyloGridSearchCV

   model = PhyloRandomForestRegressor(random_state=42)

   search = PhyloGridSearchCV(
       estimator=model,
       param_grid={
           "n_estimators": [50, 100, 200],
           "eigenvector_variance": [0.8, 0.9, 0.95],
       },
       tree=tree,
       species_names=names,
       n_jobs=-1,
   )
   search.fit(X, y)

   print(f"Best params: {search.best_params_}")
   print(f"Best score: {search.best_score_:.3f}")

   # Predict with best model
   preds = search.predict(X)
```

**Step 2: Commit**

```bash
git add docs/usage/index.rst
git commit -m "docs: add hyperparameter tuning section"
```

---

### Task 7: Final verification

**Step 1: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 2: Verify imports**

Run: `python -c "from treeml import PhyloGridSearchCV, PhyloRandomizedSearchCV; print('OK')"`
Expected: prints `OK`
