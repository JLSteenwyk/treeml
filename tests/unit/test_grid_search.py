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


# --- Adapter tests ---

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


# --- PhyloGridSearchCV tests ---

def test_grid_search_fit():
    from treeml import PhyloRandomForestRegressor
    from treeml.cv._search import PhyloGridSearchCV
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
    from treeml.cv._search import PhyloGridSearchCV
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
    from treeml.cv._search import PhyloGridSearchCV
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
    from treeml.cv._search import PhyloGridSearchCV
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
    from treeml.cv._search import PhyloGridSearchCV
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
    from treeml.cv._search import PhyloGridSearchCV
    X, y, _, tree, names = _make_test_data()
    search = PhyloGridSearchCV(
        estimator=PhyloRandomForestRegressor(random_state=42),
        param_grid={"n_estimators": [10]},
        tree=tree,
        species_names=names,
    )
    search.fit(X, y)
    assert hasattr(search, "best_params_")


def test_grid_search_custom_cv():
    from treeml import PhyloRandomForestRegressor
    from treeml.cv._clade import PhyloCladeCV
    from treeml.cv._search import PhyloGridSearchCV
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
    from treeml.cv._search import PhyloGridSearchCV
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
    from treeml.cv._search import PhyloGridSearchCV
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
    from treeml.cv._search import PhyloGridSearchCV
    X, _, _, tree, names = _make_test_data()
    search = PhyloGridSearchCV(
        estimator=PhyloRandomForestRegressor(random_state=42),
        param_grid={"n_estimators": [10]},
        tree=tree,
        species_names=names,
    )
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):
        search.predict(X)


# --- PhyloRandomizedSearchCV tests ---

def test_randomized_search_fit():
    from treeml import PhyloRandomForestRegressor
    from treeml.cv._search import PhyloRandomizedSearchCV
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
    from treeml.cv._search import PhyloRandomizedSearchCV
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
    from treeml.cv._search import PhyloRandomizedSearchCV
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
