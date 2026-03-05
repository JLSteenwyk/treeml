import numpy as np
import pytest
from Bio import Phylo
from io import StringIO

from treeml.estimators._gradient_boosting_regressor import PhyloGradientBoostingRegressor


def _make_test_data():
    nwk = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0,(E:1.5,F:0.5):1.0);"
    tree = Phylo.read(StringIO(nwk), "newick")
    names = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(42)
    X = rng.standard_normal((6, 3))
    y = rng.standard_normal(6)
    return X, y, tree, names


def test_fit_returns_self():
    X, y, tree, names = _make_test_data()
    model = PhyloGradientBoostingRegressor(n_estimators=10, random_state=42)
    result = model.fit(X, y, tree=tree, species_names=names)
    assert result is model


def test_predict_with_tree():
    X, y, tree, names = _make_test_data()
    model = PhyloGradientBoostingRegressor(n_estimators=10, random_state=42)
    model.fit(X, y, tree=tree, species_names=names)
    preds = model.predict(X, tree=tree, species_names=names)
    assert preds.shape == (6,)


def test_predict_without_tree_warns():
    X, y, tree, names = _make_test_data()
    model = PhyloGradientBoostingRegressor(n_estimators=10, random_state=42)
    model.fit(X, y, tree=tree, species_names=names)
    with pytest.warns(UserWarning, match="without phylogenetic correction"):
        preds = model.predict(X)
    assert preds.shape == (6,)


def test_predict_without_eigenvectors():
    X, y, tree, names = _make_test_data()
    model = PhyloGradientBoostingRegressor(
        n_estimators=10, random_state=42, include_eigenvectors=False
    )
    model.fit(X, y, tree=tree, species_names=names)
    preds = model.predict(X, tree=tree, species_names=names)
    assert preds.shape == (6,)


def test_fit_stores_training_metadata():
    X, y, tree, names = _make_test_data()
    model = PhyloGradientBoostingRegressor(n_estimators=10, random_state=42)
    model.fit(X, y, tree=tree, species_names=names)
    assert hasattr(model, "n_eigenvector_cols_")
    assert hasattr(model, "L_")
    assert hasattr(model, "inner_model_")
