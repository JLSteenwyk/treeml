import numpy as np
import pytest
from Bio import Phylo
from io import StringIO

from treeml.estimators._gradient_boosting_classifier import PhyloGradientBoostingClassifier


def _make_test_data():
    nwk = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0,(E:1.5,F:0.5):1.0);"
    tree = Phylo.read(StringIO(nwk), "newick")
    names = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(42)
    X = rng.standard_normal((6, 3))
    y = np.array([0, 1, 0, 1, 0, 1])
    return X, y, tree, names


def test_fit_returns_self():
    X, y, tree, names = _make_test_data()
    clf = PhyloGradientBoostingClassifier(n_estimators=10, random_state=42)
    result = clf.fit(X, y, tree=tree, species_names=names)
    assert result is clf


def test_predict_returns_labels():
    X, y, tree, names = _make_test_data()
    clf = PhyloGradientBoostingClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y, tree=tree, species_names=names)
    preds = clf.predict(X, tree=tree, species_names=names)
    assert preds.shape == (6,)
    assert set(preds).issubset({0, 1})


def test_predict_proba_returns_probabilities():
    X, y, tree, names = _make_test_data()
    clf = PhyloGradientBoostingClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y, tree=tree, species_names=names)
    proba = clf.predict_proba(X, tree=tree, species_names=names)
    assert proba.shape == (6, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_predict_without_tree_warns():
    X, y, tree, names = _make_test_data()
    clf = PhyloGradientBoostingClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y, tree=tree, species_names=names)
    with pytest.warns(UserWarning, match="without phylogenetic correction"):
        preds = clf.predict(X)
    assert preds.shape == (6,)


def test_without_eigenvectors():
    X, y, tree, names = _make_test_data()
    clf = PhyloGradientBoostingClassifier(
        n_estimators=10, random_state=42, include_eigenvectors=False
    )
    clf.fit(X, y, tree=tree, species_names=names)
    preds = clf.predict(X, tree=tree, species_names=names)
    assert preds.shape == (6,)
