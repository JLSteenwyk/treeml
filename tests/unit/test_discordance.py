import numpy as np
import pytest
from Bio import Phylo
from io import StringIO

from treeml.estimators._regressor import PhyloRandomForestRegressor
from treeml.estimators._classifier import PhyloRandomForestClassifier
from treeml.estimators._gradient_boosting_regressor import PhyloGradientBoostingRegressor
from treeml.estimators._svm_regressor import PhyloSVMRegressor
from treeml.estimators._ridge import PhyloRidge


def _make_test_data():
    """Species tree + 3 gene trees with same taxa but different topologies."""
    sp_nwk = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
    gt1_nwk = "((A:0.8,C:0.8):1.2,(B:1.0,D:1.0):1.0);"
    gt2_nwk = "((A:1.1,B:1.1):0.9,(C:0.9,D:0.9):1.1);"
    gt3_nwk = "((A:1.0,D:1.0):1.0,(B:1.0,C:1.0):1.0);"

    sp_tree = Phylo.read(StringIO(sp_nwk), "newick")
    gene_trees = [
        Phylo.read(StringIO(gt1_nwk), "newick"),
        Phylo.read(StringIO(gt2_nwk), "newick"),
        Phylo.read(StringIO(gt3_nwk), "newick"),
    ]
    names = ["A", "B", "C", "D"]
    rng = np.random.default_rng(42)
    X = rng.standard_normal((4, 2))
    y = rng.standard_normal(4)
    y_class = np.array([0, 1, 0, 1])
    return X, y, y_class, sp_tree, gene_trees, names


def test_regressor_fit_with_gene_trees():
    X, y, _, sp_tree, gene_trees, names = _make_test_data()
    model = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    result = model.fit(X, y, tree=sp_tree, species_names=names, gene_trees=gene_trees)
    assert result is model
    assert hasattr(model, "discordance_metadata_")
    assert model.discordance_metadata_["n_gene_trees"] == 3


def test_regressor_predict_with_gene_trees():
    X, y, _, sp_tree, gene_trees, names = _make_test_data()
    model = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y, tree=sp_tree, species_names=names, gene_trees=gene_trees)
    preds = model.predict(X, tree=sp_tree, species_names=names, gene_trees=gene_trees)
    assert preds.shape == (4,)


def test_classifier_fit_with_gene_trees():
    X, _, y_class, sp_tree, gene_trees, names = _make_test_data()
    clf = PhyloRandomForestClassifier(n_estimators=10, random_state=42)
    result = clf.fit(X, y_class, tree=sp_tree, species_names=names, gene_trees=gene_trees)
    assert result is clf


def test_classifier_predict_with_gene_trees():
    X, _, y_class, sp_tree, gene_trees, names = _make_test_data()
    clf = PhyloRandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y_class, tree=sp_tree, species_names=names, gene_trees=gene_trees)
    preds = clf.predict(X, tree=sp_tree, species_names=names, gene_trees=gene_trees)
    assert preds.shape == (4,)
    assert set(preds).issubset({0, 1})


def test_discordance_vcv_differs_from_species_vcv():
    """Discordance VCV should differ from species-tree-only VCV."""
    X, y, _, sp_tree, gene_trees, names = _make_test_data()

    model_species = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    model_species.fit(X, y, tree=sp_tree, species_names=names)

    model_discord = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    model_discord.fit(X, y, tree=sp_tree, species_names=names, gene_trees=gene_trees)

    # The Cholesky factors should differ
    assert not np.allclose(model_species.L_, model_discord.L_)


def test_gradient_boosting_with_gene_trees():
    X, y, _, sp_tree, gene_trees, names = _make_test_data()
    model = PhyloGradientBoostingRegressor(n_estimators=10, random_state=42)
    model.fit(X, y, tree=sp_tree, species_names=names, gene_trees=gene_trees)
    preds = model.predict(X, tree=sp_tree, species_names=names, gene_trees=gene_trees)
    assert preds.shape == (4,)


def test_svm_with_gene_trees():
    X, y, _, sp_tree, gene_trees, names = _make_test_data()
    model = PhyloSVMRegressor(kernel="rbf", C=1.0)
    model.fit(X, y, tree=sp_tree, species_names=names, gene_trees=gene_trees)
    preds = model.predict(X, tree=sp_tree, species_names=names, gene_trees=gene_trees)
    assert preds.shape == (4,)


def test_ridge_with_gene_trees():
    X, y, _, sp_tree, gene_trees, names = _make_test_data()
    model = PhyloRidge(alpha=1.0)
    model.fit(X, y, tree=sp_tree, species_names=names, gene_trees=gene_trees)
    preds = model.predict(X, tree=sp_tree, species_names=names, gene_trees=gene_trees)
    assert preds.shape == (4,)


def test_without_gene_trees_still_works():
    """Passing gene_trees=None should behave exactly as before."""
    X, y, _, sp_tree, _, names = _make_test_data()
    model = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y, tree=sp_tree, species_names=names, gene_trees=None)
    preds = model.predict(X, tree=sp_tree, species_names=names)
    assert preds.shape == (4,)
    assert not hasattr(model, "discordance_metadata_")
