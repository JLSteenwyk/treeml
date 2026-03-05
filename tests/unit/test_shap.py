import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from Bio import Phylo
from io import StringIO

from treeml.shap._shap import PhyloSHAPResult, phylo_shap


def _make_mock_result():
    """Create a PhyloSHAPResult with known values for testing."""
    shap_values = np.array([
        [0.1, 0.2, 0.05, 0.03],
        [0.3, -0.1, 0.02, 0.01],
        [-0.2, 0.15, 0.08, 0.04],
    ])
    return PhyloSHAPResult(
        shap_values=shap_values,
        feature_names=["feat_a", "feat_b"],
        eigenvector_names=["phylo_eigvec_0", "phylo_eigvec_1"],
        n_features_original=2,
        n_eigenvector_cols=2,
        expected_value=1.5,
    )


def _make_test_data():
    nwk = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0,(E:1.5,F:0.5):1.0);"
    tree = Phylo.read(StringIO(nwk), "newick")
    names = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(42)
    X = rng.standard_normal((6, 3))
    y = rng.standard_normal(6)
    y_class = np.array([0, 1, 0, 1, 0, 1])
    return X, y, y_class, tree, names


# --- PhyloSHAPResult dataclass tests ---

def test_feature_shap_shape():
    result = _make_mock_result()
    df = result.feature_shap
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 2)
    assert list(df.columns) == ["feat_a", "feat_b"]


def test_phylo_shap_shape():
    result = _make_mock_result()
    df = result.phylo_shap
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 2)
    assert list(df.columns) == ["phylo_eigvec_0", "phylo_eigvec_1"]


def test_feature_importance_sorted_descending():
    result = _make_mock_result()
    df = result.feature_importance
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df["mean_abs_shap"].iloc[0] >= df["mean_abs_shap"].iloc[1]


def test_phylo_contribution_between_0_and_1():
    result = _make_mock_result()
    pc = result.phylo_contribution
    assert 0.0 <= pc <= 1.0


def test_summary_includes_phylo_row():
    result = _make_mock_result()
    df = result.summary()
    assert isinstance(df, pd.DataFrame)
    assert "phylo_total" in df["feature"].values


def test_feature_names_stored():
    result = _make_mock_result()
    assert result.feature_names == ["feat_a", "feat_b"]


# --- phylo_shap() function tests ---

def test_phylo_shap_rf_regressor():
    from treeml import PhyloRandomForestRegressor
    X, y, _, tree, names = _make_test_data()
    model = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y, tree=tree, species_names=names)
    result = phylo_shap(model, X)
    assert isinstance(result, PhyloSHAPResult)
    assert result.shap_values.shape[0] == 6
    assert result.n_features_original == 3


def test_phylo_shap_rf_classifier():
    from treeml import PhyloRandomForestClassifier
    X, _, y_class, tree, names = _make_test_data()
    clf = PhyloRandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y_class, tree=tree, species_names=names)
    result = phylo_shap(clf, X)
    assert isinstance(result, PhyloSHAPResult)
    assert result.shap_values.shape[0] == 6


def test_phylo_shap_gbt():
    from treeml import PhyloGradientBoostingRegressor
    X, y, _, tree, names = _make_test_data()
    model = PhyloGradientBoostingRegressor(n_estimators=10, random_state=42)
    model.fit(X, y, tree=tree, species_names=names)
    result = phylo_shap(model, X)
    assert isinstance(result, PhyloSHAPResult)


def test_phylo_shap_svm():
    from treeml import PhyloSVMRegressor
    X, y, _, tree, names = _make_test_data()
    model = PhyloSVMRegressor(kernel="rbf", C=1.0)
    model.fit(X, y, tree=tree, species_names=names)
    result = phylo_shap(model, X)
    assert isinstance(result, PhyloSHAPResult)


def test_phylo_shap_ridge():
    from treeml import PhyloRidge
    X, y, _, tree, names = _make_test_data()
    model = PhyloRidge(alpha=1.0)
    model.fit(X, y, tree=tree, species_names=names)
    result = phylo_shap(model, X)
    assert isinstance(result, PhyloSHAPResult)


def test_phylo_shap_custom_feature_names():
    from treeml import PhyloRandomForestRegressor
    X, y, _, tree, names = _make_test_data()
    model = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y, tree=tree, species_names=names)
    result = phylo_shap(model, X, feature_names=["gene_A", "gene_B", "gene_C"])
    assert result.feature_names == ["gene_A", "gene_B", "gene_C"]


def test_phylo_shap_default_feature_names():
    from treeml import PhyloRandomForestRegressor
    X, y, _, tree, names = _make_test_data()
    model = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y, tree=tree, species_names=names)
    result = phylo_shap(model, X)
    assert result.feature_names == ["feature_0", "feature_1", "feature_2"]


def test_unfitted_model_raises():
    from treeml import PhyloRandomForestRegressor
    X, _, _, _, _ = _make_test_data()
    model = PhyloRandomForestRegressor(n_estimators=10)
    with pytest.raises(ValueError, match="not fitted"):
        phylo_shap(model, X)


# --- Plotting tests ---

def test_plot_bar_returns_figure():
    from treeml import PhyloRandomForestRegressor
    X, y, _, tree, names = _make_test_data()
    model = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y, tree=tree, species_names=names)
    result = phylo_shap(model, X)
    fig = result.plot(plot_type="bar")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_beeswarm_returns_figure():
    from treeml import PhyloRandomForestRegressor
    X, y, _, tree, names = _make_test_data()
    model = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y, tree=tree, species_names=names)
    result = phylo_shap(model, X)
    fig = result.plot(plot_type="beeswarm")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_summary_plot_returns_figure():
    from treeml import PhyloRandomForestRegressor
    X, y, _, tree, names = _make_test_data()
    model = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y, tree=tree, species_names=names)
    result = phylo_shap(model, X)
    fig = result.summary_plot()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_force_plot_returns_figure():
    from treeml import PhyloRandomForestRegressor
    X, y, _, tree, names = _make_test_data()
    model = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y, tree=tree, species_names=names)
    result = phylo_shap(model, X)
    fig = result.force_plot(sample_idx=0)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_invalid_type_raises():
    result = _make_mock_result()
    with pytest.raises(ValueError, match="Unknown plot_type"):
        result.plot(plot_type="invalid")


def test_phylo_contribution_zero_shap():
    """phylo_contribution returns 0.0 when all SHAP values are zero."""
    result = PhyloSHAPResult(
        shap_values=np.zeros((3, 4)),
        feature_names=["a", "b"],
        eigenvector_names=["e0", "e1"],
        n_features_original=2,
        n_eigenvector_cols=2,
        expected_value=0.0,
    )
    assert result.phylo_contribution == 0.0


def test_force_plot_invalid_index_raises():
    result = _make_mock_result()
    with pytest.raises(IndexError, match="sample_idx"):
        result.force_plot(sample_idx=99)


def test_feature_names_length_mismatch_raises():
    from treeml import PhyloRandomForestRegressor
    X, y, _, tree, names = _make_test_data()
    model = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y, tree=tree, species_names=names)
    with pytest.raises(ValueError, match="feature_names has"):
        phylo_shap(model, X, feature_names=["only_one"])
