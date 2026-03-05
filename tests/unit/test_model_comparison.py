import numpy as np
import pandas as pd
import pytest
from Bio import Phylo
from io import StringIO

from treeml.comparison._compare import phylo_model_comparison
from treeml.estimators._regressor import PhyloRandomForestRegressor
from treeml.estimators._ridge import PhyloRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


def _make_test_data():
    nwk = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0,(E:1.5,F:0.5):1.0);"
    tree = Phylo.read(StringIO(nwk), "newick")
    names = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(42)
    X = rng.standard_normal((6, 3))
    y = rng.standard_normal(6)
    return X, y, tree, names


def _make_classification_data():
    nwk = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0,(E:1.5,F:0.5):1.0);"
    tree = Phylo.read(StringIO(nwk), "newick")
    names = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(42)
    X = rng.standard_normal((6, 3))
    y = np.array([0, 1, 0, 1, 0, 1])
    return X, y, tree, names


def test_returns_dataframe():
    X, y, tree, names = _make_test_data()
    result = phylo_model_comparison(
        X, y, tree=tree, species_names=names, random_state=42
    )
    assert isinstance(result, pd.DataFrame)


def test_has_correct_columns():
    X, y, tree, names = _make_test_data()
    result = phylo_model_comparison(
        X, y, tree=tree, species_names=names, random_state=42
    )
    expected_cols = {"model", "uncorrected_score", "phylo_corrected_score", "delta"}
    assert set(result.columns) == expected_cols


def test_default_regressors_all_present():
    X, y, tree, names = _make_test_data()
    result = phylo_model_comparison(
        X, y, tree=tree, species_names=names, random_state=42
    )
    expected_models = {
        "RandomForest", "GradientBoosting", "SVM", "KNN",
        "ElasticNet", "Ridge", "Lasso",
    }
    assert set(result["model"]) == expected_models


def test_default_classifiers_all_present():
    X, y, tree, names = _make_classification_data()
    result = phylo_model_comparison(
        X, y, tree=tree, species_names=names, random_state=42
    )
    expected_models = {"RandomForest", "GradientBoosting", "SVM", "KNN"}
    assert set(result["model"]) == expected_models


def test_custom_models():
    X, y, tree, names = _make_test_data()
    custom = {
        "RF": (
            RandomForestRegressor(n_estimators=10, random_state=42),
            PhyloRandomForestRegressor(n_estimators=10, random_state=42),
        ),
        "Ridge": (
            Ridge(alpha=1.0),
            PhyloRidge(alpha=1.0),
        ),
    }
    result = phylo_model_comparison(
        X, y, tree=tree, species_names=names,
        models=custom, random_state=42,
    )
    assert len(result) == 2
    assert set(result["model"]) == {"RF", "Ridge"}


def test_delta_is_difference():
    X, y, tree, names = _make_test_data()
    result = phylo_model_comparison(
        X, y, tree=tree, species_names=names, random_state=42
    )
    for _, row in result.iterrows():
        if not np.isnan(row["delta"]):
            expected = row["phylo_corrected_score"] - row["uncorrected_score"]
            np.testing.assert_allclose(row["delta"], expected)
