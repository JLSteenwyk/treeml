import numpy as np
from Bio import Phylo
from io import StringIO

from treeml.importance._report import phylo_feature_importance


def _make_test_data():
    nwk = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0,(E:1.5,F:0.5):1.0);"
    tree = Phylo.read(StringIO(nwk), "newick")
    names = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(42)
    X = rng.standard_normal((6, 3))
    y = rng.standard_normal(6)
    return X, y, tree, names


def test_returns_dataframe():
    import pandas as pd
    X, y, tree, names = _make_test_data()
    report = phylo_feature_importance(
        X, y, tree=tree, species_names=names,
        n_repeats=5, random_state=42,
    )
    assert isinstance(report, pd.DataFrame)


def test_report_has_correct_columns():
    X, y, tree, names = _make_test_data()
    report = phylo_feature_importance(
        X, y, tree=tree, species_names=names,
        n_repeats=5, random_state=42,
    )
    expected_cols = {"feature", "raw_importance", "phylo_corrected_importance", "delta"}
    assert expected_cols == set(report.columns)


def test_report_has_correct_rows():
    X, y, tree, names = _make_test_data()
    feature_names = ["gene_A", "gene_B", "gene_C"]
    report = phylo_feature_importance(
        X, y, tree=tree, species_names=names,
        feature_names=feature_names, n_repeats=5, random_state=42,
    )
    assert len(report) == 3
    assert list(report["feature"]) == feature_names


def test_delta_is_difference():
    X, y, tree, names = _make_test_data()
    report = phylo_feature_importance(
        X, y, tree=tree, species_names=names,
        n_repeats=5, random_state=42,
    )
    np.testing.assert_allclose(
        report["delta"].values,
        report["phylo_corrected_importance"].values - report["raw_importance"].values,
    )


def test_default_feature_names():
    X, y, tree, names = _make_test_data()
    report = phylo_feature_importance(
        X, y, tree=tree, species_names=names,
        n_repeats=5, random_state=42,
    )
    assert list(report["feature"]) == ["feature_0", "feature_1", "feature_2"]
