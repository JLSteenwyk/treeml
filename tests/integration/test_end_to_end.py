import os

import numpy as np
import pytest
from sklearn.model_selection import cross_val_score

from treeml import (
    PhyloRandomForestRegressor,
    PhyloRandomForestClassifier,
    PhyloDistanceCV,
    PhyloCladeCV,
    phylo_feature_importance,
    load_data,
)

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_files")


@pytest.mark.integration
class TestRegressionEndToEnd:
    def test_fit_predict_with_tree(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        model = PhyloRandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y, tree=tree, species_names=names)
        preds = model.predict(X, tree=tree, species_names=names)
        assert preds.shape == y.shape

    def test_cross_val_with_phylo_distance_cv(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        model = PhyloRandomForestRegressor(n_estimators=50, random_state=42)
        cv = PhyloDistanceCV(tree=tree, species_names=names, n_splits=3)
        # cross_val_score won't pass tree to fit/predict, so we test CV splits
        splits = list(cv.split(X, y))
        assert len(splits) == 3

    def test_feature_importance_report(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        report = phylo_feature_importance(
            X, y, tree=tree, species_names=names,
            feature_names=["body_mass", "diet_type"],
            n_repeats=5, random_state=42,
        )
        assert len(report) == 2
        assert "delta" in report.columns

    def test_predict_new_species_with_tree(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        model = PhyloRandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y, tree=tree, species_names=names)

        from Bio import Phylo
        updated_tree = Phylo.read(
            os.path.join(SAMPLE_DIR, "tree_with_fox.nwk"), "newick"
        )
        # Predict for fox (new species) using original + fox features
        X_with_fox = np.vstack([X, [[20.0, 1]]])
        names_with_fox = names + ["fox"]
        preds = model.predict(
            X_with_fox, tree=updated_tree, species_names=names_with_fox
        )
        assert preds.shape == (len(names) + 1,)

    def test_predict_without_tree_degrades_gracefully(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        model = PhyloRandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y, tree=tree, species_names=names)
        with pytest.warns(UserWarning):
            preds = model.predict(X)
        assert preds.shape == y.shape


@pytest.mark.integration
class TestClassificationEndToEnd:
    def test_fit_predict(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="diet_type",
        )
        clf = PhyloRandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y, tree=tree, species_names=names)
        preds = clf.predict(X, tree=tree, species_names=names)
        assert preds.shape == y.shape

    def test_predict_proba(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="diet_type",
        )
        clf = PhyloRandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y, tree=tree, species_names=names)
        proba = clf.predict_proba(X, tree=tree, species_names=names)
        assert proba.shape[1] == 2
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_clade_cv(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="diet_type",
        )
        cv = PhyloCladeCV(tree=tree, species_names=names, n_splits=3)
        splits = list(cv.split(X, y))
        assert len(splits) >= 1
