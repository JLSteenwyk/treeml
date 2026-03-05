import os

import numpy as np
import pytest
from sklearn.model_selection import cross_val_score

from treeml import (
    PhyloRandomForestRegressor,
    PhyloRandomForestClassifier,
    PhyloGradientBoostingRegressor,
    PhyloGradientBoostingClassifier,
    PhyloSVMRegressor,
    PhyloSVMClassifier,
    PhyloKNNRegressor,
    PhyloKNNClassifier,
    PhyloElasticNet,
    PhyloRidge,
    PhyloLasso,
    PhyloDistanceCV,
    phylo_model_comparison,
    PhyloCladeCV,
    phylo_feature_importance,
    phylo_shap,
    PhyloSHAPResult,
    PhyloGridSearchCV,
    PhyloRandomizedSearchCV,
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


@pytest.mark.integration
class TestGradientBoostingRegressionEndToEnd:
    def test_fit_predict_with_tree(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        model = PhyloGradientBoostingRegressor(n_estimators=50, random_state=42)
        model.fit(X, y, tree=tree, species_names=names)
        preds = model.predict(X, tree=tree, species_names=names)
        assert preds.shape == y.shape

    def test_predict_without_tree_degrades_gracefully(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        model = PhyloGradientBoostingRegressor(n_estimators=50, random_state=42)
        model.fit(X, y, tree=tree, species_names=names)
        with pytest.warns(UserWarning):
            preds = model.predict(X)
        assert preds.shape == y.shape


@pytest.mark.integration
class TestGradientBoostingClassificationEndToEnd:
    def test_fit_predict(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="diet_type",
        )
        clf = PhyloGradientBoostingClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y, tree=tree, species_names=names)
        preds = clf.predict(X, tree=tree, species_names=names)
        assert preds.shape == y.shape

    def test_predict_proba(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="diet_type",
        )
        clf = PhyloGradientBoostingClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y, tree=tree, species_names=names)
        proba = clf.predict_proba(X, tree=tree, species_names=names)
        assert proba.shape[1] == 2
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)


@pytest.mark.integration
class TestSVMRegressionEndToEnd:
    def test_fit_predict_with_tree(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        model = PhyloSVMRegressor(kernel="rbf", C=1.0)
        model.fit(X, y, tree=tree, species_names=names)
        preds = model.predict(X, tree=tree, species_names=names)
        assert preds.shape == y.shape

    def test_predict_without_tree_degrades_gracefully(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        model = PhyloSVMRegressor(kernel="rbf", C=1.0)
        model.fit(X, y, tree=tree, species_names=names)
        with pytest.warns(UserWarning):
            preds = model.predict(X)
        assert preds.shape == y.shape


@pytest.mark.integration
class TestSVMClassificationEndToEnd:
    def test_fit_predict(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="diet_type",
        )
        clf = PhyloSVMClassifier(kernel="rbf", C=1.0, random_state=42)
        clf.fit(X, y, tree=tree, species_names=names)
        preds = clf.predict(X, tree=tree, species_names=names)
        assert preds.shape == y.shape

    def test_predict_proba(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="diet_type",
        )
        clf = PhyloSVMClassifier(kernel="rbf", C=1.0, random_state=42)
        clf.fit(X, y, tree=tree, species_names=names)
        proba = clf.predict_proba(X, tree=tree, species_names=names)
        assert proba.shape[1] == 2
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)


@pytest.mark.integration
class TestKNNRegressionEndToEnd:
    def test_fit_predict_with_tree(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        model = PhyloKNNRegressor(n_neighbors=3)
        model.fit(X, y, tree=tree, species_names=names)
        preds = model.predict(X, tree=tree, species_names=names)
        assert preds.shape == y.shape

    def test_predict_without_tree_degrades_gracefully(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        model = PhyloKNNRegressor(n_neighbors=3)
        model.fit(X, y, tree=tree, species_names=names)
        with pytest.warns(UserWarning):
            preds = model.predict(X)
        assert preds.shape == y.shape


@pytest.mark.integration
class TestKNNClassificationEndToEnd:
    def test_fit_predict(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="diet_type",
        )
        clf = PhyloKNNClassifier(n_neighbors=3)
        clf.fit(X, y, tree=tree, species_names=names)
        preds = clf.predict(X, tree=tree, species_names=names)
        assert preds.shape == y.shape

    def test_predict_proba(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="diet_type",
        )
        clf = PhyloKNNClassifier(n_neighbors=3)
        clf.fit(X, y, tree=tree, species_names=names)
        proba = clf.predict_proba(X, tree=tree, species_names=names)
        assert proba.shape[1] == 2
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)


@pytest.mark.integration
class TestElasticNetEndToEnd:
    def test_fit_predict_with_tree(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        model = PhyloElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
        model.fit(X, y, tree=tree, species_names=names)
        preds = model.predict(X, tree=tree, species_names=names)
        assert preds.shape == y.shape

    def test_predict_without_tree_degrades_gracefully(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        model = PhyloElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
        model.fit(X, y, tree=tree, species_names=names)
        with pytest.warns(UserWarning):
            preds = model.predict(X)
        assert preds.shape == y.shape


@pytest.mark.integration
class TestRidgeEndToEnd:
    def test_fit_predict_with_tree(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        model = PhyloRidge(alpha=1.0)
        model.fit(X, y, tree=tree, species_names=names)
        preds = model.predict(X, tree=tree, species_names=names)
        assert preds.shape == y.shape

    def test_predict_without_tree_degrades_gracefully(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        model = PhyloRidge(alpha=1.0)
        model.fit(X, y, tree=tree, species_names=names)
        with pytest.warns(UserWarning):
            preds = model.predict(X)
        assert preds.shape == y.shape


@pytest.mark.integration
class TestLassoEndToEnd:
    def test_fit_predict_with_tree(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        model = PhyloLasso(alpha=1.0, random_state=42)
        model.fit(X, y, tree=tree, species_names=names)
        preds = model.predict(X, tree=tree, species_names=names)
        assert preds.shape == y.shape

    def test_predict_without_tree_degrades_gracefully(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        model = PhyloLasso(alpha=1.0, random_state=42)
        model.fit(X, y, tree=tree, species_names=names)
        with pytest.warns(UserWarning):
            preds = model.predict(X)
        assert preds.shape == y.shape


@pytest.mark.integration
class TestModelComparisonEndToEnd:
    def test_regression_comparison(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        report = phylo_model_comparison(
            X, y, tree=tree, species_names=names, random_state=42
        )
        assert len(report) == 7
        assert "delta" in report.columns

    def test_classification_comparison(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="diet_type",
        )
        report = phylo_model_comparison(
            X, y, tree=tree, species_names=names, random_state=42
        )
        assert len(report) == 4
        assert "delta" in report.columns


@pytest.mark.integration
class TestPhyloSHAPEndToEnd:
    def test_shap_regression(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        model = PhyloRandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y, tree=tree, species_names=names)
        result = phylo_shap(model, X, feature_names=["body_mass", "diet_type"])
        assert isinstance(result, PhyloSHAPResult)
        assert result.shap_values.shape[0] == X.shape[0]
        assert result.feature_names == ["body_mass", "diet_type"]
        assert 0.0 <= result.phylo_contribution <= 1.0
        summary = result.summary()
        assert "phylo_total" in summary["feature"].values

    def test_shap_classification(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="diet_type",
        )
        clf = PhyloRandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y, tree=tree, species_names=names)
        result = phylo_shap(clf, X)
        assert isinstance(result, PhyloSHAPResult)
        assert result.shap_values.shape[0] == X.shape[0]


@pytest.mark.integration
class TestPhyloGridSearchEndToEnd:
    def test_grid_search_regression(self):
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


@pytest.mark.integration
class TestSerializationEndToEnd:
    def test_save_load_full_workflow(self, tmp_path):
        from treeml import PhyloRandomForestRegressor, save_model, load_model

        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        model = PhyloRandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y, tree=tree, species_names=names)
        preds_before = model.predict(X, tree=tree, species_names=names)

        path = tmp_path / "e2e_model.treeml"
        save_model(model, str(path))
        loaded = load_model(str(path))

        preds_after = loaded.predict(X, tree=tree, species_names=names)
        np.testing.assert_array_almost_equal(preds_before, preds_after)
        assert loaded.species_names_ == names
