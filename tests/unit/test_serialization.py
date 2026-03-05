import numpy as np
import pytest
from Bio import Phylo
from io import StringIO


def _make_fitted_model():
    from treeml import PhyloRandomForestRegressor
    nwk = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0,(E:1.5,F:0.5):1.0);"
    tree = Phylo.read(StringIO(nwk), "newick")
    names = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(42)
    X = rng.standard_normal((6, 3))
    y = rng.standard_normal(6)
    model = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y, tree=tree, species_names=names)
    return model, X, y, tree, names


def test_save_load_roundtrip_predictions(tmp_path):
    from treeml._serialization import save_model, load_model
    model, X, _, tree, names = _make_fitted_model()
    preds_before = model.predict(X, tree=tree, species_names=names)
    path = tmp_path / "model.treeml"
    save_model(model, str(path))
    loaded = load_model(str(path))
    preds_after = loaded.predict(X, tree=tree, species_names=names)
    np.testing.assert_array_almost_equal(preds_before, preds_after)


def test_save_load_preserves_tree_and_species(tmp_path):
    from treeml._serialization import save_model, load_model
    model, _, _, _, _ = _make_fitted_model()
    path = tmp_path / "model.treeml"
    save_model(model, str(path))
    loaded = load_model(str(path))
    assert loaded.species_names_ == model.species_names_
    assert loaded.tree_ is not None


def test_save_unfitted_raises():
    from treeml import PhyloRandomForestRegressor
    from treeml._serialization import save_model
    model = PhyloRandomForestRegressor()
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):
        save_model(model, "unused.treeml")


def test_load_invalid_file_raises(tmp_path):
    import joblib
    from treeml._serialization import load_model
    path = tmp_path / "bad.treeml"
    joblib.dump({"not": "a treeml bundle"}, str(path))
    with pytest.raises(ValueError, match="not a valid treeml model"):
        load_model(str(path))


def test_version_mismatch_warns(tmp_path):
    import joblib
    from treeml._serialization import save_model, load_model
    model, _, _, _, _ = _make_fitted_model()
    path = tmp_path / "model.treeml"
    save_model(model, str(path))
    bundle = joblib.load(str(path))
    bundle["metadata"]["treeml_version"] = "99.99.99"
    joblib.dump(bundle, str(path))
    with pytest.warns(UserWarning, match="version"):
        loaded = load_model(str(path))
    assert loaded is not None


def test_auto_appends_treeml_extension(tmp_path):
    from treeml._serialization import save_model
    model, _, _, _, _ = _make_fitted_model()
    path = tmp_path / "model"
    save_model(model, str(path))
    assert (tmp_path / "model.treeml").exists()


def test_no_append_if_already_treeml(tmp_path):
    from treeml._serialization import save_model
    model, _, _, _, _ = _make_fitted_model()
    path = tmp_path / "model.treeml"
    save_model(model, str(path))
    assert path.exists()
    assert not (tmp_path / "model.treeml.treeml").exists()


def test_save_load_with_pathlib_path(tmp_path):
    from treeml._serialization import save_model, load_model
    model, X, _, tree, names = _make_fitted_model()
    path = tmp_path / "pathlib_model.treeml"
    save_model(model, path)  # Pass Path directly, not str(path)
    loaded = load_model(path)  # Pass Path directly
    preds = loaded.predict(X, tree=tree, species_names=names)
    assert preds.shape == (6,)


def _make_fitted_classifier():
    from treeml import PhyloRandomForestClassifier
    nwk = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0,(E:1.5,F:0.5):1.0);"
    tree = Phylo.read(StringIO(nwk), "newick")
    names = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(42)
    X = rng.standard_normal((6, 3))
    y = np.array([0, 1, 0, 1, 0, 1])
    model = PhyloRandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y, tree=tree, species_names=names)
    return model, X, y, tree, names


def test_save_load_classifier(tmp_path):
    from treeml._serialization import save_model, load_model
    model, X, _, tree, names = _make_fitted_classifier()
    preds_before = model.predict(X, tree=tree, species_names=names)
    path = tmp_path / "clf.treeml"
    save_model(model, str(path))
    loaded = load_model(str(path))
    preds_after = loaded.predict(X, tree=tree, species_names=names)
    np.testing.assert_array_equal(preds_before, preds_after)


def test_save_load_ridge(tmp_path):
    from treeml import PhyloRidge
    from treeml._serialization import save_model, load_model
    nwk = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0,(E:1.5,F:0.5):1.0);"
    tree = Phylo.read(StringIO(nwk), "newick")
    names = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(42)
    X = rng.standard_normal((6, 3))
    y = rng.standard_normal(6)
    model = PhyloRidge(random_state=42)
    model.fit(X, y, tree=tree, species_names=names)
    preds_before = model.predict(X, tree=tree, species_names=names)
    path = tmp_path / "ridge.treeml"
    save_model(model, str(path))
    loaded = load_model(str(path))
    preds_after = loaded.predict(X, tree=tree, species_names=names)
    np.testing.assert_array_almost_equal(preds_before, preds_after)


def test_save_load_with_gene_trees(tmp_path):
    from treeml import PhyloRandomForestRegressor
    from treeml._serialization import save_model, load_model
    nwk = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0,(E:1.5,F:0.5):1.0);"
    tree = Phylo.read(StringIO(nwk), "newick")
    gt_nwk = "((A:0.8,C:0.8):1.2,(B:1.0,D:1.0):1.0,(E:1.0,F:1.0):1.0);"
    gt = [Phylo.read(StringIO(gt_nwk), "newick")]
    names = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(42)
    X = rng.standard_normal((6, 3))
    y = rng.standard_normal(6)
    model = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y, tree=tree, species_names=names, gene_trees=gt)
    preds_before = model.predict(X, tree=tree, species_names=names, gene_trees=gt)
    path = tmp_path / "gt_model.treeml"
    save_model(model, str(path))
    loaded = load_model(str(path))
    preds_after = loaded.predict(X, tree=tree, species_names=names, gene_trees=gt)
    np.testing.assert_array_almost_equal(preds_before, preds_after)
    assert loaded.gene_trees_ is not None
