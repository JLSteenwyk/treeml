# Serialization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `save_model()` and `load_model()` functions that persist fitted treeml estimators (including tree, species names, and all fitted attributes) to disk using joblib.

**Architecture:** Bundle the fitted estimator + metadata dict into a single joblib file with `.treeml` extension. `save_model` validates the model is fitted, `load_model` validates the bundle structure and warns on version mismatch.

**Tech Stack:** joblib (already installed as sklearn dependency), treeml estimators

---

### Task 1: Core save_model / load_model implementation

**Files:**
- Create: `treeml/_serialization.py`
- Test: `tests/unit/test_serialization.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_serialization.py`:

```python
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
    # Tamper with saved version
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
    # Should NOT create model.treeml.treeml
    assert not (tmp_path / "model.treeml.treeml").exists()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_serialization.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'treeml._serialization'`

**Step 3: Write the implementation**

Create `treeml/_serialization.py`:

```python
import warnings

import joblib
from sklearn.exceptions import NotFittedError

from treeml.version import __version__


def save_model(model, path: str) -> str:
    """Save a fitted treeml estimator to disk.

    Args:
        model: A fitted treeml estimator (must have inner_model_ attribute).
        path: File path. If it doesn't end with '.treeml', the extension is appended.

    Returns:
        The actual path the model was saved to.

    Raises:
        NotFittedError: If the model has not been fitted yet.
    """
    if not hasattr(model, "inner_model_"):
        raise NotFittedError(
            "This model is not fitted yet. Call 'fit' before saving."
        )

    if not path.endswith(".treeml"):
        path = path + ".treeml"

    bundle = {
        "model": model,
        "metadata": {
            "treeml_version": __version__,
            "estimator_class": type(model).__name__,
        },
    }
    joblib.dump(bundle, path)
    return path


def load_model(path: str):
    """Load a treeml estimator from disk.

    Args:
        path: Path to a .treeml file.

    Returns:
        The fitted treeml estimator.

    Raises:
        ValueError: If the file is not a valid treeml model bundle.
    """
    bundle = joblib.load(path)

    if not isinstance(bundle, dict) or "model" not in bundle or "metadata" not in bundle:
        raise ValueError(
            f"'{path}' is not a valid treeml model file."
        )

    metadata = bundle["metadata"]
    saved_version = metadata.get("treeml_version", "unknown")
    if saved_version != __version__:
        warnings.warn(
            f"Model was saved with treeml {saved_version} but you are "
            f"loading with treeml {__version__}. This may cause issues.",
            UserWarning,
            stacklevel=2,
        )

    return bundle["model"]
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_serialization.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add treeml/_serialization.py tests/unit/test_serialization.py
git commit -m "feat: add save_model/load_model serialization"
```

---

### Task 2: Multiple estimator types round-trip tests

**Files:**
- Modify: `tests/unit/test_serialization.py`

**Step 1: Add tests for classifier and other estimator types**

Append to `tests/unit/test_serialization.py`:

```python
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
```

**Step 2: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_serialization.py -v`
Expected: All 11 tests PASS

**Step 3: Commit**

```bash
git add tests/unit/test_serialization.py
git commit -m "test: add serialization tests for classifier, ridge, gene trees"
```

---

### Task 3: Export from __init__.py and update docs

**Files:**
- Modify: `treeml/__init__.py:19` (add import)
- Modify: `treeml/__init__.py:42` (add to __all__)
- Modify: `docs/usage/index.rst:102-106` (add Serialization section)

**Step 1: Update `treeml/__init__.py`**

Add this import after the `load_data` import (line 19):

```python
from treeml._serialization import save_model, load_model
```

Add `"save_model"` and `"load_model"` to the `__all__` list, before the closing bracket.

The `__all__` list should end with:

```python
    "load_data",
    "save_model",
    "load_model",
]
```

**Step 2: Update `docs/usage/index.rst`**

Insert before the "Data Loading" section (before line 102), add a new Serialization section:

```rst
Serialization
~~~~~~~~~~~~~

- ``save_model()`` -- Save a fitted treeml estimator to disk
- ``load_model()`` -- Load a treeml estimator from disk

Example usage:

.. code-block:: python

   from treeml import PhyloRandomForestRegressor, save_model, load_model

   model = PhyloRandomForestRegressor(n_estimators=100)
   model.fit(X, y, tree=tree, species_names=names)

   # Save
   save_model(model, "my_model.treeml")

   # Load
   loaded = load_model("my_model.treeml")
   preds = loaded.predict(X, tree=tree, species_names=names)

```

**Step 3: Verify import works**

Run: `python -c "from treeml import save_model, load_model; print('OK')"`
Expected: `OK`

**Step 4: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS (existing + 11 new serialization tests)

**Step 5: Commit**

```bash
git add treeml/__init__.py docs/usage/index.rst
git commit -m "feat: export save_model/load_model and update docs"
```

---

### Task 4: Integration test

**Files:**
- Modify: `tests/integration/test_end_to_end.py`

**Step 1: Add integration test**

Add a new test class at the end of `tests/integration/test_end_to_end.py`:

```python
class TestSerializationEndToEnd:
    def test_save_load_full_workflow(self, regression_data, tmp_path):
        from treeml import PhyloRandomForestRegressor, save_model, load_model
        X, y, tree, names = regression_data
        model = PhyloRandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y, tree=tree, species_names=names)
        preds_before = model.predict(X, tree=tree, species_names=names)

        path = tmp_path / "e2e_model.treeml"
        save_model(model, str(path))
        loaded = load_model(str(path))

        preds_after = loaded.predict(X, tree=tree, species_names=names)
        np.testing.assert_array_almost_equal(preds_before, preds_after)
        assert loaded.species_names_ == names
```

**Step 2: Run integration tests**

Run: `python -m pytest tests/integration/test_end_to_end.py -v`
Expected: All integration tests PASS

**Step 3: Commit**

```bash
git add tests/integration/test_end_to_end.py
git commit -m "test: add serialization integration test"
```
