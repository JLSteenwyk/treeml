# Phylogenetic SHAP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add SHAP-based model explanations that separate original feature contributions from phylogenetic correction contributions across all treeml estimators.

**Architecture:** A `phylo_shap()` function computes SHAP values on any fitted treeml estimator's inner model, auto-selecting TreeExplainer (RF/GBT) or KernelExplainer (all others). Returns a `PhyloSHAPResult` dataclass with split SHAP values, summary statistics, and built-in plotting. The base estimator stores tree/species_names during fit so `phylo_shap()` can re-augment features internally.

**Tech Stack:** shap, matplotlib, numpy, pandas, scikit-learn

---

### Task 1: Store fit context on base estimator

**Files:**
- Modify: `treeml/estimators/_base.py:23-37` (`_build_vcv` method)
- Test: `tests/unit/test_base_estimator.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_base_estimator.py`:

```python
def test_fit_stores_tree_and_species_names():
    """After fit, model should store tree_, species_names_, gene_trees_."""
    from treeml import PhyloRandomForestRegressor
    from Bio import Phylo
    from io import StringIO
    import numpy as np

    nwk = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
    tree = Phylo.read(StringIO(nwk), "newick")
    names = ["A", "B", "C", "D"]
    rng = np.random.default_rng(42)
    X = rng.standard_normal((4, 2))
    y = rng.standard_normal(4)

    model = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y, tree=tree, species_names=names)

    assert model.tree_ is tree
    assert model.species_names_ == names
    assert model.gene_trees_ is None


def test_fit_stores_gene_trees():
    """After fit with gene_trees, model should store gene_trees_."""
    from treeml import PhyloRandomForestRegressor
    from Bio import Phylo
    from io import StringIO
    import numpy as np

    sp_nwk = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
    gt_nwk = "((A:0.8,C:0.8):1.2,(B:1.0,D:1.0):1.0);"
    tree = Phylo.read(StringIO(sp_nwk), "newick")
    gt = [Phylo.read(StringIO(gt_nwk), "newick")]
    names = ["A", "B", "C", "D"]
    rng = np.random.default_rng(42)
    X = rng.standard_normal((4, 2))
    y = rng.standard_normal(4)

    model = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y, tree=tree, species_names=names, gene_trees=gt)

    assert model.gene_trees_ is gt
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_base_estimator.py::test_fit_stores_tree_and_species_names tests/unit/test_base_estimator.py::test_fit_stores_gene_trees -v`
Expected: FAIL with `AttributeError: 'PhyloRandomForestRegressor' object has no attribute 'tree_'`

**Step 3: Write minimal implementation**

In `treeml/estimators/_base.py`, modify `_build_vcv` (lines 23-37) to store context:

```python
def _build_vcv(
    self,
    tree,
    ordered_names: List[str],
    gene_trees: Optional[List] = None,
) -> np.ndarray:
    self.tree_ = tree
    self.species_names_ = list(ordered_names)
    self.gene_trees_ = gene_trees
    if gene_trees is not None:
        from phykit.services.tree.vcv_utils import build_discordance_vcv
        vcv, metadata = build_discordance_vcv(
            tree, gene_trees, ordered_names
        )
        self.discordance_metadata_ = metadata
        return vcv
    from phykit.services.tree.vcv_utils import build_vcv_matrix
    return build_vcv_matrix(tree, ordered_names)
```

Also modify `_augment_features` (lines 39-56) to store context when `_build_vcv` is not called directly (classifiers call `_augment_features` which calls `_build_vcv`, so this is already covered). No additional changes needed — `_build_vcv` is always called during fit.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_base_estimator.py::test_fit_stores_tree_and_species_names tests/unit/test_base_estimator.py::test_fit_stores_gene_trees -v`
Expected: PASS

**Step 5: Run full test suite to check for regressions**

Run: `python -m pytest tests/ -v`
Expected: All 135+ tests PASS

**Step 6: Commit**

```bash
git add treeml/estimators/_base.py tests/unit/test_base_estimator.py
git commit -m "feat: store tree/species_names/gene_trees on base estimator during fit"
```

---

### Task 2: Add shap dependency

**Files:**
- Modify: `setup.py:25-31` (REQUIRES list)
- Modify: `requirements.txt`

**Step 1: Add shap to setup.py**

Change REQUIRES in `setup.py` to:

```python
REQUIRES = [
    "phykit>=1.11.0",
    "numpy>=1.24.0",
    "scipy>=1.11.3",
    "scikit-learn>=1.4.2",
    "pandas>=2.0.0",
    "shap>=0.42.0",
]
```

**Step 2: Add shap to requirements.txt**

```
phykit>=1.11.0
numpy>=1.24.0
scipy>=1.11.3
scikit-learn>=1.4.2
pandas>=2.0.0
shap>=0.42.0
```

**Step 3: Verify shap is importable**

Run: `python -c "import shap; print(shap.__version__)"`
Expected: prints version number. If not installed, run: `pip install shap>=0.42.0`

**Step 4: Commit**

```bash
git add setup.py requirements.txt
git commit -m "deps: add shap to required dependencies"
```

---

### Task 3: Create PhyloSHAPResult dataclass

**Files:**
- Create: `treeml/shap/__init__.py`
- Create: `treeml/shap/_shap.py`
- Test: `tests/unit/test_shap.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_shap.py`:

```python
import numpy as np
import pandas as pd
import pytest
from Bio import Phylo
from io import StringIO

from treeml.shap._shap import PhyloSHAPResult


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
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_shap.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'treeml.shap'`

**Step 3: Write minimal implementation**

Create `treeml/shap/__init__.py`:

```python
from treeml.shap._shap import phylo_shap, PhyloSHAPResult

__all__ = ["phylo_shap", "PhyloSHAPResult"]
```

Create `treeml/shap/_shap.py`:

```python
from dataclasses import dataclass
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class PhyloSHAPResult:
    """Result container for phylogenetic SHAP analysis."""

    shap_values: np.ndarray
    feature_names: List[str]
    eigenvector_names: List[str]
    n_features_original: int
    n_eigenvector_cols: int
    expected_value: float

    @property
    def feature_shap(self) -> pd.DataFrame:
        """SHAP values for original features only."""
        return pd.DataFrame(
            self.shap_values[:, :self.n_features_original],
            columns=self.feature_names,
        )

    @property
    def phylo_shap(self) -> pd.DataFrame:
        """SHAP values for phylogenetic eigenvector columns only."""
        return pd.DataFrame(
            self.shap_values[:, self.n_features_original:],
            columns=self.eigenvector_names,
        )

    @property
    def feature_importance(self) -> pd.DataFrame:
        """Mean |SHAP| per original feature, sorted descending."""
        mean_abs = np.abs(self.shap_values[:, :self.n_features_original]).mean(axis=0)
        df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs,
        })
        return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    @property
    def phylo_contribution(self) -> float:
        """Fraction of total |SHAP| attributable to phylogenetic correction."""
        total = np.abs(self.shap_values).sum()
        if total == 0:
            return 0.0
        phylo_total = np.abs(self.shap_values[:, self.n_features_original:]).sum()
        return float(phylo_total / total)

    def summary(self) -> pd.DataFrame:
        """Per-feature mean |SHAP| with a 'phylo_total' row appended."""
        feat_abs = np.abs(self.shap_values[:, :self.n_features_original]).mean(axis=0)
        phylo_abs = np.abs(self.shap_values[:, self.n_features_original:]).mean(axis=0).sum()

        rows = [
            {"feature": name, "mean_abs_shap": val}
            for name, val in zip(self.feature_names, feat_abs)
        ]
        rows.append({"feature": "phylo_total", "mean_abs_shap": phylo_abs})
        return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    def plot(self, plot_type: str = "bar", max_features: int = 20):
        """Plot feature vs phylogenetic contributions.

        Parameters
        ----------
        plot_type : str
            "bar" for horizontal grouped bar chart, "beeswarm" for beeswarm of
            original features with phylo annotation.
        max_features : int
            Maximum number of features to display.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if plot_type == "bar":
            return self._plot_bar(max_features)
        elif plot_type == "beeswarm":
            return self._plot_beeswarm(max_features)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type!r}. Use 'bar' or 'beeswarm'.")

    def _plot_bar(self, max_features: int = 20):
        summary = self.summary()
        summary = summary.head(max_features)

        fig, ax = plt.subplots(figsize=(8, max(3, len(summary) * 0.4)))
        colors = [
            "#1f77b4" if f != "phylo_total" else "#d62728"
            for f in summary["feature"]
        ]
        ax.barh(
            range(len(summary)),
            summary["mean_abs_shap"].values,
            color=colors,
        )
        ax.set_yticks(range(len(summary)))
        ax.set_yticklabels(summary["feature"].values)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(
            f"Feature Importance (phylo contribution: {self.phylo_contribution:.1%})"
        )
        fig.tight_layout()
        return fig

    def _plot_beeswarm(self, max_features: int = 20):
        import shap
        feature_sv = self.shap_values[:, :self.n_features_original]
        n_show = min(max_features, feature_sv.shape[1])

        mean_abs = np.abs(feature_sv).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:n_show]

        expl = shap.Explanation(
            values=feature_sv[:, top_idx],
            feature_names=[self.feature_names[i] for i in top_idx],
        )
        fig, ax = plt.subplots()
        shap.plots.beeswarm(expl, show=False)
        fig = plt.gcf()
        fig.suptitle(
            f"Phylo contribution: {self.phylo_contribution:.1%}",
            fontsize=10, y=1.02,
        )
        return fig

    def summary_plot(self, **kwargs):
        """Standard SHAP beeswarm plot on all features (original + eigenvectors)."""
        import shap
        all_names = self.feature_names + self.eigenvector_names
        expl = shap.Explanation(
            values=self.shap_values,
            feature_names=all_names,
        )
        shap.plots.beeswarm(expl, show=False, **kwargs)
        return plt.gcf()

    def force_plot(self, sample_idx: int = 0, **kwargs):
        """SHAP force plot for a single sample."""
        import shap
        all_names = self.feature_names + self.eigenvector_names
        expl = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=self.expected_value,
            feature_names=all_names,
        )
        shap.plots.force(expl, show=False, matplotlib=True, **kwargs)
        return plt.gcf()
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_shap.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add treeml/shap/__init__.py treeml/shap/_shap.py tests/unit/test_shap.py
git commit -m "feat: add PhyloSHAPResult dataclass with properties and plotting"
```

---

### Task 4: Implement phylo_shap() function

**Files:**
- Modify: `treeml/shap/_shap.py` (add `phylo_shap` function)
- Test: `tests/unit/test_shap.py` (add function tests)

**Step 1: Write the failing tests**

Add to `tests/unit/test_shap.py`:

```python
from treeml.shap._shap import phylo_shap


def _make_test_data():
    nwk = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0,(E:1.5,F:0.5):1.0);"
    tree = Phylo.read(StringIO(nwk), "newick")
    names = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(42)
    X = rng.standard_normal((6, 3))
    y = rng.standard_normal(6)
    y_class = np.array([0, 1, 0, 1, 0, 1])
    return X, y, y_class, tree, names


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
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_shap.py::test_phylo_shap_rf_regressor -v`
Expected: FAIL with `ImportError` (phylo_shap not yet defined)

**Step 3: Write minimal implementation**

Add to top of `treeml/shap/_shap.py` (after existing imports):

```python
import shap
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
```

Add the `phylo_shap` function after the `PhyloSHAPResult` class:

```python
def phylo_shap(
    model,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> PhyloSHAPResult:
    """Compute SHAP values for a fitted treeml estimator.

    Separates contributions from original features vs. phylogenetic
    eigenvector corrections.

    Parameters
    ----------
    model : PhyloBaseEstimator
        A fitted treeml estimator (must have inner_model_, tree_, species_names_).
    X : array-like of shape (n_samples, n_features)
        Original feature matrix (without eigenvector augmentation).
    feature_names : list of str, optional
        Names for the original features. Defaults to feature_0, feature_1, ...

    Returns
    -------
    PhyloSHAPResult
    """
    X = np.asarray(X)

    # Validate fitted model
    if not hasattr(model, "inner_model_"):
        raise ValueError(
            "Model is not fitted. Call model.fit() before phylo_shap()."
        )

    n_features = model.n_features_original_
    n_eigvec = model.n_eigenvector_cols_

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    eigenvector_names = [f"phylo_eigvec_{i}" for i in range(n_eigvec)]

    # Re-augment X using stored tree context
    tree = model.tree_
    species_names = model.species_names_
    gene_trees = model.gene_trees_

    X_aug, _ = model._augment_features(
        X, tree, species_names, gene_trees=gene_trees
    )

    # Select SHAP explainer based on inner model type
    inner = model.inner_model_
    if isinstance(inner, (RandomForestRegressor, RandomForestClassifier,
                          GradientBoostingRegressor, GradientBoostingClassifier)):
        explainer = shap.TreeExplainer(inner)
    else:
        explainer = shap.KernelExplainer(inner.predict, X_aug)

    raw_shap = explainer.shap_values(X_aug)

    # For classifiers, TreeExplainer may return list of arrays (one per class)
    # Use positive class (index 1) for binary, or take first for regression
    if isinstance(raw_shap, list):
        if len(raw_shap) == 2:
            shap_values = np.asarray(raw_shap[1])
        else:
            shap_values = np.asarray(raw_shap[0])
    else:
        shap_values = np.asarray(raw_shap)

    # Get expected value
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        if len(ev) == 2:
            expected_value = float(ev[1])
        else:
            expected_value = float(ev[0])
    else:
        expected_value = float(ev)

    return PhyloSHAPResult(
        shap_values=shap_values,
        feature_names=feature_names,
        eigenvector_names=eigenvector_names,
        n_features_original=n_features,
        n_eigenvector_cols=n_eigvec,
        expected_value=expected_value,
    )
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_shap.py -v`
Expected: All 14 tests PASS

**Step 5: Commit**

```bash
git add treeml/shap/_shap.py tests/unit/test_shap.py
git commit -m "feat: implement phylo_shap() function with auto explainer selection"
```

---

### Task 5: Add plotting tests

**Files:**
- Test: `tests/unit/test_shap.py` (add plot tests)

**Step 1: Write the failing tests**

Add to `tests/unit/test_shap.py`:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
```

**Step 2: Run tests to verify they pass**

These should already pass because the plotting code was added in Task 3. Run:

Run: `python -m pytest tests/unit/test_shap.py -v`
Expected: All 19 tests PASS

**Step 3: Commit**

```bash
git add tests/unit/test_shap.py
git commit -m "test: add plotting tests for PhyloSHAPResult"
```

---

### Task 6: Export from treeml and update __init__.py

**Files:**
- Modify: `treeml/__init__.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_shap.py`:

```python
def test_importable_from_treeml():
    from treeml import phylo_shap, PhyloSHAPResult
    assert callable(phylo_shap)
    assert PhyloSHAPResult is not None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_shap.py::test_importable_from_treeml -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

In `treeml/__init__.py`, add after the `phylo_model_comparison` import:

```python
from treeml.shap._shap import phylo_shap, PhyloSHAPResult
```

And add to `__all__`:

```python
__all__ = [
    "__version__",
    "PhyloRandomForestClassifier",
    "PhyloRandomForestRegressor",
    "PhyloGradientBoostingClassifier",
    "PhyloGradientBoostingRegressor",
    "PhyloSVMClassifier",
    "PhyloSVMRegressor",
    "PhyloKNNClassifier",
    "PhyloKNNRegressor",
    "PhyloElasticNet",
    "PhyloRidge",
    "PhyloLasso",
    "PhyloDistanceCV",
    "PhyloCladeCV",
    "phylo_feature_importance",
    "phylo_model_comparison",
    "phylo_shap",
    "PhyloSHAPResult",
    "load_data",
]
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_shap.py::test_importable_from_treeml -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add treeml/__init__.py
git commit -m "feat: export phylo_shap and PhyloSHAPResult from treeml"
```

---

### Task 7: Add integration test

**Files:**
- Modify: `tests/integration/test_end_to_end.py`

**Step 1: Write the test**

Add to `tests/integration/test_end_to_end.py`:

```python
@pytest.mark.integration
class TestPhyloSHAPEndToEnd:
    def test_shap_regression(self):
        from treeml import phylo_shap, PhyloSHAPResult
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
        from treeml import phylo_shap, PhyloSHAPResult
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
```

**Step 2: Run tests**

Run: `python -m pytest tests/integration/test_end_to_end.py::TestPhyloSHAPEndToEnd -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_end_to_end.py
git commit -m "test: add SHAP integration tests"
```

---

### Task 8: Update docs

**Files:**
- Modify: `docs/usage/index.rst`

**Step 1: Update usage docs**

Add a SHAP Explanations section to `docs/usage/index.rst` after the Feature Importance section:

```rst
SHAP Explanations
~~~~~~~~~~~~~~~~~

- ``phylo_shap()`` -- Compute SHAP values separating feature vs. phylogenetic contributions
- ``PhyloSHAPResult`` -- Result container with properties, summary, and plotting methods

Example usage:

.. code-block:: python

   from treeml import PhyloRandomForestRegressor, phylo_shap

   model = PhyloRandomForestRegressor(n_estimators=100)
   model.fit(X, y, tree=tree, species_names=names)

   result = phylo_shap(model, X, feature_names=["body_mass", "diet_type"])

   # What fraction of predictions comes from phylogenetic correction?
   print(f"Phylogenetic contribution: {result.phylo_contribution:.1%}")

   # Summary table
   print(result.summary())

   # Built-in bar plot
   result.plot(plot_type="bar")

   # Standard SHAP beeswarm
   result.summary_plot()

   # Force plot for a single sample
   result.force_plot(sample_idx=0)
```

Also add all 11 estimators to the Estimators section (currently only lists RF):

```rst
Estimators
~~~~~~~~~~

- ``PhyloRandomForestRegressor`` -- Random Forest regressor with phylogenetic correction
- ``PhyloRandomForestClassifier`` -- Random Forest classifier with phylogenetic eigenvector features
- ``PhyloGradientBoostingRegressor`` -- Gradient Boosting regressor with phylogenetic correction
- ``PhyloGradientBoostingClassifier`` -- Gradient Boosting classifier with phylogenetic eigenvector features
- ``PhyloSVMRegressor`` -- SVM regressor with phylogenetic correction
- ``PhyloSVMClassifier`` -- SVM classifier with phylogenetic eigenvector features
- ``PhyloKNNRegressor`` -- KNN regressor with phylogenetic correction
- ``PhyloKNNClassifier`` -- KNN classifier with phylogenetic eigenvector features
- ``PhyloElasticNet`` -- Elastic Net with phylogenetic correction
- ``PhyloRidge`` -- Ridge regression with phylogenetic correction
- ``PhyloLasso`` -- Lasso regression with phylogenetic correction
```

And add a Model Comparison section:

```rst
Model Comparison
~~~~~~~~~~~~~~~~

- ``phylo_model_comparison()`` -- Compare multiple estimators with and without phylogenetic correction
```

**Step 2: Verify docs build (if sphinx is available)**

Run: `cd docs && make html 2>&1 | tail -5`
Expected: Build completes without errors

**Step 3: Commit**

```bash
git add docs/usage/index.rst
git commit -m "docs: add SHAP explanations, all estimators, and model comparison to usage docs"
```

---

### Task 9: Final verification

**Step 1: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS (previous 135 + ~22 new SHAP tests)

**Step 2: Verify imports**

Run: `python -c "from treeml import phylo_shap, PhyloSHAPResult; print('OK')"`
Expected: prints `OK`

**Step 3: Commit any remaining changes**

```bash
git add -A
git commit -m "feat: phylogenetic SHAP explanations complete"
```
