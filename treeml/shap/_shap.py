from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)


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
        import matplotlib.pyplot as plt

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
        import matplotlib.pyplot as plt

        feature_sv = self.shap_values[:, :self.n_features_original]
        n_show = min(max_features, feature_sv.shape[1])

        mean_abs = np.abs(feature_sv).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:n_show]

        expl = shap.Explanation(
            values=feature_sv[:, top_idx],
            feature_names=[self.feature_names[i] for i in top_idx],
        )
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
        import matplotlib.pyplot as plt

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
        import matplotlib.pyplot as plt

        n_samples = self.shap_values.shape[0]
        if sample_idx < 0 or sample_idx >= n_samples:
            raise IndexError(
                f"sample_idx {sample_idx} out of range for {n_samples} samples."
            )

        all_names = self.feature_names + self.eigenvector_names
        expl = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=self.expected_value,
            feature_names=all_names,
        )
        shap.plots.force(expl, show=False, matplotlib=True, **kwargs)
        return plt.gcf()


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
    import shap

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
    elif len(feature_names) != n_features:
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but model "
            f"was fitted with {n_features} features."
        )

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
