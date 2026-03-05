# Phylogenetic SHAP Design

**Goal:** Add SHAP-based model explanations to treeml that separate original feature contributions from phylogenetic correction contributions.

**Architecture:** A convenience wrapper (`phylo_shap()`) that runs SHAP on any fitted treeml estimator's inner model, auto-selects the right SHAP explainer, and returns a rich result object with split SHAP values, summary statistics, and built-in plotting.

## API

```python
from treeml import phylo_shap

result = phylo_shap(
    model,                    # any fitted treeml estimator
    X,                        # original feature matrix (n_samples x n_features)
    feature_names=None,       # optional list of original feature names
)
```

- `model` must be a fitted treeml estimator with `inner_model_`, `n_eigenvector_cols_`, `n_features_original_`, `tree_`, `species_names_`
- The function re-augments `X` internally using the stored tree/species_names
- Auto-selects `shap.TreeExplainer` for RF/GBT, `shap.KernelExplainer` for all others

## PhyloSHAPResult

Dataclass with:

- `shap_values: np.ndarray` — (n_samples, n_total_features)
- `feature_names: List[str]` — original feature names
- `eigenvector_names: List[str]` — ["phylo_eigvec_0", ...]
- `n_features_original: int`
- `n_eigenvector_cols: int`
- `expected_value: float` — SHAP base value

Properties:
- `feature_shap -> pd.DataFrame` — SHAP values for original features only
- `phylo_shap -> pd.DataFrame` — SHAP values for eigenvector columns only
- `feature_importance -> pd.DataFrame` — mean |SHAP| per original feature, sorted descending
- `phylo_contribution -> float` — fraction of total |SHAP| from phylogenetic correction

Methods:
- `summary() -> pd.DataFrame` — per-feature mean |SHAP| with "phylo_total" row
- `plot(plot_type="bar", max_features=20) -> Figure` — grouped bar (features vs phylo) or beeswarm
- `summary_plot(**kwargs)` — delegates to `shap.plots.beeswarm`
- `force_plot(sample_idx=0, **kwargs)` — delegates to `shap.plots.force`

## Model Changes

Store fit context on `PhyloBaseEstimator` during `_build_vcv()`:
- `self.tree_` — the tree
- `self.species_names_` — species names
- `self.gene_trees_` — gene_trees or None

No signature changes needed.

## Explainer Selection

| Inner model type | SHAP Explainer |
|---|---|
| RandomForest*, GradientBoosting* | TreeExplainer |
| SVR, SVC, KNeighbors*, Ridge, Lasso, ElasticNet | KernelExplainer |

For classifiers, default to positive class (index 1 for binary), store full multi-class values on result.

## Plotting

- `plot(plot_type="bar")` — horizontal grouped bars: original features + aggregated phylo bar
- `plot(plot_type="beeswarm")` — beeswarm of original features, phylo contribution as annotation
- `summary_plot()` — standard SHAP beeswarm on all features
- `force_plot()` — standard SHAP force plot for a single sample
- All return matplotlib Figure

## Dependencies

- Add `shap` to `install_requires` in setup.py and requirements.txt

## Docs

- New page for SHAP explanation (usage, examples, interpretation)
- Add to API reference
- Export `phylo_shap` and `PhyloSHAPResult` from `treeml.__init__`

## Testing

Unit tests covering: all estimator types, result shapes, property correctness, plotting, error handling for unfitted models. Integration test for full pipeline.
