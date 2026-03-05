# PhyloGridSearchCV Design

**Goal:** Add phylogenetic-aware hyperparameter tuning that passes tree/species_names/gene_trees through to estimator fit/predict during cross-validation.

**Architecture:** Composition/adapter pattern. A `_PhyloEstimatorAdapter` wraps any treeml estimator so it looks like a standard sklearn estimator (tree/species_names bound into fit/predict). `PhyloGridSearchCV` and `PhyloRandomizedSearchCV` wrap sklearn's GridSearchCV/RandomizedSearchCV, defaulting CV to PhyloDistanceCV, and unwrap best_estimator_ back to the real treeml model.

## Adapter Pattern

`_PhyloEstimatorAdapter` binds tree/species_names/gene_trees so that:
- `fit(X, y)` calls `inner.fit(X, y, tree=tree, species_names=names, gene_trees=gene_trees)`
- `predict(X)` calls `inner.predict(X, tree=tree, species_names=names, gene_trees=gene_trees)`
- `predict_proba(X)` forwards similarly (for classifiers)
- `get_params()` / `set_params()` delegate to inner model for sklearn cloning

## Public API

- `PhyloGridSearchCV(estimator, param_grid, tree, species_names, gene_trees=None, cv=None, scoring=None, n_jobs=None, refit=True, **kwargs)`
- `PhyloRandomizedSearchCV(estimator, param_distributions, tree, species_names, gene_trees=None, n_iter=10, cv=None, scoring=None, n_jobs=None, refit=True, **kwargs)`

Defaults:
- `cv=None` → `PhyloDistanceCV(tree, species_names, n_splits=5)`
- `scoring=None` → auto-detect from y (`r2` for regression, `accuracy` for classification)

Attributes exposed:
- `best_params_`, `best_score_`, `cv_results_` — from sklearn
- `best_estimator_` — unwrapped treeml model (not adapter)
- `predict()`, `predict_proba()` — forward tree/species_names automatically

## Dependencies

No new dependencies. Uses sklearn.model_selection.GridSearchCV and RandomizedSearchCV.

## Testing

Unit tests: grid search fit, best_params/score/estimator, cv_results, predict shape, default/custom CV, classifier support, gene_trees forwarding, randomized search, unfitted predict error.
Integration: end-to-end on sample data.
