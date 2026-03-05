# treeml Design Document

**Date:** 2026-03-04
**Status:** Approved

## Overview

treeml is a standalone Python package for phylogenetic machine learning. It provides
scikit-learn-compatible classifiers and regressors that account for phylogenetic
non-independence among species, phylogenetically-structured cross-validation, and
comparative feature importance analysis.

## Problem

Standard ML methods (Random Forest, SVM, etc.) assume IID observations. Species data
violates this because closely related species share evolutionary history. This leads to:
- Inflated accuracy estimates (phylogenetic leakage in train/test splits)
- Confounded feature importance (features correlated with phylogeny appear important)
- Biased predictions (phylogenetic autocorrelation treated as signal)

PGLS solves this for linear models using phylogenetic variance-covariance (VCV) matrices.
treeml extends this principle to non-linear ML methods.

## Use Cases

1. **Prediction**: Predict a phenotype for species not in the training set, given their
   gene family copy numbers and phylogenetic placement.
2. **Inference**: Identify which features (e.g., gene families) are genuinely associated
   with a phenotype after removing phylogenetic confounding.

## Architecture

### Approach: Wrapper Pattern

Wrap sklearn estimators rather than reimplement them. A `PhyloRandomForestClassifier`
holds an internal `sklearn.RandomForestClassifier` but handles phylogenetic preprocessing
(whitening y, computing eigenvectors, augmenting X) in `fit()` and `predict()`.

This was chosen over:
- **Preprocessing-only** (Transformer + CV): awkward y-transformation in sklearn pipelines,
  worse user experience
- **Full reimplementation**: massive engineering effort, slower than sklearn's C-optimized
  implementation, unclear if modifying split criteria outperforms whitening/eigenvectors

### Package Structure

```
treeml/
├── treeml/
│   ├── __init__.py              # Public API exports
│   ├── _whitening.py            # Cholesky-based phylogenetic whitening
│   ├── _eigenvectors.py         # Phylogenetic eigenvector extraction
│   ├── estimators/
│   │   ├── __init__.py
│   │   ├── _base.py             # PhyloBaseEstimator (shared phylo logic)
│   │   ├── _classifier.py       # PhyloRandomForestClassifier
│   │   └── _regressor.py        # PhyloRandomForestRegressor
│   ├── cv/
│   │   ├── __init__.py
│   │   ├── _distance.py         # PhyloDistanceCV
│   │   └── _clade.py            # PhyloCladeCV
│   └── importance/
│       ├── __init__.py
│       └── _report.py           # phylo_feature_importance()
├── tests/
│   ├── unit/
│   └── integration/
├── docs/
│   ├── conf.py                  # Sphinx + sphinx_rtd_theme (PhyKIT styling)
│   ├── Makefile
│   ├── Pipfile
│   ├── index.rst
│   ├── _static/
│   │   ├── custom.css           # PhyKIT's Oxygen font, same styling
│   │   └── img/
│   ├── _templates/
│   │   └── sidebar-top.html     # GitHub buttons, analytics
│   ├── about/index.rst
│   ├── usage/index.rst
│   ├── tutorials/index.rst
│   ├── change_log/index.rst
│   └── frequently_asked_questions/index.rst
├── .github/
│   └── workflows/
│       └── ci.yml
├── codecov.yml
├── Makefile
├── setup.py
└── README.md
```

### Dependencies

- `phykit` — VCV construction (`build_vcv_matrix`, `build_discordance_vcv`, `_nearest_psd`),
  tree validation, trait file parsing
- `numpy>=1.24.0`, `scipy>=1.11.3` — Cholesky decomposition, eigendecomposition, clustering
- `scikit-learn>=1.4.2` — base classes, Random Forest, permutation importance, CV
- `pandas>=2.0.0` — feature importance reports (optional import)

## Core Phylogenetic Correction Pipeline

### VCV Construction (delegated to PhyKIT)

```python
from phykit.services.tree.vcv_utils import build_vcv_matrix, build_discordance_vcv
```

Given a tree and ordered species names, produces the n x n VCV matrix where:
- Diagonal: root-to-tip distance (total evolutionary variance per species)
- Off-diagonal: shared path length (evolutionary covariance between species)

Users can optionally pass gene trees for discordance-aware VCV.

### Phylogenetic Whitening (`_whitening.py`)

Transform the target vector y to remove phylogenetic autocorrelation:

1. Compute VCV matrix C from tree
2. Cholesky decompose: C = L L^T
3. Transform: y_white = L^{-1} y

This is exactly what makes GLS equivalent to OLS on transformed data. After whitening,
y_white has identity covariance.

**For classification targets:** whitening does not apply to discrete y. Classifiers skip
y-whitening and rely on eigenvector features + phylogenetic CV.

### Phylogenetic Eigenvectors (`_eigenvectors.py`)

Extract phylogenetic structure as features:

1. Double-center the VCV matrix (analogous to PCoA)
2. Eigendecompose: C_centered = U Lambda U^T
3. Take top k eigenvectors (default: explain 90% of variance, user-configurable)
4. Append as columns to X

Toggle behavior:
- **Prediction mode** (default): eigenvectors included
- **Inference mode** (`include_eigenvectors=False`): eigenvectors excluded

### Pipeline Flow

**fit(X, y, tree, species_names):**
```
1. Build VCV from tree                          -> C (n x n)
2. If regressor: whiten y                       -> y_white = L^{-1} y
3. If include_eigenvectors: extract eigenvecs   -> E (n x k)
4. Augment features: X_aug = [X | E]            -> (n x (p+k))
5. Fit inner sklearn model on (X_aug, y_white)
```

**predict(X_new, tree=None, species_names=None):**
```
If tree provided:
  1. Recompute VCV from new tree (all species)
  2. Compute eigenvectors for new species
  3. Augment X_new with eigenvectors
  4. Predict with inner model
  5. If regressor: un-whiten using new L matrix
  6. Return predictions, all marked phylo_corrected=True

If no tree:
  1. Eigenvector columns filled with 0
  2. Predict with inner model
  3. If regressor: skip un-whitening
  4. Emit warning: "prediction made without phylogenetic correction"
  5. Return predictions, marked phylo_corrected=False
```

### Prediction on New Species

Two scenarios:

1. **New species WITH known tree placement**: User provides a full tree containing both
   training and new species. VCV is recomputed from scratch. Full phylogenetic correction
   applies. Users can place species using tools like pplacer or EPA-ng.

2. **New species with NO tree placement**: Graceful degradation. Eigenvectors set to 0,
   y un-whitening skipped. Warning emitted. Results include `phylo_corrected` boolean
   per species.

API:
```python
model.predict(X_new, tree=updated_tree, species_names=new_names)  # scenario 1
model.predict(X_new)                                               # scenario 2
```

## Cross-Validation

### PhyloDistanceCV (recommended default)

Ensures minimum phylogenetic gap between train and test sets.

Algorithm:
1. Compute pairwise patristic distance matrix from VCV:
   `d_ij = VCV[i,i] + VCV[j,j] - 2*VCV[i,j]`
2. Hierarchical clustering (scipy `fcluster`) with distance threshold
3. Each cluster = one fold (all species in a cluster stay together)
4. Threshold auto-tuned to produce approximately `n_splits` groups

Parameters:
- `n_splits` (default=5): target number of folds
- `min_dist`: optional explicit distance threshold override

### PhyloCladeCV

Holds out entire monophyletic clades.

Algorithm:
1. Traverse internal nodes of the tree
2. Select n internal nodes whose subtrees partition tips into roughly equal groups
3. Each subtree = one fold

Parameters:
- `n_splits` (default=5)
- `min_clade_size` (default=2)

### CV + Estimator Interaction

When training on a fold, VCV and eigenvectors are recomputed from only the training
species (tree pruned internally). Prediction on held-out fold uses the full tree
(scenario 1: known tree placement). This correctly simulates the real-world scenario.

Both splitters implement sklearn `BaseCrossValidator.split(X, y, groups)` and work
with `cross_val_score`, `GridSearchCV`, etc.

## Feature Importance

### phylo_feature_importance()

```python
phylo_feature_importance(X, y, tree, species_names, feature_names=None,
                         n_repeats=10, scoring=None) -> DataFrame
```

Algorithm:
1. Fit uncorrected model: standard RF on raw (X, y), no whitening, no eigenvectors.
   Compute permutation importance -> `raw_importance`
2. Fit phylo-corrected model: PhyloRandomForest on (X_aug, y_white) with eigenvectors.
   Compute permutation importance on ORIGINAL features only (exclude eigenvector columns)
   -> `phylo_corrected_importance`
3. Return DataFrame:
   `feature | raw_importance | phylo_corrected_importance | delta`

Interpreting delta:
- `delta ~ 0`: feature is genuinely predictive regardless of phylogeny
- `delta < 0` (raw > corrected): feature was inflated by phylogenetic confounding
- `delta > 0` (corrected > raw): feature was suppressed by phylogenetic noise

Uses sklearn `permutation_importance` (not built-in RF Gini importance, which is biased
toward high-cardinality features). Scoring defaults to accuracy for classifiers, R2 for
regressors.

## Public API

### Estimators

```python
from treeml import PhyloRandomForestClassifier, PhyloRandomForestRegressor

model = PhyloRandomForestRegressor(
    n_estimators=100,           # passed to inner sklearn RF
    include_eigenvectors=True,  # toggle phylo eigenvector features
    eigenvector_variance=0.90,  # variance threshold for num eigenvectors
    random_state=42,
    **rf_kwargs                 # any other sklearn RF parameter
)

model.fit(X, y, tree=tree, species_names=names)
model.predict(X_new, tree=updated_tree, species_names=new_names)
model.predict(X_new)  # graceful degradation

clf = PhyloRandomForestClassifier(...)
clf.fit(X, y, tree=tree, species_names=names)
clf.predict(X_new, tree=updated_tree, species_names=new_names)
clf.predict_proba(X_new, tree=updated_tree, species_names=new_names)
```

### Cross-Validation

```python
from treeml import PhyloDistanceCV, PhyloCladeCV
from sklearn.model_selection import cross_val_score

cv = PhyloDistanceCV(tree=tree, species_names=names, n_splits=5)
scores = cross_val_score(model, X, y, cv=cv)
```

### Feature Importance

```python
from treeml import phylo_feature_importance

report = phylo_feature_importance(
    X, y, tree=tree, species_names=names,
    feature_names=["gene_A", "gene_B", ...],
    n_repeats=10, scoring="r2",
)
```

### Data Loading

```python
from treeml import load_data

X, y, tree, species_names = load_data(
    trait_file="data.tsv",
    tree_file="species.nwk",
    response="phenotype",
)
```

## CI/CD (mirroring PhyKIT)

- GitHub Actions on push
- Matrix: Python 3.11, 3.12, 3.13 on macos-latest
- Separate unit and integration test coverage uploaded to Codecov as separate flags
- Makefile with `test.unit`, `test.integration`, `test.coverage` targets
- `codecov.yml` ignoring tests and setup files
- `setup.py`-based packaging
- Docs job: pipenv build + deploy to gh-pages

## Documentation (Sphinx, PhyKIT styling)

Sphinx with `sphinx_rtd_theme`, PhyKIT's custom CSS (Oxygen font), sidebar with GitHub
buttons, Google Analytics.

### Tutorials

1. **Quick start: Predicting a phenotype** — Load tree + trait matrix, fit
   PhyloRandomForestRegressor, predict. Minimal working example.
2. **Classification: Predicting discrete traits** — Binary phenotype,
   PhyloRandomForestClassifier, accuracy evaluation.
3. **Why phylogenetic correction matters** — Same dataset, with vs. without correction
   side-by-side. Demonstrates inflated accuracy from phylogenetic confounding.
4. **Phylogenetic cross-validation** — Standard k-fold vs. PhyloDistanceCV vs.
   PhyloCladeCV. Shows how standard CV overestimates performance.
5. **Feature importance: Finding real signal** — Run phylo_feature_importance(),
   interpret raw vs. corrected vs. delta columns.
6. **Predicting new species** — With tree placement vs. without. Graceful degradation.
7. **Discordance-aware correction** — Pass gene trees for discordance-aware VCV.

## Implementation Chunks

In order of priority:

1. **Package scaffolding** — repo, setup.py, CI, codecov, Makefile
2. **VCV + whitening + eigenvectors** — the phylo correction core
3. **PhyloBaseEstimator + PhyloRandomForestRegressor** — first working model
4. **PhyloRandomForestClassifier** — classification variant
5. **PhyloDistanceCV + PhyloCladeCV** — cross-validation splitters
6. **phylo_feature_importance()** — inference report
7. **load_data() + docs + tutorials + README**
