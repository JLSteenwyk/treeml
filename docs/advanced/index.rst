Advanced Usage
==============

This section describes the various features and options of treeml.

- Estimators_
- `Common Parameters`_
- `Cross-Validation`_
- `Feature Importance`_
- `SHAP Explanations`_
- `Model Comparison`_
- `Hyperparameter Tuning`_
- Serialization_
- `Data Loading`_
- `All Estimators`_

|

.. _Estimators:

Estimators
----------

treeml provides scikit-learn compatible estimators that account for phylogenetic
non-independence. All estimators accept ``tree``, ``species_names``, and optionally
``gene_trees`` arguments in their ``fit()`` and ``predict()`` methods.

treeml estimators correct for phylogenetic non-independence using three approaches:

1. **Phylogenetic eigenvectors** -- Eigenvectors are extracted from the phylogenetic
   variance-covariance (VCV) matrix via PCoA and appended to the feature matrix,
   allowing the model to learn phylogenetic structure.

2. **Phylogenetic whitening** -- The feature matrix and/or target variable are
   transformed using the Cholesky decomposition of the VCV matrix, removing
   phylogenetic autocorrelation before model training.

3. **Phylogenetic cross-validation** -- Specialized CV splitters (distance-based and
   clade-based) ensure that closely related species do not leak information between
   training and test folds.

|

Regressors
~~~~~~~~~~

Regressors support both eigenvector augmentation and target/feature whitening.

* ``PhyloRandomForestRegressor`` -- Random Forest regressor with phylogenetic correction.
  Wraps scikit-learn's ``RandomForestRegressor``.
* ``PhyloGradientBoostingRegressor`` -- Gradient Boosting regressor with phylogenetic correction.
  Wraps scikit-learn's ``GradientBoostingRegressor``.
* ``PhyloSVMRegressor`` -- SVM regressor with phylogenetic correction.
  Wraps scikit-learn's ``SVR``. Parameters: ``kernel``, ``C``, ``epsilon``.
* ``PhyloKNNRegressor`` -- KNN regressor with phylogenetic correction.
  Wraps scikit-learn's ``KNeighborsRegressor``. Parameters: ``n_neighbors``, ``weights``.
  Clamps ``n_neighbors`` to ``n_samples - 1`` automatically.
* ``PhyloRidge`` -- Ridge regression with phylogenetic correction.
  Wraps scikit-learn's ``Ridge``. Parameter: ``alpha``.
* ``PhyloLasso`` -- Lasso regression with phylogenetic correction.
  Wraps scikit-learn's ``Lasso``. Parameter: ``alpha``.
* ``PhyloElasticNet`` -- Elastic Net with phylogenetic correction.
  Wraps scikit-learn's ``ElasticNet``. Parameters: ``alpha``, ``l1_ratio``.

.. code-block:: python

	from treeml import PhyloRandomForestRegressor
	from Bio import Phylo

	tree = Phylo.read("species.nwk", "newick")
	model = PhyloRandomForestRegressor(n_estimators=100, random_state=42)
	model.fit(X, y, tree=tree, species_names=names)
	predictions = model.predict(X, tree=tree, species_names=names)

.. code-block:: python

	# Gradient Boosting
	from treeml import PhyloGradientBoostingRegressor
	model = PhyloGradientBoostingRegressor(n_estimators=200, learning_rate=0.1)
	model.fit(X, y, tree=tree, species_names=names)

.. code-block:: python

	# SVM
	from treeml import PhyloSVMRegressor
	model = PhyloSVMRegressor(kernel="rbf", C=1.0)
	model.fit(X, y, tree=tree, species_names=names)

.. code-block:: python

	# KNN
	from treeml import PhyloKNNRegressor
	model = PhyloKNNRegressor(n_neighbors=5)
	model.fit(X, y, tree=tree, species_names=names)

.. code-block:: python

	# Ridge
	from treeml import PhyloRidge
	model = PhyloRidge(alpha=1.0)
	model.fit(X, y, tree=tree, species_names=names)

.. code-block:: python

	# Lasso
	from treeml import PhyloLasso
	model = PhyloLasso(alpha=0.1)
	model.fit(X, y, tree=tree, species_names=names)

.. code-block:: python

	# Elastic Net
	from treeml import PhyloElasticNet
	model = PhyloElasticNet(alpha=0.1, l1_ratio=0.5)
	model.fit(X, y, tree=tree, species_names=names)

|

Classifiers
~~~~~~~~~~~

Classifiers support eigenvector augmentation and feature whitening. They do not
support target whitening (``whiten_target``) since the target is categorical.
All classifiers provide ``predict_proba()`` in addition to ``predict()``.

* ``PhyloRandomForestClassifier`` -- Random Forest classifier with phylogenetic eigenvector features.
  Wraps scikit-learn's ``RandomForestClassifier``.
* ``PhyloGradientBoostingClassifier`` -- Gradient Boosting classifier with phylogenetic eigenvector features.
  Wraps scikit-learn's ``GradientBoostingClassifier``.
* ``PhyloSVMClassifier`` -- SVM classifier with phylogenetic eigenvector features.
  Wraps scikit-learn's ``SVC`` (with ``probability=True``). Parameters: ``kernel``, ``C``.
* ``PhyloKNNClassifier`` -- KNN classifier with phylogenetic eigenvector features.
  Wraps scikit-learn's ``KNeighborsClassifier``. Parameters: ``n_neighbors``, ``weights``.

.. code-block:: python

	from treeml import PhyloRandomForestClassifier

	model = PhyloRandomForestClassifier(n_estimators=100, random_state=42)
	model.fit(X, y, tree=tree, species_names=names)
	predictions = model.predict(X, tree=tree, species_names=names)
	probabilities = model.predict_proba(X, tree=tree, species_names=names)

.. code-block:: python

	# Gradient Boosting classifier
	from treeml import PhyloGradientBoostingClassifier
	model = PhyloGradientBoostingClassifier(n_estimators=200, learning_rate=0.1)
	model.fit(X, y, tree=tree, species_names=names)

.. code-block:: python

	# SVM classifier
	from treeml import PhyloSVMClassifier
	model = PhyloSVMClassifier(kernel="rbf", C=1.0)
	model.fit(X, y, tree=tree, species_names=names)

.. code-block:: python

	# KNN classifier
	from treeml import PhyloKNNClassifier
	model = PhyloKNNClassifier(n_neighbors=5)
	model.fit(X, y, tree=tree, species_names=names)

|

.. _`Common Parameters`:

Common Parameters
-----------------

All phylogenetic estimators share the following parameters in addition to
their underlying scikit-learn estimator parameters.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``include_eigenvectors``
     - ``True``
     - Augment feature matrix with phylogenetic eigenvectors from PCoA on the
       double-centered VCV matrix. The eigenvectors allow the model to learn
       phylogenetic structure directly from the data.
   * - ``eigenvector_variance``
     - ``0.90``
     - Fraction of variance explained by retained eigenvectors. Higher values
       retain more eigenvectors. Must be between 0 and 1.
   * - ``whiten_features``
     - ``True``
     - Phylogenetically whiten the feature matrix using Cholesky decomposition
       of the VCV matrix. This corrects for phylogenetic autocorrelation among
       samples under a Brownian motion model of evolution.
   * - ``whiten_target``
     - ``False``
     - Phylogenetically whiten the target variable (regressors only). Uses the
       Cholesky decomposition of the VCV to remove phylogenetic signal from y
       before model training. Predictions are un-whitened automatically.

|

**fit() and predict() arguments**

All estimators accept the following arguments in ``fit()`` and ``predict()``:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Argument
     - Description
   * - ``X``
     - Feature matrix of shape (n_species, n_features).
   * - ``y``
     - Target vector of shape (n_species,). Only required for ``fit()``.
   * - ``tree``
     - Phylogenetic tree (Bio.Phylo tree object). Required for ``fit()``.
       Optional for ``predict()`` -- if omitted, predictions are made without
       phylogenetic correction and a warning is raised.
   * - ``species_names``
     - List of species names matching the rows of X, in the same order.
       Required for ``fit()``. Optional for ``predict()``.
   * - ``gene_trees``
     - Optional list of gene trees for discordance-aware VCV construction.
       When provided, the VCV matrix accounts for gene tree discordance
       rather than using only the species tree.

|

.. _`Cross-Validation`:

Cross-Validation
-----------------

Standard k-fold cross-validation can leak phylogenetic signal between folds because
closely related species may appear in both training and test sets. treeml provides
two phylogenetic-aware CV splitters that prevent this.

|

**PhyloDistanceCV**

Splits data based on phylogenetic distance using hierarchical clustering (UPGMA)
on patristic distances computed from the tree. Species in the same cluster are
assigned to the same fold, ensuring that closely related species do not appear in
both training and test sets.

Parameters:

* ``tree`` -- Phylogenetic tree (Bio.Phylo).
* ``species_names`` -- List of species names matching rows of X.
* ``n_splits`` -- Number of CV folds. *Default: 5*.
* ``min_dist`` -- Optional minimum phylogenetic distance for cluster assignment.
  If omitted, the distance threshold is automatically tuned via binary search
  to produce the requested number of folds.

.. code-block:: python

	from treeml import PhyloDistanceCV
	from sklearn.model_selection import cross_val_score

	cv = PhyloDistanceCV(tree=tree, species_names=names, n_splits=5)
	scores = cross_val_score(model, X, y, cv=cv)
	print(f"CV R² scores: {scores}")

|

**PhyloCladeCV**

Holds out entire monophyletic clades for each fold. Clades are selected to be
non-overlapping and approximately equal in size. Remaining species not assigned
to any clade are added to the smallest fold.

Parameters:

* ``tree`` -- Phylogenetic tree (Bio.Phylo).
* ``species_names`` -- List of species names matching rows of X.
* ``n_splits`` -- Number of CV folds. *Default: 5*.
* ``min_clade_size`` -- Minimum number of tips for a clade to be considered. *Default: 2*.

.. code-block:: python

	from treeml import PhyloCladeCV

	cv = PhyloCladeCV(tree=tree, species_names=names, n_splits=5)
	for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
	    print(f"Fold {fold}: {len(train_idx)} train, {len(test_idx)} test")

|

.. _`Feature Importance`:

Feature Importance
------------------

The ``phylo_feature_importance()`` function compares feature importance with and
without phylogenetic correction using permutation importance. This helps distinguish
true biological signal from phylogenetic artifact.

For each feature, permutation importance measures how much the model's score drops
when that feature is randomly shuffled. The function fits both a standard (uncorrected)
Random Forest and a phylogenetic-corrected Random Forest, then reports the importance
scores side by side.

Parameters:

* ``X`` -- Feature matrix.
* ``y`` -- Target vector.
* ``tree`` -- Phylogenetic tree.
* ``species_names`` -- List of species names.
* ``feature_names`` -- Optional list of feature names. *Default: feature_0, feature_1, ...*
* ``n_repeats`` -- Number of permutation repeats. *Default: 10*.
* ``scoring`` -- Scoring metric. *Default: 'r2' for regression, 'accuracy' for classification*.
* ``n_estimators`` -- Number of trees in the Random Forest. *Default: 100*.
* ``random_state`` -- Random state for reproducibility.

Returns a DataFrame with columns: ``feature``, ``raw_importance``,
``phylo_corrected_importance``, ``delta``.

.. code-block:: python

	from treeml import phylo_feature_importance

	report = phylo_feature_importance(
	    X, y, tree=tree, species_names=names,
	    feature_names=["body_mass", "diet_type", "habitat"]
	)
	print(report)

|

.. _`SHAP Explanations`:

SHAP Explanations
-----------------

The ``phylo_shap()`` function computes SHAP values for a fitted treeml estimator,
separating contributions from original features vs. phylogenetic eigenvector corrections.
This allows you to understand how much of each prediction is driven by the biological
features versus the phylogenetic correction.

Parameters:

* ``model`` -- A fitted treeml estimator (must have been fitted with ``fit()``).
* ``X`` -- Original feature matrix (without eigenvector augmentation).
* ``feature_names`` -- Optional list of feature names. *Default: feature_0, feature_1, ...*

Returns a ``PhyloSHAPResult`` object with the following properties and methods:

* ``feature_shap`` -- DataFrame of SHAP values for original features only.
* ``phylo_shap`` -- DataFrame of SHAP values for phylogenetic eigenvector columns only.
* ``feature_importance`` -- DataFrame of mean \|SHAP\| per feature, sorted descending.
* ``phylo_contribution`` -- Float: fraction of total \|SHAP\| attributable to phylogenetic correction.
* ``summary()`` -- DataFrame with per-feature mean \|SHAP\| and a ``phylo_total`` row.
* ``plot(plot_type="bar")`` -- Horizontal bar chart of feature vs. phylogenetic contributions.
* ``plot(plot_type="beeswarm")`` -- Beeswarm plot of original features with phylo annotation.
* ``summary_plot()`` -- Standard SHAP beeswarm plot on all features (original + eigenvectors).
* ``force_plot(sample_idx=0)`` -- SHAP force plot for a single sample.

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

|

.. _`Model Comparison`:

Model Comparison
-----------------

The ``phylo_model_comparison()`` function benchmarks multiple estimators with and without
phylogenetic correction. For each model, it fits on the full dataset and evaluates both
a standard (uncorrected) version and a phylo-corrected version, reporting scores and
the delta between them.

Parameters:

* ``X`` -- Feature matrix.
* ``y`` -- Target vector.
* ``tree`` -- Phylogenetic tree.
* ``species_names`` -- List of species names.
* ``models`` -- Optional dict of ``{name: (sklearn_model, phylo_model)}`` pairs.
  If omitted, uses all default estimators. Auto-detects regression vs classification from y.
* ``scoring`` -- Scoring metric. *Default: 'r2' for regression, 'accuracy' for classification*.
* ``n_splits`` -- Number of CV folds. *Default: 3*.
* ``random_state`` -- Random state for reproducibility.

Returns a DataFrame with columns: ``model``, ``uncorrected_score``,
``phylo_corrected_score``, ``delta``.

Default models compared:

* **Regression**: RandomForest, GradientBoosting, SVM, KNN, ElasticNet, Ridge, Lasso
* **Classification**: RandomForest, GradientBoosting, SVM, KNN

.. code-block:: python

	from treeml import phylo_model_comparison

	results = phylo_model_comparison(
	    X, y, tree=tree, species_names=names
	)
	print(results)

|

.. _`Hyperparameter Tuning`:

Hyperparameter Tuning
---------------------

treeml provides wrappers around scikit-learn's ``GridSearchCV`` and ``RandomizedSearchCV``
that automatically forward ``tree`` and ``species_names`` to the estimator's ``fit()``
and ``predict()`` during cross-validation. If no ``cv`` argument is provided, they
default to ``PhyloDistanceCV`` with 5 splits.

|

**PhyloGridSearchCV**

Exhaustive grid search over a parameter grid with phylogenetic cross-validation.

Parameters:

* ``estimator`` -- A treeml estimator.
* ``param_grid`` -- Dictionary of parameter names to lists of values.
* ``tree`` -- Phylogenetic tree.
* ``species_names`` -- List of species names.
* ``gene_trees`` -- Optional gene trees for discordance-aware correction.
* ``cv`` -- Cross-validation strategy. *Default: PhyloDistanceCV(n_splits=5)*.
* ``scoring`` -- Scoring metric. *Default: auto-detected from estimator type*.
* ``n_jobs`` -- Number of parallel jobs. *Default: None (sequential)*.
* ``refit`` -- Refit the best estimator on the full dataset. *Default: True*.

Properties after fitting: ``best_params_``, ``best_score_``, ``best_estimator_``, ``cv_results_``.

.. code-block:: python

	from treeml import PhyloRandomForestRegressor, PhyloGridSearchCV

	model = PhyloRandomForestRegressor(random_state=42)

	search = PhyloGridSearchCV(
	    estimator=model,
	    param_grid={
	        "n_estimators": [50, 100, 200],
	        "eigenvector_variance": [0.8, 0.9, 0.95],
	    },
	    tree=tree,
	    species_names=names,
	    n_jobs=-1,
	)
	search.fit(X, y)

	print(f"Best params: {search.best_params_}")
	print(f"Best score: {search.best_score_:.3f}")

	# Predict with best model
	preds = search.predict(X)

|

**PhyloRandomizedSearchCV**

Randomized search over parameter distributions with phylogenetic cross-validation.
Useful when the parameter space is large.

Additional parameters beyond PhyloGridSearchCV:

* ``param_distributions`` -- Dictionary of parameter names to distributions or lists.
* ``n_iter`` -- Number of parameter settings sampled. *Default: 10*.
* ``random_state`` -- Random state for reproducibility.

.. code-block:: python

	from treeml import PhyloRandomForestRegressor, PhyloRandomizedSearchCV
	from scipy.stats import randint

	model = PhyloRandomForestRegressor(random_state=42)

	search = PhyloRandomizedSearchCV(
	    estimator=model,
	    param_distributions={
	        "n_estimators": randint(50, 300),
	        "eigenvector_variance": [0.8, 0.9, 0.95],
	    },
	    tree=tree,
	    species_names=names,
	    n_iter=20,
	    n_jobs=-1,
	)
	search.fit(X, y)

	print(f"Best params: {search.best_params_}")
	print(f"Best score: {search.best_score_:.3f}")

|

.. _Serialization:

Serialization
--------------

treeml provides functions to save and load fitted estimators to disk using ``joblib``.
Model files use the ``.treeml`` extension and bundle the estimator with metadata
(treeml version and estimator class name).

|

**save_model**

Save a fitted treeml estimator to disk.

* ``model`` -- A fitted treeml estimator (must have been fitted with ``fit()``).
* ``path`` -- File path. If it doesn't end with ``.treeml``, the extension is appended.

Returns the actual path the model was saved to.

.. code-block:: python

	from treeml import save_model

	save_model(model, "my_model.treeml")

|

**load_model**

Load a treeml estimator from disk. A warning is raised if the model was saved
with a different version of treeml.

.. note::

	Only load model files you trust. Loading a malicious file can execute
	arbitrary code via pickle deserialization.

* ``path`` -- Path to a ``.treeml`` file.

.. code-block:: python

	from treeml import load_model

	loaded = load_model("my_model.treeml")
	preds = loaded.predict(X, tree=tree, species_names=names)

|

.. _`Data Loading`:

Data Loading
-------------

The ``load_data()`` function loads a tab-separated trait file and a phylogenetic tree
file, returning the feature matrix, target vector, tree, and species names ready for
use with treeml estimators.

Parameters:

* ``trait_file`` -- Path to tab-separated file. First row is a header
  (species, trait1, trait2, ...). Subsequent rows are data. Species not found
  in the tree are silently skipped.
* ``tree_file`` -- Path to tree file (Newick or Nexus format).
* ``response`` -- Column name to use as the target variable y. All other
  columns (except the species column) become features in X.
* ``tree_format`` -- Format of tree file. *Default: "newick"*.

Returns a tuple: ``(X, y, tree, species_names)``.

.. code-block:: python

	from treeml import load_data

	X, y, tree, species_names = load_data(
	    trait_file="traits.tsv",
	    tree_file="species.nwk",
	    response="body_mass",
	)

The trait file should look like:

.. code-block:: shell

	species	body_mass	diet_type	habitat
	Homo_sapiens	70.0	1	2
	Pan_troglodytes	52.0	1	1
	Gorilla_gorilla	160.0	1	1
	Mus_musculus	0.02	0	0

|

.. _`All Estimators`:

All Estimators
---------------------


.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Estimator
     - Description
   * - ``PhyloRandomForestRegressor``
     - Random Forest regressor with phylogenetic correction. Supports eigenvectors, feature whitening, and target whitening.
   * - ``PhyloGradientBoostingRegressor``
     - Gradient Boosting regressor with phylogenetic correction. Supports eigenvectors, feature whitening, and target whitening.
   * - ``PhyloSVMRegressor``
     - SVM regressor (``SVR``) with phylogenetic correction. Parameters: ``kernel``, ``C``, ``epsilon``.
   * - ``PhyloKNNRegressor``
     - K-Nearest Neighbors regressor with phylogenetic correction. Auto-clamps ``n_neighbors`` to ``n_samples - 1``.
   * - ``PhyloRidge``
     - Ridge regression with phylogenetic correction. Parameter: ``alpha``.
   * - ``PhyloLasso``
     - Lasso regression with phylogenetic correction. Parameter: ``alpha``.
   * - ``PhyloElasticNet``
     - Elastic Net with phylogenetic correction. Parameters: ``alpha``, ``l1_ratio``.
   * - ``PhyloRandomForestClassifier``
     - Random Forest classifier with phylogenetic eigenvector features. Provides ``predict_proba()``.
   * - ``PhyloGradientBoostingClassifier``
     - Gradient Boosting classifier with phylogenetic eigenvector features. Provides ``predict_proba()``.
   * - ``PhyloSVMClassifier``
     - SVM classifier (``SVC``) with phylogenetic eigenvector features. ``probability=True`` by default.
   * - ``PhyloKNNClassifier``
     - KNN classifier with phylogenetic eigenvector features. Provides ``predict_proba()``.
   * - ``PhyloDistanceCV``
     - Distance-based phylogenetic cross-validation splitter.
   * - ``PhyloCladeCV``
     - Clade-based phylogenetic cross-validation splitter.
   * - ``PhyloGridSearchCV``
     - Grid search with phylogenetic cross-validation.
   * - ``PhyloRandomizedSearchCV``
     - Randomized search with phylogenetic cross-validation.
   * - ``phylo_feature_importance()``
     - Compare permutation importance with and without phylogenetic correction.
   * - ``phylo_shap()``
     - Compute SHAP values separating feature vs. phylogenetic contributions.
   * - ``phylo_model_comparison()``
     - Benchmark multiple estimators with and without phylogenetic correction.
   * - ``save_model()``
     - Save a fitted treeml estimator to a ``.treeml`` file.
   * - ``load_model()``
     - Load a treeml estimator from a ``.treeml`` file.
   * - ``load_data()``
     - Load trait data and phylogenetic tree from files.
