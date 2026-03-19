.. _usage:

Usage
=====

^^^^^


.. _estimators:

Estimators
----------

treeml provides scikit-learn compatible estimators that account for phylogenetic
non-independence. All estimators accept ``tree`` and ``species_names`` arguments
in their ``fit()`` and ``predict()`` methods.

^^^^^

.. _regressors:

Regressors
~~~~~~~~~~

- :ref:`PhyloRandomForestRegressor <cmd-PhyloRandomForestRegressor>` -- Random Forest regressor with phylogenetic correction
- :ref:`PhyloGradientBoostingRegressor <cmd-PhyloGradientBoostingRegressor>` -- Gradient Boosting regressor with phylogenetic correction
- :ref:`PhyloSVMRegressor <cmd-PhyloSVMRegressor>` -- SVM regressor with phylogenetic correction
- :ref:`PhyloKNNRegressor <cmd-PhyloKNNRegressor>` -- KNN regressor with phylogenetic correction
- :ref:`PhyloRidge <cmd-PhyloRidge>` -- Ridge regression with phylogenetic correction
- :ref:`PhyloLasso <cmd-PhyloLasso>` -- Lasso regression with phylogenetic correction
- :ref:`PhyloElasticNet <cmd-PhyloElasticNet>` -- Elastic Net with phylogenetic correction

.. _classifiers:

Classifiers
~~~~~~~~~~~

- :ref:`PhyloRandomForestClassifier <cmd-PhyloRandomForestClassifier>` -- Random Forest classifier with phylogenetic eigenvector features
- :ref:`PhyloGradientBoostingClassifier <cmd-PhyloGradientBoostingClassifier>` -- Gradient Boosting classifier with phylogenetic eigenvector features
- :ref:`PhyloSVMClassifier <cmd-PhyloSVMClassifier>` -- SVM classifier with phylogenetic eigenvector features
- :ref:`PhyloKNNClassifier <cmd-PhyloKNNClassifier>` -- KNN classifier with phylogenetic eigenvector features

^^^^^

.. _estimator-parameters:

Common Parameters
~~~~~~~~~~~~~~~~~

All phylogenetic estimators share the following parameters in addition to
their underlying scikit-learn estimator parameters:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``include_eigenvectors``
     - ``True``
     - Augment feature matrix with phylogenetic eigenvectors from PCoA
   * - ``eigenvector_variance``
     - ``0.90``
     - Fraction of variance explained by retained eigenvectors
   * - ``whiten_features``
     - ``False``
     - Phylogenetically whiten the feature matrix using Cholesky decomposition
   * - ``whiten_target``
     - ``False``
     - Phylogenetically whiten the target variable (regressors only)

|

.. _cmd-PhyloRandomForestRegressor:

PhyloRandomForestRegressor
**************************

Random Forest regressor with phylogenetic correction. Wraps scikit-learn's
``RandomForestRegressor``.

.. code-block:: python

   from treeml import PhyloRandomForestRegressor
   from Bio import Phylo

   tree = Phylo.read("species.nwk", "newick")
   model = PhyloRandomForestRegressor(n_estimators=100, random_state=42)
   model.fit(X, y, tree=tree, species_names=names)
   predictions = model.predict(X, tree=tree, species_names=names)

|

.. _cmd-PhyloGradientBoostingRegressor:

PhyloGradientBoostingRegressor
******************************

Gradient Boosting regressor with phylogenetic correction. Wraps scikit-learn's
``GradientBoostingRegressor``.

.. code-block:: python

   from treeml import PhyloGradientBoostingRegressor

   model = PhyloGradientBoostingRegressor(n_estimators=200, learning_rate=0.1)
   model.fit(X, y, tree=tree, species_names=names)

|

.. _cmd-PhyloSVMRegressor:

PhyloSVMRegressor
*****************

Support Vector Machine regressor with phylogenetic correction. Wraps scikit-learn's
``SVR``.

.. code-block:: python

   from treeml import PhyloSVMRegressor

   model = PhyloSVMRegressor(kernel="rbf", C=1.0)
   model.fit(X, y, tree=tree, species_names=names)

|

.. _cmd-PhyloKNNRegressor:

PhyloKNNRegressor
*****************

K-Nearest Neighbors regressor with phylogenetic correction. Wraps scikit-learn's
``KNeighborsRegressor``.

.. code-block:: python

   from treeml import PhyloKNNRegressor

   model = PhyloKNNRegressor(n_neighbors=5)
   model.fit(X, y, tree=tree, species_names=names)

|

.. _cmd-PhyloRidge:

PhyloRidge
**********

Ridge regression with phylogenetic correction. Wraps scikit-learn's ``Ridge``.

.. code-block:: python

   from treeml import PhyloRidge

   model = PhyloRidge(alpha=1.0)
   model.fit(X, y, tree=tree, species_names=names)

|

.. _cmd-PhyloLasso:

PhyloLasso
**********

Lasso regression with phylogenetic correction. Wraps scikit-learn's ``Lasso``.

.. code-block:: python

   from treeml import PhyloLasso

   model = PhyloLasso(alpha=0.1)
   model.fit(X, y, tree=tree, species_names=names)

|

.. _cmd-PhyloElasticNet:

PhyloElasticNet
***************

Elastic Net with phylogenetic correction. Wraps scikit-learn's ``ElasticNet``.

.. code-block:: python

   from treeml import PhyloElasticNet

   model = PhyloElasticNet(alpha=0.1, l1_ratio=0.5)
   model.fit(X, y, tree=tree, species_names=names)

|

.. _cmd-PhyloRandomForestClassifier:

PhyloRandomForestClassifier
***************************

Random Forest classifier with phylogenetic eigenvector features. Wraps scikit-learn's
``RandomForestClassifier``.

.. code-block:: python

   from treeml import PhyloRandomForestClassifier

   model = PhyloRandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X, y, tree=tree, species_names=names)
   predictions = model.predict(X, tree=tree, species_names=names)

|

.. _cmd-PhyloGradientBoostingClassifier:

PhyloGradientBoostingClassifier
*******************************

Gradient Boosting classifier with phylogenetic eigenvector features. Wraps scikit-learn's
``GradientBoostingClassifier``.

.. code-block:: python

   from treeml import PhyloGradientBoostingClassifier

   model = PhyloGradientBoostingClassifier(n_estimators=200, learning_rate=0.1)
   model.fit(X, y, tree=tree, species_names=names)

|

.. _cmd-PhyloSVMClassifier:

PhyloSVMClassifier
******************

SVM classifier with phylogenetic eigenvector features. Wraps scikit-learn's ``SVC``.

.. code-block:: python

   from treeml import PhyloSVMClassifier

   model = PhyloSVMClassifier(kernel="rbf", C=1.0)
   model.fit(X, y, tree=tree, species_names=names)

|

.. _cmd-PhyloKNNClassifier:

PhyloKNNClassifier
******************

KNN classifier with phylogenetic eigenvector features. Wraps scikit-learn's
``KNeighborsClassifier``.

.. code-block:: python

   from treeml import PhyloKNNClassifier

   model = PhyloKNNClassifier(n_neighbors=5)
   model.fit(X, y, tree=tree, species_names=names)

|

^^^^^

.. _cross-validation:

Cross-Validation
----------------

treeml provides phylogenetic-aware cross-validation splitters that ensure closely
related species do not leak information between training and test folds.

.. _cmd-PhyloDistanceCV:

PhyloDistanceCV
~~~~~~~~~~~~~~~

Splits data based on phylogenetic distance using hierarchical clustering on patristic
distances from the tree.

.. code-block:: python

   from treeml import PhyloDistanceCV
   from sklearn.model_selection import cross_val_score

   cv = PhyloDistanceCV(tree=tree, species_names=names, n_splits=5)
   scores = cross_val_score(model, X, y, cv=cv)

|

.. _cmd-PhyloCladeCV:

PhyloCladeCV
~~~~~~~~~~~~

Holds out monophyletic clades for each fold, ensuring evolutionary-aware
train/test separation.

.. code-block:: python

   from treeml import PhyloCladeCV

   cv = PhyloCladeCV(tree=tree, species_names=names, n_splits=5)
   scores = cross_val_score(model, X, y, cv=cv)

|

^^^^^

.. _feature-importance:

Feature Importance
------------------

.. _cmd-phylo_feature_importance:

phylo_feature_importance
~~~~~~~~~~~~~~~~~~~~~~~~

Compare feature importance with and without phylogenetic correction using
permutation importance.

.. code-block:: python

   from treeml import phylo_feature_importance

   report = phylo_feature_importance(
       X, y, tree=tree, species_names=names,
       feature_names=["body_mass", "diet_type", "habitat"]
   )
   print(report)

|

^^^^^

.. _shap-explanations:

SHAP Explanations
-----------------

.. _cmd-phylo_shap:

phylo_shap
~~~~~~~~~~

Compute SHAP values for treeml estimators, separating feature contributions
from phylogenetic eigenvector contributions.

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

^^^^^

.. _model-comparison:

Model Comparison
----------------

.. _cmd-phylo_model_comparison:

phylo_model_comparison
~~~~~~~~~~~~~~~~~~~~~~

Compare multiple estimators with and without phylogenetic correction via
cross-validation.

.. code-block:: python

   from treeml import phylo_model_comparison

   results = phylo_model_comparison(
       X, y, tree=tree, species_names=names
   )
   print(results)

|

^^^^^

.. _hyperparameter-tuning:

Hyperparameter Tuning
---------------------

.. _cmd-PhyloGridSearchCV:

PhyloGridSearchCV
~~~~~~~~~~~~~~~~~

Grid search with phylogenetic cross-validation. Wraps scikit-learn's
``GridSearchCV`` while automatically binding tree and species_names.

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

.. _cmd-PhyloRandomizedSearchCV:

PhyloRandomizedSearchCV
~~~~~~~~~~~~~~~~~~~~~~~

Randomized search with phylogenetic cross-validation.

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

|

^^^^^

.. _serialization:

Serialization
-------------

.. _cmd-save_model:

save_model / load_model
~~~~~~~~~~~~~~~~~~~~~~~

Save and load fitted treeml estimators to disk.

.. code-block:: python

   from treeml import PhyloRandomForestRegressor, save_model, load_model

   model = PhyloRandomForestRegressor(n_estimators=100)
   model.fit(X, y, tree=tree, species_names=names)

   # Save
   save_model(model, "my_model.treeml")

   # Load
   loaded = load_model("my_model.treeml")
   preds = loaded.predict(X, tree=tree, species_names=names)

|

^^^^^

.. _data-loading:

Data Loading
------------

.. _cmd-load_data:

load_data
~~~~~~~~~

Load a tab-separated trait file and a phylogenetic tree file.

.. code-block:: python

   from treeml import load_data

   X, y, tree, species_names = load_data(
       trait_file="traits.tsv",
       tree_file="species.nwk"
   )

The trait file should be tab-separated with a header row. The first column contains
species names, the last column is the target variable, and the remaining columns are
features.

|
