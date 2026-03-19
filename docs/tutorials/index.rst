.. _tutorials:

Tutorials
=========

Step-by-step guides for using treeml.

|

1. Quick start: Predicting a phenotype
#######################################

This tutorial demonstrates how to use treeml to predict a continuous phenotype
while accounting for phylogenetic non-independence.

.. code-block:: python

   from treeml import PhyloRandomForestRegressor, load_data
   from sklearn.model_selection import cross_val_score
   from treeml import PhyloDistanceCV

   # Load data
   X, y, tree, species_names = load_data(
       trait_file="traits.tsv",
       tree_file="species.nwk"
   )

   # Create phylogenetic estimator
   model = PhyloRandomForestRegressor(n_estimators=100, random_state=42)

   # Fit with phylogenetic correction
   model.fit(X, y, tree=tree, species_names=species_names)

   # Evaluate with phylogenetic cross-validation
   cv = PhyloDistanceCV(tree=tree, species_names=species_names, n_splits=5)
   scores = cross_val_score(model, X, y, cv=cv)
   print(f"CV R² scores: {scores}")
   print(f"Mean R²: {scores.mean():.3f} ± {scores.std():.3f}")

|

2. Classification: Predicting discrete traits
##############################################

This tutorial demonstrates how to classify discrete traits (e.g., habitat type,
diet category) using phylogenetic-aware classifiers.

.. code-block:: python

   from treeml import PhyloRandomForestClassifier, PhyloDistanceCV
   from sklearn.model_selection import cross_val_score

   # Create classifier
   model = PhyloRandomForestClassifier(n_estimators=100, random_state=42)

   # Fit and evaluate
   model.fit(X, y, tree=tree, species_names=species_names)
   cv = PhyloDistanceCV(tree=tree, species_names=species_names, n_splits=5)
   scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
   print(f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

|

3. Why phylogenetic correction matters
########################################

When species are used as observations, ignoring phylogenetic relationships can
inflate performance metrics. This tutorial compares corrected vs. uncorrected
models.

.. code-block:: python

   from treeml import phylo_model_comparison

   results = phylo_model_comparison(
       X, y, tree=tree, species_names=species_names
   )
   print(results)

The ``phylo_model_comparison()`` function benchmarks multiple estimators with
and without phylogenetic correction and returns a DataFrame showing the
cross-validation scores and the delta between approaches.

|

4. Phylogenetic cross-validation
##################################

Standard k-fold cross-validation can leak phylogenetic signal between folds.
treeml provides two phylogenetic-aware CV strategies.

**Distance-based CV:** Groups species by phylogenetic distance using hierarchical
clustering on patristic distances.

.. code-block:: python

   from treeml import PhyloDistanceCV

   cv = PhyloDistanceCV(tree=tree, species_names=species_names, n_splits=5)
   for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
       print(f"Fold {fold}: {len(train_idx)} train, {len(test_idx)} test")

**Clade-based CV:** Holds out entire monophyletic clades for each fold.

.. code-block:: python

   from treeml import PhyloCladeCV

   cv = PhyloCladeCV(tree=tree, species_names=species_names, n_splits=5)
   for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
       print(f"Fold {fold}: {len(train_idx)} train, {len(test_idx)} test")

|

5. Feature importance: Finding real signal
###########################################

Compare feature importance with and without phylogenetic correction to
distinguish true biological signal from phylogenetic artifact.

.. code-block:: python

   from treeml import phylo_feature_importance

   report = phylo_feature_importance(
       X, y, tree=tree, species_names=species_names,
       feature_names=["body_mass", "diet_type", "habitat"]
   )
   print(report)

|

6. SHAP explanations
###########################

Use SHAP values to understand how much each feature and the phylogenetic
correction contribute to predictions.

.. code-block:: python

   from treeml import PhyloRandomForestRegressor, phylo_shap

   model = PhyloRandomForestRegressor(n_estimators=100)
   model.fit(X, y, tree=tree, species_names=species_names)

   result = phylo_shap(model, X, feature_names=["body_mass", "diet_type"])

   # Overall phylogenetic contribution
   print(f"Phylogenetic contribution: {result.phylo_contribution:.1%}")

   # Summary table
   print(result.summary())

   # Visualization
   result.plot(plot_type="bar")
   result.summary_plot()

|

7. Hyperparameter tuning
#################################

Use ``PhyloGridSearchCV`` or ``PhyloRandomizedSearchCV`` to find optimal
hyperparameters with phylogenetic-aware cross-validation.

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
       species_names=species_names,
       n_jobs=-1,
   )
   search.fit(X, y)

   print(f"Best params: {search.best_params_}")
   print(f"Best score: {search.best_score_:.3f}")

|
