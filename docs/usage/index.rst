Usage
=====

API Reference
-------------

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

Cross-Validation
~~~~~~~~~~~~~~~~

- ``PhyloDistanceCV`` -- Phylogenetic distance-based cross-validation
- ``PhyloCladeCV`` -- Clade-based cross-validation

Feature Importance
~~~~~~~~~~~~~~~~~~

- ``phylo_feature_importance()`` -- Comparative feature importance report

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

Model Comparison
~~~~~~~~~~~~~~~~

- ``phylo_model_comparison()`` -- Compare multiple estimators with and without phylogenetic correction

Data Loading
~~~~~~~~~~~~

- ``load_data()`` -- Load trait file and tree
