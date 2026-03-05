Usage
=====

API Reference
-------------

Estimators
~~~~~~~~~~

- ``PhyloRandomForestRegressor`` -- Random Forest regressor with phylogenetic correction
- ``PhyloRandomForestClassifier`` -- Random Forest classifier with phylogenetic eigenvector features

Cross-Validation
~~~~~~~~~~~~~~~~

- ``PhyloDistanceCV`` -- Phylogenetic distance-based cross-validation
- ``PhyloCladeCV`` -- Clade-based cross-validation

Feature Importance
~~~~~~~~~~~~~~~~~~

- ``phylo_feature_importance()`` -- Comparative feature importance report

Data Loading
~~~~~~~~~~~~

- ``load_data()`` -- Load trait file and tree
