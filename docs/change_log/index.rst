.. _change_log:


Change log
==========

Major changes to treeml are summarized here.

**0.1.1**

- Documentation improvements and expanded FAQ

**0.1.0**

Initial release with:

- PhyloRandomForestRegressor and PhyloRandomForestClassifier
- PhyloGradientBoostingRegressor and PhyloGradientBoostingClassifier
- PhyloSVMRegressor and PhyloSVMClassifier
- PhyloKNNRegressor and PhyloKNNClassifier
- PhyloRidge, PhyloLasso, and PhyloElasticNet regressors
- PhyloDistanceCV and PhyloCladeCV cross-validation splitters
- PhyloGridSearchCV and PhyloRandomizedSearchCV hyperparameter tuning
- phylo_feature_importance() comparative report
- phylo_shap() SHAP explainability with PhyloSHAPResult container
- phylo_model_comparison() benchmarking utility
- save_model() and load_model() serialization
- load_data() convenience function
- Eigenvector augmentation via PCoA on double-centered VCV matrix
- Phylogenetic whitening via Cholesky decomposition of VCV
- Discordance-aware VCV construction from gene trees
- All core calculations validated against R's ape package to machine epsilon precision
- Support for Python 3.11, 3.12, and 3.13
