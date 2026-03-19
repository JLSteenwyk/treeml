## 0.1.0

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
- Eigenvector augmentation and phylogenetic whitening
- Validated against R's ape package to machine epsilon precision
