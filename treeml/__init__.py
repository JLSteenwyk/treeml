from treeml.version import __version__
from treeml.estimators._classifier import PhyloRandomForestClassifier
from treeml.estimators._regressor import PhyloRandomForestRegressor
from treeml.estimators._gradient_boosting_classifier import PhyloGradientBoostingClassifier
from treeml.estimators._gradient_boosting_regressor import PhyloGradientBoostingRegressor
from treeml.estimators._svm_classifier import PhyloSVMClassifier
from treeml.estimators._svm_regressor import PhyloSVMRegressor
from treeml.estimators._knn_classifier import PhyloKNNClassifier
from treeml.estimators._knn_regressor import PhyloKNNRegressor
from treeml.estimators._elastic_net import PhyloElasticNet
from treeml.estimators._ridge import PhyloRidge
from treeml.estimators._lasso import PhyloLasso
from treeml.cv._distance import PhyloDistanceCV
from treeml.cv._clade import PhyloCladeCV
from treeml.importance._report import phylo_feature_importance
from treeml.comparison._compare import phylo_model_comparison
from treeml.shap._shap import phylo_shap, PhyloSHAPResult
from treeml._io import load_data

__all__ = [
    "__version__",
    "PhyloRandomForestClassifier",
    "PhyloRandomForestRegressor",
    "PhyloGradientBoostingClassifier",
    "PhyloGradientBoostingRegressor",
    "PhyloSVMClassifier",
    "PhyloSVMRegressor",
    "PhyloKNNClassifier",
    "PhyloKNNRegressor",
    "PhyloElasticNet",
    "PhyloRidge",
    "PhyloLasso",
    "PhyloDistanceCV",
    "PhyloCladeCV",
    "phylo_feature_importance",
    "phylo_model_comparison",
    "phylo_shap",
    "PhyloSHAPResult",
    "load_data",
]
