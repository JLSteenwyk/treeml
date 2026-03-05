from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.model_selection import cross_val_score, KFold

from treeml.estimators._regressor import PhyloRandomForestRegressor
from treeml.estimators._classifier import PhyloRandomForestClassifier
from treeml.estimators._gradient_boosting_regressor import PhyloGradientBoostingRegressor
from treeml.estimators._gradient_boosting_classifier import PhyloGradientBoostingClassifier
from treeml.estimators._svm_regressor import PhyloSVMRegressor
from treeml.estimators._svm_classifier import PhyloSVMClassifier
from treeml.estimators._knn_regressor import PhyloKNNRegressor
from treeml.estimators._knn_classifier import PhyloKNNClassifier
from treeml.estimators._elastic_net import PhyloElasticNet
from treeml.estimators._ridge import PhyloRidge
from treeml.estimators._lasso import PhyloLasso
from treeml.cv._distance import PhyloDistanceCV


def _default_regressors(random_state):
    return {
        "RandomForest": (
            RandomForestRegressor(n_estimators=100, random_state=random_state),
            PhyloRandomForestRegressor(n_estimators=100, random_state=random_state),
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(n_estimators=100, random_state=random_state),
            PhyloGradientBoostingRegressor(n_estimators=100, random_state=random_state),
        ),
        "SVM": (
            SVR(kernel="rbf"),
            PhyloSVMRegressor(kernel="rbf"),
        ),
        "KNN": (
            KNeighborsRegressor(n_neighbors=5),
            PhyloKNNRegressor(n_neighbors=5),
        ),
        "ElasticNet": (
            ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=random_state),
            PhyloElasticNet(alpha=1.0, l1_ratio=0.5, random_state=random_state),
        ),
        "Ridge": (
            Ridge(alpha=1.0),
            PhyloRidge(alpha=1.0),
        ),
        "Lasso": (
            Lasso(alpha=1.0, random_state=random_state),
            PhyloLasso(alpha=1.0, random_state=random_state),
        ),
    }


def _default_classifiers(random_state):
    return {
        "RandomForest": (
            RandomForestClassifier(n_estimators=100, random_state=random_state),
            PhyloRandomForestClassifier(n_estimators=100, random_state=random_state),
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(n_estimators=100, random_state=random_state),
            PhyloGradientBoostingClassifier(n_estimators=100, random_state=random_state),
        ),
        "SVM": (
            SVC(kernel="rbf", random_state=random_state),
            PhyloSVMClassifier(kernel="rbf", random_state=random_state),
        ),
        "KNN": (
            KNeighborsClassifier(n_neighbors=5),
            PhyloKNNClassifier(n_neighbors=5),
        ),
    }


def phylo_model_comparison(
    X,
    y,
    tree,
    species_names: List[str],
    models: Optional[Dict] = None,
    scoring: Optional[str] = None,
    n_splits: int = 3,
    random_state=None,
) -> pd.DataFrame:
    """Compare uncorrected vs phylo-corrected models side-by-side.

    For each model, fits on the full dataset and evaluates training score.
    Uncorrected models use raw (X, y); phylo-corrected models use
    eigenvector-augmented X and whitened y (for regressors).

    Args:
        X: Feature matrix (n_species x n_features).
        y: Target vector.
        tree: Phylogenetic tree (Bio.Phylo).
        species_names: Ordered species names matching rows of X.
        models: Optional dict of {name: (sklearn_model, phylo_model)} pairs.
            If None, uses all default estimators. Auto-detects regression
            vs classification from y.
        scoring: Scoring metric (default: 'r2' for regression,
            'accuracy' for classification).
        n_splits: Number of CV folds for PhyloDistanceCV (default: 3).
        random_state: Random state for reproducibility.

    Returns:
        DataFrame with columns: model, uncorrected_score,
        phylo_corrected_score, delta.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    unique_y = np.unique(y)
    is_classification = len(unique_y) <= 20 and np.all(
        unique_y == unique_y.astype(int)
    )

    if scoring is None:
        scoring = "accuracy" if is_classification else "r2"

    if models is None:
        if is_classification:
            models = _default_classifiers(random_state)
        else:
            models = _default_regressors(random_state)

    # CV splitters
    phylo_cv = PhyloDistanceCV(
        tree=tree, species_names=species_names, n_splits=n_splits
    )
    standard_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows = []
    for name, (raw_model, phylo_model) in models.items():
        # Uncorrected: standard CV on raw data
        try:
            raw_scores = cross_val_score(
                raw_model, X, y, cv=standard_cv, scoring=scoring
            )
            raw_mean = raw_scores.mean()
        except Exception:
            raw_mean = np.nan

        # Phylo-corrected: fit on full data, evaluate training score
        # (CV with tree passthrough isn't natively supported by sklearn)
        try:
            phylo_model.fit(X, y, tree=tree, species_names=species_names)
            preds = phylo_model.predict(X, tree=tree, species_names=species_names)

            if is_classification:
                from sklearn.metrics import accuracy_score
                phylo_score = accuracy_score(y, preds)
            else:
                from sklearn.metrics import r2_score
                phylo_score = r2_score(y, preds)
        except Exception:
            phylo_score = np.nan

        rows.append({
            "model": name,
            "uncorrected_score": raw_mean,
            "phylo_corrected_score": phylo_score,
            "delta": phylo_score - raw_mean if not (
                np.isnan(raw_mean) or np.isnan(phylo_score)
            ) else np.nan,
        })

    return pd.DataFrame(rows)
