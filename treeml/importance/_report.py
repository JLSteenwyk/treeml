from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance

from treeml.estimators._regressor import PhyloRandomForestRegressor
from treeml.estimators._classifier import PhyloRandomForestClassifier


def phylo_feature_importance(
    X,
    y,
    tree,
    species_names: List[str],
    feature_names: Optional[List[str]] = None,
    n_repeats: int = 10,
    scoring: Optional[str] = None,
    n_estimators: int = 100,
    random_state=None,
) -> pd.DataFrame:
    """Compare feature importance with and without phylogenetic correction."""
    X = np.asarray(X)
    y = np.asarray(y)
    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    unique_y = np.unique(y)
    is_classification = len(unique_y) <= 20 and np.all(unique_y == unique_y.astype(int))

    if scoring is None:
        scoring = "accuracy" if is_classification else "r2"

    # 1. Uncorrected model
    if is_classification:
        raw_model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
    else:
        raw_model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state
        )
    raw_model.fit(X, y)
    raw_result = permutation_importance(
        raw_model, X, y, n_repeats=n_repeats,
        scoring=scoring, random_state=random_state,
    )
    raw_importance = raw_result.importances_mean

    # 2. Phylo-corrected model
    if is_classification:
        phylo_model = PhyloRandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
    else:
        phylo_model = PhyloRandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state
        )
    phylo_model.fit(X, y, tree=tree, species_names=species_names)

    X_aug_for_eval, _ = phylo_model._augment_features(X, tree, species_names)

    if not is_classification:
        from treeml._whitening import phylo_whiten
        y_eval, _ = phylo_whiten(y, tree, species_names)
    else:
        y_eval = y

    phylo_result = permutation_importance(
        phylo_model.inner_model_, X_aug_for_eval, y_eval,
        n_repeats=n_repeats, scoring=scoring, random_state=random_state,
    )
    phylo_importance = phylo_result.importances_mean[:n_features]

    # 3. Build report
    delta = phylo_importance - raw_importance
    report = pd.DataFrame({
        "feature": feature_names,
        "raw_importance": raw_importance,
        "phylo_corrected_importance": phylo_importance,
        "delta": delta,
    })

    return report
