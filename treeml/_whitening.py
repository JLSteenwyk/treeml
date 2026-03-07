from typing import List, Optional, Tuple

import numpy as np
from phykit.services.tree.vcv_utils import build_vcv_matrix


def phylo_whiten(
    y: np.ndarray,
    tree,
    ordered_names: List[str],
    vcv: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Whiten target vector y using phylogenetic VCV matrix.

    Computes Cholesky decomposition of VCV: C = L L^T
    Returns y_white = L^{-1} y and the Cholesky factor L.

    If vcv is provided, uses it directly instead of computing from tree.
    """
    if vcv is None:
        C = build_vcv_matrix(tree, ordered_names)
    else:
        C = vcv
    L = np.linalg.cholesky(C)
    y_white = np.linalg.solve(L, y)
    return y_white, L


def phylo_whiten_features(
    X: np.ndarray,
    vcv: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Whiten feature matrix X using phylogenetic VCV matrix.

    Computes Cholesky decomposition of VCV: C = L L^T
    Returns X_white = L^{-1} X and the Cholesky factor L.

    This corrects for phylogenetic non-independence among samples
    by transforming features into a space where observations are
    independent under a Brownian motion model of evolution.
    """
    L = np.linalg.cholesky(vcv)
    X_white = np.linalg.solve(L, X)
    return X_white, L


def phylo_unwhiten(y_white: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Reverse the whitening transformation: y = L @ y_white."""
    return L @ y_white
