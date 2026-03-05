from typing import List, Tuple

import numpy as np
from phykit.services.tree.vcv_utils import build_vcv_matrix


def phylo_whiten(
    y: np.ndarray, tree, ordered_names: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Whiten target vector y using phylogenetic VCV matrix.

    Computes Cholesky decomposition of VCV: C = L L^T
    Returns y_white = L^{-1} y and the Cholesky factor L.
    """
    C = build_vcv_matrix(tree, ordered_names)
    L = np.linalg.cholesky(C)
    y_white = np.linalg.solve(L, y)
    return y_white, L


def phylo_unwhiten(y_white: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Reverse the whitening transformation: y = L @ y_white."""
    return L @ y_white
