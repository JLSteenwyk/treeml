from typing import Dict, List, Tuple

import numpy as np
from phykit.services.tree.vcv_utils import build_vcv_matrix


def extract_phylo_eigenvectors(
    tree,
    ordered_names: List[str],
    variance_threshold: float = 0.90,
) -> Tuple[np.ndarray, Dict]:
    """Extract phylogenetic eigenvectors from double-centered VCV matrix.

    Performs PCoA-like decomposition: double-center the VCV, eigendecompose,
    and retain eigenvectors explaining at least `variance_threshold` of variance.

    Returns (E, info) where E is (n_species x k) and info contains metadata.
    """
    C = build_vcv_matrix(tree, ordered_names)
    n = len(ordered_names)

    # Double-center the VCV matrix (Gower centering)
    row_means = C.mean(axis=1, keepdims=True)
    col_means = C.mean(axis=0, keepdims=True)
    grand_mean = C.mean()
    C_centered = C - row_means - col_means + grand_mean

    # Eigendecompose (symmetric, so use eigh)
    eigenvalues, eigenvectors = np.linalg.eigh(C_centered)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Keep only positive eigenvalues
    pos_mask = eigenvalues > 0
    eigenvalues = eigenvalues[pos_mask]
    eigenvectors = eigenvectors[:, pos_mask]

    if len(eigenvalues) == 0:
        return np.zeros((n, 0)), {
            "n_components": 0,
            "variance_explained": np.array([]),
        }

    # Determine number of components by variance threshold
    total_var = eigenvalues.sum()
    cumulative_var = np.cumsum(eigenvalues) / total_var
    n_components = int(np.searchsorted(cumulative_var, variance_threshold) + 1)
    n_components = min(n_components, len(eigenvalues))

    E = eigenvectors[:, :n_components]
    var_explained = eigenvalues[:n_components] / total_var

    return E, {
        "n_components": n_components,
        "variance_explained": var_explained,
    }
