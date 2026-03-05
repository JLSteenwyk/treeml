"""
Validate treeml calculations against canonical R packages (ape).

Compares:
1. VCV matrix:  treeml (via PhyKIT) vs ape::vcv.phylo()
2. Cholesky L:  treeml vs R's chol()
3. Whitened y:  treeml vs R's solve(L, y)
4. PCoA eigenvectors: treeml vs R's eigen() on double-centered VCV
5. Patristic distances: derived from VCV vs ape::cophenetic.phylo()
"""

import os
import numpy as np
import pandas as pd
from Bio import Phylo
from scipy.stats import pearsonr

from treeml._whitening import phylo_whiten
from treeml._eigenvectors import extract_phylo_eigenvectors
from phykit.services.tree.vcv_utils import build_vcv_matrix

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "sample_files")
VAL_DIR = os.path.dirname(__file__)

# Load tree
tree = Phylo.read(os.path.join(SAMPLE_DIR, "tree_simple.nwk"), "newick")
sp_order = sorted([t.name for t in tree.get_terminals()])

# Load trait data (alphabetical order)
trait_data = {
    "bear": 289.0, "cat": 28.0, "dog": 72.0, "monkey": 68.0,
    "raccoon": 39.2, "sea_lion": 247.0, "seal": 172.0, "weasel": 7.6,
}
y_raw = np.array([trait_data[s] for s in sp_order])

print("=" * 70)
print("treeml vs R (ape) Validation")
print("=" * 70)
print(f"Species order: {sp_order}")
print()

results = []

# --- 1. VCV Matrix ---
print("-" * 70)
print("1. VCV Matrix: treeml (PhyKIT) vs ape::vcv.phylo()")
print("-" * 70)

py_vcv = build_vcv_matrix(tree, sp_order)
r_vcv = pd.read_csv(os.path.join(VAL_DIR, "r_vcv.csv"), index_col=0).values

# Flatten and correlate
r_flat, p_flat = r_vcv.flatten(), py_vcv.flatten()
corr, pval = pearsonr(r_flat, p_flat)
max_diff = np.max(np.abs(r_vcv - py_vcv))

print(f"  Pearson r       = {corr:.15f}")
print(f"  p-value         = {pval:.2e}")
print(f"  Max abs diff    = {max_diff:.2e}")
print(f"  Mean abs diff   = {np.mean(np.abs(r_vcv - py_vcv)):.2e}")
results.append(("VCV matrix", corr, max_diff))
print()

# --- 2. Cholesky L ---
print("-" * 70)
print("2. Cholesky L: treeml vs R's chol()")
print("-" * 70)

py_L = np.linalg.cholesky(py_vcv)
r_L = pd.read_csv(os.path.join(VAL_DIR, "r_cholesky_L.csv"), index_col=0).values

corr_L, pval_L = pearsonr(r_L.flatten(), py_L.flatten())
max_diff_L = np.max(np.abs(r_L - py_L))

print(f"  Pearson r       = {corr_L:.15f}")
print(f"  p-value         = {pval_L:.2e}")
print(f"  Max abs diff    = {max_diff_L:.2e}")
results.append(("Cholesky L", corr_L, max_diff_L))
print()

# --- 3. Whitened y ---
print("-" * 70)
print("3. Whitened y: treeml vs R's solve(L, y)")
print("-" * 70)

py_y_white, py_L2 = phylo_whiten(y_raw, tree, sp_order)
r_whitened = pd.read_csv(os.path.join(VAL_DIR, "r_whitened_y.csv"))
r_y_white = r_whitened["y_white"].values

corr_yw, pval_yw = pearsonr(r_y_white, py_y_white)
max_diff_yw = np.max(np.abs(r_y_white - py_y_white))

print(f"  Pearson r       = {corr_yw:.15f}")
print(f"  p-value         = {pval_yw:.2e}")
print(f"  Max abs diff    = {max_diff_yw:.2e}")
print(f"  R whitened y:    {np.round(r_y_white, 6)}")
print(f"  Py whitened y:   {np.round(py_y_white, 6)}")
results.append(("Whitened y", corr_yw, max_diff_yw))
print()

# --- 4. Eigenvectors ---
print("-" * 70)
print("4. PCoA Eigenvectors: treeml vs R's eigen() on double-centered VCV")
print("-" * 70)

py_E, py_info = extract_phylo_eigenvectors(tree, sp_order, variance_threshold=0.99)
r_eigvecs = pd.read_csv(os.path.join(VAL_DIR, "r_eigenvectors.csv"), index_col=0).values
r_eigvals_df = pd.read_csv(os.path.join(VAL_DIR, "r_eigenvalues.csv"))
r_eigvals = r_eigvals_df["eigenvalue"].values

# Recompute eigenvalues directly for comparison
C_centered = py_vcv.copy()
row_means = C_centered.mean(axis=1, keepdims=True)
col_means = C_centered.mean(axis=0, keepdims=True)
grand_mean = C_centered.mean()
C_centered = C_centered - row_means - col_means + grand_mean
all_eigvals = np.sort(np.linalg.eigh(C_centered)[0])[::-1]
py_eigvals = all_eigvals[all_eigvals > 1e-10]

print(f"  R eigenvalues:  {np.round(r_eigvals, 4)}")
print(f"  Py eigenvalues: {np.round(py_eigvals, 4)}")

# Eigenvalues correlation
n_compare = min(len(r_eigvals), len(py_eigvals))
corr_ev, pval_ev = pearsonr(r_eigvals[:n_compare], py_eigvals[:n_compare])
max_diff_ev = np.max(np.abs(r_eigvals[:n_compare] - py_eigvals[:n_compare]))
print(f"  Eigenvalue Pearson r  = {corr_ev:.15f}")
print(f"  Eigenvalue max diff   = {max_diff_ev:.2e}")
results.append(("Eigenvalues", corr_ev, max_diff_ev))

# Eigenvectors: signs may flip, so compare abs correlation per column
n_cols = min(py_E.shape[1], r_eigvecs.shape[1])
print(f"\n  Per-eigenvector |correlation| (sign may flip):")
evec_corrs = []
for i in range(n_cols):
    c, p = pearsonr(r_eigvecs[:, i], py_E[:, i])
    evec_corrs.append(abs(c))
    print(f"    PC{i+1}: |r| = {abs(c):.15f}")
mean_evec_corr = np.mean(evec_corrs)
results.append(("Eigenvectors (mean |r|)", mean_evec_corr, None))
print()

# --- 5. Patristic Distances ---
print("-" * 70)
print("5. Patristic Distances: derived from VCV vs ape::cophenetic.phylo()")
print("-" * 70)

r_dist = pd.read_csv(os.path.join(VAL_DIR, "r_patristic_distances.csv"), index_col=0).values

# Compute patristic distances from VCV: d_ij = VCV[i,i] + VCV[j,j] - 2*VCV[i,j]
n = py_vcv.shape[0]
py_dist = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        py_dist[i, j] = py_vcv[i, i] + py_vcv[j, j] - 2 * py_vcv[i, j]

# Use upper triangle only (exclude diagonal zeros)
mask = np.triu_indices(n, k=1)
corr_d, pval_d = pearsonr(r_dist[mask], py_dist[mask])
max_diff_d = np.max(np.abs(r_dist - py_dist))

print(f"  Pearson r       = {corr_d:.15f}")
print(f"  p-value         = {pval_d:.2e}")
print(f"  Max abs diff    = {max_diff_d:.2e}")
results.append(("Patristic distances", corr_d, max_diff_d))
print()

# --- Summary ---
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Calculation':<30} {'Pearson r':>20} {'Max |diff|':>15}")
print("-" * 70)
for name, corr, diff in results:
    diff_str = f"{diff:.2e}" if diff is not None else "N/A"
    print(f"{name:<30} {corr:>20.15f} {diff_str:>15}")
print("-" * 70)

all_corrs = [r[1] for r in results]
print(f"\nAll correlations >= 0.999999: {all(c >= 0.999999 for c in all_corrs)}")
print(f"Min correlation:              {min(all_corrs):.15f}")
