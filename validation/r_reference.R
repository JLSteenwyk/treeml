library(ape)

# Load tree
tree <- read.tree("../tests/sample_files/tree_simple.nwk")

# Species order (alphabetical for reproducibility)
sp_order <- sort(tree$tip.label)
cat("Species order:", paste(sp_order, collapse=", "), "\n")

# --- 1. VCV matrix ---
vcv_mat <- vcv.phylo(tree)
# Reorder to alphabetical
vcv_mat <- vcv_mat[sp_order, sp_order]
write.csv(vcv_mat, "r_vcv.csv")
cat("VCV matrix written\n")

# --- 2. Cholesky decomposition ---
L <- t(chol(vcv_mat))  # Lower triangular (R's chol returns upper)
write.csv(L, "r_cholesky_L.csv")
cat("Cholesky L written\n")

# --- 3. Whitened y ---
# Use brain_size as the response
y_raw <- c(
  bear=289.0, cat=28.0, dog=72.0, monkey=68.0,
  raccoon=39.2, sea_lion=247.0, seal=172.0, weasel=7.6
)
y_raw <- y_raw[sp_order]
y_white <- solve(L, y_raw)
write.csv(data.frame(species=sp_order, y_raw=y_raw, y_white=y_white),
          "r_whitened_y.csv", row.names=FALSE)
cat("Whitened y written\n")

# --- 4. PCoA eigenvectors from double-centered VCV ---
n <- nrow(vcv_mat)
H <- diag(n) - matrix(1/n, n, n)  # Centering matrix
C_centered <- -0.5 * H %*% vcv_mat %*% H  # Note: PCoA uses -0.5 * H D H
# Actually for VCV (not distance), we just double-center directly
C_centered2 <- H %*% vcv_mat %*% H  # Double-center the VCV

eig <- eigen(C_centered2, symmetric=TRUE)
# Keep positive eigenvalues
pos <- eig$values > 1e-10
eigvals <- eig$values[pos]
eigvecs <- eig$vectors[, pos]

# Compute variance explained
var_explained <- cumsum(eigvals) / sum(eigvals)
cat("Eigenvalues:", eigvals, "\n")
cat("Cumulative variance explained:", var_explained, "\n")

# Write eigenvalues
write.csv(data.frame(eigenvalue=eigvals, cumvar=var_explained),
          "r_eigenvalues.csv", row.names=FALSE)

# Write eigenvectors (rows=species in sp_order, cols=PC1..PCk)
rownames(eigvecs) <- sp_order
colnames(eigvecs) <- paste0("PC", 1:ncol(eigvecs))
write.csv(eigvecs, "r_eigenvectors.csv")
cat("Eigenvectors written\n")

# --- 5. Patristic distance matrix ---
dist_mat <- cophenetic.phylo(tree)
dist_mat <- dist_mat[sp_order, sp_order]
write.csv(dist_mat, "r_patristic_distances.csv")
cat("Patristic distances written\n")

cat("\nAll R reference outputs saved.\n")
