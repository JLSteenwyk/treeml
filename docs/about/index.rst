.. _about:

About
=====

treeml provides phylogenetic machine learning estimators that are compatible with
scikit-learn. It accounts for the evolutionary non-independence of species when
training ML models on comparative data.

Standard machine learning methods assume that observations are independent. However,
species are related through evolution -- closely related species tend to have similar
traits not because of shared selective pressures but because they inherited those traits
from a common ancestor. treeml corrects for this non-independence using phylogenetic
variance-covariance matrices, eigenvector augmentation, and phylogenetic whitening,
extending phylogenetic correction to non-linear ML methods like Random Forest,
Gradient Boosting, and SVM.

|

How it Works
------------

treeml estimators account for evolutionary relationships using three approaches:

1. **Phylogenetic eigenvectors** -- Eigenvectors are extracted from the phylogenetic
   variance-covariance (VCV) matrix via PCoA and appended to the feature matrix,
   allowing the model to learn phylogenetic structure.

2. **Phylogenetic whitening** -- The target variable and/or feature matrix are
   transformed using the Cholesky decomposition of the VCV matrix, removing
   phylogenetic autocorrelation before model training.

3. **Phylogenetic cross-validation** -- Specialized CV splitters (distance-based and
   clade-based) ensure that closely related species do not leak information between
   training and test folds.

|

Validation against R
---------------------

All core phylogenetic calculations in treeml have been validated against canonical
R implementations from the ``ape`` package. Using an 8-species mammalian phylogeny,
every computation produces a **Pearson correlation of 1.0** with R's output, with
maximum absolute differences at machine epsilon (1e-13 to 1e-14).

.. list-table::
   :header-rows: 1
   :widths: 40 25 25

   * - Calculation
     - R reference
     - Pearson *r*
   * - VCV matrix
     - ``ape::vcv.phylo()``
     - 1.000000000000000
   * - Cholesky decomposition
     - ``chol()``
     - 1.000000000000000
   * - Phylogenetic whitening
     - ``solve(L, y)``
     - 1.000000000000000
   * - PCoA eigenvalues
     - ``eigen()``
     - 1.000000000000000
   * - PCoA eigenvectors (all 7)
     - ``eigen()``
     - \|r\| = 1.0 each
   * - Patristic distances
     - ``ape::cophenetic.phylo()``
     - 1.000000000000000

The validation scripts are available in the ``validation/`` directory of the
repository. ``r_reference.R`` computes reference values using R's ``ape`` package,
and ``validate_against_r.py`` compares them against treeml's output.

|

The Developer
-------------

treeml is developed and maintained by `Jacob L. Steenwyk <https://jlsteenwyk.github.io/>`_.

|

Citation
--------

If you use treeml, please cite: [TBD]
