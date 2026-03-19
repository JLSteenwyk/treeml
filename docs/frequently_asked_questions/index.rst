.. _faq:


FAQ
===

**What is phylogenetic non-independence?**

Species are related through evolution. Closely related species tend to have similar
traits not because of shared selective pressures but because they inherited those
traits from a common ancestor. Standard ML methods assume observations are independent,
which species data violates. Ignoring this non-independence can inflate performance
metrics and lead to biased inferences.

|

**Why not just use PGLS?**

PGLS (Phylogenetic Generalized Least Squares) is a linear method. It works well for
linear relationships, but many biological relationships are non-linear. treeml extends
phylogenetic correction to non-linear ML methods like Random Forest, Gradient Boosting,
and SVM while remaining compatible with scikit-learn workflows.

|

**When should I use phylogenetic correction?**

Whenever your rows are species (or populations, strains) that are related by a
phylogenetic tree and you want to avoid confounding due to shared ancestry. This
applies to both regression and classification tasks.

|

**What tree format does treeml accept?**

treeml reads trees via BioPython's ``Phylo`` module. Newick and Nexus formats are
supported. Trees should be rooted and have branch lengths for best results. treeml
will warn if the tree appears unrooted (root has more than 2 children) or if any
terminals have missing branch lengths.

|

**Can I use treeml with scikit-learn's cross_val_score and GridSearchCV?**

Yes. All treeml estimators are scikit-learn compatible. Use ``PhyloDistanceCV`` or
``PhyloCladeCV`` as the ``cv`` argument to ensure phylogenetic-aware splitting.
``PhyloGridSearchCV`` and ``PhyloRandomizedSearchCV`` provide convenience wrappers
that automatically bind tree and species_names.

|

**What is eigenvector augmentation?**

treeml extracts eigenvectors from the phylogenetic variance-covariance (VCV) matrix
using PCoA-like decomposition (double-centering followed by eigendecomposition).
These eigenvectors capture phylogenetic structure and are appended as additional
features. The ``eigenvector_variance`` parameter controls how many eigenvectors are
retained based on the fraction of variance explained (default: 90%).

|

**What is phylogenetic whitening?**

Phylogenetic whitening transforms the data using the Cholesky decomposition of the
VCV matrix. Under a Brownian motion model of evolution, this transformation removes
phylogenetic autocorrelation, making the observations independent. treeml can whiten
both features (``whiten_features=True``) and the target variable
(``whiten_target=True``, regressors only).

|

**What is the difference between PhyloDistanceCV and PhyloCladeCV?**

``PhyloDistanceCV`` groups species by phylogenetic distance using hierarchical clustering
(UPGMA) and assigns groups to folds. ``PhyloCladeCV`` holds out entire monophyletic clades,
which is a stricter separation because the entire evolutionary lineage is held out.
``PhyloCladeCV`` may produce uneven fold sizes depending on tree topology.

|

**Can I use gene trees for discordance-aware correction?**

Yes. All estimators accept an optional ``gene_trees`` argument. When provided, the VCV
matrix is constructed accounting for gene tree discordance (e.g., due to incomplete
lineage sorting) rather than using only the species tree.

|

**I am having trouble installing treeml, what should I do?**

Please install treeml using a virtual environment as directed in the installation
instructions. treeml requires Python >= 3.11. If you are still running into issues
after installing in a virtual environment, please contact the developer via email_
or open a GitHub issue_.

.. _email: https://jlsteenwyk.com/contact.html
.. _issue: https://github.com/jlsteenwyk/treeml/issues

^^^^^
