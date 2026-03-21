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

**What do the different combinations of whiten_features and include_eigenvectors do?**

treeml's two main phylogenetic correction parameters can be combined in four ways,
each with different implications for how phylogeny enters the model:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - ``whiten_features``
     - ``include_eigenvectors``
     - Effect
   * - ``False``
     - ``False``
     - **No phylogenetic correction.** Raw features only. Phylogeny is ignored entirely
       in the feature space (though phylogenetic-aware CV can still be used for honest
       evaluation). This is the standard ML baseline.
   * - ``False``
     - ``True``
     - **Eigenvector augmentation only.** Original features are preserved intact and
       phylogenetic eigenvectors are appended as additional columns. The model can
       learn to use phylogenetic structure as a predictor without distorting the
       original feature space. This is a lossless approach — no information is removed,
       and the model decides how much weight to give phylogenetic vs. genomic features.
   * - ``True``
     - ``False``
     - **Whitening only.** Features are transformed by L⁻¹X (Cholesky of VCV) to remove
       phylogenetic autocorrelation. Observations become approximately independent under
       a Brownian motion model, analogous to GLS residuals. No phylogenetic structure is
       explicitly provided to the model. This is the most aggressive correction — it
       assumes all phylogenetic structure in features is confounding.
   * - ``True``
     - ``True``
     - **Whitening + eigenvector augmentation (default).** Features are whitened to remove
       phylogenetic autocorrelation, then eigenvectors are appended to provide the model
       with explicit phylogenetic covariates. This is the PGLS analogue for ML — it
       separates phylogenetic structure from the features and re-introduces it as
       controlled covariates. However, the eigenvectors are a lossy approximation
       (typically capturing ~90% of phylogenetic variance), so some fine-grained
       feature-by-clade associations lost during whitening may not be fully recovered.

**Choosing the right combination** depends on whether phylogenetic structure in your
features is signal or confound for the trait you are predicting. When both features and
trait are strongly phylogenetically conserved, whitening can remove genuine predictive
signal. When the trait is phylogenetically labile, whitening removes confounding
structure. Eigenvector augmentation is generally safe to include in either case since the
model can learn to ignore uninformative eigenvectors. We recommend comparing multiple
combinations using phylogenetic-aware CV to determine the best approach for your dataset.

|

**Why does phylogenetic whitening sometimes hurt performance?**

When both the trait and genomic features are strongly phylogenetically conserved (e.g.,
Pagel's λ > 0.8), the shared phylogenetic structure between features and trait is
genuinely predictive — not a statistical confound. Whitening with L⁻¹X removes this
shared structure, discarding real signal. While eigenvector augmentation
(``include_eigenvectors=True``) adds back broad phylogenetic axes, it cannot fully
recover the fine-grained feature-by-clade associations that were destroyed by
whitening. In empirical tests with yeast metabolic traits, eigenvectors accounted for
only 4–8% of total model importance in phylo-corrected models, indicating they are a
lossy approximation of the removed structure.

In contrast, for traits with weaker phylogenetic signal (e.g., individual substrate
growth rates), the phylogenetic structure in features acts more as a confound than as
signal, and whitening can improve performance.

As a practical guideline:

- **Always use phylogenetic-aware CV** (``PhyloDistanceCV`` or ``PhyloCladeCV``) — this
  provides an honest estimate of model generalization regardless of whether you whiten.
- **Try both** ``whiten_features=True`` and ``whiten_features=False`` — the optimal
  choice depends on how much the phylogenetic structure in features overlaps with the
  trait signal.
- **Traits with strong phylogenetic signal** (high λ): whitening is more likely to hurt.
- **Traits with weak phylogenetic signal** (low λ): whitening is more likely to help.

|

**I am having trouble installing treeml, what should I do?**

Please install treeml using a virtual environment as directed in the installation
instructions. treeml requires Python >= 3.11. If you are still running into issues
after installing in a virtual environment, please contact the developer via email_
or open a GitHub issue_.

.. _email: https://jlsteenwyk.com/contact.html
.. _issue: https://github.com/jlsteenwyk/treeml/issues

^^^^^
