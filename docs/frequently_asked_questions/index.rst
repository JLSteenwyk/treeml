.. _faq:


FAQ
===

**What is phylogenetic non-independence?**

Species are related through evolution. Closely related species tend to have similar
traits not because of shared selective pressures but because they inherited those
traits from a common ancestor. Standard ML methods assume observations are independent,
which species data violates.

|

**Why not just use PGLS?**

PGLS (Phylogenetic Generalized Least Squares) is a linear method. It works well for
linear relationships, but many biological relationships are non-linear. treeml extends
phylogenetic correction to non-linear ML methods like Random Forest, Gradient Boosting,
and SVM.

|

**When should I use phylogenetic correction?**

Whenever your rows are species (or populations, strains) that are related by a
phylogenetic tree and you want to avoid confounding due to shared ancestry.

|

**What tree format does treeml accept?**

treeml reads trees via BioPython's ``Phylo`` module. Newick and Nexus formats are
supported. Trees should be rooted and have branch lengths for best results.

|

**Can I use treeml with scikit-learn's cross_val_score and GridSearchCV?**

Yes. All treeml estimators are scikit-learn compatible. Use ``PhyloDistanceCV`` or
``PhyloCladeCV`` as the ``cv`` argument to ensure phylogenetic-aware splitting.
``PhyloGridSearchCV`` and ``PhyloRandomizedSearchCV`` provide convenience wrappers
that automatically bind tree and species_names.

|

**I am having trouble installing treeml, what should I do?**

Please install treeml using a virtual environment as directed in the installation
instructions. If you are still running into issues after installing in a virtual
environment, please contact the developer via email_ or open a GitHub issue_.

.. _email: https://jlsteenwyk.com/contact.html
.. _issue: https://github.com/jlsteenwyk/treeml/issues
