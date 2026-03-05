treeml
======

Phylogenetic machine learning: scikit-learn estimators that account for evolutionary
non-independence among species.

Quick Start
-----------

.. code-block:: shell

   pip install treeml

.. code-block:: python

   from treeml import PhyloRandomForestRegressor, PhyloDistanceCV
   from sklearn.model_selection import cross_val_score
   from Bio import Phylo

   tree = Phylo.read("species.nwk", "newick")

   model = PhyloRandomForestRegressor(n_estimators=100)
   model.fit(X, y, tree=tree, species_names=names)

   cv = PhyloDistanceCV(tree=tree, species_names=names, n_splits=5)
   scores = cross_val_score(model, X, y, cv=cv)

.. toctree::
   :maxdepth: 4

   about/index
   usage/index
   tutorials/index
   change_log/index
   frequently_asked_questions/index
