# treeml

Phylogenetic machine learning: scikit-learn estimators that account for evolutionary non-independence among species.

## Installation

```shell
pip install treeml
```

## Quick Start

```python
from treeml import PhyloRandomForestRegressor, PhyloDistanceCV
from sklearn.model_selection import cross_val_score
from Bio import Phylo

tree = Phylo.read("species.nwk", "newick")
# X = feature matrix (n_species x p_features)
# y = target vector (n_species)

model = PhyloRandomForestRegressor(n_estimators=100)
model.fit(X, y, tree=tree, species_names=names)

cv = PhyloDistanceCV(tree=tree, species_names=names, n_splits=5)
scores = cross_val_score(model, X, y, cv=cv)
```
