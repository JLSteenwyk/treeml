import numpy as np
from Bio import Phylo
from io import StringIO

from treeml.estimators._base import PhyloBaseEstimator


def _make_test_data():
    nwk = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
    tree = Phylo.read(StringIO(nwk), "newick")
    names = ["A", "B", "C", "D"]
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    return X, tree, names


def test_build_vcv():
    X, tree, names = _make_test_data()
    est = PhyloBaseEstimator()
    vcv = est._build_vcv(tree, names)
    assert vcv.shape == (4, 4)
    np.testing.assert_allclose(vcv, vcv.T)


def test_augment_features_with_eigenvectors():
    X, tree, names = _make_test_data()
    est = PhyloBaseEstimator(include_eigenvectors=True, eigenvector_variance=0.9)
    X_aug, info = est._augment_features(X, tree, names)
    assert X_aug.shape[0] == 4
    assert X_aug.shape[1] > 2


def test_augment_features_without_eigenvectors():
    X, tree, names = _make_test_data()
    est = PhyloBaseEstimator(include_eigenvectors=False)
    X_aug, info = est._augment_features(X, tree, names)
    assert X_aug.shape == (4, 2)


def test_get_params():
    est = PhyloBaseEstimator(include_eigenvectors=True, eigenvector_variance=0.8)
    params = est.get_params()
    assert params["include_eigenvectors"] is True
    assert params["eigenvector_variance"] == 0.8


def test_fit_stores_tree_and_species_names():
    """After fit, model should store tree_, species_names_, gene_trees_."""
    from treeml import PhyloRandomForestRegressor
    from Bio import Phylo
    from io import StringIO
    import numpy as np

    nwk = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
    tree = Phylo.read(StringIO(nwk), "newick")
    names = ["A", "B", "C", "D"]
    rng = np.random.default_rng(42)
    X = rng.standard_normal((4, 2))
    y = rng.standard_normal(4)

    model = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y, tree=tree, species_names=names)

    assert model.tree_ is tree
    assert model.species_names_ == names
    assert model.gene_trees_ is None


def test_fit_stores_gene_trees():
    """After fit with gene_trees, model should store gene_trees_."""
    from treeml import PhyloRandomForestRegressor
    from Bio import Phylo
    from io import StringIO
    import numpy as np

    sp_nwk = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
    gt_nwk = "((A:0.8,C:0.8):1.2,(B:1.0,D:1.0):1.0);"
    tree = Phylo.read(StringIO(sp_nwk), "newick")
    gt = [Phylo.read(StringIO(gt_nwk), "newick")]
    names = ["A", "B", "C", "D"]
    rng = np.random.default_rng(42)
    X = rng.standard_normal((4, 2))
    y = rng.standard_normal(4)

    model = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y, tree=tree, species_names=names, gene_trees=gt)

    assert model.gene_trees_ is gt
