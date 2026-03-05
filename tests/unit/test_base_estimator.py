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
