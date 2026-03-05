import numpy as np
from Bio import Phylo
from io import StringIO

from treeml._eigenvectors import extract_phylo_eigenvectors


def _make_tree():
    nwk = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
    tree = Phylo.read(StringIO(nwk), "newick")
    names = ["A", "B", "C", "D"]
    return tree, names


def test_eigenvectors_shape():
    tree, names = _make_tree()
    E, info = extract_phylo_eigenvectors(tree, names, variance_threshold=0.9)
    assert E.shape[0] == 4
    assert E.shape[1] >= 1
    assert E.shape[1] <= 4


def test_eigenvectors_variance_threshold():
    tree, names = _make_tree()
    E_low, _ = extract_phylo_eigenvectors(tree, names, variance_threshold=0.5)
    E_high, _ = extract_phylo_eigenvectors(tree, names, variance_threshold=0.99)
    assert E_high.shape[1] >= E_low.shape[1]


def test_eigenvectors_info_contains_metadata():
    tree, names = _make_tree()
    E, info = extract_phylo_eigenvectors(tree, names, variance_threshold=0.9)
    assert "n_components" in info
    assert "variance_explained" in info
    assert info["n_components"] == E.shape[1]


def test_eigenvectors_are_orthogonal():
    tree, names = _make_tree()
    E, _ = extract_phylo_eigenvectors(tree, names, variance_threshold=0.99)
    gram = E.T @ E
    off_diag = gram - np.diag(np.diag(gram))
    np.testing.assert_allclose(off_diag, 0, atol=1e-10)


def test_eigenvectors_for_new_species():
    nwk = "((A:1.0,B:1.0):1.0,((C:1.0,D:1.0):0.5,E:1.5):1.0);"
    tree = Phylo.read(StringIO(nwk), "newick")
    names = ["A", "B", "C", "D", "E"]
    E, info = extract_phylo_eigenvectors(tree, names, variance_threshold=0.9)
    assert E.shape[0] == 5
