import numpy as np
from Bio import Phylo
from io import StringIO

from treeml._whitening import phylo_whiten, phylo_unwhiten


def _make_tree_and_vcv():
    """Helper: small 4-tip tree for predictable VCV."""
    nwk = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
    tree = Phylo.read(StringIO(nwk), "newick")
    names = ["A", "B", "C", "D"]
    return tree, names


def test_whiten_returns_correct_shape():
    tree, names = _make_tree_and_vcv()
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_white, L = phylo_whiten(y, tree, names)
    assert y_white.shape == (4,)
    assert L.shape == (4, 4)


def test_whiten_removes_covariance():
    """After whitening, the effective covariance should be identity."""
    tree, names = _make_tree_and_vcv()
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_white, L = phylo_whiten(y, tree, names)
    from phykit.services.tree.vcv_utils import build_vcv_matrix
    C = build_vcv_matrix(tree, names)
    L_inv = np.linalg.inv(L)
    result = L_inv @ C @ L_inv.T
    np.testing.assert_allclose(result, np.eye(4), atol=1e-10)


def test_unwhiten_inverts_whiten():
    tree, names = _make_tree_and_vcv()
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_white, L = phylo_whiten(y, tree, names)
    y_recovered = phylo_unwhiten(y_white, L)
    np.testing.assert_allclose(y_recovered, y, atol=1e-10)


def test_whiten_different_from_input():
    tree, names = _make_tree_and_vcv()
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_white, L = phylo_whiten(y, tree, names)
    assert not np.allclose(y_white, y)
