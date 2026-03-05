import numpy as np
from Bio import Phylo
from io import StringIO

from treeml.cv._distance import PhyloDistanceCV


def _make_tree_and_data():
    nwk = "((A:1.0,B:1.0):5.0,(C:1.0,D:1.0):5.0,(E:1.0,F:1.0):5.0);"
    tree = Phylo.read(StringIO(nwk), "newick")
    names = ["A", "B", "C", "D", "E", "F"]
    X = np.arange(12).reshape(6, 2).astype(float)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    return X, y, tree, names


def test_split_returns_correct_number_of_folds():
    X, y, tree, names = _make_tree_and_data()
    cv = PhyloDistanceCV(tree=tree, species_names=names, n_splits=3)
    splits = list(cv.split(X, y))
    assert len(splits) == 3


def test_all_indices_covered():
    X, y, tree, names = _make_tree_and_data()
    cv = PhyloDistanceCV(tree=tree, species_names=names, n_splits=3)
    all_test = set()
    for train, test in cv.split(X, y):
        all_test.update(test.tolist())
    assert all_test == {0, 1, 2, 3, 4, 5}


def test_train_test_no_overlap():
    X, y, tree, names = _make_tree_and_data()
    cv = PhyloDistanceCV(tree=tree, species_names=names, n_splits=3)
    for train, test in cv.split(X, y):
        assert len(set(train) & set(test)) == 0


def test_close_relatives_same_fold():
    X, y, tree, names = _make_tree_and_data()
    cv = PhyloDistanceCV(tree=tree, species_names=names, n_splits=3)
    for train, test in cv.split(X, y):
        test_set = set(test.tolist())
        if 0 in test_set:
            assert 1 in test_set
        if 1 in test_set:
            assert 0 in test_set


def test_get_n_splits():
    X, y, tree, names = _make_tree_and_data()
    cv = PhyloDistanceCV(tree=tree, species_names=names, n_splits=3)
    assert cv.get_n_splits() == 3
