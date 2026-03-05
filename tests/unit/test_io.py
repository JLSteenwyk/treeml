import os
import numpy as np
from Bio import Phylo

from treeml._io import load_data

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_files")


def test_load_data_returns_correct_types():
    X, y, tree, species_names = load_data(
        trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
        tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
        response="brain_size",
    )
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(species_names, list)


def test_load_data_correct_shapes():
    X, y, tree, species_names = load_data(
        trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
        tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
        response="brain_size",
    )
    n = len(species_names)
    assert X.shape[0] == n
    assert y.shape[0] == n
    assert X.shape[1] == 2  # body_mass, diet_type (brain_size is response)


def test_load_data_response_not_in_features():
    X, y, tree, species_names = load_data(
        trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
        tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
        response="brain_size",
    )
    assert y.shape == (len(species_names),)


def test_load_data_species_match_tree():
    X, y, tree, species_names = load_data(
        trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
        tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
        response="brain_size",
    )
    tree_tips = {t.name for t in tree.get_terminals()}
    assert set(species_names).issubset(tree_tips)
