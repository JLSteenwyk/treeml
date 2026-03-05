from typing import List, Tuple

import numpy as np
from Bio import Phylo


def load_data(
    trait_file: str,
    tree_file: str,
    response: str,
    tree_format: str = "newick",
) -> Tuple[np.ndarray, np.ndarray, object, List[str]]:
    """Load trait data and phylogenetic tree.

    Args:
        trait_file: Path to tab-separated file. First row is header
            (species, trait1, trait2, ...). Subsequent rows are data.
        tree_file: Path to tree file (Newick or Nexus).
        response: Column name to use as the target variable y.
        tree_format: Format of tree file (default: "newick").

    Returns:
        (X, y, tree, species_names)
    """
    tree = Phylo.read(tree_file, tree_format)
    tree_tips = {t.name for t in tree.get_terminals()}

    with open(trait_file) as f:
        lines = f.readlines()

    header = lines[0].strip().split("\t")
    trait_names = header[1:]

    if response not in trait_names:
        raise ValueError(
            f"Response '{response}' not found in trait file. "
            f"Available: {', '.join(trait_names)}"
        )

    resp_idx = trait_names.index(response)
    feature_indices = [i for i in range(len(trait_names)) if i != resp_idx]

    species_names = []
    y_values = []
    x_values = []

    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        species = parts[0]
        if species not in tree_tips:
            continue
        values = [float(v) for v in parts[1:]]
        species_names.append(species)
        y_values.append(values[resp_idx])
        x_values.append([values[i] for i in feature_indices])

    X = np.array(x_values)
    y = np.array(y_values)

    return X, y, tree, species_names
