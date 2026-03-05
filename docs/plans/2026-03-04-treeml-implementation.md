# treeml Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a scikit-learn-compatible phylogenetic ML package with Random Forest classifiers/regressors, phylogenetic cross-validation, and comparative feature importance.

**Architecture:** Wrapper pattern around sklearn estimators. PhyloBaseEstimator handles phylogenetic preprocessing (VCV construction via PhyKIT, Cholesky whitening of y, eigenvector extraction from VCV). Inner sklearn RF does the actual learning. Two CV splitters and a feature importance report complete the package.

**Tech Stack:** Python 3.11+, scikit-learn, numpy, scipy, biopython (via phykit), pandas (optional)

---

## Chunk 1: Package Scaffolding

### Task 1.1: Initialize git repo and create package skeleton

**Files:**
- Create: `treeml/__init__.py`
- Create: `treeml/version.py`
- Create: `setup.py`
- Create: `requirements.txt`
- Create: `tests/requirements.txt`
- Create: `README.md`

**Step 1: Initialize git repo**

Run: `cd /Users/jacoblsteenwyk/Desktop/kit_dev/treeml && git init`
Expected: Initialized empty Git repository

**Step 2: Create version file**

Create `treeml/version.py`:
```python
__version__ = "0.1.0"
```

**Step 3: Create package init**

Create `treeml/__init__.py`:
```python
from treeml.version import __version__

__all__ = ["__version__"]
```

**Step 4: Create setup.py**

Create `setup.py`:
```python
from os import path
from setuptools import setup, find_packages

from treeml.version import __version__

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md")) as f:
    long_description = f.read()

CLASSIFIERS = [
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering",
]

REQUIRES = [
    "phykit>=1.11.0",
    "numpy>=1.24.0",
    "scipy>=1.11.3",
    "scikit-learn>=1.4.2",
]

setup(
    name="treeml",
    description="Phylogenetic machine learning: scikit-learn estimators that account for evolutionary non-independence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jacob L. Steenwyk",
    author_email="jlsteenwyk@gmail.com",
    url="https://github.com/jlsteenwyk/treeml",
    packages=find_packages(),
    python_requires=">=3.11",
    classifiers=CLASSIFIERS,
    version=__version__,
    include_package_data=True,
    install_requires=REQUIRES,
)
```

**Step 5: Create requirements.txt**

Create `requirements.txt`:
```
phykit>=1.11.0
numpy>=1.24.0
scipy>=1.11.3
scikit-learn>=1.4.2
```

**Step 6: Create tests/requirements.txt**

Create `tests/requirements.txt`:
```
mock==5.1.0
pytest-cov==4.1.0
pytest-mock==3.14.0
pytest==8.3.5
```

**Step 7: Create README.md**

Create `README.md`:
```markdown
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
```

**Step 8: Create empty test directories and init files**

Create `tests/__init__.py` (empty), `tests/unit/__init__.py` (empty), `tests/integration/__init__.py` (empty).

**Step 9: Commit**

```bash
git add -A
git commit -m "feat: initialize treeml package skeleton"
```

---

### Task 1.2: Create Makefile

**Files:**
- Create: `Makefile`

**Step 1: Create Makefile**

Create `Makefile`:
```makefile
install:
	python setup.py install

develop:
	python setup.py develop

test: test.unit test.integration

test.unit:
	python3 -m pytest -m "not integration"

test.integration:
	python3 -m pytest -m "integration"

test.fast:
	python3 -m pytest -m "not (integration or slow)"

test.coverage: coverage.unit coverage.integration

coverage.unit:
	python -m pytest --cov=./ -m "not integration" --cov-report=xml:unit.coverage.xml

coverage.integration:
	python -m pytest --cov=./ -m "integration" --cov-report=xml:integration.coverage.xml
```

**Step 2: Verify make targets parse**

Run: `make -n test.unit`
Expected: Shows `python3 -m pytest -m "not integration"` without errors

**Step 3: Commit**

```bash
git add Makefile
git commit -m "feat: add Makefile with test and coverage targets"
```

---

### Task 1.3: Create CI workflow and codecov config

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `codecov.yml`

**Step 1: Create CI workflow**

Create `.github/workflows/ci.yml`:
```yaml
name: CI
on: push
jobs:
  test:
    runs-on: macos-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
    steps:
    - uses: actions/checkout@master
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@master
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install setuptools
        pip install -r requirements.txt
        make install
        pip install -r tests/requirements.txt
    - name: Run tests
      run: |
        make test
    - name: Generate coverage report
      if: ${{ matrix.python-version == '3.11' }}
      run: |
        make test.coverage
    - name: Upload unit test coverage to Codecov
      if: ${{ matrix.python-version == '3.11' }}
      uses: codecov/codecov-action@v1.0.7
      with:
        token: ${{ secrets.CODECOV_TOKEN }}
        file: ./unit.coverage.xml
        flags: unit
        env_vars: PYTHON
        name: codecov-unit
        fail_ci_if_error: false
    - name: Upload integration test coverage to Codecov
      if: ${{ matrix.python-version == '3.11' }}
      uses: codecov/codecov-action@v1.0.7
      with:
        token: ${{ secrets.CODECOV_TOKEN }}
        file: ./integration.coverage.xml
        flags: integration
        env_vars: PYTHON
        name: codecov-integration
        fail_ci_if_error: false
```

**Step 2: Create codecov config**

Create `codecov.yml`:
```yaml
comment: false
coverage:
  precision: 2
  round: down
  range: "70...95"
ignore:
  - "tests"
  - "setup.py"
```

**Step 3: Commit**

```bash
git add .github/workflows/ci.yml codecov.yml
git commit -m "feat: add GitHub Actions CI and Codecov config"
```

---

### Task 1.4: Create pytest config and verify tests run

**Files:**
- Create: `pytest.ini`
- Create: `tests/unit/test_version.py`

**Step 1: Create pytest.ini**

Create `pytest.ini`:
```ini
[pytest]
markers =
    integration: marks tests as integration tests
    slow: marks tests as slow
```

**Step 2: Write a smoke test**

Create `tests/unit/test_version.py`:
```python
from treeml import __version__


def test_version_is_string():
    assert isinstance(__version__, str)


def test_version_format():
    parts = __version__.split(".")
    assert len(parts) == 3
    for part in parts:
        assert part.isdigit()
```

**Step 3: Run tests to verify setup works**

Run: `python3 -m pytest tests/unit/test_version.py -v`
Expected: 2 passed

**Step 4: Commit**

```bash
git add pytest.ini tests/unit/test_version.py
git commit -m "feat: add pytest config and version smoke test"
```

---

### Task 1.5: Create test sample files

**Files:**
- Create: `tests/sample_files/tree_simple.nwk`
- Create: `tests/sample_files/traits_simple.tsv`
- Create: `tests/sample_files/traits_multi.tsv`

These test files are used throughout all subsequent tests.

**Step 1: Create a simple test tree (7 species)**

Create `tests/sample_files/tree_simple.nwk`:
```
((raccoon:19.19959,bear:6.80041):0.84600,((sea_lion:11.99700,seal:12.00300):7.52973,((monkey:100.85930,cat:47.14069):20.59201,weasel:18.87953):2.09460):3.87382,dog:25.46154);
```

**Step 2: Create a simple trait file (single response, for regression)**

Create `tests/sample_files/traits_simple.tsv`:
```
species	body_mass	brain_size	diet_type
raccoon	8.5	39.2	1
bear	250.0	289.0	1
sea_lion	200.0	247.0	0
seal	85.0	172.0	0
monkey	7.0	68.0	1
cat	4.5	28.0	0
weasel	1.0	7.6	0
dog	30.0	72.0	1
```

Note: `body_mass` and `brain_size` are continuous (regression targets), `diet_type` is binary (classification target). `body_mass` is a feature, `brain_size` is a plausible response.

**Step 3: Create a tree with an extra species for prediction tests**

Create `tests/sample_files/tree_with_fox.nwk`:
```
((raccoon:19.19959,bear:6.80041):0.84600,((sea_lion:11.99700,seal:12.00300):7.52973,((monkey:100.85930,cat:47.14069):20.59201,weasel:18.87953):2.09460):3.87382,(dog:15.0,fox:15.0):10.46154);
```

**Step 4: Commit**

```bash
git add tests/sample_files/
git commit -m "feat: add test sample files (tree, traits)"
```

---

## Chunk 2: VCV + Whitening + Eigenvectors

### Task 2.1: Phylogenetic whitening module

**Files:**
- Create: `treeml/_whitening.py`
- Create: `tests/unit/test_whitening.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_whitening.py`:
```python
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
    # L^{-1} C L^{-T} should be identity
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
    # Whitened values should differ from original
    assert not np.allclose(y_white, y)
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_whitening.py -v`
Expected: FAIL (ImportError — module doesn't exist)

**Step 3: Write minimal implementation**

Create `treeml/_whitening.py`:
```python
from typing import List, Tuple

import numpy as np
from phykit.services.tree.vcv_utils import build_vcv_matrix


def phylo_whiten(
    y: np.ndarray, tree, ordered_names: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Whiten target vector y using phylogenetic VCV matrix.

    Computes Cholesky decomposition of VCV: C = L L^T
    Returns y_white = L^{-1} y and the Cholesky factor L.
    """
    C = build_vcv_matrix(tree, ordered_names)
    L = np.linalg.cholesky(C)
    y_white = np.linalg.solve(L, y)
    return y_white, L


def phylo_unwhiten(y_white: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Reverse the whitening transformation: y = L @ y_white."""
    return L @ y_white
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_whitening.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add treeml/_whitening.py tests/unit/test_whitening.py
git commit -m "feat: add phylogenetic whitening module"
```

---

### Task 2.2: Phylogenetic eigenvector extraction module

**Files:**
- Create: `treeml/_eigenvectors.py`
- Create: `tests/unit/test_eigenvectors.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_eigenvectors.py`:
```python
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
    assert E.shape[0] == 4  # n_species
    assert E.shape[1] >= 1  # at least 1 eigenvector
    assert E.shape[1] <= 4  # at most n eigenvectors


def test_eigenvectors_variance_threshold():
    tree, names = _make_tree()
    E_low, _ = extract_phylo_eigenvectors(tree, names, variance_threshold=0.5)
    E_high, _ = extract_phylo_eigenvectors(tree, names, variance_threshold=0.99)
    # Higher threshold should retain >= as many eigenvectors
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
    # E^T E should be approximately diagonal
    gram = E.T @ E
    off_diag = gram - np.diag(np.diag(gram))
    np.testing.assert_allclose(off_diag, 0, atol=1e-10)


def test_eigenvectors_for_new_species():
    """Eigenvectors for species in an updated tree should work."""
    nwk = "((A:1.0,B:1.0):1.0,((C:1.0,D:1.0):0.5,E:1.5):1.0);"
    tree = Phylo.read(StringIO(nwk), "newick")
    names = ["A", "B", "C", "D", "E"]
    E, info = extract_phylo_eigenvectors(tree, names, variance_threshold=0.9)
    assert E.shape[0] == 5
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_eigenvectors.py -v`
Expected: FAIL (ImportError)

**Step 3: Write minimal implementation**

Create `treeml/_eigenvectors.py`:
```python
from typing import Dict, List, Tuple

import numpy as np
from phykit.services.tree.vcv_utils import build_vcv_matrix


def extract_phylo_eigenvectors(
    tree,
    ordered_names: List[str],
    variance_threshold: float = 0.90,
) -> Tuple[np.ndarray, Dict]:
    """Extract phylogenetic eigenvectors from double-centered VCV matrix.

    Performs PCoA-like decomposition: double-center the VCV, eigendecompose,
    and retain eigenvectors explaining at least `variance_threshold` of variance.

    Returns (E, info) where E is (n_species x k) and info contains metadata.
    """
    C = build_vcv_matrix(tree, ordered_names)
    n = len(ordered_names)

    # Double-center the VCV matrix (Gower centering)
    row_means = C.mean(axis=1, keepdims=True)
    col_means = C.mean(axis=0, keepdims=True)
    grand_mean = C.mean()
    C_centered = C - row_means - col_means + grand_mean

    # Eigendecompose (symmetric, so use eigh)
    eigenvalues, eigenvectors = np.linalg.eigh(C_centered)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Keep only positive eigenvalues
    pos_mask = eigenvalues > 0
    eigenvalues = eigenvalues[pos_mask]
    eigenvectors = eigenvectors[:, pos_mask]

    if len(eigenvalues) == 0:
        return np.zeros((n, 0)), {
            "n_components": 0,
            "variance_explained": np.array([]),
        }

    # Determine number of components by variance threshold
    total_var = eigenvalues.sum()
    cumulative_var = np.cumsum(eigenvalues) / total_var
    n_components = int(np.searchsorted(cumulative_var, variance_threshold) + 1)
    n_components = min(n_components, len(eigenvalues))

    E = eigenvectors[:, :n_components]
    var_explained = eigenvalues[:n_components] / total_var

    return E, {
        "n_components": n_components,
        "variance_explained": var_explained,
    }
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_eigenvectors.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add treeml/_eigenvectors.py tests/unit/test_eigenvectors.py
git commit -m "feat: add phylogenetic eigenvector extraction module"
```

---

## Chunk 3: PhyloBaseEstimator + Regressor

### Task 3.1: PhyloBaseEstimator base class

**Files:**
- Create: `treeml/estimators/__init__.py`
- Create: `treeml/estimators/_base.py`
- Create: `tests/unit/test_base_estimator.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_base_estimator.py`:
```python
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
    # VCV should be symmetric
    np.testing.assert_allclose(vcv, vcv.T)


def test_augment_features_with_eigenvectors():
    X, tree, names = _make_test_data()
    est = PhyloBaseEstimator(include_eigenvectors=True, eigenvector_variance=0.9)
    X_aug, info = est._augment_features(X, tree, names)
    assert X_aug.shape[0] == 4
    assert X_aug.shape[1] > 2  # original 2 + eigenvectors


def test_augment_features_without_eigenvectors():
    X, tree, names = _make_test_data()
    est = PhyloBaseEstimator(include_eigenvectors=False)
    X_aug, info = est._augment_features(X, tree, names)
    assert X_aug.shape == (4, 2)  # unchanged


def test_get_params():
    est = PhyloBaseEstimator(include_eigenvectors=True, eigenvector_variance=0.8)
    params = est.get_params()
    assert params["include_eigenvectors"] is True
    assert params["eigenvector_variance"] == 0.8
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_base_estimator.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

Create `treeml/estimators/__init__.py`:
```python
from treeml.estimators._classifier import PhyloRandomForestClassifier
from treeml.estimators._regressor import PhyloRandomForestRegressor

__all__ = [
    "PhyloRandomForestClassifier",
    "PhyloRandomForestRegressor",
]
```

Note: The imports above will fail until Task 3.2 and Task 4.1 are complete. That's fine — the init is written now for the final structure. Tests in this task import `_base` directly.

Create `treeml/estimators/_base.py`:
```python
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator

from treeml._eigenvectors import extract_phylo_eigenvectors
from treeml._whitening import phylo_whiten, phylo_unwhiten


class PhyloBaseEstimator(BaseEstimator):
    """Base class for phylogenetic ML estimators.

    Handles VCV construction, eigenvector extraction, and feature augmentation.
    Subclasses implement the actual model (classifier or regressor).
    """

    def __init__(
        self,
        include_eigenvectors: bool = True,
        eigenvector_variance: float = 0.90,
        **kwargs,
    ):
        self.include_eigenvectors = include_eigenvectors
        self.eigenvector_variance = eigenvector_variance

    def _build_vcv(self, tree, ordered_names: List[str]) -> np.ndarray:
        from phykit.services.tree.vcv_utils import build_vcv_matrix
        return build_vcv_matrix(tree, ordered_names)

    def _augment_features(
        self,
        X: np.ndarray,
        tree,
        ordered_names: List[str],
    ) -> Tuple[np.ndarray, Dict]:
        if not self.include_eigenvectors:
            return X, {"n_components": 0}

        E, info = extract_phylo_eigenvectors(
            tree, ordered_names, variance_threshold=self.eigenvector_variance
        )
        X_aug = np.column_stack([X, E])
        return X_aug, info

    def _augment_features_predict(
        self,
        X_new: np.ndarray,
        tree,
        species_names: Optional[List[str]],
        n_eigenvector_cols: int,
    ) -> Tuple[np.ndarray, bool]:
        """Augment features for prediction.

        Returns (X_augmented, phylo_corrected).
        """
        if n_eigenvector_cols == 0:
            return X_new, tree is not None

        if tree is not None and species_names is not None:
            E, _ = extract_phylo_eigenvectors(
                tree, species_names,
                variance_threshold=self.eigenvector_variance,
            )
            # Pad or truncate to match training eigenvector count
            if E.shape[1] < n_eigenvector_cols:
                pad = np.zeros((E.shape[0], n_eigenvector_cols - E.shape[1]))
                E = np.column_stack([E, pad])
            elif E.shape[1] > n_eigenvector_cols:
                E = E[:, :n_eigenvector_cols]
            X_aug = np.column_stack([X_new, E])
            return X_aug, True
        else:
            # No tree: fill eigenvector columns with 0
            warnings.warn(
                "No tree provided for prediction. "
                "Predictions made without phylogenetic correction.",
                UserWarning,
                stacklevel=2,
            )
            zeros = np.zeros((X_new.shape[0], n_eigenvector_cols))
            X_aug = np.column_stack([X_new, zeros])
            return X_aug, False
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_base_estimator.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add treeml/estimators/_base.py treeml/estimators/__init__.py tests/unit/test_base_estimator.py
git commit -m "feat: add PhyloBaseEstimator base class"
```

---

### Task 3.2: PhyloRandomForestRegressor

**Files:**
- Create: `treeml/estimators/_regressor.py`
- Create: `tests/unit/test_regressor.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_regressor.py`:
```python
import numpy as np
import pytest
from Bio import Phylo
from io import StringIO
from sklearn.utils.estimator_checks import parametrize_with_checks

from treeml.estimators._regressor import PhyloRandomForestRegressor


def _make_test_data():
    nwk = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0,(E:1.5,F:0.5):1.0);"
    tree = Phylo.read(StringIO(nwk), "newick")
    names = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(42)
    X = rng.standard_normal((6, 3))
    y = rng.standard_normal(6)
    return X, y, tree, names


def test_fit_returns_self():
    X, y, tree, names = _make_test_data()
    model = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    result = model.fit(X, y, tree=tree, species_names=names)
    assert result is model


def test_predict_with_tree():
    X, y, tree, names = _make_test_data()
    model = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y, tree=tree, species_names=names)
    preds = model.predict(X, tree=tree, species_names=names)
    assert preds.shape == (6,)


def test_predict_without_tree_warns():
    X, y, tree, names = _make_test_data()
    model = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y, tree=tree, species_names=names)
    with pytest.warns(UserWarning, match="without phylogenetic correction"):
        preds = model.predict(X)
    assert preds.shape == (6,)


def test_predict_without_eigenvectors():
    X, y, tree, names = _make_test_data()
    model = PhyloRandomForestRegressor(
        n_estimators=10, random_state=42, include_eigenvectors=False
    )
    model.fit(X, y, tree=tree, species_names=names)
    preds = model.predict(X, tree=tree, species_names=names)
    assert preds.shape == (6,)


def test_fit_stores_training_metadata():
    X, y, tree, names = _make_test_data()
    model = PhyloRandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y, tree=tree, species_names=names)
    assert hasattr(model, "n_eigenvector_cols_")
    assert hasattr(model, "L_")
    assert hasattr(model, "inner_model_")
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_regressor.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

Create `treeml/estimators/_regressor.py`:
```python
from typing import List, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from treeml.estimators._base import PhyloBaseEstimator
from treeml._whitening import phylo_whiten, phylo_unwhiten


class PhyloRandomForestRegressor(PhyloBaseEstimator):
    """Random Forest regressor with phylogenetic correction.

    Wraps sklearn RandomForestRegressor. Whitens the target variable y
    using the phylogenetic VCV matrix and optionally augments features
    with phylogenetic eigenvectors.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        include_eigenvectors: bool = True,
        eigenvector_variance: float = 0.90,
        random_state=None,
        **rf_kwargs,
    ):
        super().__init__(
            include_eigenvectors=include_eigenvectors,
            eigenvector_variance=eigenvector_variance,
        )
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.rf_kwargs = rf_kwargs

    def fit(self, X, y, tree=None, species_names=None):
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)

        if tree is None or species_names is None:
            raise ValueError("tree and species_names are required for fit().")

        # Whiten y
        y_white, L = phylo_whiten(y, tree, species_names)
        self.L_ = L

        # Augment features
        X_aug, eigvec_info = self._augment_features(X, tree, species_names)
        self.n_eigenvector_cols_ = eigvec_info["n_components"]
        self.n_features_original_ = X.shape[1]

        # Fit inner model
        self.inner_model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            **self.rf_kwargs,
        )
        self.inner_model_.fit(X_aug, y_white)

        return self

    def predict(self, X, tree=None, species_names=None):
        X = np.asarray(X)

        X_aug, phylo_corrected = self._augment_features_predict(
            X, tree, species_names, self.n_eigenvector_cols_
        )

        y_pred_white = self.inner_model_.predict(X_aug)

        if phylo_corrected and tree is not None and species_names is not None:
            # Recompute L from the prediction tree for un-whitening
            from treeml._whitening import phylo_whiten
            from phykit.services.tree.vcv_utils import build_vcv_matrix
            C = build_vcv_matrix(tree, species_names)
            L_pred = np.linalg.cholesky(C)
            return phylo_unwhiten(y_pred_white, L_pred)
        else:
            return y_pred_white
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_regressor.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add treeml/estimators/_regressor.py tests/unit/test_regressor.py
git commit -m "feat: add PhyloRandomForestRegressor"
```

---

## Chunk 4: PhyloRandomForestClassifier

### Task 4.1: PhyloRandomForestClassifier

**Files:**
- Create: `treeml/estimators/_classifier.py`
- Create: `tests/unit/test_classifier.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_classifier.py`:
```python
import numpy as np
import pytest
from Bio import Phylo
from io import StringIO

from treeml.estimators._classifier import PhyloRandomForestClassifier


def _make_test_data():
    nwk = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0,(E:1.5,F:0.5):1.0);"
    tree = Phylo.read(StringIO(nwk), "newick")
    names = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(42)
    X = rng.standard_normal((6, 3))
    y = np.array([0, 0, 1, 1, 0, 1])
    return X, y, tree, names


def test_fit_returns_self():
    X, y, tree, names = _make_test_data()
    clf = PhyloRandomForestClassifier(n_estimators=10, random_state=42)
    result = clf.fit(X, y, tree=tree, species_names=names)
    assert result is clf


def test_predict_returns_labels():
    X, y, tree, names = _make_test_data()
    clf = PhyloRandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y, tree=tree, species_names=names)
    preds = clf.predict(X, tree=tree, species_names=names)
    assert preds.shape == (6,)
    assert set(preds).issubset({0, 1})


def test_predict_proba_returns_probabilities():
    X, y, tree, names = _make_test_data()
    clf = PhyloRandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y, tree=tree, species_names=names)
    proba = clf.predict_proba(X, tree=tree, species_names=names)
    assert proba.shape == (6, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_predict_without_tree_warns():
    X, y, tree, names = _make_test_data()
    clf = PhyloRandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y, tree=tree, species_names=names)
    with pytest.warns(UserWarning, match="without phylogenetic correction"):
        preds = clf.predict(X)
    assert preds.shape == (6,)


def test_no_y_whitening_for_classifier():
    """Classifiers should NOT whiten y (discrete targets)."""
    X, y, tree, names = _make_test_data()
    clf = PhyloRandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y, tree=tree, species_names=names)
    # Should not have L_ attribute (no whitening)
    assert not hasattr(clf, "L_")


def test_without_eigenvectors():
    X, y, tree, names = _make_test_data()
    clf = PhyloRandomForestClassifier(
        n_estimators=10, random_state=42, include_eigenvectors=False
    )
    clf.fit(X, y, tree=tree, species_names=names)
    preds = clf.predict(X, tree=tree, species_names=names)
    assert preds.shape == (6,)
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_classifier.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

Create `treeml/estimators/_classifier.py`:
```python
from typing import List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from treeml.estimators._base import PhyloBaseEstimator


class PhyloRandomForestClassifier(PhyloBaseEstimator):
    """Random Forest classifier with phylogenetic correction.

    Wraps sklearn RandomForestClassifier. Augments features with phylogenetic
    eigenvectors. Does NOT whiten y (target is discrete).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        include_eigenvectors: bool = True,
        eigenvector_variance: float = 0.90,
        random_state=None,
        **rf_kwargs,
    ):
        super().__init__(
            include_eigenvectors=include_eigenvectors,
            eigenvector_variance=eigenvector_variance,
        )
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.rf_kwargs = rf_kwargs

    def fit(self, X, y, tree=None, species_names=None):
        X = np.asarray(X)
        y = np.asarray(y)

        if tree is None or species_names is None:
            raise ValueError("tree and species_names are required for fit().")

        # Augment features (no y-whitening for classifiers)
        X_aug, eigvec_info = self._augment_features(X, tree, species_names)
        self.n_eigenvector_cols_ = eigvec_info["n_components"]
        self.n_features_original_ = X.shape[1]

        # Fit inner model
        self.inner_model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            **self.rf_kwargs,
        )
        self.inner_model_.fit(X_aug, y)

        return self

    def predict(self, X, tree=None, species_names=None):
        X = np.asarray(X)
        X_aug, _ = self._augment_features_predict(
            X, tree, species_names, self.n_eigenvector_cols_
        )
        return self.inner_model_.predict(X_aug)

    def predict_proba(self, X, tree=None, species_names=None):
        X = np.asarray(X)
        X_aug, _ = self._augment_features_predict(
            X, tree, species_names, self.n_eigenvector_cols_
        )
        return self.inner_model_.predict_proba(X_aug)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_classifier.py -v`
Expected: 6 passed

**Step 5: Update treeml/__init__.py to export all public classes**

Edit `treeml/__init__.py`:
```python
from treeml.version import __version__
from treeml.estimators._classifier import PhyloRandomForestClassifier
from treeml.estimators._regressor import PhyloRandomForestRegressor

__all__ = [
    "__version__",
    "PhyloRandomForestClassifier",
    "PhyloRandomForestRegressor",
]
```

**Step 6: Commit**

```bash
git add treeml/estimators/_classifier.py tests/unit/test_classifier.py treeml/__init__.py
git commit -m "feat: add PhyloRandomForestClassifier"
```

---

## Chunk 5: Cross-Validation Splitters

### Task 5.1: PhyloDistanceCV

**Files:**
- Create: `treeml/cv/__init__.py`
- Create: `treeml/cv/_distance.py`
- Create: `tests/unit/test_cv_distance.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_cv_distance.py`:
```python
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
    """Species A and B are close (distance=2); they should be in the same fold."""
    X, y, tree, names = _make_tree_and_data()
    cv = PhyloDistanceCV(tree=tree, species_names=names, n_splits=3)
    for train, test in cv.split(X, y):
        test_set = set(test.tolist())
        # A=0, B=1 should either both be in test or both in train
        if 0 in test_set:
            assert 1 in test_set
        if 1 in test_set:
            assert 0 in test_set


def test_get_n_splits():
    X, y, tree, names = _make_tree_and_data()
    cv = PhyloDistanceCV(tree=tree, species_names=names, n_splits=3)
    assert cv.get_n_splits() == 3
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_cv_distance.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

Create `treeml/cv/__init__.py`:
```python
from treeml.cv._distance import PhyloDistanceCV
from treeml.cv._clade import PhyloCladeCV

__all__ = ["PhyloDistanceCV", "PhyloCladeCV"]
```

Create `treeml/cv/_distance.py`:
```python
from typing import List, Optional

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.model_selection import BaseCrossValidator

from phykit.services.tree.vcv_utils import build_vcv_matrix


class PhyloDistanceCV(BaseCrossValidator):
    """Cross-validation splitter based on phylogenetic distance.

    Clusters species by patristic distance so closely related species
    are always in the same fold. This prevents phylogenetic leakage
    between train and test sets.
    """

    def __init__(
        self,
        tree,
        species_names: List[str],
        n_splits: int = 5,
        min_dist: Optional[float] = None,
    ):
        self.tree = tree
        self.species_names = species_names
        self.n_splits = n_splits
        self.min_dist = min_dist
        self._groups = self._compute_groups()

    def _compute_groups(self) -> np.ndarray:
        vcv = build_vcv_matrix(self.tree, self.species_names)
        n = len(self.species_names)

        # Convert VCV to distance matrix
        diag = np.diag(vcv)
        dist_matrix = diag[:, None] + diag[None, :] - 2 * vcv

        # Extract condensed distance for scipy
        condensed = []
        for i in range(n):
            for j in range(i + 1, n):
                condensed.append(dist_matrix[i, j])
        condensed = np.array(condensed)

        # Hierarchical clustering
        Z = linkage(condensed, method="average")

        if self.min_dist is not None:
            groups = fcluster(Z, t=self.min_dist, criterion="distance")
        else:
            # Auto-tune threshold to get approximately n_splits groups
            # Binary search on threshold
            lo, hi = 0.0, float(condensed.max())
            best_groups = fcluster(Z, t=hi, criterion="distance")
            for _ in range(50):
                mid = (lo + hi) / 2.0
                groups = fcluster(Z, t=mid, criterion="distance")
                n_groups = len(set(groups))
                if n_groups > self.n_splits:
                    lo = mid
                elif n_groups < self.n_splits:
                    hi = mid
                else:
                    break
            best_groups = fcluster(Z, t=mid, criterion="distance")

            # If we can't get exact n_splits, use the closest we found
            groups = best_groups

        return groups

    def split(self, X=None, y=None, groups=None):
        unique_groups = sorted(set(self._groups))
        n_actual_groups = len(unique_groups)

        # Map group labels to fold indices (round-robin if more groups than splits)
        fold_for_group = {}
        for i, g in enumerate(unique_groups):
            fold_for_group[g] = i % self.n_splits

        indices = np.arange(len(self._groups))

        for fold in range(self.n_splits):
            test_mask = np.array([fold_for_group[g] == fold for g in self._groups])
            if not test_mask.any():
                continue
            train_idx = indices[~test_mask]
            test_idx = indices[test_mask]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_cv_distance.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add treeml/cv/__init__.py treeml/cv/_distance.py tests/unit/test_cv_distance.py
git commit -m "feat: add PhyloDistanceCV cross-validation splitter"
```

---

### Task 5.2: PhyloCladeCV

**Files:**
- Create: `treeml/cv/_clade.py`
- Create: `tests/unit/test_cv_clade.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_cv_clade.py`:
```python
import numpy as np
from Bio import Phylo
from io import StringIO

from treeml.cv._clade import PhyloCladeCV


def _make_tree_and_data():
    """Three clear clades: (A,B), (C,D), (E,F)."""
    nwk = "((A:1.0,B:1.0):5.0,(C:1.0,D:1.0):5.0,(E:1.0,F:1.0):5.0);"
    tree = Phylo.read(StringIO(nwk), "newick")
    names = ["A", "B", "C", "D", "E", "F"]
    X = np.arange(12).reshape(6, 2).astype(float)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    return X, y, tree, names


def test_split_returns_correct_number_of_folds():
    X, y, tree, names = _make_tree_and_data()
    cv = PhyloCladeCV(tree=tree, species_names=names, n_splits=3)
    splits = list(cv.split(X, y))
    assert len(splits) == 3


def test_all_indices_covered():
    X, y, tree, names = _make_tree_and_data()
    cv = PhyloCladeCV(tree=tree, species_names=names, n_splits=3)
    all_test = set()
    for train, test in cv.split(X, y):
        all_test.update(test.tolist())
    assert all_test == {0, 1, 2, 3, 4, 5}


def test_train_test_no_overlap():
    X, y, tree, names = _make_tree_and_data()
    cv = PhyloCladeCV(tree=tree, species_names=names, n_splits=3)
    for train, test in cv.split(X, y):
        assert len(set(train) & set(test)) == 0


def test_monophyletic_groups():
    """Each fold should hold out a monophyletic clade (A,B) or (C,D) or (E,F)."""
    X, y, tree, names = _make_tree_and_data()
    cv = PhyloCladeCV(tree=tree, species_names=names, n_splits=3)
    for train, test in cv.split(X, y):
        test_set = set(test.tolist())
        # Each test set should be one of the clades
        assert test_set in [{0, 1}, {2, 3}, {4, 5}]


def test_get_n_splits():
    X, y, tree, names = _make_tree_and_data()
    cv = PhyloCladeCV(tree=tree, species_names=names, n_splits=3)
    assert cv.get_n_splits() == 3
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_cv_clade.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

Create `treeml/cv/_clade.py`:
```python
from typing import List

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class PhyloCladeCV(BaseCrossValidator):
    """Cross-validation splitter that holds out monophyletic clades.

    Traverses internal nodes to find subtrees that partition tips
    into approximately equal-sized groups.
    """

    def __init__(
        self,
        tree,
        species_names: List[str],
        n_splits: int = 5,
        min_clade_size: int = 2,
    ):
        self.tree = tree
        self.species_names = species_names
        self.n_splits = n_splits
        self.min_clade_size = min_clade_size
        self._folds = self._compute_folds()

    def _compute_folds(self) -> List[List[int]]:
        name_to_idx = {name: i for i, name in enumerate(self.species_names)}
        n = len(self.species_names)

        # Get all internal clades and their tip sets
        clades = []
        for clade in self.tree.find_clades(order="level"):
            tips = [t.name for t in clade.get_terminals()]
            tip_indices = [name_to_idx[t] for t in tips if t in name_to_idx]
            if self.min_clade_size <= len(tip_indices) < n:
                clades.append(set(tip_indices))

        # Greedily select n_splits non-overlapping clades that partition the tips
        # Strategy: find the split of internal nodes that best partitions into n_splits
        target_size = n / self.n_splits
        clades.sort(key=lambda c: abs(len(c) - target_size))

        selected = []
        assigned = set()
        for clade in clades:
            if len(selected) >= self.n_splits:
                break
            if not (clade & assigned):  # no overlap with already assigned
                selected.append(sorted(clade))
                assigned.update(clade)

        # Assign remaining unassigned species to the smallest fold
        remaining = sorted(set(range(n)) - assigned)
        if remaining:
            if selected:
                smallest_idx = min(range(len(selected)), key=lambda i: len(selected[i]))
                selected[smallest_idx].extend(remaining)
            else:
                selected.append(remaining)

        # If we have fewer folds than requested, split the largest fold
        while len(selected) < self.n_splits:
            largest_idx = max(range(len(selected)), key=lambda i: len(selected[i]))
            fold = selected[largest_idx]
            if len(fold) < 2:
                break
            mid = len(fold) // 2
            selected[largest_idx] = fold[:mid]
            selected.append(fold[mid:])

        return selected

    def split(self, X=None, y=None, groups=None):
        all_indices = set(range(len(self.species_names)))
        for fold in self._folds:
            test_idx = np.array(fold)
            train_idx = np.array(sorted(all_indices - set(fold)))
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self._folds)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_cv_clade.py -v`
Expected: 5 passed

**Step 5: Update treeml/__init__.py**

Edit `treeml/__init__.py`:
```python
from treeml.version import __version__
from treeml.estimators._classifier import PhyloRandomForestClassifier
from treeml.estimators._regressor import PhyloRandomForestRegressor
from treeml.cv._distance import PhyloDistanceCV
from treeml.cv._clade import PhyloCladeCV

__all__ = [
    "__version__",
    "PhyloRandomForestClassifier",
    "PhyloRandomForestRegressor",
    "PhyloDistanceCV",
    "PhyloCladeCV",
]
```

**Step 6: Commit**

```bash
git add treeml/cv/_clade.py tests/unit/test_cv_clade.py treeml/__init__.py
git commit -m "feat: add PhyloCladeCV cross-validation splitter"
```

---

## Chunk 6: Feature Importance Report

### Task 6.1: phylo_feature_importance()

**Files:**
- Create: `treeml/importance/__init__.py`
- Create: `treeml/importance/_report.py`
- Create: `tests/unit/test_importance.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_importance.py`:
```python
import numpy as np
from Bio import Phylo
from io import StringIO

from treeml.importance._report import phylo_feature_importance


def _make_test_data():
    nwk = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0,(E:1.5,F:0.5):1.0);"
    tree = Phylo.read(StringIO(nwk), "newick")
    names = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(42)
    X = rng.standard_normal((6, 3))
    y = rng.standard_normal(6)
    return X, y, tree, names


def test_returns_dataframe():
    import pandas as pd
    X, y, tree, names = _make_test_data()
    report = phylo_feature_importance(
        X, y, tree=tree, species_names=names,
        n_repeats=5, random_state=42,
    )
    assert isinstance(report, pd.DataFrame)


def test_report_has_correct_columns():
    X, y, tree, names = _make_test_data()
    report = phylo_feature_importance(
        X, y, tree=tree, species_names=names,
        n_repeats=5, random_state=42,
    )
    expected_cols = {"feature", "raw_importance", "phylo_corrected_importance", "delta"}
    assert expected_cols == set(report.columns)


def test_report_has_correct_rows():
    X, y, tree, names = _make_test_data()
    feature_names = ["gene_A", "gene_B", "gene_C"]
    report = phylo_feature_importance(
        X, y, tree=tree, species_names=names,
        feature_names=feature_names, n_repeats=5, random_state=42,
    )
    assert len(report) == 3
    assert list(report["feature"]) == feature_names


def test_delta_is_difference():
    X, y, tree, names = _make_test_data()
    report = phylo_feature_importance(
        X, y, tree=tree, species_names=names,
        n_repeats=5, random_state=42,
    )
    np.testing.assert_allclose(
        report["delta"].values,
        report["phylo_corrected_importance"].values - report["raw_importance"].values,
    )


def test_default_feature_names():
    X, y, tree, names = _make_test_data()
    report = phylo_feature_importance(
        X, y, tree=tree, species_names=names,
        n_repeats=5, random_state=42,
    )
    assert list(report["feature"]) == ["feature_0", "feature_1", "feature_2"]
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_importance.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

Create `treeml/importance/__init__.py`:
```python
from treeml.importance._report import phylo_feature_importance

__all__ = ["phylo_feature_importance"]
```

Create `treeml/importance/_report.py`:
```python
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance

from treeml.estimators._regressor import PhyloRandomForestRegressor
from treeml.estimators._classifier import PhyloRandomForestClassifier


def phylo_feature_importance(
    X,
    y,
    tree,
    species_names: List[str],
    feature_names: Optional[List[str]] = None,
    n_repeats: int = 10,
    scoring: Optional[str] = None,
    n_estimators: int = 100,
    random_state=None,
) -> pd.DataFrame:
    """Compare feature importance with and without phylogenetic correction.

    Returns a DataFrame with columns:
        feature, raw_importance, phylo_corrected_importance, delta
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # Determine if classification or regression
    unique_y = np.unique(y)
    is_classification = len(unique_y) <= 20 and np.all(unique_y == unique_y.astype(int))

    if scoring is None:
        scoring = "accuracy" if is_classification else "r2"

    # 1. Uncorrected model
    if is_classification:
        raw_model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
    else:
        raw_model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state
        )
    raw_model.fit(X, y)
    raw_result = permutation_importance(
        raw_model, X, y, n_repeats=n_repeats,
        scoring=scoring, random_state=random_state,
    )
    raw_importance = raw_result.importances_mean

    # 2. Phylo-corrected model
    if is_classification:
        phylo_model = PhyloRandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
    else:
        phylo_model = PhyloRandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state
        )
    phylo_model.fit(X, y, tree=tree, species_names=species_names)

    # Build augmented X for permutation importance
    X_aug_for_eval, _ = phylo_model._augment_features(X, tree, species_names)

    # For regressor, we need to evaluate on whitened y
    if not is_classification:
        from treeml._whitening import phylo_whiten
        y_eval, _ = phylo_whiten(y, tree, species_names)
    else:
        y_eval = y

    phylo_result = permutation_importance(
        phylo_model.inner_model_, X_aug_for_eval, y_eval,
        n_repeats=n_repeats, scoring=scoring, random_state=random_state,
    )
    # Only take importance of original features (exclude eigenvector columns)
    phylo_importance = phylo_result.importances_mean[:n_features]

    # 3. Build report
    delta = phylo_importance - raw_importance
    report = pd.DataFrame({
        "feature": feature_names,
        "raw_importance": raw_importance,
        "phylo_corrected_importance": phylo_importance,
        "delta": delta,
    })

    return report
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_importance.py -v`
Expected: 5 passed

**Step 5: Update treeml/__init__.py**

Edit `treeml/__init__.py`:
```python
from treeml.version import __version__
from treeml.estimators._classifier import PhyloRandomForestClassifier
from treeml.estimators._regressor import PhyloRandomForestRegressor
from treeml.cv._distance import PhyloDistanceCV
from treeml.cv._clade import PhyloCladeCV
from treeml.importance._report import phylo_feature_importance

__all__ = [
    "__version__",
    "PhyloRandomForestClassifier",
    "PhyloRandomForestRegressor",
    "PhyloDistanceCV",
    "PhyloCladeCV",
    "phylo_feature_importance",
]
```

**Step 6: Commit**

```bash
git add treeml/importance/ tests/unit/test_importance.py treeml/__init__.py
git commit -m "feat: add phylo_feature_importance() report"
```

---

## Chunk 7: Data Loading, Docs, and Integration Tests

### Task 7.1: load_data() convenience function

**Files:**
- Create: `treeml/_io.py`
- Create: `tests/unit/test_io.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_io.py`:
```python
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
    # brain_size is response, so X has remaining columns (body_mass, diet_type)
    assert X.shape[1] == 2


def test_load_data_response_not_in_features():
    X, y, tree, species_names = load_data(
        trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
        tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
        response="brain_size",
    )
    # y should be brain_size values, X should not contain brain_size
    assert y.shape == (len(species_names),)


def test_load_data_species_match_tree():
    X, y, tree, species_names = load_data(
        trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
        tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
        response="brain_size",
    )
    tree_tips = {t.name for t in tree.get_terminals()}
    assert set(species_names).issubset(tree_tips)
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_io.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

Create `treeml/_io.py`:
```python
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
    trait_names = header[1:]  # first column is species name

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
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_io.py -v`
Expected: 4 passed

**Step 5: Update treeml/__init__.py**

Edit `treeml/__init__.py`:
```python
from treeml.version import __version__
from treeml.estimators._classifier import PhyloRandomForestClassifier
from treeml.estimators._regressor import PhyloRandomForestRegressor
from treeml.cv._distance import PhyloDistanceCV
from treeml.cv._clade import PhyloCladeCV
from treeml.importance._report import phylo_feature_importance
from treeml._io import load_data

__all__ = [
    "__version__",
    "PhyloRandomForestClassifier",
    "PhyloRandomForestRegressor",
    "PhyloDistanceCV",
    "PhyloCladeCV",
    "phylo_feature_importance",
    "load_data",
]
```

**Step 6: Commit**

```bash
git add treeml/_io.py tests/unit/test_io.py treeml/__init__.py
git commit -m "feat: add load_data() convenience function"
```

---

### Task 7.2: Integration test — full end-to-end workflow

**Files:**
- Create: `tests/integration/test_end_to_end.py`

**Step 1: Write integration test**

Create `tests/integration/test_end_to_end.py`:
```python
import os

import numpy as np
import pytest
from sklearn.model_selection import cross_val_score

from treeml import (
    PhyloRandomForestRegressor,
    PhyloRandomForestClassifier,
    PhyloDistanceCV,
    PhyloCladeCV,
    phylo_feature_importance,
    load_data,
)

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_files")


@pytest.mark.integration
class TestRegressionEndToEnd:
    def test_fit_predict_with_tree(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        model = PhyloRandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y, tree=tree, species_names=names)
        preds = model.predict(X, tree=tree, species_names=names)
        assert preds.shape == y.shape

    def test_cross_val_with_phylo_distance_cv(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        model = PhyloRandomForestRegressor(n_estimators=50, random_state=42)
        cv = PhyloDistanceCV(tree=tree, species_names=names, n_splits=3)
        # cross_val_score won't pass tree to fit/predict, so we test CV splits
        splits = list(cv.split(X, y))
        assert len(splits) == 3

    def test_feature_importance_report(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        report = phylo_feature_importance(
            X, y, tree=tree, species_names=names,
            feature_names=["body_mass", "diet_type"],
            n_repeats=5, random_state=42,
        )
        assert len(report) == 2
        assert "delta" in report.columns

    def test_predict_new_species_with_tree(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        model = PhyloRandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y, tree=tree, species_names=names)

        from Bio import Phylo
        updated_tree = Phylo.read(
            os.path.join(SAMPLE_DIR, "tree_with_fox.nwk"), "newick"
        )
        # Predict for fox (new species) using original + fox features
        X_with_fox = np.vstack([X, [[20.0, 1]]])
        names_with_fox = names + ["fox"]
        preds = model.predict(
            X_with_fox, tree=updated_tree, species_names=names_with_fox
        )
        assert preds.shape == (len(names) + 1,)

    def test_predict_without_tree_degrades_gracefully(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="brain_size",
        )
        model = PhyloRandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y, tree=tree, species_names=names)
        with pytest.warns(UserWarning):
            preds = model.predict(X)
        assert preds.shape == y.shape


@pytest.mark.integration
class TestClassificationEndToEnd:
    def test_fit_predict(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="diet_type",
        )
        clf = PhyloRandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y, tree=tree, species_names=names)
        preds = clf.predict(X, tree=tree, species_names=names)
        assert preds.shape == y.shape

    def test_predict_proba(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="diet_type",
        )
        clf = PhyloRandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y, tree=tree, species_names=names)
        proba = clf.predict_proba(X, tree=tree, species_names=names)
        assert proba.shape[1] == 2
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_clade_cv(self):
        X, y, tree, names = load_data(
            trait_file=os.path.join(SAMPLE_DIR, "traits_simple.tsv"),
            tree_file=os.path.join(SAMPLE_DIR, "tree_simple.nwk"),
            response="diet_type",
        )
        cv = PhyloCladeCV(tree=tree, species_names=names, n_splits=3)
        splits = list(cv.split(X, y))
        assert len(splits) >= 1
```

**Step 2: Run integration tests**

Run: `python3 -m pytest tests/integration/test_end_to_end.py -v -m integration`
Expected: All passed

**Step 3: Run full test suite**

Run: `make test`
Expected: All unit + integration tests pass

**Step 4: Commit**

```bash
git add tests/integration/test_end_to_end.py
git commit -m "feat: add end-to-end integration tests"
```

---

### Task 7.3: Sphinx documentation scaffolding

**Files:**
- Create: `docs/conf.py`
- Create: `docs/Makefile`
- Create: `docs/Pipfile`
- Create: `docs/index.rst`
- Create: `docs/about/index.rst`
- Create: `docs/usage/index.rst`
- Create: `docs/tutorials/index.rst`
- Create: `docs/change_log/index.rst`
- Create: `docs/frequently_asked_questions/index.rst`
- Create: `docs/_static/custom.css`
- Create: `docs/_templates/sidebar-top.html`

This task creates the documentation structure. Tutorial content is written in Task 7.4.

**Step 1: Create docs/conf.py**

Create `docs/conf.py`:
```python
import os
import sys

try:
    import sphinx_rtd_theme
except ModuleNotFoundError:
    sphinx_rtd_theme = None

sys.path.append(os.path.abspath("./_ext"))

project = "treeml"
copyright = "2026 Jacob L. Steenwyk"
author = "Jacob L. Steenwyk <jlsteenwyk@gmail.com>"

extensions = ['sphinx_rtd_theme'] if sphinx_rtd_theme is not None else []

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
language = "en"
smartquotes = False
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_ext", "plans"]
pygments_style = None

html_theme = "sphinx_rtd_theme" if sphinx_rtd_theme is not None else "alabaster"
html_theme_options = {
    "body_max_width": "900px",
    'logo_only': True,
} if sphinx_rtd_theme is not None else {}
html_show_sourcelink = False
html_static_path = ["_static"]

html_sidebars = {
    "**": [
        "sidebar-top.html",
        "navigation.html",
        "relations.html",
        "searchbox.html",
    ]
}

htmlhelp_basename = "treemldoc"


def setup(app):
    app.add_css_file("custom.css")
```

**Step 2: Create docs/Makefile**

Create `docs/Makefile`:
```makefile
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

.PHONY: help Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
```

**Step 3: Create docs/Pipfile**

Create `docs/Pipfile`:
```toml
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
sphinx = "*"
sphinx-rtd-theme = "*"

[dev-packages]

[requires]
python_version = "3.11"
```

**Step 4: Create docs/_static/custom.css**

Copy PhyKIT's `custom.css` verbatim (same Oxygen font family, styling).

**Step 5: Create docs/_templates/sidebar-top.html**

Create `docs/_templates/sidebar-top.html`:
```html
<link href='http://fonts.googleapis.com/css?family=Oxygen:300,400,700' rel='stylesheet'>
<script async defer src="https://buttons.github.io/buttons.js"></script>

<div id="logo">
    <center>
        <a href="{{ pathto(master_doc) }}"><h2>treeml</h2></a>
    </center>
</div>

<div id="gh-buttons">
    <a class="github-button" href="https://github.com/jlsteenwyk/treeml" aria-label="View jlsteenwyk/treeml on GitHub" data-style="mega">Source</a>
    <a class="github-button" href="https://github.com/jlsteenwyk/treeml/issues" data-icon="octicon-issue-opened" aria-label="Issue jlsteenwyk/treeml on GitHub" data-style="mega">Issue</a>
</div>
```

**Step 6: Create docs/index.rst**

Create `docs/index.rst`:
```rst
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
```

**Step 7: Create stub pages**

Create `docs/about/index.rst`:
```rst
About
=====

treeml provides phylogenetic machine learning estimators that are compatible with
scikit-learn. It accounts for the evolutionary non-independence of species when
training ML models on comparative data.

Citation
--------

If you use treeml, please cite: [TBD]
```

Create `docs/usage/index.rst`:
```rst
Usage
=====

API Reference
-------------

Estimators
~~~~~~~~~~

- ``PhyloRandomForestRegressor`` — Random Forest regressor with phylogenetic correction
- ``PhyloRandomForestClassifier`` — Random Forest classifier with phylogenetic eigenvector features

Cross-Validation
~~~~~~~~~~~~~~~~

- ``PhyloDistanceCV`` — Phylogenetic distance-based cross-validation
- ``PhyloCladeCV`` — Clade-based cross-validation

Feature Importance
~~~~~~~~~~~~~~~~~~

- ``phylo_feature_importance()`` — Comparative feature importance report

Data Loading
~~~~~~~~~~~~

- ``load_data()`` — Load trait file and tree
```

Create `docs/tutorials/index.rst`:
```rst
.. _tutorials:

Tutorials
=========

Step-by-step guides for using treeml.

|

1. Quick start: Predicting a phenotype
#######################################

Coming soon.

|

2. Classification: Predicting discrete traits
##############################################

Coming soon.

|

3. Why phylogenetic correction matters
########################################

Coming soon.

|

4. Phylogenetic cross-validation
##################################

Coming soon.

|

5. Feature importance: Finding real signal
###########################################

Coming soon.

|

6. Predicting new species
###########################

Coming soon.

|

7. Discordance-aware correction
#################################

Coming soon.
```

Create `docs/change_log/index.rst`:
```rst
Change Log
==========

v0.1.0
------

- Initial release
- PhyloRandomForestRegressor and PhyloRandomForestClassifier
- PhyloDistanceCV and PhyloCladeCV
- phylo_feature_importance() report
- load_data() convenience function
```

Create `docs/frequently_asked_questions/index.rst`:
```rst
Frequently Asked Questions
==========================

What is phylogenetic non-independence?
--------------------------------------

Species are related through evolution. Closely related species tend to have similar
traits not because of shared selective pressures but because they inherited those
traits from a common ancestor. Standard ML methods assume observations are independent,
which species data violates.

Why not just use PGLS?
-----------------------

PGLS is a linear method. It works great for linear relationships, but many biological
relationships are non-linear. treeml extends phylogenetic correction to non-linear
ML methods like Random Forest.

When should I use phylogenetic correction?
-------------------------------------------

Whenever your rows are species (or populations, strains) that are related by a
phylogenetic tree and you want to avoid confounding due to shared ancestry.
```

**Step 8: Commit**

```bash
git add docs/
git commit -m "feat: add Sphinx documentation scaffolding"
```

---

### Task 7.4: Run full test suite and verify package installs

**Step 1: Install package in development mode**

Run: `cd /Users/jacoblsteenwyk/Desktop/kit_dev/treeml && pip install -e .`
Expected: Successfully installed treeml

**Step 2: Run full test suite**

Run: `make test`
Expected: All unit and integration tests pass

**Step 3: Verify imports work**

Run:
```bash
python3 -c "
from treeml import (
    PhyloRandomForestClassifier,
    PhyloRandomForestRegressor,
    PhyloDistanceCV,
    PhyloCladeCV,
    phylo_feature_importance,
    load_data,
    __version__,
)
print(f'treeml v{__version__} loaded successfully')
print(f'Exports: {7} public symbols')
"
```
Expected: `treeml v0.1.0 loaded successfully`

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: treeml v0.1.0 complete"
```
