import re
from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# Read version without importing treeml (avoids triggering full import chain
# before dependencies are installed)
with open(path.join(here, "treeml", "version.py")) as f:
    __version__ = re.search(r'__version__\s*=\s*"(.+?)"', f.read()).group(1)

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
    "pandas>=2.0.0",
    "shap>=0.42.0",
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

## push new version to pypi
# rm -rf dist
# python3 setup.py sdist bdist_wheel --universal
# twine upload dist/* -r pypi
