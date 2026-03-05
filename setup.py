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
