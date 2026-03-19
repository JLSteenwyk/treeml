treeml
======

Phylogenetic machine learning: scikit-learn estimators that account for evolutionary
non-independence among species.

If you found treeml useful, please cite: [TBD]


Quick Start
-----------
These two lines represent the simplest method to rapidly install and run treeml.

.. code-block:: shell

	# install
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

Below are more detailed instructions, including alternative installation methods.

**1) Installation**

To help ensure treeml can be installed using your favorite workflow, we have made treeml available from pip and source.

**Install from pip**

To install from pip, use the following commands:

.. code-block:: shell

	# create virtual environment
	python -m venv venv
	# activate virtual environment
	source venv/bin/activate
	# install treeml
	pip install treeml

**Note: the virtual environment must be activated to use treeml.**

|

**Install from source**

Similarly, to install from source, we strongly recommend using a virtual environment. To do so, use the following commands:

.. code-block:: shell

	# download
	git clone https://github.com/JLSteenwyk/treeml.git
	cd treeml/
	# create virtual environment
	python -m venv venv
	# activate virtual environment
	source venv/bin/activate
	# install
	pip install .

To deactivate your virtual environment, use the following command:

.. code-block:: shell

	# deactivate virtual environment
	deactivate

**Note: the virtual environment must be activated to use treeml.**

|

**2) Basic Usage**

To use treeml, import the desired estimator and fit it with a phylogenetic tree:

.. code-block:: python

	from treeml import PhyloRandomForestRegressor
	from Bio import Phylo

	tree = Phylo.read("species.nwk", "newick")
	model = PhyloRandomForestRegressor(n_estimators=100)
	model.fit(X, y, tree=tree, species_names=names)
	predictions = model.predict(X, tree=tree, species_names=names)

|

.. toctree::
	:maxdepth: 4

	about/index
	usage/index
	tutorials/index
	change_log/index
	other_software/index
	frequently_asked_questions/index
