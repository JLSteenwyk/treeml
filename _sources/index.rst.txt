treeml
======

^^^^^


treeml is a phylogenetic machine learning library that provides scikit-learn compatible estimators accounting for evolutionary non-independence among species.


If you found treeml useful, please cite: [TBD]


Quick Start
-----------
These two lines represent the simplest method to rapidly install and use treeml.

.. code-block:: shell

	# install
	pip install treeml

.. code-block:: python

	from treeml import PhyloRandomForestRegressor
	from Bio import Phylo

	tree = Phylo.read("species.nwk", "newick")
	model = PhyloRandomForestRegressor(n_estimators=100)
	model.fit(X, y, tree=tree, species_names=names)
	predictions = model.predict(X, tree=tree, species_names=names)

Below are more detailed instructions, including alternative installation methods.

**1) Installation**

To help ensure treeml can be installed using your favorite workflow, we have made treeml available from pip, source, and the anaconda cloud.

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
	make install

To deactivate your virtual environment, use the following command:

.. code-block:: shell

	# deactivate virtual environment
	deactivate

**Note: the virtual environment must be activated to use treeml.**

If you run into permission errors when executing *make install*, create a
virtual environment for your installation:

.. code-block:: shell

	git clone https://github.com/JLSteenwyk/treeml.git
	cd treeml/
	python -m venv venv
	source venv/bin/activate
	make install

Note: the virtual environment must be activated to use treeml.

|

**Install from anaconda**

To install via anaconda, execute the following command:

.. code-block:: shell

	conda install bioconda::treeml

Visit here for more information: https://anaconda.org/bioconda/treeml

|

**2) Usage**

To use treeml, import the desired estimator and fit it with a phylogenetic tree:

.. code-block:: python

	from treeml import PhyloRandomForestRegressor
	from Bio import Phylo

	tree = Phylo.read("species.nwk", "newick")
	model = PhyloRandomForestRegressor(n_estimators=100)
	model.fit(X, y, tree=tree, species_names=names)
	predictions = model.predict(X, tree=tree, species_names=names)

|

^^^^

.. toctree::
	:maxdepth: 4

	about/index
	advanced/index
	tutorials/index
	change_log/index
	other_software/index
	frequently_asked_questions/index

^^^^
