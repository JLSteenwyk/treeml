About
=====

^^^^^

**Tree** **M**\achine **L**\earning (**treeml**) was developed by
`Jacob L. Steenwyk <https://jlsteenwyk.github.io/>`_ to address a fundamental challenge
in comparative biology: species are not independent data points due to shared evolutionary history.

Standard machine learning methods assume that observations are independent. However, closely related
species tend to have similar traits not because of shared selective pressures but because they
inherited those traits from a common ancestor. This phylogenetic non-independence can inflate
performance metrics, introduce confounding, and lead to biased inferences when species-level data
are used as training observations.

In light of these challenges, we developed treeml, which extends phylogenetic correction to non-linear
machine learning methods. treeml achieves this by (1) augmenting feature matrices with phylogenetic
eigenvectors, (2) whitening features and targets using the Cholesky decomposition of the phylogenetic
variance-covariance matrix, and (3) providing phylogenetic-aware cross-validation splitters that
prevent related species from leaking information between training and test folds. Our validation
against R's ``ape`` package demonstrates that all core calculations match to machine epsilon
(:ref:`see details here <validation>`).

|

The Developer
--------------

^^^^^

treeml is developed and maintained by `Jacob L. Steenwyk <https://jlsteenwyk.github.io/>`_.

|

|JLSteenwyk|

|GoogleScholarSteenwyk| |GitHubSteenwyk| |TwitterSteenwyk|

`Jacob L. Steenwyk <https://jlsteenwyk.github.io/>`_ is an Assistant Professor in the
`Department of Ecology and Evolutionary Biology
<https://www.colorado.edu/ebio/>`_ at the
`University of Colorado Boulder <https://www.colorado.edu/>`_.
His research focuses on understanding the evolutionary genomics of diverse organisms.
Beyond research, Steenwyk aims to make education more accessible
through diverse avenues of community engagement. Find out more information at his
`personal website <http://jlsteenwyk.github.io/>`_.

.. |JLSteenwyk| image:: ../_static/img/Steenwyk.jpg
   :width: 35%

.. |GoogleScholarSteenwyk| image:: ../_static/img/GoogleScholar.png
   :target: https://scholar.google.com/citations?user=VXV2j6gAAAAJ&hl=en
   :width: 4.5%

.. |TwitterSteenwyk| image:: ../_static/img/Twitter.png
   :target: https://twitter.com/jlsteenwyk
   :width: 4.5%

.. |GitHubSteenwyk| image:: ../_static/img/Github.png
   :target: https://github.com/JLSteenwyk
   :width: 4.5%

|

.. _validation:

Validation against R
---------------------

^^^^^

All core phylogenetic calculations in treeml have been validated against canonical
R implementations from the ``ape`` package. Using an 8-species mammalian phylogeny,
every computation produces a **Pearson correlation of 1.0** with R's output, with
maximum absolute differences at machine epsilon (1e-13 to 1e-14).

.. list-table::
   :header-rows: 1
   :widths: 40 25 25

   * - Calculation
     - R reference
     - Pearson *r*
   * - VCV matrix
     - ``ape::vcv.phylo()``
     - 1.000000000000000
   * - Cholesky decomposition
     - ``chol()``
     - 1.000000000000000
   * - Phylogenetic whitening
     - ``solve(L, y)``
     - 1.000000000000000
   * - PCoA eigenvalues
     - ``eigen()``
     - 1.000000000000000
   * - PCoA eigenvectors (all 7)
     - ``eigen()``
     - \|r\| = 1.0 each
   * - Patristic distances
     - ``ape::cophenetic.phylo()``
     - 1.000000000000000

The validation scripts are available in the ``validation/`` directory of the
repository. ``r_reference.R`` computes reference values using R's ``ape`` package,
and ``validate_against_r.py`` compares them against treeml's output.

|

Citation
--------

^^^^^

If you use treeml, please cite: [TBD]
