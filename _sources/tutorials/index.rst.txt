.. _tutorials:

Tutorials
=========

Step-by-step guides for using treeml.

|

1. Quick start: Predicting a continuous phenotype
##################################################

This tutorial demonstrates how to use treeml to predict a continuous phenotype
while accounting for phylogenetic non-independence.

**Step 1: Load data**

Prepare a tab-separated trait file and a Newick tree file. The trait file should
have a header row with species names in the first column and trait values in
subsequent columns.

.. code-block:: shell

	cat traits.tsv

	species	body_mass	brain_mass	lifespan
	Homo_sapiens	70.0	1.35	79.0
	Pan_troglodytes	52.0	0.38	50.0
	Gorilla_gorilla	160.0	0.50	40.0
	Mus_musculus	0.02	0.0004	2.5
	Rattus_norvegicus	0.3	0.002	3.0
	Bos_taurus	750.0	0.45	20.0
	Sus_scrofa	80.0	0.18	15.0
	Canis_familiaris	30.0	0.07	12.0

.. code-block:: python

	from treeml import load_data

	X, y, tree, species_names = load_data(
	    trait_file="traits.tsv",
	    tree_file="species.nwk",
	    response="lifespan",
	)

	print(f"Features shape: {X.shape}")
	print(f"Target shape: {y.shape}")
	print(f"Species: {species_names}")

|

**Step 2: Create a phylogenetic estimator**

.. code-block:: python

	from treeml import PhyloRandomForestRegressor

	model = PhyloRandomForestRegressor(
	    n_estimators=100,
	    eigenvector_variance=0.90,
	    whiten_features=True,
	    whiten_target=False,
	    random_state=42,
	)

|

**Step 3: Fit the model**

.. code-block:: python

	model.fit(X, y, tree=tree, species_names=species_names)

|

**Step 4: Make predictions**

.. code-block:: python

	predictions = model.predict(X, tree=tree, species_names=species_names)
	for sp, pred, actual in zip(species_names, predictions, y):
	    print(f"{sp}: predicted={pred:.1f}, actual={actual:.1f}")

|

**Step 5: Evaluate with phylogenetic cross-validation**

.. code-block:: python

	from treeml import PhyloDistanceCV
	from sklearn.model_selection import cross_val_score

	cv = PhyloDistanceCV(tree=tree, species_names=species_names, n_splits=3)
	scores = cross_val_score(model, X, y, cv=cv)
	print(f"CV R² scores: {scores}")
	print(f"Mean R²: {scores.mean():.3f} ± {scores.std():.3f}")

|

2. Classification: Predicting discrete traits
##############################################

This tutorial demonstrates how to classify discrete traits (e.g., habitat type,
diet category) using phylogenetic-aware classifiers.

.. code-block:: python

	from treeml import PhyloRandomForestClassifier, PhyloDistanceCV
	from sklearn.model_selection import cross_val_score

	# Create classifier
	model = PhyloRandomForestClassifier(n_estimators=100, random_state=42)

	# Fit
	model.fit(X, y, tree=tree, species_names=species_names)

	# Predict class labels
	predictions = model.predict(X, tree=tree, species_names=species_names)

	# Predict class probabilities
	probabilities = model.predict_proba(X, tree=tree, species_names=species_names)

	# Evaluate with phylogenetic CV
	cv = PhyloDistanceCV(tree=tree, species_names=species_names, n_splits=3)
	scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
	print(f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

|

3. Why phylogenetic correction matters
########################################

When species are used as observations, ignoring phylogenetic relationships can
inflate performance metrics. This tutorial compares corrected vs. uncorrected
models using ``phylo_model_comparison()``.

.. code-block:: python

	from treeml import phylo_model_comparison

	results = phylo_model_comparison(
	    X, y, tree=tree, species_names=species_names
	)
	print(results)

The function benchmarks multiple estimators (Random Forest, Gradient Boosting,
SVM, KNN, Ridge, Lasso, Elastic Net for regression) with and without phylogenetic
correction and returns a DataFrame showing the cross-validation scores and the
delta between approaches.

A positive delta indicates that phylogenetic correction improved model performance.
A negative delta suggests that the uncorrected model performed better, which can
happen when the phylogenetic signal in the data is weak.

|

4. Phylogenetic cross-validation
##################################

Standard k-fold cross-validation can leak phylogenetic signal between folds.
treeml provides two phylogenetic-aware CV strategies.

**Distance-based CV:** Groups species by phylogenetic distance using hierarchical
clustering on patristic distances. The distance threshold is automatically tuned
to produce the requested number of folds.

.. code-block:: python

	from treeml import PhyloDistanceCV

	cv = PhyloDistanceCV(tree=tree, species_names=species_names, n_splits=5)
	for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
	    print(f"Fold {fold}: {len(train_idx)} train, {len(test_idx)} test")

**Clade-based CV:** Holds out entire monophyletic clades for each fold. This is
a stricter form of phylogenetic CV because the entire evolutionary lineage is
held out, preventing any closely related species from appearing in the training set.

.. code-block:: python

	from treeml import PhyloCladeCV

	cv = PhyloCladeCV(tree=tree, species_names=species_names, n_splits=5)
	for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
	    print(f"Fold {fold}: {len(train_idx)} train, {len(test_idx)} test")

|

5. Feature importance: Finding real signal
###########################################

Compare feature importance with and without phylogenetic correction to
distinguish true biological signal from phylogenetic artifact. Features whose
importance drops substantially after phylogenetic correction may have been
inflated by shared ancestry rather than reflecting true predictive relationships.

.. code-block:: python

	from treeml import phylo_feature_importance

	report = phylo_feature_importance(
	    X, y, tree=tree, species_names=species_names,
	    feature_names=["body_mass", "brain_mass"],
	    n_repeats=10,
	)
	print(report)

The ``delta`` column shows the difference between phylo-corrected and raw
importance. A negative delta suggests the feature's importance was inflated
by phylogenetic confounding.

|

6. SHAP explanations
###########################

Use SHAP values to understand how much each feature and the phylogenetic
correction contribute to predictions. The ``phylo_shap()`` function separates
SHAP values into original feature contributions and phylogenetic eigenvector
contributions.

.. code-block:: python

	from treeml import PhyloRandomForestRegressor, phylo_shap

	model = PhyloRandomForestRegressor(n_estimators=100)
	model.fit(X, y, tree=tree, species_names=species_names)

	result = phylo_shap(model, X, feature_names=["body_mass", "brain_mass"])

	# Overall phylogenetic contribution
	print(f"Phylogenetic contribution: {result.phylo_contribution:.1%}")

	# Summary table
	print(result.summary())

	# Bar chart comparing feature vs phylogenetic contributions
	result.plot(plot_type="bar")

	# Standard SHAP beeswarm plot (all features + eigenvectors)
	result.summary_plot()

	# Force plot for a single species
	result.force_plot(sample_idx=0)

|

7. Hyperparameter tuning
#################################

Use ``PhyloGridSearchCV`` or ``PhyloRandomizedSearchCV`` to find optimal
hyperparameters with phylogenetic-aware cross-validation. These wrappers
automatically bind tree and species_names to the estimator during CV.

.. code-block:: python

	from treeml import PhyloRandomForestRegressor, PhyloGridSearchCV

	model = PhyloRandomForestRegressor(random_state=42)

	search = PhyloGridSearchCV(
	    estimator=model,
	    param_grid={
	        "n_estimators": [50, 100, 200],
	        "eigenvector_variance": [0.8, 0.9, 0.95],
	    },
	    tree=tree,
	    species_names=species_names,
	    n_jobs=-1,
	)
	search.fit(X, y)

	print(f"Best params: {search.best_params_}")
	print(f"Best score: {search.best_score_:.3f}")

For large parameter spaces, use randomized search:

.. code-block:: python

	from treeml import PhyloRandomizedSearchCV
	from scipy.stats import randint

	search = PhyloRandomizedSearchCV(
	    estimator=model,
	    param_distributions={
	        "n_estimators": randint(50, 300),
	        "eigenvector_variance": [0.8, 0.9, 0.95],
	    },
	    tree=tree,
	    species_names=species_names,
	    n_iter=20,
	    n_jobs=-1,
	)
	search.fit(X, y)

|

8. Saving and loading models
#################################

Save fitted models for later use and load them back.

.. code-block:: python

	from treeml import PhyloRandomForestRegressor, save_model, load_model

	# Fit a model
	model = PhyloRandomForestRegressor(n_estimators=100)
	model.fit(X, y, tree=tree, species_names=species_names)

	# Save
	save_model(model, "my_model.treeml")

	# Load
	loaded = load_model("my_model.treeml")
	preds = loaded.predict(X, tree=tree, species_names=species_names)

|

9. Discordance-aware correction
#################################

When gene tree discordance is expected (e.g., due to incomplete lineage sorting),
you can pass gene trees to construct a discordance-aware VCV matrix instead of
using the species tree alone.

.. code-block:: python

	from treeml import PhyloRandomForestRegressor
	from Bio import Phylo

	species_tree = Phylo.read("species.nwk", "newick")
	gene_trees = [Phylo.read(f"gene_{i}.nwk", "newick") for i in range(100)]

	model = PhyloRandomForestRegressor(n_estimators=100)
	model.fit(
	    X, y,
	    tree=species_tree,
	    species_names=species_names,
	    gene_trees=gene_trees,
	)

	predictions = model.predict(
	    X,
	    tree=species_tree,
	    species_names=species_names,
	    gene_trees=gene_trees,
	)

|
