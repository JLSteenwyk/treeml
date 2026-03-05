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
