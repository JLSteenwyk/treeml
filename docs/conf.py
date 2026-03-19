# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

import os
import sys

# -- Path setup --------------------------------------------------------------

sys.path.append(os.path.abspath("./_ext"))

# -- Project information -----------------------------------------------------

project = "treeml"
copyright = "2026 Jacob L. Steenwyk"
author = "Jacob L. Steenwyk <jlsteenwyk@gmail.com>"

# -- General configuration ---------------------------------------------------

extensions = ["sphinx.ext.githubpages"]

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
language = "en"
smartquotes = False
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_ext", "plans"]
pygments_style = None


# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "body_max_width": "900px",
    "logo_only": True,
}
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

# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "treemldoc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {}

latex_documents = [
    (
        master_doc,
        "treeml.tex",
        "treeml Documentation",
        "Jacob L. Steenwyk \\textless{}jlsteenwyk@gmail.com\\textgreater{}",
        "manual",
    ),
]


# -- Options for manual page output ------------------------------------------

man_pages = [(master_doc, "treeml", "treeml Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (
        master_doc,
        "treeml",
        "treeml Documentation",
        author,
        "treeml",
        "Phylogenetic machine learning.",
        "Miscellaneous",
    ),
]


# -- Options for Epub output -------------------------------------------------

epub_title = project
epub_exclude_files = ["search.html"]


# -- Setup -------------------------------------------------
def setup(app):
    app.add_css_file("custom.css")
