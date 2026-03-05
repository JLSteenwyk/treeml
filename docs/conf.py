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
