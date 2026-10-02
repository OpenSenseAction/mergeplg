from __future__ import annotations

import importlib.metadata
import re

project = "mergeplg"
copyright = "2024, Christian Chwala"
author = "Christian Chwala"
release = importlib.metadata.version("mergeplg")

# Keep sidebar version compact (e.g. 0.1.0) while preserving full build
# metadata in ``release`` for explicit display elsewhere.
_version_match = re.match(r"^\d+\.\d+\.\d+", release)
version = _version_match.group(0) if _version_match else release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "nbsphinx",
]

source_suffix = [".rst", ".md"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

html_theme = "furo"

myst_enable_extensions = [
    "colon_fence",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

always_document_param_types = True
