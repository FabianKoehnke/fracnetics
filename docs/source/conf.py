# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../fracnetics'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Fracnetics'
author = 'Fabian Köhnke'
copyright = '2025, Fabian Köhnke'
autodoc_typehints = 'description'
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'breathe',
    'sphinx_gallery.gen_gallery',
]

# ---- Breathe (C++ Integration) ----
breathe_projects = {
    "Fracnetics": "../cpp/xml"
}
breathe_default_project = "Fracnetics"


templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# ---- sphinx-gallery ----
sphinx_gallery_conf = {
    'examples_dirs': '../../examples',   # Python examples
    'gallery_dirs': 'auto_examples',
}
