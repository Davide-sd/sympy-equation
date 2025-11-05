import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from doc_utils import param_formatter

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'sympy_equation'
copyright = '2025, Davide Sandonà'
author = 'Davide Sandonà'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.doctest",
    "numpydoc",
    "sphinx_rtd_theme",
    "sphinx_design",
    "nbsphinx",
]

# in order to show the icons on sponsorship buttons
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css"
]

# hide the table inside classes autodoc
numpydoc_show_class_members = False

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


# -- param.Parameterized -----------------------------------------------------
# Inspired by:
# https://github.com/holoviz-dev/nbsite/blob/master/nbsite/paramdoc.py
def setup(app):
    app.connect("autodoc-process-docstring", param_formatter, priority=-100)
