# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../'))  # Assumes conf.py is inside 'source'
sys.path.insert(0, os.path.abspath('../Predictors'))
sys.path.insert(0, os.path.abspath('../tools'))

# -- Project information -----------------------------------------------------
project = 'Forecasting framework'
copyright = '2024, Gianluca Ferro'
author = 'Gianluca Ferro'
release = '1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',  # Enables autodoc extension to extract docstrings
    'sphinx.ext.napoleon'  # Enables support for Google and NumPy style docstrings
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'alabaster'
html_static_path = ['_static']

html_css_files = [
    'custom.css',  # Removed the leading './' which might cause path issues
]

# Specify source encoding
source_encoding = 'utf-8-sig'

# Master document
master_doc = 'index'  # Ensure this is set to the name of your master document, typically 'index.rst'

# Path to the documentation root, which should contain your rst files
# Assuming 'docs' is a directory inside 'source'
source_suffix = '.rst'
