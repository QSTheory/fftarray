
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../examples/"))

# ---------------------------- Project information --------------------------- #

project = "FFTArray"
copyright = "2024, The FFTArray authors. NumPy, Jax and PyFFTW are copyright the respective authors."
author = "The FFTArray authors"

version = ""
release = ""

# --------------------------- General configuration -------------------------- #

# TODO: copied from jax, test if needed
# needs_sphinx = "2.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_autodoc_typehints",
    "myst_nb",
    # "sphinx_remove_toctrees",
    # "sphinx_copybutton",
    "sphinx_design",
    "nbsphinx",
    "nbsphinx_link"
]


napoleon_numpy_docstring = True
napolean_use_rtype = False

autosummary_generate = True
autosummary_overwrite = True
autosummary_import_members = True

nbsphinx_allow_errors = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "panel": ("https://panel.holoviz.org/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "pyfftw": ("https://pyfftw.readthedocs.io/en/latest/", None)
}

templates_path = ['_templates']

source_suffix = ['.rst', '.ipynb', '.md']

exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store'
    'build/html',
    'build/jupyter_execute',
    'notebooks/README.md',
    'README.md',
    'notebooks/*.md'
]

pygments_style = None

html_theme = 'sphinx_book_theme'

html_theme_options = {
    'show_toc_level': 2,
    'repository_url': 'https://gitlab.projekt.uni-hannover.de/iqo-tsquad-github/fftarray',
    'use_repository_button': True,
    'navigation_with_keys': False,
}

html_static_path = ['_static']

html_css_files = [
    'style.css',
]


# ----------------------------------- myst ----------------------------------- #
myst_heading_anchors = 3  # auto-generate 3 levels of heading anchors
myst_enable_extensions = ['dollarmath']
nb_execution_mode = "force"
nb_execution_allow_errors = False
nb_merge_streams = True

# TODO: copied from jax, test if needed
nb_execution_timeout = 100


