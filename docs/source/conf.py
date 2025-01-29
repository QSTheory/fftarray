
import os
import sys

sys.path.insert(0, os.path.abspath("../../src/"))
sys.path.insert(0, os.path.abspath("../../examples/"))

# ---------------------------- Project information --------------------------- #

project = "fftarray"
copyright = "2024, The fftaray authors. NumPy and Jax are copyright the respective authors."
author = "The fftarray authors"

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
    "myst_nb",
    # "sphinx_remove_toctrees",
    "sphinx_copybutton",
    "sphinx_design",
    "nbsphinx",
    "nbsphinx_link"
]

add_module_names = False

napoleon_numpy_docstring = True
napolean_use_rtype = False
napoleon_use_param = False

autosummary_generate = True
autosummary_overwrite = True
autosummary_import_members = True

autodoc_typehints = "both"
autodoc_typehints_format = "short"

nbsphinx_allow_errors = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/generated/", None),
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

html_theme_options = dict(
    repository_url='https://github.com/QSTheory/fftarray',
    repository_branch='main',
    navigation_with_keys=False,  # pydata/pydata-sphinx-theme#1492
    navigation_depth=4,
    path_to_docs='docs',
    use_edit_page_button=True,
    use_repository_button=True,
    use_issues_button=True,
    home_page_in_toc=False,
)

html_static_path = ['_static']

html_css_files = [
    'style.css',
]


# ----------------------------------- myst ----------------------------------- #
myst_heading_anchors = 3  # auto-generate 3 levels of heading anchors
myst_enable_extensions = ['dollarmath', 'colon_fence']
nb_execution_mode = "force"
nb_execution_allow_errors = False
nb_merge_streams = True

# TODO: copied from jax, test if needed
nb_execution_timeout = 100


