
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
    "sphinx_autodoc_typehints",
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

autosummary_generate = True
autosummary_overwrite = True
autosummary_import_members = True

nbsphinx_allow_errors = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "panel": ("https://panel.holoviz.org/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None)
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

html_theme = 'furo'

html_theme_options = {
    # 'show_toc_level': 2,
    # 'repository_url': 'https://github.com/QSTheory/fftarray',
    # 'use_repository_button': True,
    "source_repository": "https://github.com/QSTheory/fftarray",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/QSTheory/fftarray",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
    "source_branch": "main",
    "source_directory": "docs/",
}

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


