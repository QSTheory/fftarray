# Runs all the notebooks as parts of the tests.
# originally based on https://jupyter-notebook.readthedocs.io/en/5.7.6/examples/Notebook/Importing%20Notebooks.html

import io
import types
import os

import pytest
from IPython import get_ipython
from nbformat import read
from IPython.core.interactiveshell import InteractiveShell

import jax


from tests.helpers import XPS
import fftarray as fa

@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("nb_path", [
    "Gaussians",
    "multi_dimensional",
    "Bragg_beam_splitter",
    "Derivative",
])
def test_notebooks(xp, nb_path):
    run_notebook(xp=xp, nb_path=nb_path)


def test_two_species_notebook():
    """
        The two-species example explicitly utilizes jax.
    """
    run_notebook(xp=jax.numpy, nb_path="two_species_groundstate")

def run_notebook(xp, nb_path) -> None:
    # load the notebook object
    with io.open(f"examples/{nb_path}.ipynb", 'r', encoding='utf-8') as f:
        nb = read(f, 4)

    shell = InteractiveShell.instance()
    # create the module
    mod = types.ModuleType("notebook")
    mod.__file__ = nb_path
    mod.__dict__['get_ipython'] = get_ipython
    mod.__dict__["xp"] = xp

    # extra work to ensure that magics that would affect the user_ns
    # actually affect the notebook module's ns
    save_user_ns = shell.user_ns
    shell.user_ns = mod.__dict__

    os.environ["TEST_MODE"] = "True"
    # Force the matplotlib backend to be non-interactive.
    # Otherwise this test fails for example on a macOS notebook.
    os.environ["MPLBACKEND"] = "agg"

    try:
        for cell in nb.cells:
            if cell.cell_type == 'code':
                # Set the Array API implementation to test with.
                with fa.default_xp(xp):
                    # transform the input to executable Python
                    code = shell.input_transformer_manager.transform_cell(cell.source)
                    # run the code in themodule
                    exec("import sys", mod.__dict__)
                    exec("sys.path.append('examples')", mod.__dict__)
                    exec(code, mod.__dict__)
    finally:
        shell.user_ns = save_user_ns