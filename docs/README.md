# Documentation

The documentation is generated via [Sphinx](https://www.sphinx-doc.org) using the [book theme](https://sphinx-book-theme.readthedocs.io/en/latest/).

## Building the documentation

**To be able to build the documentation**, the documentation dependencies have to be installed:
```shell
python -m pip install -e ".[doc]"
```
**To build the documentation**, simply execute the Makefile inside the `docs` folder:
```shell
cd docs
make
```
By default, this will build the documentation for all versions. To only build it locally, execute `make local`.

The homepage of the documentation can be found in `build/html/main/index.html` (replace `main` by `local` for the local version).

## Remark on docstrings

The docstrings in this project are written in **numpy style**. Please read the [numpy style documentation](https://numpydoc.readthedocs.io/en/latest/format.html) to get to know the syntax and the different sections.

If you are using vscode, there is an extension called [Python Docstring Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) which automatically generates the docstring from the function's definition in the numpy format (the numpy style has to be specified in the extension's settings).
