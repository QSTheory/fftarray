# Documentation

The documentation is generated via [Sphinx](https://www.sphinx-doc.org) using
the [book theme](https://sphinx-book-theme.readthedocs.io/en/latest/).

## Building the documentation

**To be able to build the documentation**, the documentation dependencies have
to be installed:
```shell
python -m pip install -e ".[doc]"
```
**To build the documentation**, simply execute the Makefile inside the `docs`
folder:
```shell
cd docs
make
```
By default, this will build the documentation for all versions (see next
section). To only build it for your current version, execute `make local`.
The homepage of the documentation can be found in `build/html/local/index.html`.

### Building for all versions

In `helpers/generate_versions.py`, `build/html/versions.json` is created,
containing a list of all the versions along with their names and links.
Then, for each of these versions, the files from the corresponding Git branch or
version tag are saved to `build/html/src/{version}` using `git archive`.
In this folder, the documentation is built with `make local`.
The resulting folder, which contains the HTML files
(`build/html/src/{version}/docs/build/html/local`) is then moved to the
collective documentation (`build/html/{version}`).

## Remark on docstrings

The docstrings in this project are written in **numpy style**. Please read the
[numpy style
documentation](https://numpydoc.readthedocs.io/en/latest/format.html) to get to
know the syntax and the different sections.

If you are using vscode, there is an extension called [Python Docstring
Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)
which automatically generates the docstring from the function's definition in
the numpy format (the numpy style has to be specified in the extension's
settings).
