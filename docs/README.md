# Documentation

The documentation is generated via [Sphinx](https://www.sphinx-doc.org) using the [book theme](https://sphinx-book-theme.readthedocs.io/en/latest/).

## Building the documentation

**To be able to build the documentation**, the development dependencies have to be installed:
```shell
python -m pip install -e ".[dev]"
```
**To build the documentation**, simply execute `./build_docs.sh` inside the `docs` folder:
```shell
./build_docs.sh
```
This will create two versions: the public and development version.

The homepage of the documentation can be found in `build/html/public/index.html` (exchange `public` by `development` for the development version) and can be opened locally.

### Included `README.md` files
The documentation imports the `README.md` from the main project directory into the "Gettings started" page. The `README.md` of all example subfolders are imported into the "Examples" page.

## Versioning

The documentation is built in two different versions. The version can be toggled at the bottom of the sidebar.

The public version only includes modules and functions that are of use for the user. Other methods should be marked private by
- the function name: function name begins with an underscore, e.g. `_private_func()`
- the docstring: docstring contains `:meta private:` in a separate line

### reStructuredText files
In reStructuredText files, a section can be marked private using the tag `.. only:: development`. Everything beneath this tag (and indented) will only appear in the development version. The same applies to the public version: `.. only:: public`.

### private.txt
Whole files can be marked private by including their path (relative to `docs`) into `private.txt`. This makes them only appear in the development version.

### exclude.txt
If a file should not be included in either the public nor the development version, add its path (relative to `docs`) to `exclude.txt`.

## Remark on docstrings

The docstrings in this project are written in **numpy style**. Please read the [numpy style documentation](https://numpydoc.readthedocs.io/en/latest/format.html) to get to know the syntax and the different sections.

If you are using vscode, there is an extension called [Python Docstring Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) which automatically generates the docstring from the function's definition in the numpy format (the numpy style has to be specified in the extension's settings).

### Package summaries

The package summaries that include a table of the package's direct imports are created by a docstring inside the package's `__init__.py` file.
