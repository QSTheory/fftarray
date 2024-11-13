# fftarray

[**Installation**](#installation) | ...

## Installation

There are different versions of fftarray available for installation, enabling different capabilities and thus, coming with different external packages as requirements.
The bare version features the core capabilities. For example, there are no helper methods to define a `FFTDimension` or export into/import from an `xarray.DataArray`. There is also no automatic installation of required packages for accelerated FFT implementations on GPUs (`jax`). Additionally, there is a version to enable the execution of the examples.

You can install each version of fftarray from the GitHub repository directly via SSH (recommended) or HTTPS.
```shell
## Bare installation
python -m pip install 'fftarray @ git+ssh://git@github.com/QSTheory/fftarray.git' # SSH
python -m pip install 'fftarray @ git+https://github.com/QSTheory/fftarray.git' # HTTPS
```
**Available versions**
```shell
## JAX support (GPU acceleration)
python -m pip install 'fftarray[jax] @ git+ssh://git@github.com/QSTheory/fftarray.git' # SSH
## Some helper methods (FFT constraint solver + xarray import/export)
python -m pip install 'fftarray[helpers] @ git+ssh://git@github.com/QSTheory/fftarray.git' # SSH
## Examples
python -m pip install 'fftarray[examples] @ git+ssh://git@github.com/QSTheory/fftarray.git' # SSH
```
You can also combine different additions:
```shell
## JAX support + helper methods
python -m pip install 'fftarray[helpers,jax] @ git+ssh://git@github.com/QSTheory/fftarray.git' # SSH
```
