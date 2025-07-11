# FFTArray: A Python Library for the Implementation of Discretized Multi-Dimensional Fourier Transforms

[**Intro**](#intro) | [**Installation**](#installation) | ...

## Intro
For a more through description of the library we reommend the [publication](todo) and the [documentation](todo).

### Adding Coordinate Grids to the FFT
As a physicist we often just need a discretized version of the (continuous) Fourier transform.
The Discrete Fourier Transform (and its fast implementations in form of Fast Fourier Transforms) is however defined without a coordinate grid.
To turn it into a discretized approximation of the Fourier transform one needs to add coordinate grids which then add additional scale and phase factor before and after each (inverse) discrete Fourier transform.
Additionally the sample spacing and number of samples in position space define the sample spacing in frequency space and vice versa which usually needs to be ensured by hand.

FFTArray provides an easy to use general discretized Fourier transform by managing the coordinate grids in multiple dimensions which are ensured to always be correct.
Arrays with sampled values are combined with the dimension metadata as well in which space the values currently are:
```{code-cell} ipython3
import numpy as np
import fftarray as fa

dim_x = fa.dim_from_constraints(
    name="x",
    n=1024,
    pos_extent=2*np.pi,
    pos_min=0,
    freq_middle=0,
)
sin_x = fa.sin((50*2*np.pi)*x)
sin_x_in_freq_space = sin_x.into_space("freq")
```
For a quick getting started see [First steps](todo).

### Built for implementing spectral Fourier solvers

Spectral Fourier solvers like the split-step method require many consecutive (inverse) Fourier transforms.
In these cases the additional scale and phase factors can be optimized out.
By only applying these phase factors lazily FFTArray enables this use-case without performance impact while still enabling all the comforts of having always the correct phase factors applied.

### GPU support via the Python Array API Standard

Via the [Python Array API Standard](https://data-apis.org/array-api/latest/) FFTArray is able to support many different array libraries to enable for example hardware acceleration via GPUs.

## Installation

The required dependencies of FFTArray are kept small to ensure compatibility with many different environments.
For most use cases we recommend installing the optional constraint solver with the `z3` option:
```shell
pip install fftarray[z3]
```

Any array library besides NumPy like for example [JAX](https://github.com/jax-ml/jax?tab=readme-ov-file#installation) should be installed following their respective documentation.
Since each of them have different approaches on how to handle for example GPU support on different operating systems we do not recommend installing them via the optional dependency groups of FFTArray.



