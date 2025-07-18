# FFTArray: A Python Library for the Implementation of Discretized Multi-Dimensional Fourier Transforms

[**Intro**](#intro) | [**Installation**](#installation)

## Intro
FFTArray is a Python library that handles multidimensional arrays and their representation in dual spaces (original and frequency domain) and provides the following highlight features:
- **From formulas to code**: The user can directly map analytical equations involving Fourier transforms to code without mixing discretization details with physics. This enables rapid prototyping of diverse physical models and solver strategies.
- **Seamless multidimensionality**: Dimensions are broadcast by name which enables a uniform API to seamlessly transition from single- to multi-dimensional systems.
- **High performance**: Avoidable scale and phase factors in the Fourier transform are automatically skipped. Via the [Python Array API Standard](https://data-apis.org/array-api/latest/), FFTArray supports many different array libraries to enable for example hardware acceleration via GPUs.

Below we give a quick introduction to the basic functionality of the library.
For a more thorough description of FFTArray, we recommend reading the [publication](todo) and the [documentation](https://qstheory.github.io/fftarray/main).

### Adding Coordinate Grids to the FFT

The continuous Fourier transform is defined as:

$$
\begin{aligned}
    \mathcal{F}&: \ G(f) = \int_{-\infty}^{\infty}dx \ g(x)\ e^{- 2 \pi i fx},\quad \forall\ f\in \mathbb R,\\
    \widehat{\mathcal{F}}&: \ g(x) = \int_{-\infty}^{\infty}df\ G(f)\ e^{2 \pi i fx},\quad \forall\ x \in \mathbb R.
\end{aligned}
$$

When discretizing it on a finite grid in position and frequency space, one does not only get the Fast Fourier transform (FFT) but some additional phase and scale factors:

$$
\begin{aligned}
    x_n &:= x_\mathrm{min} + n  \Delta x, \quad n = 0, \ldots, N-1 ,\\
    \quad f_m &:= f_\mathrm{min} + m \Delta f, \quad m = 0, \ldots, N-1,
\end{aligned}
$$

$$
\begin{aligned}
    \text{(gdFT)} \quad G_m
    &= \Delta x \ \sum_{n=0}^{N-1} g_n \ \exp \left({-2 \pi i \ \left( f_\mathrm{min} + m \Delta f \right) \left( x_\mathrm{min} + n \Delta x \right) }\right) \\
    &= \Delta x
        \ {\textcolor{green}{\exp \left({\textcolor{green}{-} 2\pi i \ x_\mathrm{min} \  m \Delta f}\right)}}
        \ {\textcolor{green}{\exp \left({\textcolor{green}{-} 2\pi i \ x_\mathrm{min} \ f_\mathrm{min}}\right)}}
        \ \ \textcolor{black}{\mathrm{fft}} \left(
            g_n \ {\textcolor{green}{\exp \left({\textcolor{green}{-} 2\pi i \ f_\mathrm{min} \ n \Delta x}\right)}}
        \right),
\end{aligned}
$$

$$
\begin{aligned}
    \text{(gdIFT)} \quad g_n
    &= \Delta f \ \sum_{m=0}^{N-1} G_m \ \exp  \left({2 \pi i \ \left( f_\mathrm{min} + m \Delta f \right) \left( x_\mathrm{min} + n \Delta x \right) } \right) \\
    &= {\textcolor{green}{\exp \left({\textcolor{green}{+} 2\pi i \ f_\mathrm{min} \ n \Delta x}\right)}}
        \ \ \textcolor{black}{\mathrm{ifft}} \left(
            G_m \ {\textcolor{green}{\exp \left({\textcolor{green}{+} 2\pi i \ x_\mathrm{min} \  m \Delta f}\right)}}
            \ {\textcolor{green}{\exp \left({\textcolor{green}{+} 2\pi i \ x_\mathrm{min} \ f_\mathrm{min}}\right)}} / \Delta x
        \right).
\end{aligned}
$$

$\mathrm{fft}$ and $\mathrm{ifft}$ follow here the definition of NumPy's default behavior in its [Discrete Fourier Transform](https://numpy.org/doc/stable/reference/routines.fft.html) module.


Keeping track of these coordinate-dependent scale and phase factors is tedious and error-prone.
Additionally the sample spacing and number of samples in position space define the sample spacing in frequency space and vice versa via $1 = N \Delta x \Delta f$ which usually needs to be ensured by hand.

FFTArray automatically takes care of these and provides an easy to use general discretized Fourier transform by managing the coordinate grids in multiple dimensions, ensuring those are always correct.
Arrays with sampled values are combined with the dimension metadata as well as in which space the values currently are:
```python
import fftarray as fa

dim_x = fa.dim_from_constraints(name="x", n=1024, pos_middle=0., d_pos=0.01, freq_middle=0)
dim_y = fa.dim_from_constraints(name="y", n=2048, pos_min=-5., pos_max=6.,freq_middle=0)

arr_x = fa.coords_from_dim(dim_x, "pos")
arr_y = fa.coords_from_dim(dim_y, "pos")

arr_gauss_2d = fa.exp(-(arr_x**2 + arr_y**2)/0.2)
arr_gauss_2d_in_freq_space = arr_gauss_2d.into_space("freq")
```
For a quick getting started, see [First steps](https://qstheory.github.io/fftarray/main/first_steps.html).

### Built for Implementing Spectral Fourier Solvers

Spectral Fourier solvers like the split-step method require many consecutive (inverse) Fourier transforms.
In these cases the additional scale and phase factors can be optimized out.
By only applying these phase factors lazily, FFTArray handles this use-case with minimal performance impact while still providing the comfort of ensuring the application of all required phase factors.
For quantum mechanics, especially for simulating matter waves, the [matterwave package](https://github.com/QSTheory/matterwave) provides a collection of helpers built on top of FFTArray.

## Installation

The required dependencies of FFTArray are kept minimal to ensure compatibility with different environments.
For most use cases we recommend installing the optional constraint solver for easy Dimension definition with the `dimsolver` option:
```shell
pip install fftarray[dimsolver]
```

Any array library besides NumPy like for example [JAX](https://github.com/jax-ml/jax?tab=readme-ov-file#installation) should be installed following their respective documentation.
Since each of them have different approaches on how to handle for example GPU support on different operating systems we do not recommend installing them via the optional dependency groups of FFTArray.



