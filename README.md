# FFTArray: A Python Library for the Implementation of Discretized Multi-Dimensional Fourier Transforms

[**Intro**](#intro) | [**Installation**](#installation)

## Intro
FFTArray is a Python package that provides at a high level:
- **From formulas to code**: The user can directly map analytical equations involving Fourier transforms to code without mixing discretization details with physics. This enables rapid prototyping of diverse physical models and solver strategies.
- **Seamless multidimensionality**: Dimensions are broadcast by name which enables a uniform API to seamlessly transition from single- to multi-dimensional systems.
- **High performance**: Unecessary scale and phase factors are automatically skipped and via the [Python Array API Standard](https://data-apis.org/array-api/latest/) FFTArray is able to support many different array libraries to enable for example hardware acceleration via GPUs.

Below is very quick introduction to get started with the library.
For a more thorough description of the library we recommend reading the [publication](todo) and the [documentation](todo).

### Adding Coordinate Grids to the FFT

$$
\newcommand{\freqShift}[1]{\textcolor{orange}{e^{#1 i \freqMin \, n \Delta x}}}
\newcommand{\posL}[0]{x}
\newcommand{\posIdx}[0]{n}
\newcommand{\posD}[0]{\posL_\posIdx}
\newcommand{\posMin}[0]{\posL_\mathrm{min}}

\newcommand{\freqL}[0]{f}
\newcommand{\freqIdx}[0]{m}
\newcommand{\freqD}[0]{\freqL_\freqIdx}
\newcommand{\freqMin}[0]{\freqL_\mathrm{min}}

\newcommand{\samplesPos}[0]{g_n}
\newcommand{\samplesPosFFT}[0]{g_n^\mathrm{fft}}
\newcommand{\samplesPosInt}[2]{#1_n^\mathrm{int}(#2)}
%\newcommand{\samplesPosInt}[1]{g_n^#1}
\newcommand{\samplesFreq}[0]{G_m}
\newcommand{\samplesFreqFFT}[0]{G_m^\mathrm{fft}}
\newcommand{\samplesFreqInt}[2]{#1_m^\mathrm{int}(#2)}
%\newcommand{\samplesFreqInt}[1]{G_m^#1}
%\newcommand{\freqCan}{-\left\lfloor{0.5N}\right\rfloor \, \Delta \freqL}
\newcommand{\freqCan}{-\mathrm{floor}(0.5N) \, \Delta \freqL}
\newcommand{\freqCanSym}{\freqMin^\text{sym}}
\newcommand{\posCan}{-\mathrm{floor}(0.5N) \, \Delta \posL}
\newcommand{\posCanSym}{\posMin^\text{sym}}

\newcommand{\fou}[0]{\mathcal{F}}
\newcommand{\ifou}[0]{\widehat{\mathcal{F}}}
\newcommand{\fft}[0]{\texttt{fft}_m}
\newcommand{\ifft}[0]{\texttt{ifft}_m}

% DFT (and numpy) phase factor definition
\newcommand{\dftNoNExp}[1]{\textcolor{green}{e^{#1 2\pi i \ m \Delta \freqL \ n \Delta \posL}}}
\newcommand{\fftExp}[1]{\textcolor{green}{e^{#1 2\pi i \ \frac{m n}{N}}}}
\newcommand{\fftExpPM}[0]{\textcolor{green}{e^{\pm 2\pi i \ \frac{m n}{N}}}}

% three non-DFT phase factor definitions
\newcommand{\freqZeroShift}[1]{\textcolor{green}{e^{\textcolor{green}{#1} 2\pi i \ \freqMin \ n \Delta x}}}
\newcommand{\xZeroShift}[1]{\textcolor{green}{e^{\textcolor{green}{#1} 2\pi i \ \posMin \  m \Delta \freqL}}}
\newcommand{\phaseCorrection}[1]{\textcolor{green}{e^{\textcolor{green}{#1} 2\pi i \ \posMin \ \freqMin}}}

\newcommand{\xShift}[0]{x_\mathrm{shift}}
\newcommand{\fShift}[0]{f_\mathrm{shift}}
$$

The continuous Fourier transform is defined as:
$$
\begin{align}
    \fou&: \ G(\freqL) = \int_{-\infty}^{\infty}dx \ g(x)\ e^{- 2 \pi i \freqL\posL},\quad \forall\ \freqL\in \mathbb R,\\
    \ifou&: \ g(\posL) = \int_{-\infty}^{\infty}d\freqL\ G(\freqL)\ e^{2 \pi i \freqL\posL},\quad \forall\ x \in \mathbb R.
\end{align}
$$

When discretizing it on a finite grid in position and frequency space one does not only get a fast Fourier transform (FFT) but also some additional phase and scale factors:
$$
\begin{align}
    \posD &\coloneqq \posMin + \posIdx  \Delta \posL, \quad \posIdx = 0, \ldots, N-1 ,\\
    \quad \freqD &\coloneqq \freqMin + \freqIdx \Delta \freqL, \quad \freqIdx = 0, \ldots, N-1,
\end{align}
$$

$$
\begin{align}
    \text{(gdFT)} \quad \samplesFreq
    &= \Delta \posL \ \sum_{n=0}^{N-1} \samplesPos \ e^{-2 \pi i \ \left( \freqMin + \freqIdx \Delta \freqL \right) \left( \posMin + \posIdx \Delta \posL \right) } \\
    % &= \Delta \posL
    %     \ \xZeroShift{-}
    %     \ \phaseCorrection{-}
    %     \ \sum_{n=0}^{N-1} \samplesPos\
    %     \ \underbrace{\dftNoNExp{-}}_{\fftExp{-}}
    %     \ \freqZeroShift{-} \\
        % \\
    &= \Delta \posL
        \ \xZeroShift{-}
        \ \phaseCorrection{-}
        \ \textcolor{black}{\fft} \left(
            \samplesPos \ \freqZeroShift{-}
        \right),
\end{align}
$$

$$
\begin{align}
    \text{(gdIFT)} \quad \samplesPos
    &= \Delta \freqL \ \sum_{m=0}^{N-1} \samplesFreq \ e^{2 \pi i \ \left( \freqMin + \freqIdx \Delta \freqL \right) \left( \posMin + \posIdx \Delta \posL \right) } \\
    % &= \Delta \freqL
    %     \ \freqZeroShift{+}
    %     %\ N
    %     %\ \frac{1}{N}
    %     \ \sum_{m=0}^{N-1} \samplesFreq \
    %     \ \underbrace{\dftNoNExp{+}}_{\fftExp{+}}
    %     \ \xZeroShift{+}
    %     \ \phaseCorrection{+} \\
    &= \freqZeroShift{+}
        \ \textcolor{black}{\ifft} \left(
            \samplesFreq \ \xZeroShift{+} \ \phaseCorrection{+} / \Delta \posL
        \right).
\end{align}
$$

Keeping track of these coordinate-dependent scale and phase factors is tedious and error-prone.
Additionally the sample spacing and number of samples in position space define the sample spacing in frequency space and vice versa via $1 = N \Delta \posL \Delta \freqL$ which usually needs to be ensured by hand.

FFTArray automatically takes care of these and provides an easy to use general discretized Fourier transform by managing the coordinate grids in multiple dimensions which are ensured to always be correct.
Arrays with sampled values are combined with the dimension metadata as well in which space the values currently are:
```{code-cell} ipython3
import numpy as np
import fftarray as fa

dim_x = fa.dim_from_constraints(name="x", n=1024, pos_extent=12., pos_min=-7., freq_middle=0)
dim_y = fa.dim_from_constraints(name="y", n=2048, pos_min=-5., pos_max=6.,freq_middle=0)

arr_x = fa.coords_from_dim(dim_x, "pos")
arr_y = fa.coords_from_dim(dim_y, "pos")

arr_gauss_2d = fa.exp(-(arr_x**2 + arr_y**2)/0.2)
arr_gauss_2d_in_freq_space = arr_gauss_2d.into_space("freq")
```
For a quick getting started see [First steps](todo).

### Built for implementing spectral Fourier solvers

Spectral Fourier solvers like the split-step method require many consecutive (inverse) Fourier transforms.
In these cases the additional scale and phase factors can be optimized out.
By only applying these phase factors lazily FFTArray enables this use-case with minimal performance impact while still enabling all the comforts of having always the correct phase factors applied.
For quantum mechanics especially of matter waves the [matterwave package](https://github.com/QSTheory/matterwave) provides a collection of helpers built on top of FFTArray.

## Installation

The required dependencies of FFTArray are kept small to ensure compatibility with many different environments.
For most use cases we recommend installing the optional constraint solver for easy Dimension with the `z3` option:
```shell
pip install fftarray[z3]
```

Any array library besides NumPy like for example [JAX](https://github.com/jax-ml/jax?tab=readme-ov-file#installation) should be installed following their respective documentation.
Since each of them have different approaches on how to handle for example GPU support on different operating systems we do not recommend installing them via the optional dependency groups of FFTArray.



