---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: dev
  language: python
  name: python3
---

# First steps

To get started with FFTArray first create a `Dimension`:
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
print(dim_x)
```
The `Dimension` object `dim_x` encapsulates a discretely sampled dimension in both position and frequency space.
Internally it only stores `n`, `d_pos`, `pos_min` and `freq_min`, but can compute many other parameters about the sampled grids, for more details see {py:class}`fftarray.Dimension`.

{py:func}`dim_from_constraints <fftarray.dim_from_constraints>` accepts any combination of grid parameters which yields a unique solution for the position and frequency grid.
In the example above the constraint solver automatically computes `d_pos` from the passed in `n` and `pos_extent`.

Now let us create an actual array of values:
```{code-cell} ipython3
x: fa.Array = fa.coords_from_dim(dim_x, "pos")
print(x)
```
{py:func}`coords_from_dim <fftarray._src.creation_functions.coords_from_dim>` in this case fills each value in the array with its coordinate along the `x`.
For other ways of creating arrays see the {py:mod}`Array creation functions<fftarray._src.creation_functions>`.

The `x` variable can now be treated like x in an analytic expression.
We can for example compute the sine of `x`:
```{code-cell} ipython3
sin_x = fa.sin((50*2*np.pi)*x)
```

And now we can get its frequency representation with a simple `sin_x.into_space("freq")`.
For plotting we need to "unwrap" the `fa.Array` objects into plain NumPy arrays, which we can do with the {py:func}`values <fftarray.Array.values>` method:
```{code-cell} ipython3
from bokeh.plotting import figure, show
from bokeh.io import output_notebook

output_notebook(hide_banner=True)

fig = figure(
    width=400,
    height=200,
    x_axis_label="Frequency [Hz]",
    y_axis_label="Amplitude",
)

fig.line(
    # dim_x.values("freq") is a little helper to directly
    # get a bare array containing the coordinates of the dimension.
    x=dim_x.values("freq"),
    y=np.abs(sin_x.into_space("freq").values("freq"))
)
show(fig)
```
The above plot shows two peaks at 50Hz and -50Hz.
This is not due to discretization but also a property of the continuous Fourier Transform:
The Fourier transform $G(f): \mathbb{R} \to \mathbb{C}$ of a real-valued function $g(x): \mathbb{R} \to \mathbb{R}$ is conjugate symmetric, i.e., $G(f)=\overline{G(-f)}$.

## Notes on aliasing and periodicity

When sampling a function at discrete points one needs to take care that the sample spacing is small enough to capture all high frequency detail of the sampled function.
This is formalized in the Nyquist Shannon Sampling Theorem.
If the sampling is too coarse, high frequencies appear as wrong low frequency aliases in the sampled function.
Additionally each discretely sampled function is periodic in the other space.
Depending on what operations are done with such a sampled function this periodicity can appear as boundary effects where for example a moving wave function which exits during propagation the domain on one side appears again on the other side.
This can also happen in frequency space, if it is for example accelerated by a potential.
For more notes and further literature about the sampling theorem we refer to the second chapter of the [FFTArray publication](unpublished)

## Next steps

The examples show more advanced usage of FFTArray like working with multiple dimensions, computing derivatives and implementing atom optical simulations.
For usage of FFTArray with quantum mechanical problems like atom optics we also recommend checking out our [matterwave](https://github.com/QSTheory/matterwave) library which is a collection of helpers built on top of FFTArray.