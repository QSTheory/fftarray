# Working with JAX

This chapter contains some notes and tips about using FFTArray with [JAX](https://github.com/jax-ml/jax).
In general we assume familiarity with how JAX works and refer to their [documentation](https://docs.jax.dev/en/latest/).
When working in eager mode without using any tracing, JAX behaves very similarly to other array libraries with FFTArray.

Specifically for scientific computing we want to highlight that JAX by default enforces single-precision numbers.
Double precision numbers can be activated in multiple ways, see [JAX - The Sharp Bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision).
One way is to add the following snippet directly after import:
```python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
```


## Tracing

In order to be able to pass instances of the `Dimension` and `Array` class into and out of traced functions, one needs to register their PyTree implementations:
```python continuation
import fftarray as fa
try:
    fa.jax_register_pytree_nodes()
except ValueError:
    # Already registered.
    pass
```

## Fourier transforms, lazy evaluation and jax.lax.scan

The current `space`, `eager` and `factors_applied` attributes of an `Array` are static at trace time.
Therefore when passing an `Array` as a carry through scan, these attributes must match for the initial value and the return value.
For example:
```python continuation
from typing import Tuple

import pytest

dim_x = fa.dim(name="x", n=8, d_pos=0.4, pos_min=0, freq_min=0)
arr: fa.Array = fa.coords_from_dim(dim_x, "pos", xp=jnp)

def step_fun(arr: fa.Array, _) -> Tuple[fa.Array, None]:
    # This is a simple example for a more complicated operation which might change
    # the space and factors_applied of the array.
    arr = arr.into_space("freq") + 1.
    return arr, None

with pytest.raises(TypeError):
    arr_res, _ = jax.lax.scan(
        step_fun,
        init=arr,
        length=3,
    )
```
which results in this error:
```
Traceback (most recent call last):
  File [...], in <module>
    arr, _ = jax.lax.scan(
TypeError: scan body function carry input and carry output must have the same pytree
structure, but they differ:

The input carry arr is a <class 'fftarray._src.array.Array'> with pytree metadata
(('pos',), (False,), (True,), <module 'jax.numpy' from [...]>) but the corresponding
component of the carry output is a <class 'fftarray._src.array.Array'> with pytree
metadata (('freq',), (False,), (False,), <module 'jax.numpy' from [...]>),
so the pytree node metadata does not match.

Revise the function so that the carry output has the same pytree structure as the carry input.
```

The second part of `pytree metadata (('freq',), (False,), (False,)` lists the `space`, `eager` and `factors_applied` property of the returned `Array` per dimension which has to match the attributes of the input `Array`:
```python continuation
arr_res, _ = jax.lax.scan(
    step_fun,
    init=arr.into_space("freq").into_eager(False).into_factors_applied(False),
    length=3,
)
```

## Indexing
Currently, `fa.Array` indexing does not support dynamically traced indexers:
```python continuation
@jax.jit
def dynamic_indexing(arr, indexer: int):
    """
        This function passes the index through dynamically.
    """
    return arr.isel(x=indexer)

with pytest.raises(NotImplementedError):
    dynamic_indexing(arr, 3)

```

However, you can still perform indexing within jitted functions when using **static indexers**. You can achieve this by either using concrete values defined independently of your jitted function arguments or by marking the indexer argument as static.

```python continuation
# Concrete index value computed at trace time
from functools import partial

@jax.jit
def compile_time_indexing(arr):
    """
        The index is computed at compile time.
    """
    concrete_index_value = 1+2
    selected_by_index = arr.isel(x=concrete_index_value)
    selected_by_position = arr.sel(x=1.3, method="nearest")
    return selected_by_index+selected_by_position

compile_time_indexing(arr) # no error

# Static index value passed in statically at trace time.
@partial(jax.jit, static_argnames=["indexer", "pos_indexer"])
def static_indexer(arr, indexer: int, pos_indexer: float):
    "The indexer args are marked as static and therefore fixed at compile time."
    selected_by_index = arr.isel(x=indexer)
    selected_by_position = arr.sel(x=pos_indexer, method="nearest")
    return selected_by_index+selected_by_position

static_indexer(arr, 3, 1.3) # no error
```

## Dynamic Coordinate Systems
By default, all members of the `Dimension` class are marked as static during tracing.
Both, {py:func}`dim<fftarray.dim>` and {py:func}`dim_from_constraints<fftarray.dim_from_constraints>` have the optional parameter `dynamically_traced_coords: bool = False`.
When setting this to `True`, the members `d_pos`, `pos_min` and `freq_min` are turned into dynamic values during JAX tracing.
This enables for example to change coordinates during tracing:
```python continuation
dim_x_dynamic = fa.dim(
    name="x",
    n=8,
    d_pos=0.4,
    pos_min=0,
    freq_min=0,
    dynamically_traced_coords=True,
)

def step_fun_dim(dim_x_dynamic: fa.Dimension, _) -> Tuple[fa.Dimension, None]:
    # Modify the passed in dimension so that it is different in the next iteration.
    dim_x_dynamic = fa.dim(
        name="x",
        n=8,
        d_pos=dim_x_dynamic.d_pos,
        pos_min=dim_x_dynamic.pos_min + 0.1,
        freq_min=dim_x_dynamic.freq_min,
        dynamically_traced_coords=True,
    )
    return dim_x_dynamic, None

dim_x_dynamic, _ = jax.lax.scan(
    step_fun_dim,
    init=dim_x_dynamic,
    length=3,
)
```
However, this comes with some caveats:

- Only {py:func}`dim<fftarray.dim>` accepts tracer values because the z3 solver in {py:func}`dim_from_constraints<fftarray.dim_from_constraints>` requires concrete floats.
- Dynamic coordinates currently prevent indexing by position during tracing.
- `Dimension` objects returned from a traced function have JAX scalars as their attributes so all coordinate properties (except for `n` and the created coordinate grids) are then also JAX scalars and no longer Python scalars.
This can lead for example to interoperability problems when using them with other libraries.


Since the coordinates of `Dimension` are replaced by tracers they cannot be compared and therefore also not combined during tracing if they are passed in as separate arguments:
```python continuation
import pytest

fa.set_default_xp(jax.numpy)
dim_x = fa.Dimension("x", 4, 0.5, 0., 0., dynamically_traced_coords=True)

@jax.jit
def my_fun(dim1: fa.Dimension) -> fa.Array:
    arr1 = fa.coords_from_dim(dim1, "pos")
    arr2 = fa.coords_from_dim(dim1, "pos")

    # Works, because both arrays use the same dimension with the same tracers.
    return arr1+arr2

my_fun(dim_x)

@jax.jit
def my_fun_not_dynamic(dim1: fa.Dimension, dim2: fa.Dimension) -> fa.Array:
    arr1 = fa.coords_from_dim(dim1, "pos")
    arr2 = fa.coords_from_dim(dim2, "pos")

    # Addition requires all dimensions with the same name to be equal, this is explicitly checked before the operation.
    # The check for equality fails with a `jax.errors.TracerBoolConversionError` because the coordinate grids' values of the `Dimension`s are only known at runtime.
    # If `dynamically_traced_coords` above were set to False, the exact values of `dim1` and `dim2` were available at trace time and therefore this addition would succeed.
    return arr1+arr2



with pytest.raises(jax.errors.TracerBoolConversionError):
    my_fun_not_dynamic(dim_x, dim_x)
```

This can be solved by passing each `Dimension` instance exactly once into the jitted function.
Note that when passing the same `Dimension` object as part of two different `FFTArray` objects, each `Dimension` instance gets its own distinct tracer.
For example two `FFTArray` objects which contain a `Dimension` named `"x"` could not be combined inside a jitted function if they were passed in as parameters:
```python continuation
@jax.jit
def my_fun(arr1: fa.Array, arr2: fa.Array) -> fa.Array:
    # This check fails with a `jax.errors.TracerBoolConversionError`
    # because the equality of the two dimensions can only be determined
    # at run time.
    return arr1+arr2

dim_x = fa.Dimension("x", 4, 0.5, 0., 0., dynamically_traced_coords=True)
arr1 = fa.coords_from_dim(dim_x, "pos")
arr2 = fa.coords_from_dim(dim_x, "pos")
with pytest.raises(jax.errors.TracerBoolConversionError):
    my_fun(arr1, arr2)
```

