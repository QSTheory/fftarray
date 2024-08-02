The indexing behaviour and syntax as well as the following docmentation are inspired by xarray, see [xarray: Indexing and selecting data](https://docs.xarray.dev/en/stable/user-guide/indexing.html).

## Quick Overview
In total, FFTArray objects can be indexed via four different methods, two dimension lookup methods (positional, by name) and two index lookup methods (by integer, by label). In our case, label always refers to the FFTDimension coordinate values of type float (and should not be understood as any other objects like str).

The main difference between `FFTArray` indexing and `xarray` indexing lies in the fact that an `FFTArray` object can be in different internal states, i.e., each dimension can be either in position or frequency space and will always be indexed within this space. This is especially important when indexing by label, i.e., by position or frequency coordinate values.

In the following overview, we understand `fft_arr` as an FFTArray object with two dimensions in the order `("x", "y")`. FFTArray supports four possible ways of indexing, here shown by selecting the first `y` coordinate (`y=1e-3`).

Dimension lookup | Index lookup | FFTArray syntax |
--- | --- | --- |
Positional | By integer | `fft_arr[:,0]` |
Positional | By label | `fft_arr.loc[:,1e-3]` |
By name | By integer | `fft_arr[dict(y=0)]` or `arr.isel(y=0)` |
By name | By label | `fft_arr.loc[dict(y=1e-3)]` or `arr.sel(y=1e-3)` |

## Some Examples
```python
x_dim = FFTDimension(name="x", n=8, d_pos=0.4, pos_min=0, freq_min=0)
y_dim = FFTDimension(name="y", n=8, d_pos=0.4, pos_min=0, freq_min=0)
z_dim = FFTDimension(name="z", n=8, d_pos=0.4, pos_min=0, freq_min=0)

backend = NumpyBackend()
fft_arr = (
    x_dim.fft_array(space="pos", backend=backend) +
    y_dim.fft_array(space="pos", backend=backend) +
    z_dim.fft_array(space="pos", backend=backend)
)
```
### Indexing by Integer
First, we will have a look at indexing by integer for both dimension lookup methods: positional and by name. Positional indexing by integer is the most common form of indexing that you also know from other array libraries such as `numpy`. Here, we can additionally look up the dimension by name as each of our `FFTArray` dimensions has a unique name (supplied by the `FFTDimension` name).
```python
# The following ways of indexing all have the same result.
# Each of them reduces the x-dimension to index 3
# and slices the y-dimension from index 1 to (not including) 5.
# The third z-dimension is not indexed and therefore fully kept.
indexed_fft_arr: FFTArray = fft_arr[3,:5,:]
indexed_fft_arr: FFTArray = fft_arr[3,:5]
indexed_fft_arr: FFTArray = fft_arr[3,:5,...] # ellipsis fills up non-mentioned dimensions (useful for only indexing late or early dimensions)
indexed_fft_arr: FFTArray = fft_arr[dict(x=3, y=slice(None,5))]
indexed_fft_arr: FFTArray = fft_arr.isel(x=3, y=slice(None,5))
indexed_fft_arr: FFTArray = fft_arr.isel(dict(x=3, y=slice(None,5)))

# There are some special cases to keep in mind when indexing by integer.
# These examples also evaluate to the same value as above.
indexed_fft_arr: FFTArray = fft_arr[dict(x=3, y=slice(0,5))] # slice(None,x) = slice(0,x)
indexed_fft_arr: FFTArray = fft_arr[dict(x=-5, y=slice(0,5))] # for array with dim.n = 8: index -5 = index 3
indexed_fft_arr: FFTArray = fft_arr[dict(x=3, y=slice(-100,5))] # slice objects with start < -dim.n are mapped to None
indexed_fft_arr: FFTArray = fft_arr[dict(x=3, y=slice(0,-3))] # slice indices are individually mapped to a valid region if possible
```
### Indexing by Label/Coordinate
Now, we will have a look at indexing by label for both dimension lookup methods: positional and by name. Label indexing is applicable to FFTArrays because our dimensions have coordinate values (of type float) which are as unique as indices. As above, we can additionally look up the dimension by name as each of our `FFTArray` dimensions has a unique name (supplied by the `FFTDimension` name).
```python
# The following ways of indexing return exactly the same result.
# Each of them reduces the x-dimension to coordinate 1.2
# and slices the y-dimension to only include coordinates between 1 and 2.5.
# The third z-dimension is not indexed and therefore fully kept.

indexed_fft_arr: FFTArray = fft_arr.loc[1.2,1:2.5,:]
indexed_fft_arr: FFTArray = fft_arr.loc[1.2,1:2.5]
indexed_fft_arr: FFTArray = fft_arr.loc[1.2,1:2.5,...]
indexed_fft_arr: FFTArray = fft_arr.loc[dict(x=1.2, y=slice(1,2.5))]
indexed_fft_arr: FFTArray = fft_arr.sel(x=1.2, y=slice(1,2.5))
indexed_fft_arr: FFTArray = fft_arr.sel(dict(x=1.2, y=slice(1,2.5)))

# Additionally, with fft_arr.sel one can also choose a method for coordinate search.
# This leads to the following indexing commands yielding the same result.
# However, we do not support a method when using slice objects for indexing.
# Please keep in mind that in these examples we always index in
# position space where all dimensions have d_pos = 0.4.
indexed_fft_arr: FFTArray = fft_arr.sel(x=0.4)
indexed_fft_arr: FFTArray = fft_arr.sel(x=0.3, method="nearest")
indexed_fft_arr: FFTArray = fft_arr.sel(x=0.5, method="nearest")
indexed_fft_arr: FFTArray = fft_arr.sel(x=0.1, method="bfill") # equivalent: method="backfill"
indexed_fft_arr: FFTArray = fft_arr.sel(x=0.3, method="bfill") # equivalent: method="backfill"
indexed_fft_arr: FFTArray = fft_arr.sel(x=0.4, method="bfill") # equivalent: method="backfill"
indexed_fft_arr: FFTArray = fft_arr.sel(x=0.5, method="ffill") # equivalent: method="pad"
indexed_fft_arr: FFTArray = fft_arr.sel(x=0.75, method="ffill") # equivalent: method="pad"
indexed_fft_arr: FFTArray = fft_arr.sel(x=0.4, method="ffill") # equivalent: method="pad"

```
## Some Sharp JAX-Bits
Currently, `FFTArray` indexing does not support dynamically traced indexers.

This is what you can **not** do. This applies to all indexing methods.

```python

fft_arr: FFTArray

def indexing(fft_arr, indexer: int):
    return fft_arr.isel(x=indexer)

dynamic_indexer = jax.jit(indexing)

dynamic_indexer(fft_arr, 3)  # NotImplementedError

```

However, you can still perform indexing **by integer** within jitted functions when using **static indexers**. You can achieve this by either using concrete values defined independently of your jitted function arguments or by marking the indexer argument as static.

```python

fft_arr: FFTArray

### Concrete index value
concerete_index_value = 3
def indexing(fft_arr):
    return fft_arr.isel(x=concerete_index_value)

static_indexer = jax.jit(indexing)

static_indexer(fft_arr) # no error

### Mark index arg as static
def indexing(fft_arr, indexer: int):
    return fft_arr.isel(x=indexer)

static_indexer = jax.jit(indexing, static_argnames="indexer")

static_indexer(fft_arr, 3) # no error

```
