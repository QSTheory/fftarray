
import pytest
import numpy as np
import jax

from fftarray.fft_array import FFTDimension
from fftarray.backends.tensor_lib import TensorLib
from fftarray.backends.jax_backend import JaxTensorLib
from fftarray.backends.np_backend import NumpyTensorLib
from fftarray.backends.pyfftw_backend import PyFFTWTensorLib
from fftarray.xr_helpers import as_xr_pos

jax.config.update("jax_enable_x64", True)

TENSOR_LIBS = [NumpyTensorLib, JaxTensorLib, PyFFTWTensorLib]

"""
Relevant functions/classes for indexing
- class LocFFTArrayIndexer
- method FFTArray.__getitem__
- property FFTArray.loc = LocFFTArrayIndexer(self)
- method FFTArray.sel
- method FFTArray.isel
- method FFTDimension._index_from_coord
- method FFTDimension._dim_from_slice
- method FFTDimension._dim_from_start_and_n
"""

@pytest.mark.parametrize("tlib_class", TENSOR_LIBS)
@pytest.mark.parametrize("do_jit", [False, True])
def test_fftdim_single_element_indexing(tlib_class: TensorLib, do_jit: bool) -> None:
    if do_jit and type(tlib_class) != JaxTensorLib:
        return

    dim = FFTDimension("x",
        n=4,
        d_pos=1,
        pos_min=0.5,
        freq_min=0.,
    )

    def test_functions(dim):
        return (
            dim._index_from_coord(0.5, method = None, space="pos") == 0,
            dim._index_from_coord(2.5, method = None, space="pos") == 2,
            dim._index_from_coord(0.4, method = "nearest", space="pos") == 0,
            dim._index_from_coord(2.6, method = "nearest", space="pos") == 2,
        )

    if do_jit:
        test_functions = jax.jit(test_functions)

    results = test_functions(dim)

    assert results[0] == 0
    assert results[1] == 2
    assert results[2] == 0
    assert results[3] == 2

@pytest.mark.parametrize("tlib_class", TENSOR_LIBS)
@pytest.mark.parametrize("do_jit", [False, True])
def test_1d_fftarray_indexing(tlib_class: TensorLib, do_jit: bool) -> None:
    if do_jit and type(tlib_class) != JaxTensorLib:
        return

    dim = FFTDimension("x",
        n=4,
        d_pos=1,
        pos_min=0.5,
        freq_min=0.,
    )
    fftarray_1d = dim.fft_array(space="pos", tlib=tlib_class())

@pytest.mark.parametrize("tlib", TENSOR_LIBS)
@pytest.mark.parametrize("do_jit", [False, True])
def test_indexing(tlib, do_jit: bool):
    if do_jit and type(tlib) != JaxTensorLib:
        return

    x_dim = FFTDimension("x",
        n=4,
        d_pos=1,
        pos_min=0.5,
        freq_min=0.,
        default_tlib=tlib(precision="default"),
    )
    y_dim = FFTDimension("y",
        n=4,
        d_pos=2,
        pos_min=-2,
        freq_min=0.,
        default_tlib=tlib(precision="default"),
    )

    arr_2d = x_dim.fft_array(space="pos") + y_dim.fft_array(space="pos")**2
    xr_arr = as_xr_pos(arr_2d)

    assert np.array_equal(arr_2d.values[0:3:2,:], xr_arr.values[0:3:2,:])
    assert np.array_equal(
        arr_2d.isel(x=1,y=slice(0,2,None)).transpose("x", "y"),
        xr_arr.isel(x=1,y=slice(0,2,None)).expand_dims({"x": 1}).transpose("x", "y")
    )
    assert np.array_equal(
        arr_2d.sel(x=(1,3),y=3.4, method="nearest").transpose("x", "y"),
        xr_arr.sel(y=3.4, method="nearest")
            .where(xr_arr.x > 1, drop=True)
            .where(xr_arr.x < 3, drop=True)
            .expand_dims({"y": 1}).transpose("x", "y")
    )

    assert np.array_equal(
        arr_2d.loc[:, 0].transpose("x", "y").values,
        xr_arr.loc[:,0].expand_dims({"y": 1}).transpose("x", "y")
    )
    assert np.array_equal(
        arr_2d.loc[(1,3), 2].transpose("x", "y").values,
        xr_arr.sel(y=2)
            .where(xr_arr.x > 1, drop=True)
            .where(xr_arr.x < 3, drop=True)
            .expand_dims({"y": 1}).transpose("x", "y")
    )

    def test_jittable(arr_2d):
        return (
            arr_2d.sel(x=1,y=3.4, method="nearest"),
            arr_2d.sel(x=-100,y=3.4, method="nearest"),
            arr_2d.loc[:],
            arr_2d.isel(x=3, y=2),
        )
    if do_jit:
        test_jittable = jax.jit(test_jittable)

    jit_res = test_jittable(arr_2d=arr_2d)
    assert np.array_equal(jit_res[2], xr_arr.sel(x=1,y=3.4, method="nearest").expand_dims({"x": 1, "y": 1}))
    assert np.array_equal(jit_res[3], xr_arr.sel(x=-100,y=3.4, method="nearest").expand_dims({"x": 1, "y": 1}))
    assert np.array_equal(jit_res[4], xr_arr.loc[:])
    assert np.array_equal(jit_res[5], xr_arr.isel(x=3, y=2).expand_dims({"x": 1, "y": 1}))
