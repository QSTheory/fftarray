
from functools import reduce
import itertools
from typing import Hashable, List, Literal, Mapping, Optional, Tuple, Union
import pytest
import numpy as np
import jax
import xarray as xr

from fftarray.fft_array import FFTArray, FFTDimension, Space
from fftarray.backends.tensor_lib import TensorLib
from fftarray.backends.jax_backend import JaxTensorLib
from fftarray.backends.np_backend import NumpyTensorLib
from fftarray.backends.pyfftw_backend import PyFFTWTensorLib
from fftarray.xr_helpers import as_xr_pos

jax.config.update("jax_enable_x64", True)

TENSOR_LIBS = [NumpyTensorLib, JaxTensorLib, PyFFTWTensorLib]

TEST_FFTDIM = FFTDimension(
    name="x", n=8, d_pos=1, pos_min=0, freq_min=0
)
STANDARD_TEST_DATAARRAY = xr.DataArray(
    data=np.linspace(0, 7, num=8),
    dims=["x"],
    coords={'x': np.linspace(0, 7, num=8)},
)

pos_values = TEST_FFTDIM.pos_min + np.arange(TEST_FFTDIM.n)*TEST_FFTDIM.d_pos
freq_values = TEST_FFTDIM.freq_min + np.arange(TEST_FFTDIM.n)*TEST_FFTDIM.d_freq

STANDARD_TEST_DATASET = xr.Dataset(
    data_vars={
        "pos": (["pos_coord"], pos_values),
        "freq": (["freq_coord"], freq_values),
    },
    coords={
        "pos_coord": pos_values,
        "freq_coord": freq_values,
    }
)

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

    tlib = tlib_class()

    dim = FFTDimension("x",
        n=4,
        d_pos=1,
        pos_min=0.5,
        freq_min=0.,
    )

    def test_functions(dim):
        return (
            dim._index_from_coord(0.5, method = None, space="pos", tlib=tlib),
            dim._index_from_coord(2.5, method = None, space="pos", tlib=tlib),
            dim._index_from_coord(0.4, method = "nearest", space="pos", tlib=tlib),
            dim._index_from_coord(2.6, method = "nearest", space="pos", tlib=tlib),
        )

    if do_jit:
        test_functions = jax.jit(test_functions)

    results = test_functions(dim)

    assert results[0] == 0
    assert results[1] == 2
    assert results[2] == 0
    assert results[3] == 2

valid_test_slices = [
    slice(None, None), slice(0, None), slice(None, -1), slice(-8, None),
    slice(1,4), slice(-3,-1), slice(-3,6), slice(-1, None), slice(None,20)
]

@pytest.mark.parametrize("valid_slice", valid_test_slices)
@pytest.mark.parametrize("space", ["pos", "freq"])
# TODO: test also for do_jit == True
@pytest.mark.parametrize("do_jit", [False])
def test_valid_fftdim_dim_from_slice(do_jit: bool, space: Space, valid_slice: slice) -> None:

    if do_jit:
        @jax.jit
        def test_function(_slice):
            return TEST_FFTDIM._dim_from_slice_jax(range=_slice, space=space)
    else:
        def test_function(_slice):
            return TEST_FFTDIM._dim_from_slice(range=_slice, space=space)

    result_dim = test_function(valid_slice)

    assert np.array_equal(
        result_dim.np_array(space),
        TEST_FFTDIM.np_array(space)[valid_slice]
    )

invalid_slices = [
    slice(1, 1), slice(1, 0), slice(-2, 0), slice(7, -1),
    slice(0, 6, 2), slice(None, None, 2), slice(0., 5.),
    slice(10,None)
]

@pytest.mark.parametrize("space", ["pos", "freq"])
@pytest.mark.parametrize("invalid_slice", invalid_slices)
def test_errors_fftdim_dim_from_slice(space: Space, invalid_slice: slice) -> None:

    with pytest.raises(IndexError):
        TEST_FFTDIM._dim_from_slice(invalid_slice, space=space)


coord_test_samples = [
    -5, -1.5, -1, -0.5, 0, 0.3, 0.5, 0.7, 1, 1.3, 7.5, 8, 8.5, 9,
    (-5,10), (None, None), (0,7), (0,8), (0.5,0.1)
]

@pytest.mark.parametrize("tlib_class", TENSOR_LIBS)
@pytest.mark.parametrize("method", ["nearest", "pad", "ffill", "backfill", "bfill", None])
@pytest.mark.parametrize("valid_coord", coord_test_samples)
@pytest.mark.parametrize("space", ["pos", "freq"])
@pytest.mark.parametrize("do_jit", [True, False])
def test_valid_index_from_coord(
    do_jit: bool,
    space: Space,
    valid_coord: Union[float,slice],
    method: Literal["nearest", "pad", "ffill", "backfill", "bfill", None],
    tlib_class: TensorLib
) -> None:

    if do_jit:
        tlib = tlib_class()
        if isinstance(tlib, JaxTensorLib) and method=='nearest':
            @jax.jit
            def test_function(_coord):
                return TEST_FFTDIM._index_from_coord(coord=_coord, space=space, method=method, tlib=tlib_class())
        else:
            return
    else:
        def test_function(_coord):
            return TEST_FFTDIM._index_from_coord(coord=_coord, space=space, method=method, tlib=tlib_class())

    try:
        try:
            dim_index_result = test_function(valid_coord)
        except (KeyError, NotImplementedError) as e:
            dim_index_result = type(e)
        try:
            if isinstance(valid_coord, tuple):
                valid_coord = slice(valid_coord[0], valid_coord[1])
            xr_result_coord = STANDARD_TEST_DATASET[space].sel({f"{space}_coord": valid_coord}, method=method)
            xr_result_dim_index = STANDARD_TEST_DATASET[space].isel({f"{space}_coord": dim_index_result})
            np.testing.assert_array_equal(
                xr_result_coord.data,
                xr_result_dim_index.data
            )
        except (KeyError, NotImplementedError) as e:
            xr_result = type(e)
            assert dim_index_result == xr_result
    except Exception as e:
        raise e

def make_xr_indexer(indexer, space: Space):
    return {
        f"{name}_{space}": [index] if isinstance(index, int) else index
        for name, index in indexer.items()
    }

integer_indexers_test_samples = [
    {"x": 1, "y": 1, "z": 1}, {"x": 1, "y": 1, "z": slice(None, None)},
    {"x": 1, "y": 1}, {"x": -20}, {"z": 5}, {"random": 1}, {},
    {"x": slice(-20,5), "y": slice(-6,6), "z": slice(None, 4)}
]

@pytest.mark.parametrize("indexers", integer_indexers_test_samples)
@pytest.mark.parametrize("tlib_class", TENSOR_LIBS)
@pytest.mark.parametrize("space", ["pos", "freq"])
# TODO: think about making this also jittable
@pytest.mark.parametrize("do_jit", [False])
def test_3d_fft_array_indexing_by_integer(
    do_jit: bool,
    space: Space,
    tlib_class: TensorLib,
    indexers: Optional[Mapping[Hashable, Union[int, slice]]],
) -> None:

    fft_array, xr_dataset = generate_test_fftarray_xrdataset(
        ["x", "y", "z"],
        dimension_length=8,
        tlib=tlib_class()
    )

    if do_jit:
        tlib = tlib_class()
        if isinstance(tlib, JaxTensorLib):
            @jax.jit
            def test_function_isel(_indexers) -> FFTArray:
                return fft_array.into(space=space).isel(_indexers)
            @jax.jit
            def test_function_square_brackets(_indexers) -> FFTArray:
                return fft_array.into(space=space)[_indexers]
        else:
            return
    else:
        def test_function_isel(_indexers) -> FFTArray:
            return fft_array.into(space=space).isel(_indexers)
        def test_function_square_brackets(_indexers) -> FFTArray:
            return fft_array.into(space=space)[_indexers]

    try:
        fft_array_result_isel = test_function_isel(indexers)
    except Exception as e:
        fft_array_result_isel = type(e)
    try:
        fft_array_result_square_brackets = test_function_square_brackets(indexers)
    except Exception as e:
        fft_array_result_square_brackets = type(e)
    try:
        xr_indexer = make_xr_indexer(indexers, space)
        xr_result = xr_dataset[space].isel(xr_indexer).data
    except Exception as e:
        xr_result = type(e)
        assert fft_array_result_isel == xr_result
        assert fft_array_result_square_brackets == xr_result
        return

    np.testing.assert_array_equal(
        fft_array_result_isel.values,
        xr_result.data
    )
    np.testing.assert_array_equal(
        fft_array_result_square_brackets.values,
        xr_result.data
    )

label_indexers_test_samples = [
    {"x": 1, "y": 1, "z": 1}, {"x": 1, "y": 1, "z": slice(None, None)},
    {"x": 1, "y": 1}, {"x": -20}, {"z": 5}, {"random": 1}, {},
    {"x": slice(-20,5), "y": slice(-6,6), "z": slice(None, 4)},
]

@pytest.mark.parametrize("indexers", label_indexers_test_samples)
@pytest.mark.parametrize("tlib_class", TENSOR_LIBS)
@pytest.mark.parametrize("space", ["pos", "freq"])
# TODO: think about making this also jittable
@pytest.mark.parametrize("do_jit", [False])
def test_3d_fft_array_indexing_by_label(
    do_jit: bool,
    space: Space,
    tlib_class: TensorLib,
    indexers: Optional[Mapping[Hashable, Union[int, slice]]],
) -> None:

    fft_array, xr_dataset = generate_test_fftarray_xrdataset(
        ["x", "y", "z"],
        dimension_length=8,
        tlib=tlib_class()
    )

    if do_jit:
        tlib = tlib_class()
        if isinstance(tlib, JaxTensorLib):
            @jax.jit
            def test_function_sel(_indexers) -> FFTArray:
                return fft_array.into(space=space).sel(_indexers)
            @jax.jit
            def test_function_square_brackets(_indexers) -> FFTArray:
                return fft_array.into(space=space).loc[_indexers]
        else:
            return
    else:
        def test_function_sel(_indexers) -> FFTArray:
            return fft_array.into(space=space).sel(_indexers)
        def test_function_square_brackets(_indexers) -> FFTArray:
            return fft_array.into(space=space).loc[_indexers]

    try:
        fft_array_result_sel = test_function_sel(indexers)
    except Exception as e:
        fft_array_result_sel = type(e)
    try:
        fft_array_result_loc_square_brackets = test_function_square_brackets(indexers)
    except Exception as e:
        fft_array_result_loc_square_brackets = type(e)
    try:
        xr_indexer = make_xr_indexer(indexers, space)
        xr_result = xr_dataset[space].sel(xr_indexer).data
    except Exception as e:
        xr_result = type(e)
        if xr_result in [KeyError, ValueError]:
            xr_result = (KeyError, ValueError)
        else:
            xr_result = [xr_result]
        assert fft_array_result_sel in xr_result
        assert fft_array_result_loc_square_brackets in xr_result
        return

    np.testing.assert_array_equal(
        fft_array_result_sel.values,
        xr_result.data
    )
    np.testing.assert_array_equal(
        fft_array_result_loc_square_brackets.values,
        xr_result.data
    )

def generate_test_fftarray_xrdataset(
    dimension_names: List[str],
    dimension_length: Union[int, List[int]],
    tlib: TensorLib,
) -> Tuple[FFTArray, xr.Dataset]:

    if isinstance(dimension_length, int):
        dimension_length = [dimension_length]*len(dimension_names)

    dims = [
        FFTDimension(name=dim_name, n=dim_length, d_pos=1, pos_min=0, freq_min=0)
        for dim_name, dim_length in zip(dimension_names, dimension_length)
    ]

    fft_array = reduce(lambda x,y: x+y, [dim.fft_array(tlib=tlib, space="pos") for dim in dims])

    pos_coords = {
        f"{dim.name}_pos": dim.np_array(space="pos")
        for dim in dims
    }
    freq_coords = {
        f"{dim.name}_freq": dim.np_array(space="freq")
        for dim in dims
    }

    xr_dataset = xr.Dataset(
        data_vars={
            'pos': ([f"{name}_pos" for name in dimension_names], fft_array.values),
            'freq': ([f"{name}_freq" for name in dimension_names], fft_array.into(space="freq").values),
        },
        coords=pos_coords | freq_coords
    )

    return (fft_array, xr_dataset)



@pytest.mark.parametrize("tlib_class", TENSOR_LIBS)
@pytest.mark.parametrize("do_jit", [False, True])
# def test_1d_fftarray_indexing(tlib_class: TensorLib, do_jit: bool) -> None:
def not_implemented(tlib_class: TensorLib, do_jit: bool) -> None:
    if do_jit and type(tlib_class) != JaxTensorLib:
        return

    tlib=tlib_class()

    dim = FFTDimension("x",
        n=4,
        d_pos=1,
        pos_min=0.,
        freq_min=0.,
    )
    fftarray_1d = 2*dim.fft_array(space="pos", tlib=tlib)

    def test_functions(fftarray: FFTArray):
        return (
            # fftarray[0],
            # fftarray[1],
            # fftarray[0:3],
            fftarray.__getitem__(0),
            fftarray.isel(x=0)
        )

    if do_jit:
        test_functions = jax.jit(test_functions)

    results = test_functions(fftarray_1d)

    assert results[0].values == 0
    assert results[1].values == 0
    # assert



@pytest.mark.parametrize("tlib_class", TENSOR_LIBS)
@pytest.mark.parametrize("do_jit", [False, True])
def not_implemented(tlib_class, do_jit: bool):
    if do_jit and type(tlib_class) != JaxTensorLib:
        return

    tlib = tlib_class()

    x_dim = FFTDimension("x",
        n=4,
        d_pos=1,
        pos_min=0.5,
        freq_min=0.,
    )
    y_dim = FFTDimension("y",
        n=4,
        d_pos=2,
        pos_min=-2,
        freq_min=0.,
    )

    arr_2d = x_dim.fft_array(tlib, space="pos") + y_dim.fft_array(tlib, space="pos")**2
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
