
from functools import reduce
from typing import Dict, Hashable, List, Literal, Mapping, Tuple, TypeVar, Union
import pytest
import numpy as np
import jax
import xarray as xr

from fftarray.fft_array import FFTArray, FFTDimension, Space
from fftarray.backends.backend import Backend
from fftarray.backends.jax import JaxBackend
from fftarray.backends.numpy import NumpyBackend
from fftarray.backends.pyfftw import PyFFTWBackend

jax.config.update("jax_enable_x64", True)

EllipsisType = TypeVar('EllipsisType')

BACKENDS = [NumpyBackend, JaxBackend, PyFFTWBackend]

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

@pytest.mark.parametrize("backend_class", BACKENDS)
def test_fftdim_single_element_indexing(backend_class) -> None:

    backend = backend_class()

    dim = FFTDimension("x",
        n=4,
        d_pos=1,
        pos_min=0.5,
        freq_min=0.,
    )

    def test_functions(dim):
        return (
            dim._index_from_coord(0.5, method = None, space="pos", backend=backend),
            dim._index_from_coord(2.5, method = None, space="pos", backend=backend),
            dim._index_from_coord(0.4, method = "nearest", space="pos", backend=backend),
            dim._index_from_coord(2.6, method = "nearest", space="pos", backend=backend),
        )

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
def test_valid_fftdim_dim_from_slice(space: Space, valid_slice: slice) -> None:

    result_dim = TEST_FFTDIM._dim_from_slice(range=valid_slice, space=space)

    np.testing.assert_array_equal(
        result_dim.np_array(space),
        TEST_FFTDIM.np_array(space)[valid_slice],
        strict=True
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

invalid_substepping_slices = [
    slice(None, None, 2), slice(None, None, 3),
    slice(None, None, 0), slice(None, None, -1), slice(None, None, -2)
]

@pytest.mark.parametrize("as_dict", [True, False])
@pytest.mark.parametrize("backend_class", BACKENDS)
@pytest.mark.parametrize("invalid_slice", invalid_substepping_slices)
@pytest.mark.parametrize("space", ["pos", "freq"])
def test_errors_fftarray_index_substepping(
    space: Space,
    invalid_slice: slice,
    backend_class,
    as_dict: bool,
) -> None:

    fft_arr = TEST_FFTDIM.fft_array(backend=backend_class(), space=space)

    if as_dict:
        invalid_slice = {"x": invalid_slice} # type: ignore

    with pytest.raises(IndexError):
        fft_arr[invalid_slice]
    with pytest.raises(IndexError):
        fft_arr.loc[invalid_slice]

    if as_dict:
        with pytest.raises(IndexError):
            fft_arr.sel(invalid_slice) # type: ignore
        with pytest.raises(IndexError):
            fft_arr.isel(invalid_slice) # type: ignore
    else:
        with pytest.raises(IndexError):
            fft_arr.sel(x=invalid_slice)
        with pytest.raises(IndexError):
            fft_arr.isel(x=invalid_slice)

invalid_tuples = [
    (Ellipsis, Ellipsis),
    (slice(None, None), slice(None, None), slice(None, None)),
    (Ellipsis, slice(None, None), slice(None, None)),
    (slice(None, None), slice(None, None), Ellipsis),
]

@pytest.mark.parametrize("backend_class", BACKENDS)
@pytest.mark.parametrize("invalid_tuple", invalid_tuples)
@pytest.mark.parametrize("space", ["pos", "freq"])
def test_errors_fftarray_invalid_indexes(
    space: Space,
    invalid_tuple: tuple,
    backend_class,
) -> None:

    fft_arr, _ = generate_test_fftarray_xrdataset(
        ["x", "y"],
        dimension_length=8,
        backend=backend_class()
    )
    fft_arr = fft_arr.into(space=space)

    with pytest.raises(IndexError):
        fft_arr[invalid_tuple]
    with pytest.raises(IndexError):
        fft_arr.loc[invalid_tuple]

coord_test_samples = [
    -5, -1.5, -1, -0.5, 0, 0.3, 0.5, 0.7, 1, 1.3, 7.5, 8, 8.5, 9,
    slice(-5,10), slice(None, None), slice(0,7), slice(0,8), slice(0.5,0.1)
]

@pytest.mark.parametrize("backend_class", BACKENDS)
@pytest.mark.parametrize("method", ["nearest", "pad", "ffill", "backfill", "bfill", None])
@pytest.mark.parametrize("valid_coord", coord_test_samples)
@pytest.mark.parametrize("space", ["pos", "freq"])
def test_valid_index_from_coord(
    space: Space,
    valid_coord: Union[float,slice],
    method: Literal["nearest", "pad", "ffill", "backfill", "bfill", None],
    backend_class
) -> None:

    def test_function(_coord):
        return TEST_FFTDIM._index_from_coord(coord=_coord, space=space, method=method, backend=backend_class())

    try:
        dim_index_result = test_function(valid_coord)
    except (KeyError, NotImplementedError) as e:
        dim_index_result = type(e)
    try:
        xr_result_coord = STANDARD_TEST_DATASET[space].sel({f"{space}_coord": valid_coord}, method=method)
        xr_result_dim_index = STANDARD_TEST_DATASET[space].isel({f"{space}_coord": dim_index_result})
        np.testing.assert_array_equal(
            xr_result_coord.data,
            xr_result_dim_index.data
        )
    except (KeyError, NotImplementedError) as e:
        xr_result = type(e)
        assert dim_index_result == xr_result

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
@pytest.mark.parametrize("backend_class", BACKENDS)
@pytest.mark.parametrize("space", ["pos", "freq"])
def test_3d_fft_array_indexing_by_integer(
    space: Space,
    backend_class,
    indexers: Mapping[Hashable, Union[int, slice]],
) -> None:

    fft_array, xr_dataset = generate_test_fftarray_xrdataset(
        ["x", "y", "z"],
        dimension_length=8,
        backend=backend_class()
    )

    def test_function_isel(_indexers) -> FFTArray:
        return fft_array.into(space=space).isel(_indexers)
    def test_function_square_brackets(_indexers) -> FFTArray:
        return fft_array.into(space=space)[_indexers]

    try:
        fft_array_result_isel = test_function_isel(indexers) # type: ignore
    except Exception as e:
        fft_array_result_isel = type(e) # type: ignore
    try:
        fft_array_result_square_brackets = test_function_square_brackets(indexers) # type: ignore
    except Exception as e:
        fft_array_result_square_brackets = type(e) # type: ignore
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

tuple_indexers = [
    (..., slice(None, None)),
    (slice(None, None), ...),
    (...,),
    ...,
    (slice(None,5), ),
    (slice(None,1), ..., slice(None,2)),
    (slice(None, None), slice(None, None))
]

@pytest.mark.parametrize("indexers", tuple_indexers)
@pytest.mark.parametrize("backend_class", BACKENDS)
@pytest.mark.parametrize("space", ["pos", "freq"])
def test_3d_fft_array_positional_indexing(
    space: Space,
    backend_class,
    indexers: Tuple[Union[int, float, slice, EllipsisType]],
) -> None:

    fft_array, xr_dataset = generate_test_fftarray_xrdataset(
        ["x", "y", "z"],
        dimension_length=8,
        backend=backend_class()
    )

    def test_function_loc_square_brackets(_indexers) -> FFTArray:
        return fft_array.into(space=space).loc[_indexers]
    def test_function_square_brackets(_indexers) -> FFTArray:
        return fft_array.into(space=space)[_indexers]

    fft_array_result_square_brackets = test_function_square_brackets(indexers) # type: ignore
    xr_result_square_bracket = xr_dataset[space][indexers].data

    np.testing.assert_array_equal(
        fft_array_result_square_brackets.values,
        xr_result_square_bracket.data
    )

    fft_array_result_loc_square_brackets = test_function_loc_square_brackets(indexers) # type: ignore
    xr_result_loc_square_bracket = xr_dataset[space].loc[indexers].data

    np.testing.assert_array_equal(
        fft_array_result_loc_square_brackets.values,
        xr_result_loc_square_bracket.data
    )

label_indexers_test_samples = [
    {"x": 3, "y": 1, "z": 4}, {"x": 0, "y": 2, "z": slice(None, None)},
    {"x": 1, "y": 4}, {"x": -25}, {"z": 5}, {"random": 1}, {},
    {"x": slice(-23,5), "y": slice(-6,6), "z": slice(None, 3)},
]

@pytest.mark.parametrize("indexers", label_indexers_test_samples)
@pytest.mark.parametrize("backend_class", BACKENDS)
@pytest.mark.parametrize("space", ["pos", "freq"])
@pytest.mark.parametrize("method", ["nearest", "pad", "ffill", "backfill", "bfill", None, "unsupported"])
def test_3d_fft_array_label_indexing(
    space: Space,
    backend_class,
    indexers: Mapping[Hashable, Union[int, slice]],
    method: Literal["nearest", "pad", "ffill", "backfill", "bfill", None],
) -> None:

    fft_array, xr_dataset = generate_test_fftarray_xrdataset(
        ["x", "y", "z"],
        dimension_length=8,
        backend=backend_class()
    )

    try:
        fft_array_result = fft_array.into(space=space).sel(indexers, method=method) # type: ignore
    except Exception as e:
        fft_array_result = type(e) # type: ignore

    try:
        xr_indexer = make_xr_indexer(indexers, space)
        xr_result = xr_dataset[space].sel(xr_indexer, method=method).data # type: ignore
    except Exception as e:
        xr_result = type(e) # type: ignore
        if xr_result in [KeyError, ValueError]:
            xr_result = (KeyError, ValueError)
        else:
            xr_result = [xr_result]
        assert fft_array_result in xr_result
        return

    np.testing.assert_array_equal(
        fft_array_result.values,
        xr_result.data
    )


@pytest.mark.parametrize("indexers", label_indexers_test_samples)
@pytest.mark.parametrize("index_by", ["label", "integer"])
@pytest.mark.parametrize("space", ["pos", "freq"])
def test_3d_fft_array_indexing(
    space: Space,
    index_by: Literal["label", "integer"],
    indexers: Mapping[Hashable, Union[int, slice]],
) -> None:

    backend = JaxBackend()
    fft_array, xr_dataset = generate_test_fftarray_xrdataset(
        ["x", "y", "z"],
        dimension_length=8,
        backend=backend
    )

    def test_function_sel(_indexers) -> FFTArray:
        if index_by == "label":
            return fft_array.into(space=space).sel(_indexers)
        else:
            return fft_array.into(space=space).isel(_indexers)

    def test_function_square_brackets(_indexers) -> FFTArray:
        if index_by == "label":
            return fft_array.into(space=space).loc[_indexers]
        else:
            return fft_array.into(space=space)[_indexers]

    fft_error = False
    try:
        fft_array_result_sel = test_function_sel(indexers)
    except Exception as e:
        fft_array_result_sel = type(e) # type: ignore
        fft_error = True
    try:
        fft_array_result_loc_square_brackets = test_function_square_brackets(indexers)
    except Exception as e:
        fft_array_result_loc_square_brackets = type(e) # type: ignore
        fft_error = True
    try:
        xr_indexer = make_xr_indexer(indexers, space)
        if index_by == "label":
            xr_result = xr_dataset[space].sel(xr_indexer).data
        else:
            xr_result = xr_dataset[space].isel(xr_indexer).data
    except Exception as e:
        xr_result = type(e)
        if xr_result in [KeyError, ValueError]:
            xr_result = (KeyError, ValueError)
        else:
            xr_result = [xr_result]

    if fft_error:
        assert (
            fft_array_result_sel in xr_result
        )
        assert (
            fft_array_result_loc_square_brackets in xr_result
        )
    else:
        np.testing.assert_array_equal(
            fft_array_result_sel.values,
            xr_result.data
        )
        np.testing.assert_array_equal(
            fft_array_result_loc_square_brackets.values,
            xr_result.data
        )

valid_indexers = [
    {"x": slice(None, 5), "y": 4},
    {"x": 3},
    {"x": slice(None, None), "y": slice(None, None)},
    {}
]

space_combinations = [
    {"x": "pos", "y": "pos"}, {"x": "pos", "y": "freq"},
    {"x": "freq", "y": "pos"}, {"x": "freq", "y": "freq"}
]

@pytest.mark.parametrize("indexers", valid_indexers)
@pytest.mark.parametrize("backend_class", BACKENDS)
@pytest.mark.parametrize("space_combination", space_combinations)
def test_fftarray_state_management(
    space_combination: Dict[str, Space],
    backend_class,
    indexers: Mapping[Hashable, Union[int, slice]],
) -> None:
    """
    Tests if the indexed FFTArray has the correct internal properties,
    especially if _factors_applied is True afterwards.
    Also checks, that the values correspond to _factors_applied True.
    For the special case of empty indexing, it checks that _factors_applied is
    the same as the original FFTArray.
    """

    dims = {
        dim_name: FFTDimension(name=dim_name, n=8, d_pos=1, pos_min=0, freq_min=0)
        for dim_name in space_combination
    }
    fft_arrays = {
        dim_name: dims[dim_name].fft_array(space=space, backend=backend_class(), eager=False)
        for dim_name, space in space_combination.items()
    }

    fft_array_2d = fft_arrays["x"] + fft_arrays["y"]

    space_comb_list = [space_combination[dim_name] for dim_name in ["x", "y"]]
    diff_space_comb = [
        "pos" if space_comb == "freq" else "freq"
        for space_comb in space_comb_list
    ]

    try:
        # Test FFTArray[]
        fft_raw_values = fft_array_2d[indexers].values
        fft_different_internal = fft_array_2d.into(space=diff_space_comb).into(space=space_comb_list)
        fft_indexed = fft_different_internal[indexers]
        fft_indexed_values = fft_indexed.values

        np.testing.assert_array_equal(fft_raw_values, fft_indexed_values)
        assert (
            all(fft_indexed._factors_applied) or
            (len(indexers) == 0 and fft_indexed._factors_applied == fft_different_internal._factors_applied)
        )
        assert fft_array_2d.eager == fft_indexed.eager
        assert fft_different_internal.space == fft_indexed.space

        # Test FFTArray.isel()
        fft_raw_values = fft_array_2d.isel(indexers).values
        fft_different_internal = fft_array_2d.into(space=diff_space_comb).into(space=space_comb_list)
        fft_indexed = fft_different_internal.isel(indexers)
        fft_indexed_values = fft_indexed.values

        np.testing.assert_array_equal(fft_raw_values, fft_indexed_values)
        assert (
            all(fft_indexed._factors_applied) or
            (len(indexers) == 0 and fft_indexed._factors_applied == fft_different_internal._factors_applied)
        )
        assert fft_array_2d.eager == fft_indexed.eager
        assert fft_different_internal.space == fft_indexed.space

        # Test FFTArray.loc[]
        fft_raw_values = fft_array_2d.loc[indexers].values
        fft_different_internal = fft_array_2d.into(space=diff_space_comb).into(space=space_comb_list)
        fft_indexed = fft_different_internal.loc[indexers]
        fft_indexed_values = fft_indexed.values

        np.testing.assert_array_equal(fft_raw_values, fft_indexed_values)
        assert (
            all(fft_indexed._factors_applied) or
            (len(indexers) == 0 and fft_indexed._factors_applied == fft_different_internal._factors_applied)
        )
        assert fft_array_2d.eager == fft_indexed.eager
        assert fft_different_internal.space == fft_indexed.space

        # Test FFTArray.sel()
        fft_raw_values = fft_array_2d.sel(indexers, method="nearest").values
        fft_different_internal = fft_array_2d.into(space=diff_space_comb).into(space=space_comb_list)
        fft_indexed = fft_different_internal.sel(indexers)
        fft_indexed_values = fft_indexed.values

        np.testing.assert_array_equal(fft_raw_values, fft_indexed_values)
        assert (
            all(fft_indexed._factors_applied) or
            (len(indexers) == 0 and fft_indexed._factors_applied == fft_different_internal._factors_applied)
        )
        assert fft_array_2d.eager == fft_array_2d.eager
        assert fft_different_internal.space == fft_array_2d.space
    except (KeyError, NotImplementedError):
        return

def generate_test_fftarray_xrdataset(
    dimension_names: List[str],
    dimension_length: Union[int, List[int]],
    backend: Backend,
) -> Tuple[FFTArray, xr.Dataset]:

    if isinstance(dimension_length, int):
        dimension_length = [dimension_length]*len(dimension_names)

    dims = [
        FFTDimension(name=dim_name, n=dim_length, d_pos=1, pos_min=0, freq_min=0)
        for dim_name, dim_length in zip(dimension_names, dimension_length)
    ]

    fft_array = reduce(lambda x,y: x+y, [dim.fft_array(backend=backend, space="pos") for dim in dims])

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

@jax.jit
def index_with_tracer_getitem(obj, idx):
    return obj[idx]
@jax.jit
def index_with_tracer_loc(obj, idx):
    return obj.loc[idx]
@jax.jit
def index_with_tracer_isel(obj, idx):
    return obj.isel(idx)
@jax.jit
def index_with_tracer_sel(obj, idx):
    return obj.sel(idx)

def test_invalid_tracer_index() -> None:
    fft_arr = TEST_FFTDIM.fft_array(backend=JaxBackend(), space="pos")
    tracer_index = jax.numpy.array(3)

    with pytest.raises(NotImplementedError):
        index_with_tracer_getitem(fft_arr, {'x': tracer_index})
    with pytest.raises(NotImplementedError):
        index_with_tracer_loc(fft_arr, {'x': tracer_index})
    with pytest.raises(NotImplementedError):
        index_with_tracer_isel(fft_arr, {'x': tracer_index})
    with pytest.raises(NotImplementedError):
        index_with_tracer_sel(fft_arr, {'x': tracer_index})

def test_jit_static_indexing() -> None:

    fft_arr, xr_dataset = generate_test_fftarray_xrdataset(["x"], dimension_length=8, backend=JaxBackend())

    def test_function_isel(_indexers) -> FFTArray:
        return fft_arr.isel(x=_indexers)

    def test_function_square_brackets(_indexers) -> FFTArray:
        return fft_arr[slice(*_indexers)]

    test_function_isel = jax.jit(test_function_isel, static_argnums=(0,))
    test_function_square_brackets = jax.jit(test_function_square_brackets, static_argnums=(0,))

    isel_indexer = 3
    sq_brackets_indexer = (1,4)

    fft_array_result_isel = test_function_isel(isel_indexer) # type: ignore
    fft_array_result_square_brackets = test_function_square_brackets(sq_brackets_indexer) # type: ignore

    xr_result_isel = xr_dataset["pos"].isel(x_pos=isel_indexer).data
    xr_result_square_brackets = xr_dataset["pos"][slice(*sq_brackets_indexer)].data

    np.testing.assert_array_equal(
        fft_array_result_isel.values,
        xr_result_isel
    )

    np.testing.assert_array_equal(
        fft_array_result_square_brackets.values,
        xr_result_square_brackets
    )


def test_invalid_kw_and_pos_indexers() -> None:

    fft_arr, _ = generate_test_fftarray_xrdataset(["x", "y"], dimension_length=8, backend=NumpyBackend())

    with pytest.raises(ValueError):
        fft_arr.sel({'x': 3}, y=3)
    with pytest.raises(ValueError):
        fft_arr.isel({'x': 3}, y=3)

@pytest.mark.parametrize("index_method", ["sel", "isel"])
def test_missing_dims(
    index_method: Literal["sel", "isel"]
) -> None:

    fft_arr, _ = generate_test_fftarray_xrdataset(["x", "y"], dimension_length=8, backend=NumpyBackend())

    with pytest.raises(ValueError):
        getattr(fft_arr, index_method)({"x": 3}, missing_dims="unsupported")

    with pytest.raises(ValueError):
        getattr(fft_arr, index_method)({"unknown_dim": 3})
    with pytest.raises(ValueError):
        getattr(fft_arr, index_method)({"unknown_dim": 3}, missing_dims="raise")
    with pytest.warns(UserWarning):
        getattr(fft_arr, index_method)({"unknown_dim": 3}, missing_dims="warn")

    getattr(fft_arr, index_method)({"x": 3}, missing_dims="raise")
    getattr(fft_arr, index_method)({"unknown_dim": 3}, missing_dims="ignore")

