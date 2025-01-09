
from functools import reduce
from typing import Dict, List, Literal, Mapping, Tuple, TypeVar, Union
import pytest
import numpy as np
import xarray as xr

import fftarray as fa
from fftarray.array import Array, Space
from fftarray.tests.helpers import XPS

EllipsisType = TypeVar('EllipsisType')


TEST_DIM = fa.dim(
    name="x", n=8, d_pos=1, pos_min=0, freq_min=0
)
STANDARD_TEST_DATAARRAY = xr.DataArray(
    data=np.linspace(0, 7, num=8),
    dims=["x"],
    coords={'x': np.linspace(0, 7, num=8)},
)

pos_values = TEST_DIM.pos_min + np.arange(TEST_DIM.n)*TEST_DIM.d_pos
freq_values = TEST_DIM.freq_min + np.arange(TEST_DIM.n)*TEST_DIM.d_freq

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
- class LocArrayIndexer
- method Array.__getitem__
- property Array.loc = LocArrayIndexer(self)
- method Array.sel
- method Array.isel
- method Dimension._index_from_coord
- method Dimension._dim_from_slice
- method Dimension._dim_from_start_and_n
"""
def test_dim_single_element_indexing() -> None:
    dim = fa.dim("x",
        n=4,
        d_pos=1,
        pos_min=0.5,
        freq_min=0.,
    )

    def test_functions(dim):
        return (
            dim._index_from_coord(0.5, method = None, space="pos"),
            dim._index_from_coord(2.5, method = None, space="pos"),
            dim._index_from_coord(0.4, method = "nearest", space="pos"),
            dim._index_from_coord(2.6, method = "nearest", space="pos"),
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
def test_valid_dim_dim_from_slice(space: Space, valid_slice: slice) -> None:

    result_dim = TEST_DIM._dim_from_slice(range=valid_slice, space=space)

    np.testing.assert_array_equal(
        result_dim.np_array(space),
        TEST_DIM.np_array(space)[valid_slice],
        strict=True
    )

invalid_slices = [
    slice(1, 1), slice(1, 0), slice(-2, 0), slice(7, -1),
    slice(0, 6, 2), slice(None, None, 2), slice(0., 5.),
    slice(10,None)
]

@pytest.mark.parametrize("space", ["pos", "freq"])
@pytest.mark.parametrize("invalid_slice", invalid_slices)
def test_errors_dim_dim_from_slice(space: Space, invalid_slice: slice) -> None:

    with pytest.raises(IndexError):
        TEST_DIM._dim_from_slice(invalid_slice, space=space)

invalid_substepping_slices = [
    slice(None, None, 2), slice(None, None, 3),
    slice(None, None, 0), slice(None, None, -1), slice(None, None, -2)
]

@pytest.mark.parametrize("as_dict", [True, False])
@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("invalid_slice", invalid_substepping_slices)
@pytest.mark.parametrize("space", ["pos", "freq"])
def test_errors_array_index_substepping(
    space: Space,
    invalid_slice: slice,
    xp,
    as_dict: bool,
) -> None:

    arr = fa.coords_from_dim(TEST_DIM, space, xp=xp)

    if as_dict:
        invalid_slice = {"x": invalid_slice} # type: ignore

    with pytest.raises(IndexError):
        arr[invalid_slice]
    with pytest.raises(IndexError):
        arr.loc[invalid_slice]

    if as_dict:
        with pytest.raises(IndexError):
            arr.sel(invalid_slice) # type: ignore
        with pytest.raises(IndexError):
            arr.isel(invalid_slice) # type: ignore
    else:
        with pytest.raises(IndexError):
            arr.sel(x=invalid_slice)
        with pytest.raises(IndexError):
            arr.isel(x=invalid_slice)

invalid_tuples = [
    (Ellipsis, Ellipsis),
    (slice(None, None), slice(None, None), slice(None, None)),
    (Ellipsis, slice(None, None), slice(None, None)),
    (slice(None, None), slice(None, None), Ellipsis),
]

@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("invalid_tuple", invalid_tuples)
@pytest.mark.parametrize("space", ["pos", "freq"])
def test_errors_array_invalid_indexes(
    space: Space,
    invalid_tuple: tuple,
    xp,
) -> None:

    arr, _ = generate_test_array_xrdataset(
        ["x", "y"],
        dimension_length=8,
        xp=xp
    )
    arr = arr.into_space(space)

    with pytest.raises(IndexError):
        arr[invalid_tuple]
    with pytest.raises(IndexError):
        arr.loc[invalid_tuple]

coord_test_samples = [
    -5, -1.5, -1, -0.5, 0, 0.3, 0.5, 0.7, 1, 1.3, 7.5, 8, 8.5, 9,
    slice(-5,10), slice(None, None), slice(0,7), slice(0,8), slice(0.5,0.1)
]

@pytest.mark.parametrize("method", ["nearest", "pad", "ffill", "backfill", "bfill", None])
@pytest.mark.parametrize("valid_coord", coord_test_samples)
@pytest.mark.parametrize("space", ["pos", "freq"])
def test_valid_index_from_coord(
    space: Space,
    valid_coord: Union[float,slice],
    method: Literal["nearest", "pad", "ffill", "backfill", "bfill", None],
) -> None:

    def test_function(_coord):
        return TEST_DIM._index_from_coord(coord=_coord, space=space, method=method)

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
    {"x": slice(-7,5), "y": slice(-6,6), "z": slice(None, 4)}
]

@pytest.mark.parametrize("indexers", integer_indexers_test_samples)
@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("space", ["pos", "freq"])
def test_3d_array_indexing_by_integer(
    space: Space,
    xp,
    indexers: Mapping[str, Union[int, slice]],
) -> None:

    arr, xr_dataset = generate_test_array_xrdataset(
        ["x", "y", "z"],
        dimension_length=8,
        xp=xp
    )

    def test_function_isel(_indexers) -> Array:
        return arr.into_space(space).isel(_indexers)
    def test_function_square_brackets(_indexers) -> Array:
        return arr.into_space(space)[_indexers]

    try:
        arr_result_isel = test_function_isel(indexers) # type: ignore
    except Exception as e:
        arr_result_isel = type(e) # type: ignore
    try:
        arr_result_square_brackets = test_function_square_brackets(indexers) # type: ignore
    except Exception as e:
        arr_result_square_brackets = type(e) # type: ignore
    try:
        xr_indexer = make_xr_indexer(indexers, space)
        xr_result = xr_dataset[space].isel(xr_indexer).data
    except Exception as e:
        xr_result = type(e)
        assert arr_result_isel == xr_result
        assert arr_result_square_brackets == xr_result
        return

    np.testing.assert_array_equal(
        arr_result_isel.values(space),
        xr_result
    )
    np.testing.assert_array_equal(
        arr_result_square_brackets.values(space),
        xr_result
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
@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("space", ["pos", "freq"])
def test_3d_array_positional_indexing(
    space: Space,
    xp,
    indexers: Tuple[Union[int, float, slice, EllipsisType]],
) -> None:

    arr, xr_dataset = generate_test_array_xrdataset(
        ["x", "y", "z"],
        dimension_length=8,
        xp=xp
    )

    def test_function_loc_square_brackets(_indexers) -> Array:
        return arr.into_space(space).loc[_indexers]
    def test_function_square_brackets(_indexers) -> Array:
        return arr.into_space(space)[_indexers]

    arr_result_square_brackets = test_function_square_brackets(indexers) # type: ignore
    xr_result_square_bracket = xr_dataset[space][indexers].data

    np.testing.assert_array_equal(
        arr_result_square_brackets.values(space),
        xr_result_square_bracket
    )

    arr_result_loc_square_brackets = test_function_loc_square_brackets(indexers) # type: ignore
    xr_result_loc_square_bracket = xr_dataset[space].loc[indexers].data

    np.testing.assert_array_equal(
        arr_result_loc_square_brackets.values(space),
        xr_result_loc_square_bracket
    )

label_indexers_test_samples = [
    {"x": 3, "y": 1, "z": 4}, {"x": 0, "y": 2, "z": slice(None, None)},
    {"x": 1, "y": 4}, {"x": -25}, {"z": 5}, {"random": 1}, {},
    {"x": slice(-7,5), "y": slice(-6,6), "z": slice(None, 3)},
]

@pytest.mark.parametrize("indexers", label_indexers_test_samples)
@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("space", ["pos", "freq"])
@pytest.mark.parametrize("method", ["nearest", "pad", "ffill", "backfill", "bfill", None, "unsupported"])
def test_3d_array_label_indexing(
    space: Space,
    xp,
    indexers: Mapping[str, Union[int, slice]],
    method: Literal["nearest", "pad", "ffill", "backfill", "bfill", None],
) -> None:

    arr, xr_dataset = generate_test_array_xrdataset(
        ["x", "y", "z"],
        dimension_length=8,
        xp=xp
    )

    try:
        arr_result = arr.into_space(space).sel(indexers, method=method)
    except Exception as e:
        arr_result = type(e) # type: ignore

    try:
        xr_indexer = make_xr_indexer(indexers, space)
        xr_result = xr_dataset[space].sel(xr_indexer, method=method).data
    except Exception as e:
        xr_result = type(e)
        if xr_result in [KeyError, ValueError]:
            xr_result = (KeyError, ValueError)
        else:
            xr_result = [xr_result]
        assert arr_result in xr_result
        return

    np.testing.assert_array_equal(
        arr_result.values(space),
        xr_result
    )


@pytest.mark.parametrize("indexers", label_indexers_test_samples)
@pytest.mark.parametrize("index_by", ["label", "integer"])
@pytest.mark.parametrize("space", ["pos", "freq"])
@pytest.mark.parametrize("xp", XPS)
def test_3d_array_indexing(
    space: Space,
    index_by: Literal["label", "integer"],
    indexers: Mapping[str, Union[int, slice]],
    xp,
) -> None:

    arr, xr_dataset = generate_test_array_xrdataset(
        ["x", "y", "z"],
        dimension_length=8,
        xp=xp,
    )

    def test_function_sel(_indexers) -> Array:
        if index_by == "label":
            return arr.into_space(space).sel(_indexers)
        else:
            return arr.into_space(space).isel(_indexers)

    def test_function_square_brackets(_indexers) -> Array:
        if index_by == "label":
            return arr.into_space(space).loc[_indexers]
        else:
            return arr.into_space(space)[_indexers]

    arr_error = False
    try:
        arr_result_sel = test_function_sel(indexers)
    except Exception as e:
        arr_result_sel = type(e) # type: ignore
        arr_error = True
    try:
        arr_result_loc_square_brackets = test_function_square_brackets(indexers)
    except Exception as e:
        arr_result_loc_square_brackets = type(e) # type: ignore
        arr_error = True
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

    if arr_error:
        assert (
            arr_result_sel in xr_result
        )
        assert (
            arr_result_loc_square_brackets in xr_result
        )
    else:
        np.testing.assert_array_equal(
            arr_result_sel.values(space),
            xr_result
        )
        np.testing.assert_array_equal(
            arr_result_loc_square_brackets.values(space),
            xr_result
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
@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("space_combination", space_combinations)
def test_array_state_management(
    space_combination: Dict[str, Space],
    xp,
    indexers: Mapping[str, Union[int, slice]],
) -> None:
    """
    Tests if the indexed Array has the correct internal properties,
    especially if _factors_applied is True afterwards.
    Also checks, that the values correspond to _factors_applied True.
    For the special case of empty indexing, it checks that _factors_applied is
    the same as the original Array.
    """

    dims = {
        dim_name: fa.dim(name=dim_name, n=8, d_pos=1, pos_min=0, freq_min=0)
        for dim_name in space_combination
    }
    arrs = {
        dim_name: fa.coords_from_dim(dims[dim_name], space, xp=xp).into_eager(False)
        for dim_name, space in space_combination.items()
    }

    arr_2d = arrs["x"] + arrs["y"]

    space_comb_list = [space_combination[dim_name] for dim_name in ["x", "y"]]
    diff_space_comb: List[Space] = [
        "pos" if space_comb == "freq" else "freq"
        for space_comb in space_comb_list
    ]

    try:
        # Test Array[]
        arr_raw_values = arr_2d[indexers].values(arr_2d.space)
        arr_different_internal = arr_2d.into_space(diff_space_comb).into_space(space_comb_list)
        arr_indexed = arr_different_internal[indexers]
        arr_indexed_values = arr_indexed.values(arr_indexed.space)

        np.testing.assert_allclose(arr_raw_values, arr_indexed_values, atol=1e-16)
        assert (
            all(arr_indexed._factors_applied) or
            (len(indexers) == 0 and arr_indexed._factors_applied == arr_different_internal._factors_applied)
        )
        assert arr_2d.eager == arr_indexed.eager
        assert arr_different_internal.space == arr_indexed.space

        # Test Array.isel()
        arr_raw_values = arr_2d.isel(indexers).values(arr_2d.space)
        arr_different_internal = arr_2d.into_space(diff_space_comb).into_space(space_comb_list)
        arr_indexed = arr_different_internal.isel(indexers)
        arr_indexed_values = arr_indexed.values(arr_indexed.space)

        np.testing.assert_allclose(arr_raw_values, arr_indexed_values, atol=1e-16)
        assert (
            all(arr_indexed._factors_applied) or
            (len(indexers) == 0 and arr_indexed._factors_applied == arr_different_internal._factors_applied)
        )
        assert arr_2d.eager == arr_indexed.eager
        assert arr_different_internal.space == arr_indexed.space

        # Test Array.loc[]
        arr_raw_values = arr_2d.loc[indexers].values(arr_2d.space)
        arr_different_internal = arr_2d.into_space(diff_space_comb).into_space(space_comb_list)
        arr_indexed = arr_different_internal.loc[indexers]
        arr_indexed_values = arr_indexed.values(arr_indexed.space)

        np.testing.assert_allclose(arr_raw_values, arr_indexed_values, atol=1e-16)
        assert (
            all(arr_indexed._factors_applied) or
            (len(indexers) == 0 and arr_indexed._factors_applied == arr_different_internal._factors_applied)
        )
        assert arr_2d.eager == arr_indexed.eager
        assert arr_different_internal.space == arr_indexed.space

        # Test Array.sel()
        arr_raw_values = arr_2d.sel(indexers, method="nearest").values(arr_2d.space)
        arr_different_internal = arr_2d.into_space(diff_space_comb).into_space(space_comb_list)
        arr_indexed = arr_different_internal.sel(indexers)
        arr_indexed_values = arr_indexed.values(arr_indexed.space)

        np.testing.assert_allclose(arr_raw_values, arr_indexed_values, atol=1e-16)
        assert (
            all(arr_indexed._factors_applied) or
            (len(indexers) == 0 and arr_indexed._factors_applied == arr_different_internal._factors_applied)
        )
        assert arr_2d.eager == arr_2d.eager
        assert arr_different_internal.space == arr_2d.space
    except (KeyError, NotImplementedError):
        return

def generate_test_array_xrdataset(
    dimension_names: List[str],
    dimension_length: Union[int, List[int]],
    xp,
) -> Tuple[Array, xr.Dataset]:

    if isinstance(dimension_length, int):
        dimension_length = [dimension_length]*len(dimension_names)

    dims = [
        fa.dim(name=dim_name, n=dim_length, d_pos=1, pos_min=0, freq_min=0)
        for dim_name, dim_length in zip(dimension_names, dimension_length, strict=True)
    ]

    arr = reduce(lambda x,y: x+y, [fa.coords_from_dim(dim, "pos", xp=xp) for dim in dims])

    pos_coords = {
        f"{dim.name}_pos": dim.np_array("pos")
        for dim in dims
    }
    freq_coords = {
        f"{dim.name}_freq": dim.np_array("freq")
        for dim in dims
    }

    xr_dataset = xr.Dataset(
        data_vars={
            'pos': ([f"{name}_pos" for name in dimension_names], np.array(arr.values("pos"))),
            'freq': ([f"{name}_freq" for name in dimension_names], np.array(arr.values("freq"))),
        },
        coords=pos_coords | freq_coords
    )

    return (arr, xr_dataset)

try:
    import jax
    import jax.numpy as jnp
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
        arr = fa.coords_from_dim(TEST_DIM, "pos", xp=jnp)
        tracer_index = jax.numpy.array(3)

        with pytest.raises(NotImplementedError):
            index_with_tracer_getitem(arr, {'x': tracer_index})
        with pytest.raises(NotImplementedError):
            index_with_tracer_loc(arr, {'x': tracer_index})
        with pytest.raises(NotImplementedError):
            index_with_tracer_isel(arr, {'x': tracer_index})
        with pytest.raises(NotImplementedError):
            index_with_tracer_sel(arr, {'x': tracer_index})

    def test_jit_static_indexing() -> None:

        arr, xr_dataset = generate_test_array_xrdataset(["x"], dimension_length=8, xp=jnp)

        def test_function_isel(_indexers) -> Array:
            return arr.isel(x=_indexers)

        def test_function_square_brackets(_indexers) -> Array:
            return arr[slice(*_indexers)]

        test_function_isel = jax.jit(test_function_isel, static_argnums=(0,))
        test_function_square_brackets = jax.jit(test_function_square_brackets, static_argnums=(0,))

        isel_indexer = 3
        sq_brackets_indexer = (1,4)

        arr_result_isel = test_function_isel(isel_indexer)
        arr_result_square_brackets = test_function_square_brackets(sq_brackets_indexer)

        xr_result_isel = xr_dataset["pos"].isel(x_pos=isel_indexer).data
        xr_result_square_brackets = xr_dataset["pos"][slice(*sq_brackets_indexer)].data

        np.testing.assert_array_equal(
            arr_result_isel.values("pos"),
            xr_result_isel
        )

        np.testing.assert_array_equal(
            arr_result_square_brackets.values("pos"),
            xr_result_square_brackets
        )
except ImportError:
    pass


def test_invalid_kw_and_pos_indexers() -> None:
    arr, _ = generate_test_array_xrdataset(["x", "y"], dimension_length=8, xp=np)

    with pytest.raises(ValueError):
        arr.sel({'x': 3}, y=3)
    with pytest.raises(ValueError):
        arr.isel({'x': 3}, y=3)

@pytest.mark.parametrize("index_method", ["sel", "isel"])
def test_missing_dims(
    index_method: Literal["sel", "isel"]
) -> None:

    arr, _ = generate_test_array_xrdataset(["x", "y"], dimension_length=8, xp=np)

    with pytest.raises(ValueError):
        getattr(arr, index_method)({"x": 3}, missing_dims="unsupported")

    with pytest.raises(ValueError):
        getattr(arr, index_method)({"unknown_dim": 3})
    with pytest.raises(ValueError):
        getattr(arr, index_method)({"unknown_dim": 3}, missing_dims="raise")
    with pytest.warns(UserWarning):
        getattr(arr, index_method)({"unknown_dim": 3}, missing_dims="warn")

    getattr(arr, index_method)({"x": 3}, missing_dims="raise")
    getattr(arr, index_method)({"unknown_dim": 3}, missing_dims="ignore")

