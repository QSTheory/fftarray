from typing import Iterable, Literal, Optional, Dict, Any, Union, List

import numpy as np
import pytest
import fftarray as fa

from tests.helpers import XPS, XPS_ROTATED_PAIRS, get_dims, dtypes_names_pairs, dtype_names_numeric_core, DTYPE_NAME


@pytest.mark.parametrize("xp_target, xp_other", XPS_ROTATED_PAIRS)
@pytest.mark.parametrize("xp_source", ["values", "direct"])
@pytest.mark.parametrize("init_dtype_name, result_dtype_name", dtypes_names_pairs)
@pytest.mark.parametrize("ndims", [0,1,2])
@pytest.mark.parametrize("defensive_copy", [False, True])
@pytest.mark.parametrize("eager", [False, True])
def test_from_array_object(
        xp_target,
        xp_other,
        xp_source: Literal["values", "direct"],
        init_dtype_name: DTYPE_NAME,
        result_dtype_name: Optional[DTYPE_NAME],
        ndims: int,
        defensive_copy: bool,
        eager: bool,
    ) -> None:
    """
        Test array creation from an Array API array.
        This has two cases for xp derivation:
        1) From the passed in array.
        2) Via direct override.
        The default xp is only used when constructing an Array from Python values
        which is tested in the list creation tests.
    """
    dims = get_dims(ndims)
    shape = tuple(dim.n for dim in dims)

    array_args: Dict[str, Any] = dict(
        dims=dims,
        space="pos",
        defensive_copy=defensive_copy,
    )

    if result_dtype_name is None:
        # result_dtype_name is None means that the dtype is inferred from
        # the passed in values.
        # Therefore the expected dtype is the same than the one that is used
        # to create the array.
        result_dtype = getattr(xp_target, init_dtype_name)
    else:
        # In this case we explicitly override the dtype
        # in the creation of the array.
        result_dtype = getattr(xp_target, result_dtype_name)
        array_args["dtype"] = result_dtype


    match xp_source:
        case "values":
            # We derive xp from the passed in values => create it with the target_xp
            xp_init = xp_target

        case "direct":
            # We override the xp from the passed in values => create it with the other_xp,
            # so that we explicitly use that case.
            xp_init = xp_other
            array_args["xp"] = xp_target

    values = xp_init.full(shape, 1., dtype=getattr(xp_init, init_dtype_name))
    array_args["values"] = values

    # Eager is always inferred from the default setting since there is no override parameter.
    with fa.default_eager(eager):
        arr = fa.array(array_args.pop("values"), array_args.pop("dims"), array_args.pop("space"), **array_args)

    assert arr.xp == xp_target
    assert arr.dtype == result_dtype
    assert arr.shape == shape
    assert arr.eager == (eager,)*ndims

    # Do the exact same path that the values in the test pass through
    # just directly in the array API namespace.
    # That way we ensure that type promotion and conversion via fftarray
    # work the same way as with the underlying libraries.
    values = xp_init.full(shape, 1., dtype=getattr(xp_init, init_dtype_name))
    values_ref = xp_target.asarray(values, copy=True)

    try:
        # For array libraries with immutable arrays (e.g. jax), we assume this fails.
        # In these cases, we skip testing immutability ourself.
        values += 2
    except(TypeError):
        pass

    if defensive_copy:
        assert xp_target.all(arr.values("pos") == values_ref)
    # If not copy, we cannot test for inequality because aliasing behavior
    # is not defined and for jax for example an inequality check would fail.

    if ndims > 0:
        wrong_shape = list(shape)
        wrong_shape[0] = 10
        values = xp_target.full(tuple(wrong_shape), 1., dtype=result_dtype)
        with pytest.raises(ValueError):
            arr = fa.array(values, dims, "pos")

@pytest.mark.parametrize("xp_target, xp_other", XPS_ROTATED_PAIRS)
@pytest.mark.parametrize("xp_source", ["default", "direct"])
@pytest.mark.parametrize("defensive_copy", [False, True])
@pytest.mark.parametrize("dtype_name", dtype_names_numeric_core)
@pytest.mark.parametrize("eager", [False, True])
def test_from_list(
        xp_target,
        xp_other,
        xp_source: Literal["default", "direct"],
        defensive_copy: bool,
        dtype_name: DTYPE_NAME,
        eager: bool,
    ) -> None:
    """
        Test array creation from a list.
        This has two cases for xp derivation:
        1) From default xp.
        2) Via direct override.
    """

    dtype = getattr(xp_target, dtype_name)
    x_dim = fa.dim("x", n=3, d_pos=0.1, pos_min=0, freq_min=0)
    y_dim = fa.dim("y", n=2, d_pos=0.1, pos_min=0, freq_min=0)

    array_args = dict(
        defensive_copy=defensive_copy,
        dtype=dtype,
    )


    match xp_source:
        case "default":
            default_xp = xp_target
        case "direct":
            default_xp = xp_other
            array_args["xp"] = xp_target

    check_array_from_list(
        xp_target=xp_target,
        default_xp=default_xp,
        dims=[x_dim],
        vals_list = [1,2,3],
        array_args=array_args,
        dtype=dtype,
        eager=eager,
    )
    check_array_from_list(
        xp_target=xp_target,
        default_xp=default_xp,
        dims=[x_dim, y_dim],
        vals_list = [[1,4],[2,5],[3,6]],
        array_args=array_args,
        dtype=dtype,
        eager=eager,
    )


    with fa.default_eager(eager):
        with fa.default_xp(default_xp):
            # Test that inhomogeneous list triggers the correct error.
            with pytest.raises(ValueError):
                fa.array([1,[2]], [x_dim], "pos", **array_args)

def check_array_from_list(
        xp_target,
        default_xp,
        dims: Iterable[fa.Dimension],
        vals_list,
        array_args: Dict[str, Any],
        dtype,
        eager: bool,
    ) -> None:
    ref_vals = xp_target.asarray(vals_list, dtype=dtype)

    with fa.default_eager(eager):
        with fa.default_xp(default_xp):
            arr = fa.array(vals_list, dims, "pos", **array_args)
    arr_vals = arr.values("pos")

    assert arr.xp == xp_target
    assert arr.shape == ref_vals.shape
    assert arr.dtype == ref_vals.dtype
    assert arr.eager == (eager,)*len(arr.shape)
    assert type(arr_vals) is type(ref_vals)
    np.testing.assert_equal(
        np.array(arr_vals),
        np.array(vals_list),
    )


@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("fill_value, direct_dtype_name",
    [
        pytest.param(5, None),
        pytest.param(5., None),
        pytest.param(5.+1.j, None),
        pytest.param(5, "uint32"),
        pytest.param(5, "int64"),
        pytest.param(5, "float64"),
        pytest.param(5, "complex64"),
        pytest.param(5., "float64"),
        pytest.param(5., "complex64"),
        pytest.param(5.+1.j, "complex64"),
        pytest.param(5.+1.j, "complex128"),
    ]
)
@pytest.mark.parametrize("ndims", [0,1,2])
@pytest.mark.parametrize("eager", [False, True])
def test_full_scalar(
        xp,
        fill_value,
        direct_dtype_name: Optional[DTYPE_NAME],
        ndims: int,
        eager: bool,
    ) -> None:

    if direct_dtype_name is None:
        # dtype not specified, thus should be inferred by xp from the fill_value
        direct_dtype = None
        expected_dtype = xp.full(1, fill_value).dtype
    else:
        # dtype is explicity specified in array creation, overwrites fill_value dtype
        direct_dtype = getattr(xp, direct_dtype_name)
        expected_dtype = direct_dtype

    check_full(
        xp=xp,
        fill_value=fill_value,
        direct_dtype=direct_dtype,
        expected_dtype=expected_dtype,
        ndims=ndims,
        eager=eager,
    )

@pytest.mark.parametrize("xp, xp_other", XPS_ROTATED_PAIRS)
@pytest.mark.parametrize("xp_source", ["fill_value", "direct"])
@pytest.mark.parametrize("init_dtype_name, direct_dtype_name", dtypes_names_pairs)
@pytest.mark.parametrize("ndims", [0,1,2])
@pytest.mark.parametrize("eager", [True, False])
def test_full_array(
        xp,
        xp_other,
        xp_source: Literal["fill_value", "direct"],
        init_dtype_name: DTYPE_NAME,
        direct_dtype_name: Optional[DTYPE_NAME],
        ndims: int,
        eager: bool,
    ) -> None:

    match xp_source:
        case "fill_value":
            init_xp = xp
        case "direct":
            init_xp = xp_other

    # Define xp array according to xp_source with fill_value using init_dtype_name
    fill_value = init_xp.asarray(5, dtype=getattr(init_xp, init_dtype_name))

    if direct_dtype_name is None:
        # This case means the result dtype should be inferred from the fill value.
        expected_dtype = getattr(xp, init_dtype_name)
        direct_dtype = None
    else:
        # This case means the result dtype is explicitly specified by direct_dtype_name.
        expected_dtype = getattr(xp, direct_dtype_name)
        direct_dtype = expected_dtype

    check_full(
        xp=xp,
        fill_value=fill_value,
        direct_dtype=direct_dtype,
        expected_dtype=expected_dtype,
        ndims=ndims,
        eager=eager,
    )

def check_full(
        xp,
        fill_value,
        direct_dtype,
        expected_dtype,
        ndims: int,
        eager: bool,
    ) -> None:

    dims_list = get_dims(ndims)
    shape = tuple(dim.n for dim in dims_list)

    if len(dims_list) == 1:
        dims: Union[fa.Dimension, List[fa.Dimension]] = dims_list[0]
    else:
        dims = dims_list

    with fa.default_eager(eager):
        arr = fa.full(dims, "pos", fill_value, xp=xp, dtype=direct_dtype)

    arr_values = arr.values("pos")
    ref_arr = xp.full(shape, xp.asarray(fill_value, dtype=direct_dtype))
    assert arr.dtype == expected_dtype
    assert arr.eager == (eager,)*ndims
    assert arr.factors_applied == (True,)*ndims

    assert type(ref_arr) is type(arr_values)
    np.testing.assert_array_equal(
        np.array(ref_arr),
        np.array(arr_values),
    )