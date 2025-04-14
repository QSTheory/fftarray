from typing import (
    Iterable, Literal, Optional, Dict, Any, Union, List, get_args
)

import array_api_strict
import array_api_compat
import numpy as np
import pytest
import fftarray as fa

from tests.helpers import (
    XPS_WITH_DEFAULT_DEVICE_PAIRS, XPS_ROTATED_PAIRS, XPS_DEVICE_PAIRS, XPS_NON_DEFAULT_DEVICE_PAIRS,
    get_dims, dtypes_names_pairs, dtype_names_numeric_core, DTYPE_NAME, get_test_array,
    assert_fa_array_exact_equal,
)

def test_no_xp_conversion() -> None:
    """
        Tests that fa.array raises a ValueError if the user tries
        to implicitly convert between two different array libraries.
    """
    with pytest.raises(ValueError):
        fa.array(np.array(1), [], [], xp=array_api_strict)

@pytest.mark.parametrize("xp, device_param, device_res", XPS_DEVICE_PAIRS)
@pytest.mark.parametrize("init_dtype_name, result_dtype_name", dtypes_names_pairs)
@pytest.mark.parametrize("ndims", [0,1,2])
@pytest.mark.parametrize("defensive_copy", [False, True])
@pytest.mark.parametrize("eager", [False, True])
def test_from_array_object(
        xp,
        device_param: Optional[Any],
        device_res: Any,
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
        spaces="pos",
        defensive_copy=defensive_copy,
        device=device_param,
    )

    if result_dtype_name is None:
        # result_dtype_name is None means that the dtype is inferred from
        # the passed in values.
        # Therefore the expected dtype is the same than the one that is used
        # to create the array.
        result_dtype = getattr(xp, init_dtype_name)
    else:
        # In this case we explicitly override the dtype
        # in the creation of the array.
        result_dtype = getattr(xp, result_dtype_name)
        array_args["dtype"] = result_dtype


    values = xp.full(shape, 1., dtype=getattr(xp, init_dtype_name), device=device_param)
    array_args["values"] = values

    # Eager is always inferred from the default setting since there is no override parameter.
    with fa.default_eager(eager):
        arr = fa.array(array_args.pop("values"), array_args.pop("dims"), array_args.pop("spaces"), **array_args)

    assert arr.xp == xp
    assert arr.device == device_res
    assert arr.dtype == result_dtype
    assert arr.shape == shape
    assert arr.eager == (eager,)*ndims

    # Do the exact same path that the values in the test pass through
    # just directly in the array API namespace.
    # That way we ensure that type promotion and conversion via fftarray
    # work the same way as with the underlying libraries.
    values_ref = xp.asarray(values, copy=True, device=device_param)

    try:
        # For array libraries with immutable arrays (e.g. jax), we assume this fails.
        # In these cases, we skip testing immutability ourself.
        values += 2
    except(TypeError):
        pass

    if defensive_copy:
        assert xp.all(arr.values("pos") == values_ref)
    # If not copy, we cannot test for inequality because aliasing behavior
    # is not defined and for jax for example an inequality check would fail.

    if ndims > 0:
        wrong_shape = list(shape)
        wrong_shape[0] = 10
        values = xp.full(tuple(wrong_shape), 1., dtype=result_dtype)
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


@pytest.mark.parametrize("xp, device_param, device_res", XPS_DEVICE_PAIRS)
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
        device_param: Optional[Any],
        device_res: Any,
        fill_value,
        direct_dtype_name: Optional[DTYPE_NAME],
        ndims: int,
        eager: bool,
    ) -> None:

    if direct_dtype_name is None:
        direct_dtype = None
    else:
        # dtype is explicity specified in array creation,
        # overwrites the default dtype of the array library.
        direct_dtype = getattr(xp, direct_dtype_name)

    check_full(
        xp=xp,
        device_param=device_param,
        device_res=device_res,
        fill_value=fill_value,
        direct_dtype=direct_dtype,
        ndims=ndims,
        eager=eager,
    )

@pytest.mark.parametrize("xp, device_param, device_res", XPS_DEVICE_PAIRS)
@pytest.mark.parametrize("init_dtype_name, direct_dtype_name", dtypes_names_pairs)
@pytest.mark.parametrize("ndims", [0,1,2])
@pytest.mark.parametrize("eager", [True, False])
def test_full_array(
        xp,
        device_param: Optional[Any],
        device_res: Any,
        init_dtype_name: DTYPE_NAME,
        direct_dtype_name: Optional[DTYPE_NAME],
        ndims: int,
        eager: bool,
    ) -> None:
    # Define xp array according to xp with fill_value using init_dtype_name
    fill_value = xp.asarray(5, dtype=getattr(xp, init_dtype_name))

    if direct_dtype_name is None:
        direct_dtype = None
    else:
        # dtype is explicity specified in array creation,
        # overwrites the default dtype of the array library.
        direct_dtype = getattr(xp, direct_dtype_name)

    check_full(
        xp=xp,
        device_param=device_param,
        device_res=device_res,
        fill_value=fill_value,
        direct_dtype=direct_dtype,
        ndims=ndims,
        eager=eager,
    )

def check_full(
        xp,
        device_param: Optional[Any],
        device_res: Any,
        fill_value,
        direct_dtype,
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
        arr = fa.full(dims, "pos", fill_value, xp=xp, dtype=direct_dtype, device=device_param)

    arr_values = arr.values("pos")
    ref_arr = xp.full(shape, fill_value, dtype=direct_dtype, device=device_param)
    assert arr.dtype == ref_arr.dtype
    assert arr.device == device_res
    assert arr.eager == (eager,)*ndims
    assert arr.factors_applied == (True,)*ndims

    assert type(ref_arr) is type(arr_values)

    # FIXME: Technically this does not guarantuee convertability into NumPy
    # for any Array API compatible library.
    default_device = xp.__array_namespace_info__().default_device()
    np.testing.assert_equal(
        np.array(array_api_compat.to_device(ref_arr, default_device)),
        np.array(array_api_compat.to_device(arr_values, default_device)),
    )


@pytest.mark.parametrize("xp, device_param, device_res", XPS_NON_DEFAULT_DEVICE_PAIRS)
@pytest.mark.parametrize("device_source", ["arr", "direct"])
def test_coords_from_arr_devices(
            xp,
            device_param: Optional[Any],
            device_res: Any,
            device_source: Literal["arr", "direct"],
        ) -> None:
    """
        Test that non-default devices get correctly propagated and overridden.
        The default device route is already tested in ``test_coords_from_arr_dtype``.
    """
    check_coords_from_arr(
        xp=xp,
        device_param=device_param,
        device_res=device_res,
        device_source=device_source,
        dtype_name="complex128",
        dtype_override_name=None,
        res_dtype_name="float64",
    )


@pytest.mark.parametrize("xp, device_res", XPS_WITH_DEFAULT_DEVICE_PAIRS)
@pytest.mark.parametrize("dtype_name, dtype_override_name, res_dtype_name", [
    ("int64", None, None),
    ("float32", None, "float32"),
    ("float32", "complex64", None),
    ("float32", "float64", "float64"),
    ("complex128", None, "float64"),
])
def test_coords_from_arr_dtype(
            xp,
            device_res: Any,
            dtype_name: DTYPE_NAME,
            dtype_override_name: Optional[DTYPE_NAME],
            res_dtype_name: Optional[DTYPE_NAME],
        ) -> None:
    """
        This test tests different dtype combinations with the default device.
    """
    check_coords_from_arr(
        xp=xp,
        device_param=None,
        device_res=device_res,
        device_source="arr",
        dtype_name=dtype_name,
        dtype_override_name=dtype_override_name,
        res_dtype_name=res_dtype_name,
    )

def check_coords_from_arr(
            xp,
            device_param: Optional[Any],
            device_res: Any,
            device_source: Literal["arr", "direct"],
            dtype_name: DTYPE_NAME,
            dtype_override_name: Optional[DTYPE_NAME],
            res_dtype_name: Optional[DTYPE_NAME],
        ) -> None:
    """
        Test that ``coords_from_arr`` returns the correct coordinate array
        while correctly inferring dtype, device and xp from the given template array
        while honoring passed in overrides.
        ``device_param`` defines the device parameter which determines the final device.
        It can be passed either into the template array with ``device_source="arr"`` or
        as an override when ``device_source="direct"``.
        The other function (``coords_from_arr`` or the template array generation) gets passed
        ``device=None``.
        This way the correct behavior is tested when ``device_param`` is set to a non-default
        device, i.e., ``device_res`` is the expected resulting device after ``coords_from_arr``.
        ``res_dtype_name=None`` means that ``coords_from_arr`` is expected to raise
        a ``ValueError``.
        ``dtype_override_name=None`` means that ``coords_from_arr`` is expected to
        return the default float type of the used ``xp``.
    """
    space: fa.Space = "pos"
    dtype = getattr(xp, dtype_name)

    if dtype_override_name is None:
        dtype_override = None
    else:
        dtype_override = getattr(xp, dtype_override_name)

    if res_dtype_name is None:
        res_dtype = dtype
    else:
        res_dtype = getattr(xp, res_dtype_name)

    match device_source:
        case "arr":
            template_arr_device_arg = device_param
            call_device_param = None
        case "direct":
            template_arr_device_arg = None
            call_device_param = device_param

    dim = fa.dim("x", n=3, d_pos=0.1, pos_min=0, freq_min=0)
    template_arr: fa.Array = get_test_array(
        xp=xp,
        device=template_arr_device_arg,
        dim=dim,
        space=space,
        dtype=dtype,
        factors_applied=True,
    )

    if res_dtype_name is None:
        with pytest.raises(ValueError):
            fa.coords_from_arr(template_arr, "x", space, dtype=dtype_override, device=call_device_param)
        return

    ref_values = fa.coords_from_dim(dim, space, xp=xp, dtype=res_dtype, device=device_res)
    arr_from_arr = fa.coords_from_arr(template_arr, "x", space, dtype=dtype_override, device=call_device_param)
    assert_fa_array_exact_equal(ref_values, arr_from_arr)

@pytest.mark.parametrize("xp_target, xp_other", XPS_ROTATED_PAIRS)
@pytest.mark.parametrize("xp_source", ["default", "direct"])
@pytest.mark.parametrize("direct_dtype_name", [None, "float64", "int64", "complex64"])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("space", get_args(fa.Space))
def test_coords_from_dim(
        xp_target,
        xp_other,
        xp_source: Literal["default", "direct"],
        direct_dtype_name: Optional[DTYPE_NAME],
        eager: bool,
        space: fa.Space,
    ) -> None:
    """
        Test Array creation from a Dimension.
        This has two cases for xp derivation:
        1) From default xp.
        2) Via direct override.
    """

    if direct_dtype_name is None:
        direct_dtype = None
        expected_dtype = xp_target.full(1, 2.).dtype
    else:
        direct_dtype = getattr(xp_target, direct_dtype_name)
        expected_dtype = direct_dtype

    dim = fa.dim("x", n=3, d_pos=0.1, pos_min=0, freq_min=0)
    ref_values = xp_target.asarray(dim.values(space, xp=np), dtype=direct_dtype)

    array_args = {"dtype": direct_dtype}

    match xp_source:
        case "default":
            default_xp = xp_target
        case "direct":
            default_xp = xp_other
            array_args["xp"] = xp_target

    with fa.default_eager(eager):
        with fa.default_xp(default_xp):
            if direct_dtype is not None and not xp_target.isdtype(direct_dtype, ("real floating")):
                with pytest.raises(ValueError):
                    arr = fa.coords_from_dim(dim, space, **array_args)
                return
            arr = fa.coords_from_dim(dim, space, **array_args)

    arr_values = arr.values(space, xp=np)
    np.testing.assert_equal(arr_values, ref_values)
    assert arr.shape == (dim.n,)
    assert arr.xp == xp_target
    assert arr.dtype == expected_dtype
    assert arr.spaces == (space,)
    assert arr.eager == (eager,)
    assert arr.factors_applied == (True,)
