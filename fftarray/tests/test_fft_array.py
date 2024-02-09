from typing import List, Type
from functools import reduce
from itertools import product

import pytest
from hypothesis import given, strategies as st, note, settings, assume
import numpy as np
import jax

from fftarray.fft_array import FFTArray, FFTDimension
from fftarray.fft_constraint_solver import fft_dim_from_constraints
from fftarray.backends.jax_backend import JaxTensorLib
from fftarray.backends.np_backend import NumpyTensorLib
from fftarray.backends.pyfftw_backend import PyFFTWTensorLib
from fftarray.backends.tensor_lib import TensorLib, PrecisionSpec
from fftarray.xr_helpers import as_xr_pos

jax.config.update("jax_enable_x64", True)

def assert_scalars_almost_equal_nulp(x, y, nulp = 1):
    np.testing.assert_array_almost_equal_nulp(np.array([x]), np.array([y]), nulp = nulp)

tensor_libs: List[Type[TensorLib]] = [NumpyTensorLib, JaxTensorLib, PyFFTWTensorLib]
precisions: List[PrecisionSpec] = ["fp32", "fp64", "default"]

@pytest.mark.parametrize("tlib", tensor_libs)
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

    assert x_dim._index_from_coord(0.5, method = None, space="pos") == 0
    assert x_dim._index_from_coord(2.5, method = None, space="pos") == 2
    assert x_dim._index_from_coord(0.4, method = "nearest", space="pos") == 0
    assert x_dim._index_from_coord(2.6, method = "nearest", space="pos") == 2


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

    def test_jittable(x_dim, arr_2d):
        return (
            x_dim._index_from_coord(0.4, method = "nearest", space="pos"),
            x_dim._index_from_coord(2.6, method = "nearest", space="pos"),
            arr_2d.sel(x=1,y=3.4, method="nearest"),
            arr_2d.sel(x=-100,y=3.4, method="nearest"),
            arr_2d.loc[:],
            arr_2d.isel(x=3, y=2),
        )
    if do_jit:
        test_jittable = jax.jit(test_jittable)

    jit_res = test_jittable(x_dim=x_dim, arr_2d=arr_2d)
    assert jit_res[0] == 0
    assert jit_res[1] == 2
    assert np.array_equal(jit_res[2], xr_arr.sel(x=1,y=3.4, method="nearest").expand_dims({"x": 1, "y": 1}))
    assert np.array_equal(jit_res[3], xr_arr.sel(x=-100,y=3.4, method="nearest").expand_dims({"x": 1, "y": 1}))
    assert np.array_equal(jit_res[4], xr_arr.loc[:])
    assert np.array_equal(jit_res[5], xr_arr.isel(x=3, y=2).expand_dims({"x": 1, "y": 1}))


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("tensor_lib", tensor_libs)
@pytest.mark.parametrize("precision", ("fp32", "fp64", "default"))
@pytest.mark.parametrize("override", (None, "fp32", "fp64", "default"))
@pytest.mark.parametrize("eager", [False, True])
def test_dtype(tensor_lib, precision, override, eager: bool):
    tlib = tensor_lib(precision=precision)
    tlib_override = tensor_lib(precision=override)
    x_dim = FFTDimension("x",
        n=4,
        d_pos=1,
        pos_min=0.,
        freq_min=0.,
        default_tlib=tlib,
        default_eager=eager,
    )

    if override is None:
        assert x_dim.fft_array(space="pos").values.dtype == tlib.real_type
    else:
        assert x_dim.fft_array(space="pos", tlib=tlib_override).values.dtype == tlib_override.real_type
        assert x_dim.fft_array(space="pos").into(tlib=tlib_override).values.dtype == tlib_override.real_type


    if override is None:
        assert x_dim.fft_array(space="freq").values.dtype == tlib.real_type
    else:
        assert x_dim.fft_array(space="freq", tlib=tlib_override).values.dtype == tlib_override.real_type
        assert x_dim.fft_array(space="freq").into(tlib=tlib_override).values.dtype == tlib_override.real_type

    assert x_dim.fft_array(space="pos").into(space="freq").values.dtype == tlib.complex_type
    assert x_dim.fft_array(space="freq").into(space="pos").values.dtype == tlib.complex_type

    assert np.abs(x_dim.fft_array(space="pos").into(space="freq")).values.dtype == tlib.real_type # type: ignore
    assert np.abs(x_dim.fft_array(space="freq").into(space="pos")).values.dtype == tlib.real_type # type: ignore

    if override is not None:
        assert x_dim.fft_array(space="pos").into(space="freq", tlib=tlib_override).values.dtype == tlib_override.complex_type
        assert x_dim.fft_array(space="freq").into(space="pos", tlib=tlib_override).values.dtype == tlib_override.complex_type


@pytest.mark.parametrize("tensor_lib", tensor_libs)
@pytest.mark.parametrize("override", tensor_libs)
def test_backend_override(tensor_lib, override):
    x_dim = FFTDimension("x",
        n=4,
        d_pos=1,
        pos_min=0.,
        freq_min=0.,
        default_tlib=tensor_lib(),
    )

    x_dim_override = FFTDimension("x",
        n=4,
        d_pos=1,
        pos_min=0.,
        freq_min=0.,
        default_tlib=override(),
    )

    assert type(x_dim.fft_array(space="pos", tlib=override()).values) == type(x_dim_override.fft_array(space="pos").values)
    assert type(x_dim.fft_array(space="freq", tlib=override()).values) == type(x_dim_override.fft_array(space="freq").values)
    assert type(x_dim.fft_array(space="pos", tlib=override()).into(space="freq").values) == type(x_dim_override.fft_array(space="freq").values)
    assert type(x_dim.fft_array(space="freq", tlib=override()).into(space="pos").values) == type(x_dim_override.fft_array(space="freq").values)

    assert type(x_dim.fft_array(space="pos").into(tlib=override()).values) == type(x_dim_override.fft_array(space="pos").values)
    assert type(x_dim.fft_array(space="freq").into(tlib=override()).values) == type(x_dim_override.fft_array(space="freq").values)
    assert type(x_dim.fft_array(space="pos").into(tlib=override()).into(space="freq").values) == type(x_dim_override.fft_array(space="freq").values)
    assert type(x_dim.fft_array(space="freq").into(tlib=override()).into(space="pos").values) == type(x_dim_override.fft_array(space="pos").values)

    assert type(x_dim.fft_array(space="pos").into(space="freq", tlib=override()).values) == type(x_dim_override.fft_array(space="freq").values)
    assert type(x_dim.fft_array(space="freq").into(space="pos", tlib=override()).values) == type(x_dim_override.fft_array(space="freq").values)


def test_broadcasting(nulp: int = 1) -> None:
    x_dim = FFTDimension("x", n=4, d_pos=1, pos_min=0., freq_min=0.)
    y_dim = FFTDimension("y", n=8, d_pos=1, pos_min=0., freq_min=0.)

    x_ref = np.arange(0., 4.)
    y_ref = np.arange(0., 8.)
    np.testing.assert_array_almost_equal_nulp(np.array(x_dim.fft_array(space="pos")), x_ref, nulp = 0)
    np.testing.assert_array_almost_equal_nulp(np.array(y_dim.fft_array(space="pos")), y_ref, nulp = 0)

    x_ref_broadcast = x_ref.reshape(1,-1)
    y_ref_broadcast = y_ref.reshape(-1,1)
    np.testing.assert_array_almost_equal_nulp((x_dim.fft_array(space="pos") + y_dim.fft_array(space="pos")).transpose("x", "y").values, (x_ref_broadcast+y_ref_broadcast).transpose(), nulp = 0)
    np.testing.assert_array_almost_equal_nulp((x_dim.fft_array(space="pos") + y_dim.fft_array(space="pos")).transpose("y", "x").values, x_ref_broadcast+y_ref_broadcast, nulp = 0)

@given(
    value=st.one_of([
        st.integers(min_value=np.iinfo(np.int32).min, max_value=np.iinfo(np.int32).max),
        st.complex_numbers(allow_infinity=False, allow_nan=False, allow_subnormal=False),
        st.floats(allow_infinity=False, allow_nan=False, allow_subnormal=False)
    ]),
    init_factors_applied=st.booleans(),
    eager=st.booleans(),
    init_space=st.sampled_from(["pos", "freq"]),
    tlib=st.sampled_from(tensor_libs),
    precision=st.sampled_from(precisions),
    ndims=st.integers(min_value=1, max_value=4)
)
@settings(max_examples=500, deadline=None)
def test_attributes(value, init_factors_applied, eager, tlib, precision, init_space, ndims):
    dims = [
        FFTDimension(f"{ndim}", n=4, d_pos=0.1, pos_min=-0.2, freq_min=-2.1, default_eager=eager, default_tlib=tlib(precision=precision))
    for ndim in range(ndims)]
    note(dims)
    fftarr_values = np.full(tuple([dim.n for dim in dims]), value)
    note(fftarr_values.dtype)
    note(fftarr_values)

    if precision == "fp32":
        if isinstance(value, float):
            assume(np.abs(value) >= np.finfo(np.float32).smallest_normal and np.abs(value) <= np.finfo(np.float32).max)
        if isinstance(value, complex):
            assume(np.abs(value.real) >= np.finfo(np.float32).smallest_normal and np.abs(value.real) <= np.finfo(np.float32).max)
            assume(np.abs(value.imag) >= np.finfo(np.float32).smallest_normal and np.abs(value.imag) <= np.finfo(np.float32).max)
    if precision == "fp64":
        if isinstance(value, complex):
            assume(np.abs(value) >= np.finfo(np.float64).smallest_normal and np.abs(value) <= np.finfo(np.float64).max)
        if isinstance(value, complex):
            assume(np.abs(value.real) >= np.finfo(np.float64).smallest_normal and np.abs(value.real) <= np.finfo(np.float64).max)
            assume(np.abs(value.imag) >= np.finfo(np.float64).smallest_normal and np.abs(value.imag) <= np.finfo(np.float64).max)

    # test eager dim and eager fftarray error
    # with pytest.raises(Exception):
    #     FFTArray(
    #         values=fftarr_values,
    #         dims=dims,
    #         space=init_space,
    #         eager=not eager,
    #         factors_applied=init_factor_applied
    #     )

    fftarr = FFTArray(
        values=fftarr_values,
        dims=dims,
        space=init_space,
        eager=eager,
        factors_applied=init_factors_applied
    )
    note(fftarr)

    if init_factors_applied:
        # fftarray must be handled the same way as applying the operations to the values numpy array
        np.testing.assert_array_equal(fftarr.values, fftarr_values, strict=True)
    elif init_space == "pos":
        # factors not applied, pos space
        assert_invariance_to_factors_equivalence(fftarr, fftarr_values, exact=True)
    elif ndims == 1:
        # factors not applied, freq space
        assert_invariance_to_factors_equivalence(fftarr, fftarr_values/(dims[0].n*dims[0].d_freq), exact=False)

    assert_single_operand_fun_equivalence(fftarr, init_factors_applied)
    assert_dual_operand_fun_equivalence(fftarr, init_factors_applied)

def is_inf_or_nan(x):
    """Check if (real or imag of) x is inf or nan"""
    return (np.isinf(x.real).any() or np.isinf(x.imag).any() or np.isnan(x.real).any() or np.isnan(x.imag).any())

def assert_equal_op(arr, values, op, exact=True):
    """Helper function to test equality between an FFTArray and a values array.
    `op` denotes the operation acting on the FFTArray and on the values before
    comparison.
    `exact` denotes whether the comparison is performed using nulp (number of
    unit in the last place for tolerance) or using the less stringent
    `numpy.testing.allclose`.
    """
    arr_op = op(arr).values
    values_op = op(values)

    if arr_op.dtype != values_op.dtype:
        note(f"Changing type to {values_op.dtype}")
        arr_op = arr_op.astype(values_op.dtype)
        values_op = values_op.astype(values_op.dtype)

    if is_inf_or_nan(values_op) or (exact==False and is_inf_or_nan(arr_op)):
        return

    if exact and ("int" in str(values.dtype) or arr.tlib.precision == "fp64"):
        if "int" in str(values.dtype):
            np.testing.assert_array_equal(arr_op, values_op, strict=True)
        if "float" in str(values.dtype):
            np.testing.assert_array_almost_equal_nulp(arr_op, values_op, nulp=4)
        if "complex" in str(values.dtype):
            assert_array_almost_equal_nulp_complex(arr_op, values_op, nulp=4)
    else:
        rtol = 1e-6 if isinstance(arr.tlib, JaxTensorLib) else 1e-7
        np.testing.assert_allclose(arr_op, values_op, rtol=rtol)

def assert_array_almost_equal_nulp_complex(x, y, nulp):
    """Compare two arrays of complex numbers. Simply compares the real and
    imaginary part.
    """
    np.testing.assert_array_almost_equal_nulp(x.real, y.real, nulp)
    np.testing.assert_array_almost_equal_nulp(x.imag, y.imag, nulp)

def assert_single_operand_fun_equivalence(arr, exact):
    """Test whether applying operands to the FFTArray (and then getting the
    values) is equivalent to applying the same operands to the values array:

        operand(FFTArray).values == operand(FFTArray.values)

    """
    values = arr.values
    note("f(x) = x")
    assert_equal_op(arr, values, lambda x: x, exact)
    note("f(x) = abs(x)")
    assert_equal_op(arr, values, lambda x: np.abs(x), exact)
    note("f(x) = x**2")
    assert_equal_op(arr, values, lambda x:  x**2, False) # Exact comparison fails
    note("f(x) = x**3")
    assert_equal_op(arr, values, lambda x:  x**3, False) # Exact comparison fails

def assert_invariance_to_factors_equivalence(arr, values, exact):
    """Test whether the absolute of the FFTArray initialized with
    factors_applied=False and space="pos" is equivalent to the values it was
    initialized with. This should be true as the factors in position space are
    only phases (which drop out in np.abs).
    """
    assert_equal_op(arr, values, lambda x: np.abs(x), exact)

def assert_dual_operand_fun_equivalence(arr, exact):
    """Test whether a dual operation on an FFTArray, e.g., the
    sum/multiplication of two, is equivalent to applying this operand to its
    values.

        operand(FFTArray, FFTArray).values = operand(FFTArray.values, FFTArray.values)

    """
    values = arr.values
    note("f(x,y) = x+y")
    assert_equal_op(arr, values, lambda x: x+x, exact)
    note("f(x,y) = x*y")
    assert_equal_op(arr, values, lambda x: x*x, False) # Exact comparison fails

# @pytest.mark.parametrize("eager", [False, True])
# def test_lazy_0(eager: bool) -> None:
#     dim_pos_x = fft_dim_from_constraints("x", n = 4, d_pos = 1., pos_min = 0.3, freq_min = 0.7, default_eager=eager)
#     dim_pos_y = fft_dim_from_constraints("y", n = 4, d_pos = 1., pos_min = 1.3, freq_min = 1.7, default_eager=eager)
#     dim_freq_x = fft_dim_from_constraints("x", n = 4, d_freq = 1., pos_min = 0.7, freq_min = 0.3, default_eager=eager)
#     dim_freq_y = fft_dim_from_constraints("y", n = 4, d_freq = 1., pos_min = 1.7, freq_min = 1.3, default_eager=eager)

#     ref_values = np.arange(4).reshape(4,1)+0.3 + np.arange(4).reshape(1,4)+1.3
#     arrs = [
#         (dim_pos_x.fft_array(space="pos") + dim_pos_y.fft_array(space="pos")).transpose("x", "y"),
#         (dim_freq_x.fft_array(space="freq") + dim_freq_y.fft_array(space="freq")).transpose("x", "y"),
#     ]
#     for arr in arrs:
#         np.testing.assert_array_almost_equal(arr.into(space="freq").into(space="pos").into(space="freq").values, arr.into(space="freq").values)
#         np.testing.assert_array_almost_equal(arr.values, ref_values)

# def _get_fft_arr(dims: List[FFTDimension], per_dim_values) -> FFTArray:
#     return reduce(lambda x,y: x+y, [
#         FFTArray(
#             values=per_dim_values,
#             dims=[dim],
#             space="pos",
#             eager=False,
#             factors_applied=True,
#         )
#         for dim in dims
#     ])

# arrs = []
# for tlib, precision in product(tensor_libs, precisions):
#     tlib_obj = tlib(precision=precision)
#     x_dim = FFTDimension("x",
#         n=4,
#         d_pos=1,
#         pos_min=0.5,
#         freq_min=0.,
#         default_tlib=tlib_obj,
#     )
#     y_dim = FFTDimension("y",
#         n=4,
#         d_pos=2,
#         pos_min=-2,
#         freq_min=0.,
#         default_tlib=tlib_obj,
#     )
#     for dims in [[x_dim], [x_dim, y_dim]]:
#         arrs.append(_get_fft_arr(dims, tlib_obj.array([0., 1., 2., 3.])))
#         arrs.append(_get_fft_arr(dims, tlib_obj.array([0., 1., 2., 3.]) + 1.j)) # type: ignore
#         arrs.append(_get_fft_arr(dims, tlib_obj.array([0, 1, 2, 3])))


# @pytest.mark.parametrize("arr", arrs)
# def test_lazy_1(arr):
#     assert_single_operand_fun_equivalence(arr)
#     assert_dual_operand_fun_equivalence(arr)
