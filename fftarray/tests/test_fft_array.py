from typing import List, Type, Any, Callable, Union, Tuple

import pytest
from hypothesis import given, strategies as st, note, settings
import numpy as np
import jax

from fftarray.fft_array import FFTArray, FFTDimension, Space
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

@pytest.mark.parametrize("tlib_class", tensor_libs)
@pytest.mark.parametrize("do_jit", [False, True])
def test_indexing(tlib_class, do_jit: bool) -> None:
    if do_jit and type(tlib_class) != JaxTensorLib:
        return

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

    tlib=tlib_class(precision="default")

    arr_2d = x_dim.fft_array(tlib, space="pos") + y_dim.fft_array(tlib, space="pos")**2
    xr_arr = as_xr_pos(arr_2d)

    assert x_dim._index_from_coord(0.5, method = None, space="pos", tlib=tlib) == 0
    assert x_dim._index_from_coord(2.5, method = None, space="pos", tlib=tlib) == 2
    assert x_dim._index_from_coord(0.4, method = "nearest", space="pos", tlib=tlib) == 0
    assert x_dim._index_from_coord(2.6, method = "nearest", space="pos", tlib=tlib) == 2


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
            x_dim._index_from_coord(0.4, method = "nearest", space="pos", tlib=arr_2d.tlib),
            x_dim._index_from_coord(2.6, method = "nearest", space="pos", tlib=arr_2d.tlib),
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
def test_dtype(tensor_lib, precision, override, eager: bool) -> None:
    tlib = tensor_lib(precision=precision)
    tlib_override = tensor_lib(precision=override)
    x_dim = FFTDimension("x",
        n=4,
        d_pos=1,
        pos_min=0.,
        freq_min=0.,
    )

    if override is None:
        assert x_dim.fft_array(tlib, space="pos").values.dtype == tlib.real_type
    else:
        assert x_dim.fft_array(tlib_override, space="pos", eager=eager).values.dtype == tlib_override.real_type
        assert x_dim.fft_array(tlib, space="pos", eager=eager).into(tlib=tlib_override).values.dtype == tlib_override.real_type


    if override is None:
        assert x_dim.fft_array(tlib, space="freq", eager=eager).values.dtype == tlib.real_type
    else:
        assert x_dim.fft_array(tlib_override, space="freq", eager=eager).values.dtype == tlib_override.real_type
        assert x_dim.fft_array(tlib, space="freq", eager=eager).into(tlib=tlib_override).values.dtype == tlib_override.real_type

    assert x_dim.fft_array(tlib, space="pos", eager=eager).into(space="freq").values.dtype == tlib.complex_type
    assert x_dim.fft_array(tlib, space="freq", eager=eager).into(space="pos").values.dtype == tlib.complex_type

    assert np.abs(x_dim.fft_array(tlib, space="pos", eager=eager).into(space="freq")).values.dtype == tlib.real_type # type: ignore
    assert np.abs(x_dim.fft_array(tlib, space="freq", eager=eager).into(space="pos")).values.dtype == tlib.real_type # type: ignore

    if override is not None:
        assert x_dim.fft_array(tlib, space="pos", eager=eager).into(space="freq", tlib=tlib_override).values.dtype == tlib_override.complex_type
        assert x_dim.fft_array(tlib, space="freq", eager=eager).into(space="pos", tlib=tlib_override).values.dtype == tlib_override.complex_type


@pytest.mark.parametrize("tensor_lib", tensor_libs)
@pytest.mark.parametrize("override", tensor_libs)
def test_backend_override(tensor_lib, override) -> None:
    x_dim = FFTDimension("x",
        n=4,
        d_pos=1,
        pos_min=0.,
        freq_min=0.,
    )

    assert type(x_dim.fft_array(tensor_lib(), space="pos").into(tlib=override()).values) == type(x_dim.fft_array(override(), space="pos").values)
    assert type(x_dim.fft_array(tensor_lib(), space="freq").into(tlib=override()).values) == type(x_dim.fft_array(override(), space="freq").values)
    assert type(x_dim.fft_array(tensor_lib(), space="pos").into(tlib=override()).into(space="freq").values) == type(x_dim.fft_array(override(), space="freq").values)
    assert type(x_dim.fft_array(tensor_lib(), space="freq").into(tlib=override()).into(space="pos").values) == type(x_dim.fft_array(override(), space="pos").values)

    assert type(x_dim.fft_array(tensor_lib(), space="pos").into(space="freq", tlib=override()).values) == type(x_dim.fft_array(override(), space="freq").values)
    assert type(x_dim.fft_array(tensor_lib(), space="freq").into(space="pos", tlib=override()).values) == type(x_dim.fft_array(override(), space="freq").values)


def test_broadcasting(nulp: int = 1) -> None:
    x_dim = FFTDimension("x", n=4, d_pos=1, pos_min=0., freq_min=0.)
    y_dim = FFTDimension("y", n=8, d_pos=1, pos_min=0., freq_min=0.)

    x_ref = np.arange(0., 4.)
    y_ref = np.arange(0., 8.)
    np.testing.assert_array_almost_equal_nulp(np.array(x_dim.fft_array(tlib=NumpyTensorLib(), space="pos")), x_ref, nulp = 0)
    np.testing.assert_array_almost_equal_nulp(np.array(y_dim.fft_array(tlib=NumpyTensorLib(), space="pos")), y_ref, nulp = 0)

    x_ref_broadcast = x_ref.reshape(1,-1)
    y_ref_broadcast = y_ref.reshape(-1,1)
    np.testing.assert_array_almost_equal_nulp((x_dim.fft_array(tlib=NumpyTensorLib(), space="pos") + y_dim.fft_array(tlib=NumpyTensorLib(), space="pos")).transpose("x", "y").values, (x_ref_broadcast+y_ref_broadcast).transpose(), nulp = 0)
    np.testing.assert_array_almost_equal_nulp((x_dim.fft_array(tlib=NumpyTensorLib(), space="pos") + y_dim.fft_array(tlib=NumpyTensorLib(), space="pos")).transpose("y", "x").values, x_ref_broadcast+y_ref_broadcast, nulp = 0)

@given(
    value=st.one_of([
        st.integers(min_value=np.iinfo(np.int32).min, max_value=np.iinfo(np.int32).max),
        st.complex_numbers(allow_infinity=False, allow_nan=False, allow_subnormal=False, width=64),
        st.floats(allow_infinity=False, allow_nan=False, allow_subnormal=False, width=32)
    ]),
    init_factors_applied=st.booleans(),
    eager=st.booleans(),
    init_space=st.sampled_from(["pos", "freq"]),
    tlib=st.sampled_from(tensor_libs),
    precision=st.sampled_from(precisions),
    ndims=st.integers(min_value=1, max_value=4)
)
@settings(max_examples=1000, deadline=None)
def test_attributes(value, init_factors_applied, eager, tlib, precision, init_space, ndims):
    dims = [
        FFTDimension(f"{ndim}", n=4, d_pos=0.1, pos_min=-0.2, freq_min=-2.1)
    for ndim in range(ndims)]
    note(dims)
    fftarr_values = np.full(tuple([dim.n for dim in dims]), value)
    note(fftarr_values.dtype)
    note(fftarr_values)

    fftarr = FFTArray(
        values=fftarr_values,
        dims=dims,
        space=init_space,
        eager=eager,
        factors_applied=init_factors_applied,
        tlib=tlib(precision=precision)
    )
    note(fftarr)

    # -- basic tests
    if init_factors_applied:
        # fftarray must be handled the same way as applying the operations to the values numpy array
        np.testing.assert_array_equal(fftarr.values, fftarr_values, strict=True)
    elif init_space == "pos":
        # factors not applied, pos space
        assert_invariance_to_factors_equivalence(fftarr, fftarr_values, exact=True)
    elif ndims == 1:
        # factors not applied, freq space
        assert_invariance_to_factors_equivalence(fftarr, fftarr_values/(dims[0].n*dims[0].d_freq), exact=False)

    # -- test operands
    assert_single_operand_fun_equivalence(fftarr, init_factors_applied)
    assert_dual_operand_fun_equivalence(fftarr, init_factors_applied)
    assert_special_behavior_lazy(fftarr)

    # Max 4 dimensions (or PyFFTWTensorLib)
    if (ndims < 4 or isinstance(tlib, PyFFTWTensorLib)):

        # -- test ffts
        assert_fft_ifft_invariance(fftarr, init_space)

        # -- test eager, factors_applied logic
        assert_fftarray_eager_factors_applied(fftarr)

    # Multidimensional
    if ndims > 1:
        # -- test fftarray sel
        assert_fftarray_sel_order(fftarr)


def is_inf_or_nan(x):
    """Check if (real or imag of) x is inf or nan"""
    return (np.isinf(x.real).any() or np.isinf(x.imag).any() or np.isnan(x.real).any() or np.isnan(x.imag).any())

def assert_equal_op(arr: FFTArray, values: Any, op: Callable[[Any],Any], exact=True):
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

def assert_array_almost_equal_nulp_complex(x: Any, y: Any, nulp: int):
    """Compare two arrays of complex numbers. Simply compares the real and
    imaginary part.
    """
    np.testing.assert_array_almost_equal_nulp(x.real, y.real, nulp)
    np.testing.assert_array_almost_equal_nulp(x.imag, y.imag, nulp)

def assert_single_operand_fun_equivalence(arr: FFTArray, exact: bool):
    """Test whether applying operands to the FFTArray (and then getting the
    values) is equivalent to applying the same operands to the values array:

        operand(FFTArray).values == operand(FFTArray.values)

    """
    values = arr.values
    note("f(x) = x")
    assert_equal_op(arr, values, lambda x: x, exact)
    note("f(x) = pi*x")
    assert_equal_op(arr, values, lambda x: np.pi*x, exact)
    note("f(x) = abs(x)")
    assert_equal_op(arr, values, lambda x: np.abs(x), exact)
    note("f(x) = x**2")
    assert_equal_op(arr, values, lambda x:  x**2, False) # Exact comparison fails
    note("f(x) = x**3")
    assert_equal_op(arr, values, lambda x:  x**3, False) # Exact comparison fails

def assert_invariance_to_factors_equivalence(arr: FFTArray, values: Any, exact: bool):
    """Test whether the absolute of the FFTArray initialized with
    factors_applied=False and space="pos" is equivalent to the values it was
    initialized with. This should be true as the factors in position space are
    only phases (which drop out in np.abs).
    """
    assert_equal_op(arr, values, lambda x: np.abs(x), exact)

def assert_dual_operand_fun_equivalence(arr: FFTArray, exact: bool):
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

def get_other_space(space: Union[Space, Tuple[Space, ...]]):
    """Returns the other space. If input space is "pos", "freq" is returned and
    vice versa. If space is a tuple of Space, a tuple is returned.
    """
    if isinstance(space, str):
        if space == "pos":
            return "freq"
        return "pos"
    return [get_other_space(s) for s in space]

def assert_special_behavior_lazy(arr: FFTArray):
    """Tests whether the factors are only applied when necessary. E.g., they
    should not be applied when taking the absolute value of an FFTArray (but
    the resulting FFTArray should behave as they have been applied)
    """
    note("abs(x)._factors_applied = True")
    arr_abs = np.abs(arr)
    np.testing.assert_array_equal(arr_abs._factors_applied, True)

def assert_fft_ifft_invariance(arr: FFTArray, init_space: Space):
    """Tests whether ifft(fft(*)) is an identity.

       ifft(fft(FFTArray)) == FFTArray

    """
    note("ifft(fft(x)) == x")
    other_space = get_other_space(init_space)
    arr_fft = arr.into(space=other_space)
    arr_fft_ifft = arr_fft.into(space=init_space)
    if is_inf_or_nan(arr_fft_ifft.values):
        # edge cases (very large numbers) result in inf after fft
        return
    rtol = 1e-5 if arr.tlib.precision == "fp32" else 1e-6
    np.testing.assert_allclose(arr.values, arr_fft_ifft.values, rtol=rtol, atol=1e-38)

def assert_fftarray_sel_order(arr: FFTArray):
    """Tests whether the selection order matters. Assuming an input FFTArray of
    dimensions A and B. Then

        FFTArray.sel(A==a).sel(B==b) == FFTArray.sel(B==b).sel(A==a)

    should be true.
    """
    note("fftarray.sel(A).sel(B) == fftarray.sel(B).sel(A)")
    dimA = arr.dims[0]
    dimB = arr.dims[1]
    arrA = arr.sel(**{dimA.name: getattr(dimA, f"{arr.space[0]}_middle")})
    arrB = arr.sel(**{dimB.name: getattr(dimB, f"{arr.space[1]}_middle")})
    arrAB = arrA.sel(**{dimB.name: getattr(dimB, f"{arr.space[1]}_middle")})
    arrBA = arrB.sel(**{dimA.name: getattr(dimA, f"{arr.space[0]}_middle")})
    np.testing.assert_allclose(arrAB.values, arrBA.values)

def assert_fftarray_eager_factors_applied(arr: FFTArray):
    """Tests whether the FFTArray after performing an FFT has the correct
    properties. If the initial FFTArray was eager, then the final FFTArray also
    must be eager and have _factors_apllied=True. If the initial FFTArray was
    not eager, then the final FFTArray should have eager=False and
    _factors_applied=False.
    """
    note("Testing eager and factors_applied ...")
    init_eager = arr.eager
    arr_fft = arr.into(space=get_other_space(arr.space))
    final_eager = arr_fft.eager
    final_factors_applied = arr_fft._factors_applied
    np.testing.assert_array_equal(init_eager, final_eager)
    for ffapplied, feager in zip(final_factors_applied, final_eager):
        assert (feager and ffapplied) or (not feager and not ffapplied)
