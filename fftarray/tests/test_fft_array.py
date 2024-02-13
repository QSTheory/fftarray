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
spaces: List[Space] = ["pos", "freq"]

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
    if override is None:
        tlib_override = None
    else:
        tlib_override = tensor_lib(precision=override)

    x_dim = FFTDimension("x",
        n=4,
        d_pos=1,
        pos_min=0.,
        freq_min=0.,
    )

    if tlib_override is None:
        assert x_dim.fft_array(tlib, space="pos").values.dtype == tlib.real_type
    else:
        assert x_dim.fft_array(tlib_override, space="pos", eager=eager).values.dtype == tlib_override.real_type
        assert x_dim.fft_array(tlib, space="pos", eager=eager).into(tlib=tlib_override).values.dtype == tlib_override.real_type


    if tlib_override is None:
        assert x_dim.fft_array(tlib, space="freq", eager=eager).values.dtype == tlib.real_type
    else:
        assert x_dim.fft_array(tlib_override, space="freq", eager=eager).values.dtype == tlib_override.real_type
        assert x_dim.fft_array(tlib, space="freq", eager=eager).into(tlib=tlib_override).values.dtype == tlib_override.real_type

    assert x_dim.fft_array(tlib, space="pos", eager=eager).into(space="freq").values.dtype == tlib.complex_type
    assert x_dim.fft_array(tlib, space="freq", eager=eager).into(space="pos").values.dtype == tlib.complex_type

    assert np.abs(x_dim.fft_array(tlib, space="pos", eager=eager).into(space="freq")).values.dtype == tlib.real_type # type: ignore
    assert np.abs(x_dim.fft_array(tlib, space="freq", eager=eager).into(space="pos")).values.dtype == tlib.real_type # type: ignore

    if tlib_override is not None:
        assert x_dim.fft_array(tlib, space="pos", eager=eager).into(space="freq", tlib=tlib_override).values.dtype == tlib_override.complex_type
        assert x_dim.fft_array(tlib, space="freq", eager=eager).into(space="pos", tlib=tlib_override).values.dtype == tlib_override.complex_type

    # For non-float and non-complex dtypes, we do not force the tlib precision types
    # onto the values. Therefore, the FFTArray.values dtype should not be affected by the
    # tlib_override precision for both integer values and boolean values

    int_arr = FFTArray(
        values=tlib.array([1,2,3,4]),
        dims=[x_dim],
        space="pos",
        eager=eager,
        tlib=tlib,
        factors_applied=True,
    )
    assert int_arr.values.dtype == int_arr.into(tlib=tlib_override).values.dtype

    bool_arr = FFTArray(
        values=tlib.array([False, True, False, False]),
        dims=[x_dim],
        space="pos",
        eager=eager,
        tlib=tlib,
        factors_applied=True,
    )
    assert bool_arr.values.dtype == bool_arr.into(tlib=tlib_override).values.dtype


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


@pytest.mark.parametrize("tensor_lib", tensor_libs)
@pytest.mark.parametrize("space", spaces)
def test_sel_order(tensor_lib, space):
    """Tests whether the selection order matters. Assuming an input FFTArray of
    dimensions A and B. Then

        FFTArray.sel(A==a).sel(B==b) == FFTArray.sel(B==b).sel(A==a)

    should be true.
    """
    xdim = FFTDimension("x", n=4, d_pos=0.1, pos_min=-0.2, freq_min=-2.1)
    ydim = FFTDimension("y", n=8, d_pos=0.03, pos_min=-0.5, freq_min=-4.7)
    arr = xdim.fft_array(tlib=tensor_lib(), space=space) + ydim.fft_array(tlib=tensor_lib(), space=space)
    arr_selx = arr.sel(**{"x": getattr(xdim, f"{space}_middle")})
    arr_sely = arr.sel(**{"y": getattr(ydim, f"{space}_middle")})
    arr_selx_sely = arr_selx.sel(**{"y": getattr(ydim, f"{space}_middle")})
    arr_sely_selx = arr_sely.sel(**{"x": getattr(xdim, f"{space}_middle")})
    np.testing.assert_allclose(arr_selx_sely.values, arr_sely_selx.values)

def get_hypothesis_array(draw, st_type, lengths):
    if len(lengths) > 1:
        return [get_hypothesis_array(draw, st_type, lengths[1:]) for _ in range(lengths[0])]
    return draw(st.lists(st_type, min_size=lengths[0], max_size=lengths[0]))

@st.composite
def fftarray_strategy(draw):
    ndims = draw(st.integers(min_value=1, max_value=4))
    value = st.one_of([
        st.integers(min_value=np.iinfo(np.int32).min, max_value=np.iinfo(np.int32).max),
        st.complex_numbers(allow_infinity=False, allow_nan=False, allow_subnormal=False, width=64),
        st.floats(allow_infinity=False, allow_nan=False, allow_subnormal=False, width=32)
    ])
    factors_applied = draw(st.lists(st.booleans(), min_size=ndims, max_size=ndims))
    eager = draw(st.lists(st.booleans(), min_size=ndims, max_size=ndims))
    init_space = draw(st.sampled_from(["pos", "freq"]))
    tlib = draw(st.sampled_from(tensor_libs))
    precision = draw(st.sampled_from(precisions))

    tensor_lib = tlib(precision=precision)
    note(tensor_lib)
    dims = [
        FFTDimension(f"{ndim}", n=draw(st.integers(min_value=2, max_value=8)), d_pos=0.1, pos_min=-0.2, freq_min=-2.1)
    for ndim in range(ndims)]
    note(dims)
    fftarr_values = tensor_lib.array(get_hypothesis_array(draw, value, [dim.n for dim in dims]))
    note(fftarr_values.dtype)
    note(fftarr_values)

    return FFTArray(
        values=fftarr_values,
        dims=dims,
        space=init_space,
        eager=eager,
        factors_applied=factors_applied,
        tlib=tensor_lib
    )

@pytest.mark.slow
@settings(max_examples=1000, deadline=None)
@given(fftarray_strategy())
def test_fftarray_lazyness(fftarr):
    """Tests the lazyness of a FFTArray, i.e., the correct behavior of
    factors_applied and eager.
    """
    note(fftarr)
    # -- basic tests
    assert_basic_lazy_logic(fftarr, note)
    # -- test operands
    assert_single_operand_fun_equivalence(fftarr, all(fftarr._factors_applied), note)
    assert_dual_operand_fun_equivalence(fftarr, all(fftarr._factors_applied), note)
    # Jax and Numpy only support FFT for dim<4
    if (len(fftarr.dims) < 4 or isinstance(fftarr.tlib, PyFFTWTensorLib)):
        # -- test eager, factors_applied logic
        assert_fftarray_eager_factors_applied(fftarr, note)

@pytest.mark.parametrize("tensor_lib", tensor_libs)
@pytest.mark.parametrize("space", spaces)
@pytest.mark.parametrize("eager", [True, False])
def test_fftarray_lazyness_reduced(tensor_lib, space, eager):
    """Tests the lazyness of a FFTArray, i.e., the correct behavior of
    factors_applied and eager. This is the reduced/faster version of the test
    using hypothesis.
    """
    xdim = FFTDimension("x", n=4, d_pos=0.1, pos_min=-0.2, freq_min=-2.1)
    ydim = FFTDimension("y", n=8, d_pos=0.03, pos_min=-0.5, freq_min=-4.7)
    tlib=tensor_lib(precision="default")
    fftarr = xdim.fft_array(tlib, space, eager) + ydim.fft_array(tlib, space, eager)
    assert_basic_lazy_logic(fftarr, print)
    assert_single_operand_fun_equivalence(fftarr, all(fftarr._factors_applied), print)
    assert_dual_operand_fun_equivalence(fftarr, all(fftarr._factors_applied), print)
    assert_fftarray_eager_factors_applied(fftarr, print)

def assert_basic_lazy_logic(arr, log):
    """Tests whether FFTArray.values is equal to the internal _values for the
    special cases where factors_applied=True, space="pos" and comparing the
    absolute values, and where space="freq" and comparing values to
    _values/(n*d_freq).
    """
    if all(arr._factors_applied):
        # fftarray must be handled the same way as applying the operations to the values numpy array
        log("factors_applied=True -> x.values == x._values")
        np.testing.assert_array_equal(arr.values, arr._values, strict=True)

    log("space='pos' -> abs(x.values) == abs(x._values)")
    log("space='freq' -> abs(x.values) == abs(x._values)/(n*d_freq)")
    scale = 1
    for dim, space, fa in zip(arr.dims, arr.space, arr._factors_applied):
        if space == "freq" and not fa:
            scale *= 1/(dim.n*dim.d_freq)
    rtol = 1e-6 if arr.tlib.precision == "fp32" else 1e-12
    np.testing.assert_allclose(np.abs(arr.values), np.abs(arr._values)*scale, rtol=rtol)

def is_inf_or_nan(x):
    """Check if (real or imag of) x is inf or nan"""
    return (np.isinf(x.real).any() or np.isinf(x.imag).any() or np.isnan(x.real).any() or np.isnan(x.imag).any())

def assert_equal_op(arr: FFTArray, values: Any, op: Callable[[Any],Any], precise: bool, log):
    """Helper function to test equality between an FFTArray and a values array.
    `op` denotes the operation acting on the FFTArray and on the values before
    comparison.
    `precise` denotes whether the comparison is performed using nulp (number of
    unit in the last place for tolerance) or using the less stringent
    `numpy.testing.allclose`.
    """
    arr_op = op(arr).values
    values_op = op(values)

    if arr_op.dtype != values_op.dtype:
        log(f"Changing type to {values_op.dtype}")
        arr_op = arr_op.astype(values_op.dtype)
        values_op = values_op.astype(values_op.dtype)

    if is_inf_or_nan(values_op) or (precise==False and is_inf_or_nan(arr_op)):
        return

    if precise and ("int" in str(values.dtype) or arr.tlib.precision == "fp64"):
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

def assert_single_operand_fun_equivalence(arr: FFTArray, precise: bool, log):
    """Test whether applying operands to the FFTArray (and then getting the
    values) is equivalent to applying the same operands to the values array:

        operand(FFTArray).values == operand(FFTArray.values)

    """
    values = arr.values
    log("f(x) = x")
    assert_equal_op(arr, values, lambda x: x, precise, log)
    log("f(x) = pi*x")
    assert_equal_op(arr, values, lambda x: np.pi*x, precise, log)
    log("f(x) = abs(x)")
    assert_equal_op(arr, values, lambda x: np.abs(x), precise, log)
    log("f(x) = x**2")
    assert_equal_op(arr, values, lambda x:  x**2, precise, log)
    log("f(x) = x**3")
    assert_equal_op(arr, values, lambda x:  x**3, precise, log)
    log("f(x) = exp(x)")
    assert_equal_op(arr, values, lambda x:  np.exp(x), precise, log)
    log("f(x) = sqrt(x)")
    assert_equal_op(arr, values, lambda x:  np.sqrt(x), False, log) # precise comparison fails

def assert_dual_operand_fun_equivalence(arr: FFTArray, precise: bool, log):
    """Test whether a dual operation on an FFTArray, e.g., the
    sum/multiplication of two, is equivalent to applying this operand to its
    values.

        operand(FFTArray, FFTArray).values = operand(FFTArray.values, FFTArray.values)

    """
    values = arr.values
    log("f(x,y) = x+y")
    assert_equal_op(arr, values, lambda x: x+x, precise, log)
    log("f(x,y) = x-2*y")
    assert_equal_op(arr, values, lambda x: x-2*x, precise, log)
    log("f(x,y) = x*y")
    assert_equal_op(arr, values, lambda x: x*x, precise, log)
    log("f(x,y) = x/y")
    assert_equal_op(arr, values, lambda x: x/x, precise, log)
    log("f(x,y) = x**y")
    # integers to negative integer powers are not allowed
    if "int" in str(values.dtype):
        assert_equal_op(arr, values, lambda x: x**np.abs(x), precise, log)
    else:
        assert_equal_op(arr, values, lambda x: x**x, precise, log)

def get_other_space(space: Union[Space, Tuple[Space, ...]]):
    """Returns the other space. If input space is "pos", "freq" is returned and
    vice versa. If space is a `Tuple[Space]`, a tuple is returned.
    """
    if isinstance(space, str):
        if space == "pos":
            return "freq"
        return "pos"
    return tuple(get_other_space(s) for s in space)

def assert_fftarray_eager_factors_applied(arr: FFTArray, log):
    """Tests whether the factors are only applied when necessary and whether
    the FFTArray after performing an FFT has the correct properties. If the
    initial FFTArray was eager, then the final FFTArray also must be eager and
    have _factors_applied=True. If the initial FFTArray was not eager, then the
    final FFTArray should have eager=False and _factors_applied=False.
    """

    log("arr._factors_applied == (arr**2)._factors_applied")
    arr_sq = arr * arr
    np.testing.assert_array_equal(arr_sq.eager, arr.eager) # type: ignore
    np.testing.assert_array_equal(arr_sq._factors_applied, arr._factors_applied) # type: ignore

    log("abs(x)._factors_applied == True")
    arr_abs = np.abs(arr)
    np.testing.assert_array_equal(arr_abs.eager, arr.eager) # type: ignore
    np.testing.assert_array_equal(arr_abs._factors_applied, True) # type: ignore

    log("(x*abs(x))._factors_applied == x._factors_applied")
    # if both _factors_applied=True, the resulting FFTArray will also have it
    # True, otherwise False (if not eager)
    arr_abs_sq = arr * arr_abs
    np.testing.assert_array_equal(arr_abs_sq.eager, arr.eager) # type: ignore
    np.testing.assert_array_equal(arr_abs_sq._factors_applied, arr._factors_applied) # type: ignore

    log("(x+abs(x))._factors_applied == (x._factors_applied or x._eager)")
    arr_abs_sum = arr + arr_abs
    np.testing.assert_array_equal(arr_abs_sum.eager, arr.eager) # type: ignore
    for ea, ifa, ffa in zip(arr_abs_sum.eager, arr._factors_applied, arr_abs_sum._factors_applied):
        # True+True=True
        # False+True=eager
        assert (ifa == ffa) or (ffa == ea)

    log("fft(x)._factors_applied ...")
    arr_fft = arr.into(space=get_other_space(arr.space))
    np.testing.assert_array_equal(arr.eager, arr_fft.eager)
    for ffapplied, feager in zip(arr_fft._factors_applied, arr_fft.eager):
        assert (feager and ffapplied) or (not feager and not ffapplied)

@pytest.mark.parametrize("tensor_lib", tensor_libs)
@pytest.mark.parametrize("space", spaces)
def test_fft_ifft_invariance(tensor_lib, space: Space):
    """Tests whether ifft(fft(*)) is an identity.

       ifft(fft(FFTArray)) == FFTArray

    """
    xdim = FFTDimension("x", n=4, d_pos=0.1, pos_min=-0.2, freq_min=-2.1)
    ydim = FFTDimension("y", n=8, d_pos=0.03, pos_min=-0.4, freq_min=-4.2)
    arr = xdim.fft_array(tlib=tensor_lib(), space=space) + ydim.fft_array(tlib=tensor_lib(), space=space)
    other_space = get_other_space(space)
    arr_fft = arr.into(space=other_space)
    arr_fft_ifft = arr_fft.into(space=space)
    if is_inf_or_nan(arr_fft_ifft.values):
        # edge cases (very large numbers) result in inf after fft
        return
    rtol = 1e-5 if arr.tlib.precision == "fp32" else 1e-6
    np.testing.assert_allclose(arr.values, arr_fft_ifft.values, rtol=rtol, atol=1e-38)
