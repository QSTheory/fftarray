from typing import List, Type, Any, Callable, Union, Tuple

import pytest
from hypothesis import given, strategies as st, note, settings
import numpy as np
import jax
import jax.numpy as jnp

from fftarray.fft_array import FFTArray, FFTDimension, Space
from fftarray.backends.jax import JaxBackend
from fftarray.backends.numpy import NumpyBackend
from fftarray.backends.pyfftw import PyFFTWBackend
from fftarray.backends.backend import Backend, PrecisionSpec

jax.config.update("jax_enable_x64", True)

def assert_scalars_almost_equal_nulp(x, y, nulp = 1):
    np.testing.assert_array_almost_equal_nulp(np.array([x]), np.array([y]), nulp = nulp)

backends: List[Type[Backend]] = [NumpyBackend, JaxBackend, PyFFTWBackend]
precisions: List[PrecisionSpec] = ["fp32", "fp64", "default"]
spaces: List[Space] = ["pos", "freq"]

# Currently only tests the values type
def test_fft_array_constructor():
    """Tests whether the type checking of the FFTArray input values works.
    An FFTArray can only be initialized if the values array type is compatible
    with the provided Backend.
    """
    dim = FFTDimension("x", n=4, d_pos=0.1, pos_min=0., freq_min=0.)
    values = [1,2,3,4]
    np_arr = np.array(values)
    jnp_arr = jnp.array(values)

    failing_sets = [
        (values, [NumpyBackend(), JaxBackend(), PyFFTWBackend()]),
        (np_arr, [JaxBackend()]),
        (jnp_arr, [NumpyBackend(), PyFFTWBackend()]),
    ]
    for arr, backends in failing_sets:
        for backend in backends:
            with pytest.raises(ValueError):
                _ = FFTArray(
                    values=arr,
                    dims=[dim],
                    space="pos",
                    eager=False,
                    factors_applied=False,
                    backend=backend,
                )

    working_sets = [
        (np_arr, NumpyBackend()),
        (np_arr, PyFFTWBackend()),
        (jnp_arr, JaxBackend()),
    ]
    for arr, backend in working_sets:
            _ = FFTArray(
                values=arr,
                dims=[dim],
                space="pos",
                eager=False,
                factors_applied=False,
                backend=backend,
            )


@pytest.mark.parametrize("backend_class", backends)
@pytest.mark.parametrize("space", ["pos", "freq"])
def test_comparison(backend_class, space) -> None:
    x_dim = FFTDimension("x",
        n=4,
        d_pos=1,
        pos_min=0.5,
        freq_min=0.,
    )
    x = x_dim.fft_array(backend_class(), space=space)
    x_sq = x**2

    # Eplicitly test the operators to check that the forwarding to array_ufunc is correct
    for a, b in [(0.5, x), (x, x_sq), (x, 0.5), (x, x_sq)]:
        np.testing.assert_array_equal(a < b, np.array(a) < np.array(b), strict=True)
        np.testing.assert_array_equal(a <= b, np.array(a) <= np.array(b), strict=True)
        np.testing.assert_array_equal(a > b, np.array(a) > np.array(b), strict=True)
        np.testing.assert_array_equal(a >= b, np.array(a) >= np.array(b), strict=True)
        np.testing.assert_array_equal(a != b, np.array(a) != np.array(b), strict=True)
        np.testing.assert_array_equal(a == b, np.array(a) == np.array(b), strict=True)

@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("precision", ("fp32", "fp64", "default"))
@pytest.mark.parametrize("override", (None, "fp32", "fp64", "default"))
@pytest.mark.parametrize("eager", [False, True])
def test_dtype(backend, precision, override, eager: bool) -> None:
    backend = backend(precision=precision)
    if override is None:
        backend_override = None
    else:
        backend_override = backend(precision=override)

    x_dim = FFTDimension("x",
        n=4,
        d_pos=1,
        pos_min=0.,
        freq_min=0.,
    )

    if backend_override is None:
        assert x_dim.fft_array(backend, space="pos").values.dtype == backend.real_type
    else:
        assert x_dim.fft_array(backend_override, space="pos", eager=eager).values.dtype == backend_override.real_type
        assert x_dim.fft_array(backend, space="pos", eager=eager).into(backend=backend_override).values.dtype == backend_override.real_type


    if backend_override is None:
        assert x_dim.fft_array(backend, space="freq", eager=eager).values.dtype == backend.real_type
    else:
        assert x_dim.fft_array(backend_override, space="freq", eager=eager).values.dtype == backend_override.real_type
        assert x_dim.fft_array(backend, space="freq", eager=eager).into(backend=backend_override).values.dtype == backend_override.real_type

    assert x_dim.fft_array(backend, space="pos", eager=eager).into(space="freq").values.dtype == backend.complex_type
    assert x_dim.fft_array(backend, space="freq", eager=eager).into(space="pos").values.dtype == backend.complex_type

    assert np.abs(x_dim.fft_array(backend, space="pos", eager=eager).into(space="freq")).values.dtype == backend.real_type # type: ignore
    assert np.abs(x_dim.fft_array(backend, space="freq", eager=eager).into(space="pos")).values.dtype == backend.real_type # type: ignore

    if backend_override is not None:
        assert x_dim.fft_array(backend, space="pos", eager=eager).into(space="freq", backend=backend_override).values.dtype == backend_override.complex_type
        assert x_dim.fft_array(backend, space="freq", eager=eager).into(space="pos", backend=backend_override).values.dtype == backend_override.complex_type

    # For non-float and non-complex dtypes, we do not force the backend precision types
    # onto the values. Therefore, the FFTArray.values dtype should not be affected by the
    # backend_override precision for both integer values and boolean values

    int_arr = FFTArray(
        values=backend.array([1,2,3,4]),
        dims=[x_dim],
        space="pos",
        eager=eager,
        backend=backend,
        factors_applied=True,
    )
    assert int_arr.values.dtype == int_arr.into(backend=backend_override).values.dtype

    bool_arr = FFTArray(
        values=backend.array([False, True, False, False]),
        dims=[x_dim],
        space="pos",
        eager=eager,
        backend=backend,
        factors_applied=True,
    )
    assert bool_arr.values.dtype == bool_arr.into(backend=backend_override).values.dtype


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("override", backends)
def test_backend_override(backend, override) -> None:
    x_dim = FFTDimension("x",
        n=4,
        d_pos=1,
        pos_min=0.,
        freq_min=0.,
    )

    assert type(x_dim.fft_array(backend(), space="pos").into(backend=override()).values) == type(x_dim.fft_array(override(), space="pos").values)
    assert type(x_dim.fft_array(backend(), space="freq").into(backend=override()).values) == type(x_dim.fft_array(override(), space="freq").values)
    assert type(x_dim.fft_array(backend(), space="pos").into(backend=override()).into(space="freq").values) == type(x_dim.fft_array(override(), space="freq").values)
    assert type(x_dim.fft_array(backend(), space="freq").into(backend=override()).into(space="pos").values) == type(x_dim.fft_array(override(), space="pos").values)

    assert type(x_dim.fft_array(backend(), space="pos").into(space="freq", backend=override()).values) == type(x_dim.fft_array(override(), space="freq").values)
    assert type(x_dim.fft_array(backend(), space="freq").into(space="pos", backend=override()).values) == type(x_dim.fft_array(override(), space="freq").values)


def test_broadcasting(nulp: int = 1) -> None:
    x_dim = FFTDimension("x", n=4, d_pos=1, pos_min=0., freq_min=0.)
    y_dim = FFTDimension("y", n=8, d_pos=1, pos_min=0., freq_min=0.)

    x_ref = np.arange(0., 4.)
    y_ref = np.arange(0., 8.)
    np.testing.assert_array_almost_equal_nulp(np.array(x_dim.fft_array(backend=NumpyBackend(), space="pos")), x_ref, nulp = 0)
    np.testing.assert_array_almost_equal_nulp(np.array(y_dim.fft_array(backend=NumpyBackend(), space="pos")), y_ref, nulp = 0)

    x_ref_broadcast = x_ref.reshape(1,-1)
    y_ref_broadcast = y_ref.reshape(-1,1)
    np.testing.assert_array_almost_equal_nulp((x_dim.fft_array(backend=NumpyBackend(), space="pos") + y_dim.fft_array(backend=NumpyBackend(), space="pos")).transpose("x", "y").values, (x_ref_broadcast+y_ref_broadcast).transpose(), nulp = 0)
    np.testing.assert_array_almost_equal_nulp((x_dim.fft_array(backend=NumpyBackend(), space="pos") + y_dim.fft_array(backend=NumpyBackend(), space="pos")).transpose("y", "x").values, x_ref_broadcast+y_ref_broadcast, nulp = 0)


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("space", spaces)
def test_sel_order(backend, space):
    """Tests whether the selection order matters. Assuming an input FFTArray of
    dimensions A and B. Then

        FFTArray.sel(A==a).sel(B==b) == FFTArray.sel(B==b).sel(A==a)

    should be true.
    """
    xdim = FFTDimension("x", n=4, d_pos=0.1, pos_min=-0.2, freq_min=-2.1)
    ydim = FFTDimension("y", n=8, d_pos=0.03, pos_min=-0.5, freq_min=-4.7)
    arr = xdim.fft_array(backend=backend(), space=space) + ydim.fft_array(backend=backend(), space=space)
    arr_selx = arr.sel(**{"x": getattr(xdim, f"{space}_middle")})
    arr_sely = arr.sel(**{"y": getattr(ydim, f"{space}_middle")})
    arr_selx_sely = arr_selx.sel(**{"y": getattr(ydim, f"{space}_middle")})
    arr_sely_selx = arr_sely.sel(**{"x": getattr(xdim, f"{space}_middle")})
    np.testing.assert_allclose(arr_selx_sely.values, arr_sely_selx.values)


def draw_hypothesis_fft_array_values(draw, st_type, shape):
    """Creates multi-dimensional array with shape `shape` whose values are drawn
    using `draw` from `st_type`."""
    if len(shape) > 1:
        return [draw_hypothesis_fft_array_values(draw, st_type, shape[1:]) for _ in range(shape[0])]
    return draw(st.lists(st_type, min_size=shape[0], max_size=shape[0]))

@st.composite
def fftarray_strategy(draw):
    """Initializes an FFTArray using hypothesis."""
    ndims = draw(st.integers(min_value=1, max_value=4))
    value = st.one_of([
        st.integers(min_value=np.iinfo(np.int32).min, max_value=np.iinfo(np.int32).max),
        st.complex_numbers(allow_infinity=False, allow_nan=False, allow_subnormal=False, width=64),
        st.floats(allow_infinity=False, allow_nan=False, allow_subnormal=False, width=32)
    ])
    factors_applied = draw(st.lists(st.booleans(), min_size=ndims, max_size=ndims))
    note(f"factors_applied={factors_applied}") # TODO: remove when FFTArray.__repr__ is implemented
    eager = draw(st.lists(st.booleans(), min_size=ndims, max_size=ndims))
    note(f"eager={eager}") # TODO: remove when FFTArray.__repr__ is implemented
    init_space = draw(st.sampled_from(["pos", "freq"]))
    note(f"space={init_space}") # TODO: remove when FFTArray.__repr__ is implemented
    backend = draw(st.sampled_from(backends))
    precision = draw(st.sampled_from(precisions))

    backend = backend(precision=precision)
    note(backend)
    dims = [
        FFTDimension(f"{ndim}", n=draw(st.integers(min_value=2, max_value=8)), d_pos=0.1, pos_min=-0.2, freq_min=-2.1)
    for ndim in range(ndims)]
    note(dims)
    fftarr_values = backend.array(draw_hypothesis_fft_array_values(draw, value, [dim.n for dim in dims]))
    note(fftarr_values.dtype)
    note(fftarr_values)

    return FFTArray(
        values=fftarr_values,
        dims=dims,
        space=init_space,
        eager=eager,
        factors_applied=factors_applied,
        backend=backend
    )

@pytest.mark.slow
@settings(max_examples=1000, deadline=None)
@given(fftarray_strategy())
def test_fftarray_lazyness(fftarr):
    """Tests the lazyness of an FFTArray, i.e., the correct behavior of
    factors_applied and eager.
    """
    note(fftarr)
    # -- basic tests
    assert_basic_lazy_logic(fftarr, note)
    # -- test operands
    assert_single_operand_fun_equivalence(fftarr, all(fftarr._factors_applied), note)
    assert_dual_operand_fun_equivalence(fftarr, all(fftarr._factors_applied), note)
    # Jax only supports FFT for dim<4
    if len(fftarr.dims) < 4 or not isinstance(fftarr.backend, JaxBackend):
        # -- test eager, factors_applied logic
        assert_fftarray_eager_factors_applied(fftarr, note)

@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("precision", precisions)
@pytest.mark.parametrize("space", spaces)
@pytest.mark.parametrize("eager", [True, False])
@pytest.mark.parametrize("factors_applied", [True, False])
def test_fftarray_lazyness_reduced(backend, precision, space, eager, factors_applied):
    """Tests the lazyness of an FFTArray, i.e., the correct behavior of
    factors_applied and eager. This is the reduced/faster version of the test
    using hypothesis.
    """
    xdim = FFTDimension("x", n=4, d_pos=0.1, pos_min=-0.2, freq_min=-2.1)
    ydim = FFTDimension("y", n=8, d_pos=0.03, pos_min=-0.5, freq_min=-4.7)
    backend = backend(precision=precision)
    fftarr = xdim.fft_array(backend, space, eager) + ydim.fft_array(backend, space, eager)
    fftarr._factors_applied = (factors_applied, factors_applied)
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
    rtol = 1e-6 if arr.backend.precision == "fp32" else 1e-12
    np.testing.assert_allclose(np.abs(arr.values), np.abs(arr._values)*scale, rtol=rtol)

def is_inf_or_nan(x):
    """Check if (real or imag of) x is inf or nan"""
    return (np.isinf(x.real).any() or np.isinf(x.imag).any() or np.isnan(x.real).any() or np.isnan(x.imag).any())

def internal_and_public_values_should_differ(arr: FFTArray):
    """Returns boolean, whether `fftarray.values` should differ from
    `fftarray._values`.
    This is the case if `factors_applied=False` and the values are non-zero
    (along at least one dimension).
    Note that the position space needs to be treated separately as the phase
    factor for the first coordinate is 1 (and thus does not change `_values`).
    """
    for factor, space, i, in zip(arr._factors_applied, arr.space, range(len(arr.dims))):
        if not factor:
            # factor needs to be applied
            if space == "pos":
                # check if the values are non-zero
                # the phase factor in position space is 1 for the first
                # coordinate and, thus, is excluded from the check
                if (arr.backend.numpy_ufuncs.take(arr._values, np.arange(1,arr.dims[i].n), axis=i)!=0).any():
                    return True
            else:
                # for space="freq", the factor includes scale unequal 1, so all
                # values along this dimension must be non-zero
                if np.any(arr._values!=0, axis=i).any():
                    return True
    return False

def assert_equal_op(
        arr: FFTArray,
        values: Any,
        op: Callable[[Any],Any],
        precise: bool,
        op_forces_factors_applied: bool,
        log
    ):
    """Helper function to test equality between an FFTArray and a values array.
    `op` denotes the operation acting on the FFTArray and on the values before
    comparison.
    `precise` denotes whether the comparison is performed using nulp (number of
    unit in the last place for tolerance) or using the less stringent
    `numpy.testing.allclose`.
    If `op_forces_factors_applied` is False, it will be tested whether
    op(FFTArray)._values deviates from op(FFTArray).values (which is the case
    if the factors have not been applied after operation and if the values are
    non-zero). If it is True, it is tested if they are equal.
    """
    arr_op = op(arr).values
    values_op = op(values)

    if arr_op.dtype != values_op.dtype:
        log(f"Changing type to {values_op.dtype}")
        arr_op = arr_op.astype(values_op.dtype)
        values_op = values_op.astype(values_op.dtype)

    if is_inf_or_nan(values_op) or (precise==False and is_inf_or_nan(arr_op)):
        return

    rtol = 1e-6 if arr.backend.precision == "fp32" else 1e-7
    if precise and ("int" in str(values.dtype) or arr.backend.precision == "fp64"):
        if "int" in str(values.dtype):
            np.testing.assert_array_equal(arr_op, values_op, strict=True)
        if "float" in str(values.dtype):
            np.testing.assert_array_almost_equal_nulp(arr_op, values_op, nulp=4)
        if "complex" in str(values.dtype):
            assert_array_almost_equal_nulp_complex(arr_op, values_op, nulp=4)
    else:
        np.testing.assert_allclose(arr_op, values_op, rtol=rtol)

    _arr_op = op(arr)._values
    if op_forces_factors_applied:
        # _values should have factors applied
        np.testing.assert_allclose(_arr_op, values_op, rtol=rtol)
    else:
        # arr._values can differ from arr.values
        if internal_and_public_values_should_differ(arr):
            with pytest.raises(AssertionError):
                np.testing.assert_allclose(_arr_op, values_op, rtol=rtol)
        else:
            np.testing.assert_allclose(_arr_op, values_op, rtol=rtol)


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
    assert_equal_op(arr, values, lambda x: x, precise, False, log)
    log("f(x) = pi*x")
    assert_equal_op(arr, values, lambda x: np.pi*x, precise, False, log)
    log("f(x) = abs(x)")
    assert_equal_op(arr, values, np.abs, precise, True, log)
    log("f(x) = x**2")
    assert_equal_op(arr, values, lambda x: x**2, precise, True, log)
    log("f(x) = x**3")
    assert_equal_op(arr, values, lambda x: x**3, precise, True, log)
    log("f(x) = exp(x)")
    assert_equal_op(arr, values, np.exp, False, True, log) # precise comparison fails
    log("f(x) = sqrt(x)")
    assert_equal_op(arr, values, np.sqrt, False, True, log) # precise comparison fails

def assert_dual_operand_fun_equivalence(arr: FFTArray, precise: bool, log):
    """Test whether a dual operation on an FFTArray, e.g., the
    sum/multiplication of two, is equivalent to applying this operand to its
    values.

        operand(FFTArray, FFTArray).values = operand(FFTArray.values, FFTArray.values)

    """
    values = arr.values
    log("f(x,y) = x+y")
    assert_equal_op(arr, values, lambda x: x+x, precise, False, log)
    log("f(x,y) = x-2*y")
    assert_equal_op(arr, values, lambda x: x-2*x, precise, False, log)
    log("f(x,y) = x*y")
    assert_equal_op(arr, values, lambda x: x*x, precise, False, log)
    log("f(x,y) = x/y")
    assert_equal_op(arr, values, lambda x: x/x, precise, False, log)
    log("f(x,y) = x**y")
    # integers to negative integer powers are not allowed
    if "int" in str(values.dtype):
        assert_equal_op(arr, values, lambda x: x**np.abs(x), precise, True, log)
    else:
        assert_equal_op(arr, values, lambda x: x**x, precise, True, log)

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
    # True, otherwise False
    # given abs(x)._factors_applied=True, we test the patterns
    # True*True=True, False*True=False
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

@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("space", spaces)
def test_fft_ifft_invariance(backend, space: Space):
    """Tests whether ifft(fft(*)) is an identity.

       ifft(fft(FFTArray)) == FFTArray

    """
    xdim = FFTDimension("x", n=4, d_pos=0.1, pos_min=-0.2, freq_min=-2.1)
    ydim = FFTDimension("y", n=8, d_pos=0.03, pos_min=-0.4, freq_min=-4.2)
    arr = xdim.fft_array(backend=backend(), space=space) + ydim.fft_array(backend=backend(), space=space)
    other_space = get_other_space(space)
    arr_fft = arr.into(space=other_space)
    arr_fft_ifft = arr_fft.into(space=space)
    if is_inf_or_nan(arr_fft_ifft.values):
        # edge cases (very large numbers) result in inf after fft
        return
    rtol = 1e-5 if arr.backend.precision == "fp32" else 1e-6
    np.testing.assert_allclose(arr.values, arr_fft_ifft.values, rtol=rtol, atol=1e-38)

@pytest.mark.parametrize("space", spaces)
@pytest.mark.parametrize("dtc", [True, False])
@pytest.mark.parametrize("sel_method", ["nearest", "pad", "ffill", "backfill", "bfill"])
def test_grid_manipulation_in_jax_scan(space: Space, dtc: bool, sel_method: str) -> None:
    """Tests FFTDimension's `dynamically_traced_coords` property on the level of
    an `FFTArray`.

    Allowed by dynamic, error for static:
    - change FFTDimension properties of an FFTArray inside a `jax.lax.scan` step
    function

    Allowed by static, error for dynamic:
    - selection by coordinate
    """
    xdim = FFTDimension("x", n=4, d_pos=0.1, pos_min=-0.2, freq_min=-2.1, dynamically_traced_coords=dtc)
    ydim = FFTDimension("y", n=8, d_pos=0.03, pos_min=-0.4, freq_min=-4.2, dynamically_traced_coords=dtc)
    fftarr = xdim.fft_array(backend=JaxBackend(), space=space) + ydim.fft_array(backend=JaxBackend(), space=space)

    def jax_scan_step_fun_dynamic(carry, *_):
        # dynamic should support resizing and shifting of the grid
        # static should throw an error
        newdims = list(carry._dims)
        newdims[0]._pos_min = 0.
        newdims[0]._d_pos = newdims[0]._d_pos/2.
        newdims[1]._freq_min = newdims[1]._freq_min + 2.
        carry._dims = tuple(newdims)
        return carry, None

    def jax_scan_step_fun_static(carry, *_):
        # static should support coordinate selection
        # dynamic should throw an error
        xval = carry._dims[0]._pos_min + carry._dims[0]._d_pos
        carry_sel = carry.sel(x=xval, method=sel_method)
        return carry, None

    if dtc:
        jax.lax.scan(jax_scan_step_fun_dynamic, fftarr, jnp.arange(3))
        # internal logic in sel throws NotImplementedError for jitted index
        with pytest.raises(NotImplementedError):
            jax.lax.scan(jax_scan_step_fun_static, fftarr, jnp.arange(3))
    else:
        jax.lax.scan(jax_scan_step_fun_static, fftarr, jnp.arange(3))
        with pytest.raises(TypeError):
            jax.lax.scan(jax_scan_step_fun_dynamic, fftarr, jnp.arange(3))

def test_different_dimension_dynamic_prop() -> None:
    """Tests tracing of an FFTArray whose dimensions have different
    `dynamically_traced_coords`.
    """
    x_dim = FFTDimension(name="x", pos_min=0, freq_min=0, d_pos=1, n=8, dynamically_traced_coords=False)
    y_dim = FFTDimension(name="y", pos_min=0, freq_min=0, d_pos=1, n=4, dynamically_traced_coords=True)
    fftarr = x_dim.fft_array(backend=JaxBackend(), space="pos") + y_dim.fft_array(backend=JaxBackend(), space="pos")

    def jax_scan_step_fun_valid(carry, *_):
        xval = carry._dims[0]._pos_min + carry._dims[0]._d_pos # static dimension
        new_dims = list(carry._dims)
        new_dims[1]._pos_min = 0.123 # dynamic dimension
        carry_sel = carry.sel(x=xval, method="nearest")
        carry._dims = tuple(new_dims)
        return carry, carry_sel

    jax.lax.scan(jax_scan_step_fun_valid, fftarr, jnp.arange(3))

    def jax_scan_step_fun_invalid_change(carry, *_):
        new_dims = list(carry._dims)
        new_dims[0]._pos_min = 0.123
        carry._dims = tuple(new_dims)
        return carry, None

    def jax_scan_step_fun_invalid_sel(carry, *_):
        yval = carry._dims[1]._pos_min + carry._dims[1]._d_pos
        carry_sel = carry.sel(y=0, method="nearest")
        return carry, carry_sel

    with pytest.raises(TypeError):
        jax.lax.scan(jax_scan_step_fun_invalid_change, fftarr, jnp.arange(3))

    with pytest.raises(NotImplementedError):
        jax.lax.scan(jax_scan_step_fun_invalid_sel, fftarr, jnp.arange(3))

