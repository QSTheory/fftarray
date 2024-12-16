from typing import List, Literal, Any, Callable, Union, Tuple

import array_api_compat
import pytest
from hypothesis import given, strategies as st, note, settings
import numpy as np

import fftarray as fa
from fftarray.fft_array import FFTArray, Space

from fftarray.tests.helpers import XPS

from fftarray._utils.defaults import DEFAULT_DTYPE

PrecisionSpec = Literal["float32", "float64"]

def assert_scalars_almost_equal_nulp(x, y, nulp = 1):
    np.testing.assert_array_almost_equal_nulp(np.array([x]), np.array([y]), nulp = nulp)

precisions: List[PrecisionSpec] = ["float32", "float64"]
spaces: List[Space] = ["pos", "freq"]



@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("space", ["pos", "freq"])
def test_comparison(xp, space) -> None:
    x_dim = fa.dim("x",
        n=4,
        d_pos=1,
        pos_min=0.5,
        freq_min=0.,
    )
    x = fa.array_from_dim(dim=x_dim, xp=xp, space=space)
    x_sq = x**2

    x = x.np_array(space=space)
    x_sq = x_sq.np_array(space=space)

    # Eplicitly test the operators to check that the forwarding to array_ufunc is correct
    for a, b in [(0.5, x), (x, x_sq), (x, 0.5), (x, x_sq)]:
        np.testing.assert_array_equal(a < b, np.array(a) < np.array(b), strict=True)
        np.testing.assert_array_equal(a <= b, np.array(a) <= np.array(b), strict=True)
        np.testing.assert_array_equal(a > b, np.array(a) > np.array(b), strict=True)
        np.testing.assert_array_equal(a >= b, np.array(a) >= np.array(b), strict=True)
        np.testing.assert_array_equal(a != b, np.array(a) != np.array(b), strict=True)
        np.testing.assert_array_equal(a == b, np.array(a) == np.array(b), strict=True)

def get_complex_name(
        dtype_name: Literal["float32", "float64"]
    ) -> Literal["complex64", "complex128"]:
    match dtype_name:
        case "float32":
            return "complex64"
        case "float64":
            return "complex128"
        case _:
            raise ValueError(f"Passed in unsupported ' {dtype_name}'.")

@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("init", ("float32", "float64"))
@pytest.mark.parametrize("override", (None, "float32", "float64"))
@pytest.mark.parametrize("eager", [False, True])
def test_dtype(xp, init, override, eager: bool) -> None:
    init_dtype_real = getattr(xp, init)
    # TODO: This does not work in numpy < 2.0
    init_dtype_complex = getattr(xp, get_complex_name(init))

    if override is not None:
        override_dtype_real = getattr(xp, override)
        override_dtype_complex = getattr(xp, get_complex_name(override))

    x_dim = fa.dim("x",
        n=4,
        d_pos=1,
        pos_min=0.,
        freq_min=0.,
    )

    if override is None:
        assert fa.array_from_dim(dim=x_dim, space="pos", dtype=init_dtype_real, xp=xp).values(space="pos").dtype == init_dtype_real
    else:
        assert fa.array_from_dim(dim=x_dim, dtype=override_dtype_real, xp=xp, space="pos", eager=eager).values(space="pos").dtype == override_dtype_real
        assert fa.array_from_dim(dim=x_dim, dtype=init_dtype_real, xp=xp, space="pos", eager=eager).astype(dtype=override_dtype_real).values(space="pos").dtype == override_dtype_real


    if override is None:
        assert fa.array_from_dim(dim=x_dim, dtype=init_dtype_real, xp=xp, space="freq", eager=eager).values(space="freq").dtype == init_dtype_real
    else:
        assert fa.array_from_dim(dim=x_dim, dtype=override_dtype_real, xp=xp, space="freq", eager=eager).values(space="freq").dtype == override_dtype_real
        assert fa.array_from_dim(dim=x_dim, dtype=init_dtype_real, xp=xp, space="freq", eager=eager).astype(dtype=override_dtype_real).values(space="freq").dtype == override_dtype_real

    assert fa.array_from_dim(dim=x_dim, dtype=init_dtype_real, xp=xp, space="pos", eager=eager).into(space="freq").values(space="freq").dtype == init_dtype_complex
    assert fa.array_from_dim(dim=x_dim, dtype=init_dtype_real, xp=xp, space="freq", eager=eager).into(space="pos").values(space="pos").dtype == init_dtype_complex

    assert fa.abs(fa.array_from_dim(dim=x_dim, dtype=init_dtype_real, xp=xp, space="pos", eager=eager).into(space="freq")).values(space="freq").dtype == init_dtype_real # type: ignore
    assert fa.abs(fa.array_from_dim(dim=x_dim, dtype=init_dtype_real, xp=xp, space="freq", eager=eager).into(space="pos")).values(space="pos").dtype == init_dtype_real # type: ignore

    if override is not None:
        assert fa.array_from_dim(dim=x_dim, dtype=init_dtype_real, xp=xp, space="pos", eager=eager).astype(dtype=override_dtype_real).into(space="freq").values(space="freq").dtype == override_dtype_complex
        assert fa.array_from_dim(dim=x_dim, dtype=init_dtype_real, xp=xp, space="freq", eager=eager).astype(dtype=override_dtype_real).into(space="pos").values(space="pos").dtype == override_dtype_complex


@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("xp_override", XPS)
def test_backend_override(xp, xp_override) -> None:
    x_dim = fa.dim("x",
        n=4,
        d_pos=1,
        pos_min=0.,
        freq_min=0.,
    )

    assert type(fa.array_from_dim(dim=x_dim, xp=xp, space="pos").asxp(xp=xp_override).values(space="pos")) == type(fa.array_from_dim(dim=x_dim, xp=xp_override, space="pos").values(space="pos"))
    assert type(fa.array_from_dim(dim=x_dim, xp=xp, space="freq").asxp(xp=xp_override).values(space="freq")) == type(fa.array_from_dim(dim=x_dim, xp=xp_override, space="freq").values(space="freq"))
    assert type(fa.array_from_dim(dim=x_dim, xp=xp, space="pos").asxp(xp=xp_override).into(space="freq").values(space="freq")) == type(fa.array_from_dim(dim=x_dim, xp=xp_override, space="freq").values(space="freq"))
    assert type(fa.array_from_dim(dim=x_dim, xp=xp, space="freq").asxp(xp=xp_override).into(space="pos").values(space="pos")) == type(fa.array_from_dim(dim=x_dim, xp=xp_override, space="pos").values(space="pos"))


def test_broadcasting() -> None:
    x_dim = fa.dim("x", n=4, d_pos=1, pos_min=0., freq_min=0.)
    y_dim = fa.dim("y", n=8, d_pos=1, pos_min=0., freq_min=0.)

    x_ref = np.arange(0., 4.)
    y_ref = np.arange(0., 8.)
    np.testing.assert_array_almost_equal_nulp(fa.array_from_dim(dim=x_dim, xp=np, space="pos").np_array(space="pos"), x_ref, nulp = 0)
    np.testing.assert_array_almost_equal_nulp(fa.array_from_dim(dim=y_dim, xp=np, space="pos").np_array(space="pos"), y_ref, nulp = 0)

    x_ref_broadcast = x_ref.reshape(1,-1)
    y_ref_broadcast = y_ref.reshape(-1,1)
    np.testing.assert_array_almost_equal_nulp((fa.array_from_dim(dim=x_dim, xp=np, space="pos") + fa.array_from_dim(dim=y_dim, xp=np, space="pos")).transpose("x", "y").values(space="pos"), (x_ref_broadcast+y_ref_broadcast).transpose(), nulp = 0)
    np.testing.assert_array_almost_equal_nulp((fa.array_from_dim(dim=x_dim, xp=np, space="pos") + fa.array_from_dim(dim=y_dim, xp=np, space="pos")).transpose("y", "x").values(space="pos"), x_ref_broadcast+y_ref_broadcast, nulp = 0)


@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("space", spaces)
def test_sel_order(xp, space):
    """Tests whether the selection order matters. Assuming an input FFTArray of
    dimensions A and B. Then

        FFTArray.sel(A==a).sel(B==b) == FFTArray.sel(B==b).sel(A==a)

    should be true.
    """
    xdim = fa.dim("x", n=4, d_pos=0.1, pos_min=-0.2, freq_min=-2.1)
    ydim = fa.dim("y", n=8, d_pos=0.03, pos_min=-0.5, freq_min=-4.7)
    arr = fa.array_from_dim(dim=xdim, xp=xp, space=space) + fa.array_from_dim(dim=ydim, xp=xp, space=space)
    arr_selx = arr.sel(**{"x": getattr(xdim, f"{space}_middle")})
    arr_sely = arr.sel(**{"y": getattr(ydim, f"{space}_middle")})
    arr_selx_sely = arr_selx.sel(**{"y": getattr(ydim, f"{space}_middle")})
    arr_sely_selx = arr_sely.sel(**{"x": getattr(xdim, f"{space}_middle")})
    np.testing.assert_allclose(arr_selx_sely.values(space=space), arr_sely_selx.values(space=space))

# TODO:
import jax.numpy as jnp


# TODO: Mark as not parallelizable
def test_defaults() -> None:
    assert fa.get_default_eager() is False
    assert fa.get_default_xp() == array_api_compat.array_namespace(np.asarray(0.))
    assert fa.get_default_dtype_name() == "float64"

    xdim = fa.dim("x", n=4, d_pos=0.1, pos_min=0., freq_min=0.)

    check_defaults(xdim, xp=np, dtype_name="float64", eager=False)

    fa.set_default_xp(jnp)
    fa.set_default_dtype_name("float32")
    check_defaults(xdim, xp=jnp, dtype_name="float32", eager=False)

    fa.set_default_eager(True)
    check_defaults(xdim, xp=jnp, dtype_name="float32", eager=True)

    # Reset global state for other tests
    fa.set_default_eager(False)
    fa.set_default_xp(np)
    fa.set_default_dtype_name("float64")



def test_defaults_context() -> None:
    xdim = fa.dim("x", n=4, d_pos=0.1, pos_min=0., freq_min=0.)

    with fa.default_xp(jnp):
        with fa.default_dtype_name("float32"):
            check_defaults(xdim, xp=jnp, dtype_name="float32", eager=False)
    check_defaults(xdim, xp=np, dtype_name="float64", eager=False)
    with fa.default_xp(jnp):
        with fa.default_dtype_name("float32"):
            with fa.default_eager(eager=True):
                check_defaults(xdim, xp=jnp, dtype_name="float32", eager=True)
            check_defaults(xdim, xp=jnp, dtype_name="float32", eager=False)


def check_defaults(dim: fa.FFTDimension, xp, dtype_name: DEFAULT_DTYPE, eager: bool) -> None:
    xp_compat = array_api_compat.array_namespace(xp.asarray(0))
    values = 0.1*xp.arange(4, dtype=dtype_name)
    arr_from_dim = fa.array_from_dim(dim=dim, space="pos")
    arr_direct = fa.array(dims=dim, space="pos", values=values)
    manual_arr = FFTArray(
        values=values,
        dims=(dim,),
        space=("pos",),
        eager=(eager,),
        xp=xp_compat,
        factors_applied=(True,),
    )

    assert fa.get_default_xp() == xp_compat
    assert fa.get_default_dtype_name() == dtype_name
    assert fa.get_default_eager() == eager

    for arr in [arr_from_dim, arr_direct]:
        assert (manual_arr==arr).values(space="pos").all()
        assert arr.eager == (eager,)
        assert arr.xp == xp_compat
        assert arr.values("pos").dtype == dtype_name


def test_bool() -> None:
    xdim = fa.dim("x", n=4, d_pos=0.1, pos_min=-0.2, freq_min=-2.1)
    arr = fa.array_from_dim(dim=xdim, xp=np, space="pos")
    with pytest.raises(ValueError):
        bool(arr)

def draw_hypothesis_fft_array_values(draw, st_type, shape):
    """Creates multi-dimensional array with shape `shape` whose values are drawn
    using `draw` from `st_type`."""
    if len(shape) > 1:
        return [draw_hypothesis_fft_array_values(draw, st_type, shape[1:]) for _ in range(shape[0])]
    return draw(st.lists(st_type, min_size=shape[0], max_size=shape[0]))

@st.composite
def fftarray_strategy(draw) -> FFTArray:
    """Initializes an FFTArray using hypothesis."""
    ndims = draw(st.integers(min_value=1, max_value=4))
    value = st.one_of([
        # st.integers(min_value=np.iinfo(np.int32).min, max_value=np.iinfo(np.int32).max),
        st.complex_numbers(allow_infinity=False, allow_nan=False, allow_subnormal=False, width=64),
        st.floats(allow_infinity=False, allow_nan=False, allow_subnormal=False, width=32)
    ])
    factors_applied = draw(st.lists(st.booleans(), min_size=ndims, max_size=ndims))
    note(f"factors_applied={factors_applied}") # TODO: remove when FFTArray.__repr__ is implemented
    eager = draw(st.lists(st.booleans(), min_size=ndims, max_size=ndims))
    note(f"eager={eager}") # TODO: remove when FFTArray.__repr__ is implemented
    init_space = draw(st.sampled_from(["pos", "freq"]))
    note(f"space={init_space}") # TODO: remove when FFTArray.__repr__ is implemented
    xp = draw(st.sampled_from(XPS))
    dtype = getattr(xp, draw(st.sampled_from(precisions)))

    note(xp)
    note(dtype)
    dims = [
        fa.dim(f"{ndim}", n=draw(st.integers(min_value=2, max_value=8)), d_pos=0.1, pos_min=-0.2, freq_min=-2.1)
    for ndim in range(ndims)]
    note(dims)
    fftarr_values = xp.asarray(np.array(draw_hypothesis_fft_array_values(draw, value, [dim.n for dim in dims])))
    note(fftarr_values.dtype)
    note(fftarr_values)

    if not all(factors_applied):
        fftarr_values = xp.astype(fftarr_values, xp.complex128)
    return fa.array(
        values=fftarr_values,
        dims=dims,
        space=init_space,
        eager=eager,
        factors_applied=factors_applied,
    )

@pytest.mark.slow
@settings(max_examples=1000, deadline=None, print_blob=True)
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
    # TODO: Block this off more generally? Array API does not seem to define
    # an upper limit to the number of dimensions (nor do the JAX docs for that matter).
    if len(fftarr.dims) < 4 or not fftarr.xp==jnp:
        # -- test eager, factors_applied logic
        assert_fftarray_eager_factors_applied(fftarr, note)

@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("precision", precisions)
@pytest.mark.parametrize("space", spaces)
@pytest.mark.parametrize("eager", [True, False])
@pytest.mark.parametrize("factors_applied", [True, False])
def test_fftarray_lazyness_reduced(xp, precision, space, eager, factors_applied):
    """Tests the lazyness of an FFTArray, i.e., the correct behavior of
    factors_applied and eager. This is the reduced/faster version of the test
    using hypothesis.
    """
    xdim = fa.dim("x", n=4, d_pos=0.1, pos_min=-0.2, freq_min=-2.1)
    ydim = fa.dim("y", n=8, d_pos=0.03, pos_min=-0.5, freq_min=-4.7)
    dtype = getattr(xp, precision)
    fftarr = fa.array_from_dim(dim=xdim, xp=xp, dtype=dtype, space=space, eager=eager) + fa.array_from_dim(dim=ydim, xp=xp, dtype=dtype, space=space, eager=eager)
    # TODO: This tests either float without factors or complex with factors.
    if factors_applied:
        fftarr=fftarr.as_factors_applied(factors_applied)
    # assert_basic_lazy_logic(fftarr, print)
    assert_basic_lazy_logic(fftarr, print)
    assert_dual_operand_fun_equivalence(fftarr, all(fftarr._factors_applied), print)
    assert_fftarray_eager_factors_applied(fftarr, print)

@pytest.mark.parametrize("xp", XPS)
def test_immutability(xp) -> None:
    xdim = fa.dim("x", n=4, d_pos=0.1, pos_min=-0.2, freq_min=-2.1)
    arr = fa.array_from_dim(dim=xdim, xp=xp, dtype=xp.float64, space="pos")
    values = arr.values(space="pos")
    assert arr.values(space="pos")[0] == -0.2
    try:
        # For array libraries with immutable arrays (e.g. jax), we assume this fails.
        # In these cases, we skip testing immutability ourself.
        values[0] = 10
    except:
        pass

    assert arr.values(space="pos")[0] == -0.2
    arr_2 = arr.into("freq").into("pos")
    values_2 = arr_2.values(space="pos")
    try:
        values_2[0] = 10
    except:
        pass
    assert arr_2.values(space="pos")[0] == -0.2

def is_precision(arr, precision: Literal["float32", "float64"]) -> bool:
    if isinstance(arr, FFTArray):
        arr = arr._values
    xp = array_api_compat.array_namespace(arr)
    dtype = arr.dtype
    match precision:
        case "float32":
            return xp.float32 == dtype or xp.complex64 == dtype
        case "float64":
            return xp.float64 == dtype or xp.complex128 == dtype
        case _:
            raise ValueError("Passed unsupported precision '{precision}'.")

def assert_basic_lazy_logic(arr, log):
    """Tests whether FFTArray.values() is equal to the internal _values for the
    special cases where factors_applied=True, space="pos" and comparing the
    absolute values, and where space="freq" and comparing values to
    _values/(n*d_freq).
    """
    if all(arr._factors_applied):
        # fftarray must be handled the same way as applying the operations to the values numpy array
        log("factors_applied=True -> x.values(space=x.space) == x._values(space=x.space)")
        np.testing.assert_array_equal(arr.values(space=arr.space), arr._values, strict=True)

    log("space='pos' -> abs(x.values(space='pos')) == abs(x._values)")
    log("space='freq' -> abs(x.values(space='freq')) == abs(x._values)/(n*d_freq)")
    scale = 1
    for dim, space, fa in zip(arr.dims, arr.space, arr._factors_applied):
        if space == "freq" and not fa:
            scale *= 1/(dim.n*dim.d_freq)
    rtol = 1e-6 if is_precision(arr, "float32") else 1e-12
    np.testing.assert_allclose(np.abs(arr.values(space=arr.space)), np.abs(arr._values)*scale, rtol=rtol)

def is_inf_or_nan(x):
    """Check if (real or imag of) x is inf or nan"""
    xp = array_api_compat.array_namespace(x)
    return (xp.any(xp.isinf(x)) or xp.any(xp.isnan(x)))

def internal_and_public_values_should_differ(arr: FFTArray):
    """Returns boolean, whether `arr.values(arr.space)` should differ from
    `arr._values`.
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
                if arr.xp.any(arr.xp.take(arr._values, arr.xp.arange(1,arr.dims[i].n), axis=i)!=0):
                    return True
            else:
                # for space="freq", the factor includes scale unequal 1, so all
                # values along this dimension must be non-zero
                if arr.xp.any(arr._values!=0):
                    return True
    return False

def assert_equal_op(
        arr: FFTArray,
        values: Any,
        ops: Union[Callable[[Any],Any], Tuple[Callable[[Any],Any], Callable[[Any],Any]]],
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
    op(FFTArray)._values deviates from op(FFTArray).values() (which is the case
    if the factors have not been applied after operation and if the values are
    non-zero). If it is True, it is tested if they are equal.
    """
    if isinstance(ops, tuple):
        np_op, fa_op = ops
    else:
        np_op = ops
        fa_op = ops

    f_arr_op = fa_op(arr)
    arr_op = f_arr_op.values(space=arr.space)
    values_op = np_op(values)

    xp = array_api_compat.array_namespace(arr_op, values_op)
    if arr_op.dtype != values_op.dtype:
        log(f"Changing type to {values_op.dtype}")
        arr_op = xp.astype(arr_op, values_op.dtype)
        # TODO: Why is this necessary?
        values_op = xp.astype(values_op, values_op.dtype)

    if is_inf_or_nan(values_op) or (precise==False and is_inf_or_nan(arr_op)):
        return

    rtol = 1e-6 if is_precision(arr, "float32") else 1e-7
    if precise and ("int" in str(values.dtype) or is_precision(arr, "float64")):
        if "int" in str(arr_op.dtype):
            np.testing.assert_array_equal(arr_op, values_op, strict=True)
        if "float" in str(arr_op.dtype):
            np.testing.assert_array_almost_equal_nulp(arr_op, values_op, nulp=4)
        if "complex" in str(arr_op.dtype):
            assert_array_almost_equal_nulp_complex(arr_op, values_op, nulp=4)
    else:
        np.testing.assert_allclose(arr_op, values_op, rtol=rtol, atol=1e-38)

    _arr_op = fa_op(arr)._values
    if op_forces_factors_applied:
        # _values should have factors applied
        np.testing.assert_allclose(_arr_op, values_op, rtol=rtol, atol=1e-38)
    else:
        # arr._values can differ from arr.values()
        if internal_and_public_values_should_differ(arr):
            with pytest.raises(AssertionError):
                np.testing.assert_allclose(_arr_op, values_op, rtol=rtol)
        else:
            np.testing.assert_allclose(_arr_op, values_op, rtol=rtol, atol=1e-38)


def assert_array_almost_equal_nulp_complex(x: Any, y: Any, nulp: int):
    """Compare two arrays of complex numbers. Simply compares the real and
    imaginary part.
    """
    xp = array_api_compat.array_namespace(x,y)
    np.testing.assert_array_almost_equal_nulp(xp.real(x), xp.real(y), nulp)
    np.testing.assert_array_almost_equal_nulp(xp.imag(x), xp.imag(y), nulp)

def assert_single_operand_fun_equivalence(arr: FFTArray, precise: bool, log):
    """Test whether applying operands to the FFTArray (and then getting the
    values) is equivalent to applying the same operands to the values array:

        operand(FFTArray).values() == operand(FFTArray.values())

    """
    values = arr.values(space=arr.space)
    xp = arr.xp
    log("f(x) = x")
    assert_equal_op(arr, values, lambda x: x, precise, False, log)
    log("f(x) = pi*x")
    assert_equal_op(arr, values, lambda x: np.pi*x, precise, False, log)
    log("f(x) = abs(x)")
    assert_equal_op(arr, values, (xp.abs, fa.abs), precise, True, log)
    log("f(x) = x**2")
    assert_equal_op(arr, values, lambda x: x**2, precise, True, log)
    log("f(x) = x**3")
    assert_equal_op(arr, values, lambda x: x**3, precise, True, log)
    log("f(x) = exp(x)")
    assert_equal_op(arr, values, (xp.exp, fa.exp), False, True, log) # precise comparison fails
    log("f(x) = sqrt(x)")
    assert_equal_op(arr, values, (xp.sqrt, fa.sqrt), False, True, log) # precise comparison fails

def assert_dual_operand_fun_equivalence(arr: FFTArray, precise: bool, log):
    """Test whether a dual operation on an FFTArray, e.g., the
    sum/multiplication of two, is equivalent to applying this operand to its
    values.

        operand(FFTArray, FFTArray).values() = operand(FFTArray.values(), FFTArray.values())

    """
    values = arr.values(space=arr.space)
    xp = arr.xp

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
        assert_equal_op(arr, values, (lambda x: x**xp.abs(x), lambda x: x**fa.abs(x)), precise, True, log)
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
    arr_abs = fa.abs(arr)
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

@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("space", spaces)
@pytest.mark.parametrize("eager", [True, False])
@pytest.mark.parametrize("factors_applied_1", [True, False])
@pytest.mark.parametrize("factors_applied_2", [True, False])
def test_transform_application(
        xp,
        space: Space,
        eager: bool,
        factors_applied_1: bool,
        factors_applied_2: bool,
    ):
    """
    Tests whether `factors_applied` is correctly handled in addition and
    multiplication for a single dimension and for two different dimensions.
    """
    x_dim = fa.dim("x", n=5, d_pos=0.1, pos_min=0, freq_min=0)
    y_dim = fa.dim("y", n=8, d_pos=0.1, pos_min=0, freq_min=0)

    x_arr = fa.array_from_dim(x_dim, space, xp=xp, eager=eager).as_factors_applied(factors_applied_1)
    x2_arr = fa.array_from_dim(x_dim, space, xp=xp, eager=eager).as_factors_applied(factors_applied_2)
    y_arr = fa.array_from_dim(y_dim, space, xp=xp, eager=eager).as_factors_applied(factors_applied_2)

    x2_add = x_arr + x2_arr
    xy_add = x_arr + y_arr
    if factors_applied_1 == factors_applied_2:
        assert x2_add.factors_applied == (factors_applied_1,)
    else:
        assert x2_add.factors_applied == (eager,)

    if eager:
        assert xy_add.factors_applied == (True, True)
    else:
        assert xy_add.factors_applied == (factors_applied_1, factors_applied_2)

    assert xp.all(x2_add.values(space) == x_arr.values(space) + x2_arr.values(space))
    np.testing.assert_allclose(
        np.array(xy_add.values(space)),
        np.array(xp.reshape(x_arr.values(space), shape=(-1,1)) + xp.reshape(y_arr.values(space), shape=(1,-1))),
    )

    x2_mul = x_arr * x2_arr
    xy_mul = x_arr * y_arr
    if factors_applied_1 == True and factors_applied_2 == True:
        assert x2_mul.factors_applied == (True,)
    else:
        assert x2_mul.factors_applied == (False,)
    assert xy_mul.factors_applied == (factors_applied_1, factors_applied_2)
    np.testing.assert_allclose(
        np.array(x2_mul.values(space)),
        np.array(x_arr.values(space) * x2_arr.values(space)),
    )
    np.testing.assert_allclose(
        np.array(xy_mul.values(space)),
        np.array(xp.reshape(x_arr.values(space), shape=(-1,1)) * xp.reshape(y_arr.values(space), shape=(1,-1))),
    )


@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("space", spaces)
def test_fft_ifft_invariance(xp, space: Space):
    """Tests whether ifft(fft(*)) is an identity.

       ifft(fft(FFTArray)) == FFTArray

    """
    xdim = fa.dim("x", n=4, d_pos=0.1, pos_min=-0.2, freq_min=-2.1)
    ydim = fa.dim("y", n=8, d_pos=0.03, pos_min=-0.4, freq_min=-4.2)
    arr = fa.array_from_dim(dim=xdim, xp=xp, space=space) + fa.array_from_dim(dim=ydim, xp=xp, space=space)
    other_space = get_other_space(space)
    arr_fft = arr.into(space=other_space)
    arr_fft_ifft = arr_fft.into(space=space)
    if is_inf_or_nan(arr_fft_ifft.values(space=arr_fft_ifft.space)):
        # edge cases (very large numbers) result in inf after fft
        return
    rtol = 1e-5 if is_precision(arr, "float32") else 1e-6
    np.testing.assert_allclose(arr.values(space=arr.space), arr_fft_ifft.values(space=arr_fft_ifft.space), rtol=rtol, atol=1e-38)


@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("spaces", [("pos", "freq"), ("freq", "pos")])
@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_np_array(xp, spaces: Tuple[Space, Space], precision: PrecisionSpec):
    """Tests if `FFTArray.np_array` returns the values as a NumPy array and if it has the correct precision.
    """
    xdim = fa.dim("x", n=4, d_pos=0.1, pos_min=-0.2, freq_min=-2.1)
    arr = fa.array_from_dim(dim=xdim, xp=xp, dtype=getattr(xp, precision), space=spaces[0])

    np_arr_same = arr.np_array(space=spaces[0])
    assert isinstance(np_arr_same, np.ndarray)
    if precision == "float32":
        assert np_arr_same.dtype == np.float32
    elif precision == "float64":
        assert np_arr_same.dtype == np.float64

    np_arr_other = arr.np_array(space=spaces[1])
    assert isinstance(np_arr_other, np.ndarray)
    if precision == "float32":
        assert np_arr_other.dtype == np.complex64
    elif precision == "float64":
        assert np_arr_other.dtype == np.complex128

try:
    import jax
    import jax.numpy as jnp
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

        xdim = fa.dim("x", n=4, d_pos=0.1, pos_min=-0.2, freq_min=-2.1, dynamically_traced_coords=dtc)
        ydim = fa.dim("y", n=8, d_pos=0.03, pos_min=-0.4, freq_min=-4.2, dynamically_traced_coords=dtc)
        fftarr = fa.array_from_dim(dim=xdim, xp=jnp, space=space) + fa.array_from_dim(dim=ydim, xp=jnp, space=space)

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
except:
    ImportError

def test_different_dimension_dynamic_prop() -> None:
    """Tests tracing of an FFTArray whose dimensions have different
    `dynamically_traced_coords`.
    """
    x_dim = fa.dim(name="x", pos_min=0, freq_min=0, d_pos=1, n=8, dynamically_traced_coords=False)
    y_dim = fa.dim(name="y", pos_min=0, freq_min=0, d_pos=1, n=4, dynamically_traced_coords=True)
    fftarr = fa.array_from_dim(dim=x_dim, xp=jnp, space="pos") + fa.array_from_dim(dim=y_dim, xp=jnp, space="pos")

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

