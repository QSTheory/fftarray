from typing import List, Literal, Any, Callable, Union, Tuple

import array_api_compat
import pytest

from hypothesis import given, strategies as st, note, settings
import numpy as np
import jax.numpy as jnp

import fftarray as fa

from tests.helpers import XPS, get_other_space

PrecisionSpec = Literal["float32", "float64"]

def assert_scalars_almost_equal_nulp(x, y, nulp = 1):
    np.testing.assert_array_almost_equal_nulp(np.array([x]), np.array([y]), nulp = nulp)

precisions: List[PrecisionSpec] = ["float32", "float64"]
spaces: List[fa.Space] = ["pos", "freq"]


@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("space", ["pos", "freq"])
def test_comparison(xp, space) -> None:
    x_dim = fa.dim("x",
        n=4,
        d_pos=1,
        pos_min=0.5,
        freq_min=0.,
    )
    x = fa.coords_from_dim(x_dim, space, xp=xp)
    x_sq = x**2

    x = x.np_array(space)
    x_sq = x_sq.np_array(space)

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
def test_dtype(xp, init, override) -> None:
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
        assert fa.coords_from_dim(x_dim, "pos", dtype=init_dtype_real, xp=xp).values("pos").dtype == init_dtype_real
    else:
        assert fa.coords_from_dim(x_dim, "pos", dtype=override_dtype_real, xp=xp).values("pos").dtype == override_dtype_real
        assert fa.coords_from_dim(x_dim, "pos", dtype=init_dtype_real, xp=xp).into_dtype(override_dtype_real).values("pos").dtype == override_dtype_real


    if override is None:
        assert fa.coords_from_dim(x_dim, "freq", dtype=init_dtype_real, xp=xp).values("freq").dtype == init_dtype_real
    else:
        assert fa.coords_from_dim(x_dim, "freq", dtype=override_dtype_real, xp=xp).values("freq").dtype == override_dtype_real
        assert fa.coords_from_dim(x_dim, "freq", dtype=init_dtype_real, xp=xp).into_dtype(override_dtype_real).values("freq").dtype == override_dtype_real

    assert fa.coords_from_dim(x_dim, "pos", dtype=init_dtype_real, xp=xp).into_space("freq").values("freq").dtype == init_dtype_complex
    assert fa.coords_from_dim(x_dim, "freq", dtype=init_dtype_real, xp=xp).into_space("pos").values("pos").dtype == init_dtype_complex

    assert fa.abs(fa.coords_from_dim(x_dim, "pos", dtype=init_dtype_real, xp=xp).into_space("freq")).values("freq").dtype == init_dtype_real # type: ignore
    assert fa.abs(fa.coords_from_dim(x_dim, "freq", dtype=init_dtype_real, xp=xp).into_space("pos")).values("pos").dtype == init_dtype_real # type: ignore

    if override is not None:
        assert fa.coords_from_dim(x_dim, "pos", dtype=init_dtype_real, xp=xp).into_dtype(override_dtype_real).into_space("freq").values("freq").dtype == override_dtype_complex
        assert fa.coords_from_dim(x_dim, "freq", dtype=init_dtype_real, xp=xp).into_dtype(override_dtype_real).into_space("pos").values("pos").dtype == override_dtype_complex


@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("xp_override", XPS)
def test_backend_override(xp, xp_override) -> None:
    x_dim = fa.dim("x",
        n=4,
        d_pos=1,
        pos_min=0.,
        freq_min=0.,
    )

    assert type(fa.coords_from_dim(x_dim, "pos", xp=xp).into_xp(xp_override).values("pos")) is type(fa.coords_from_dim(x_dim, "pos", xp=xp_override).values("pos"))
    assert type(fa.coords_from_dim(x_dim, "freq", xp=xp).into_xp(xp_override).values("freq")) is type(fa.coords_from_dim(x_dim, "freq", xp=xp_override).values("freq"))
    assert type(fa.coords_from_dim(x_dim, "pos", xp=xp).into_xp(xp_override).into_space("freq").values("freq")) is type(fa.coords_from_dim(x_dim, "freq", xp=xp_override).values("freq"))
    assert type(fa.coords_from_dim(x_dim, "freq", xp=xp).into_xp(xp_override).into_space("pos").values("pos")) is type(fa.coords_from_dim(x_dim, "pos", xp=xp_override).values("pos"))


def test_broadcasting() -> None:
    x_dim = fa.dim("x", n=4, d_pos=1, pos_min=0., freq_min=0.)
    y_dim = fa.dim("y", n=8, d_pos=1, pos_min=0., freq_min=0.)

    x_ref = np.arange(0., 4.)
    y_ref = np.arange(0., 8.)
    np.testing.assert_array_almost_equal_nulp(fa.coords_from_dim(x_dim, "pos", xp=np).np_array("pos"), x_ref, nulp = 0)
    np.testing.assert_array_almost_equal_nulp(fa.coords_from_dim(y_dim, "pos", xp=np).np_array("pos"), y_ref, nulp = 0)

    x_ref_broadcast = x_ref.reshape(1,-1)
    y_ref_broadcast = y_ref.reshape(-1,1)
    np.testing.assert_array_almost_equal_nulp(
        fa.permute_dims(fa.coords_from_dim(x_dim, "pos", xp=np) + fa.coords_from_dim(y_dim, "pos", xp=np), ("x", "y")).values("pos"),
        (x_ref_broadcast+y_ref_broadcast).transpose(),
        nulp = 0
    )
    np.testing.assert_array_almost_equal_nulp(
        fa.permute_dims(fa.coords_from_dim(x_dim, "pos", xp=np) + fa.coords_from_dim(y_dim, "pos", xp=np), ("y", "x")).values("pos"),
        x_ref_broadcast+y_ref_broadcast,
        nulp = 0
    )


@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("space", spaces)
def test_sel_order(xp, space) -> None:
    """Tests whether the selection order matters. Assuming an input Array of
    dimensions A and B. Then

        Array.sel(A==a).sel(B==b) == Array.sel(B==b).sel(A==a)

    should be true.
    """
    xdim = fa.dim("x", n=4, d_pos=0.1, pos_min=-0.2, freq_min=-2.1)
    ydim = fa.dim("y", n=8, d_pos=0.03, pos_min=-0.5, freq_min=-4.7)
    arr = fa.coords_from_dim(xdim, space, xp=xp) + fa.coords_from_dim(ydim, space, xp=xp)
    arr_selx = arr.sel(**{"x": getattr(xdim, f"{space}_middle")})
    arr_sely = arr.sel(**{"y": getattr(ydim, f"{space}_middle")})
    arr_selx_sely = arr_selx.sel(**{"y": getattr(ydim, f"{space}_middle")})
    arr_sely_selx = arr_sely.sel(**{"x": getattr(xdim, f"{space}_middle")})
    np.testing.assert_allclose(arr_selx_sely.values(space), arr_sely_selx.values(space))


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


def check_defaults(dim: fa.Dimension, xp, dtype_name: fa.DEFAULT_DTYPE, eager: bool) -> None:
    xp_compat = array_api_compat.array_namespace(xp.asarray(0))
    values = 0.1*xp.arange(4, dtype=dtype_name)
    arr_from_dim = fa.coords_from_dim(dim, "pos")
    arr_direct = fa.array(values, dim, "pos")
    manual_arr = fa.Array(
        values=values,
        dims=(dim,),
        spaces=("pos",),
        eager=(eager,),
        xp=xp_compat,
        factors_applied=(True,),
    )

    assert fa.get_default_xp() == xp_compat
    assert fa.get_default_dtype_name() == dtype_name
    assert fa.get_default_eager() == eager

    for arr in [arr_from_dim, arr_direct]:
        assert (manual_arr==arr).values("pos").all()
        assert arr.eager == (eager,)
        assert arr.xp == xp_compat
        assert arr.values("pos").dtype == dtype_name


def test_bool() -> None:
    xdim = fa.dim("x", n=4, d_pos=0.1, pos_min=-0.2, freq_min=-2.1)
    arr = fa.coords_from_dim(xdim, "pos", xp=np)
    with pytest.raises(ValueError):
        bool(arr)

def draw_hypothesis_array_values(draw, st_type, shape):
    """Creates multi-dimensional array with shape `shape` whose values are drawn
    using `draw` from `st_type`."""
    if len(shape) > 1:
        return [draw_hypothesis_array_values(draw, st_type, shape[1:]) for _ in range(shape[0])]
    return draw(st.lists(st_type, min_size=shape[0], max_size=shape[0]))

@st.composite
def array_strategy(draw) -> fa.Array:
    """Initializes an Array using hypothesis."""
    ndims = draw(st.integers(min_value=1, max_value=4))
    value = st.one_of([
        # st.integers(min_value=np.iinfo(np.int32).min, max_value=np.iinfo(np.int32).max),
        st.complex_numbers(allow_infinity=False, allow_nan=False, allow_subnormal=False, width=64),
        st.floats(allow_infinity=False, allow_nan=False, allow_subnormal=False, width=32)
    ])
    factors_applied = draw(st.lists(st.booleans(), min_size=ndims, max_size=ndims))
    note(f"factors_applied={factors_applied}") # TODO: remove when Array.__repr__ is implemented
    eager = draw(st.lists(st.booleans(), min_size=ndims, max_size=ndims))
    note(f"eager={eager}") # TODO: remove when Array.__repr__ is implemented
    init_space = draw(st.sampled_from(["pos", "freq"]))
    note(f"spaces={init_space}") # TODO: remove when Array.__repr__ is implemented
    xp = draw(st.sampled_from(XPS))
    dtype = getattr(xp, draw(st.sampled_from(precisions)))

    note(xp)
    note(dtype)
    dims = [
        fa.dim(f"{ndim}", n=draw(st.integers(min_value=2, max_value=8)), d_pos=0.1, pos_min=-0.2, freq_min=-2.1)
    for ndim in range(ndims)]
    note(dims)
    arr_values = xp.asarray(np.array(draw_hypothesis_array_values(draw, value, [dim.n for dim in dims])))
    note(arr_values.dtype)
    note(arr_values)

    if not all(factors_applied):
        arr_values = xp.astype(arr_values, xp.complex128)
    return (
        fa.array(arr_values, dims, init_space)
        .into_factors_applied(factors_applied)
        .into_eager(eager)
    )

@pytest.mark.slow
@settings(max_examples=1000, deadline=None, print_blob=True)
@given(array_strategy())
def test_array_lazyness(arr):
    """Tests the lazyness of an Array, i.e., the correct behavior of
    factors_applied and eager.
    """
    note(arr)
    # -- basic tests
    assert_basic_lazy_logic(arr, note)
    # -- test operands
    assert_single_operand_fun_equivalence(arr, all(arr._factors_applied), note)
    assert_dual_operand_fun_equivalence(arr, all(arr._factors_applied), note)
    # Jax only supports FFT for dim<4
    # TODO: Block this off more generally? Array API does not seem to define
    # an upper limit to the number of dimensions (nor do the JAX docs for that matter).
    if len(arr.dims) < 4 or not arr.xp==jnp:
        # -- test eager, factors_applied logic
        assert_array_eager_factors_applied(arr, note)

@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("precision", precisions)
@pytest.mark.parametrize("space", spaces)
@pytest.mark.parametrize("eager", [True, False])
@pytest.mark.parametrize("factors_applied", [True, False])
def test_array_lazyness_reduced(xp, precision, space, eager, factors_applied) -> None:
    """Tests the lazyness of an Array, i.e., the correct behavior of
    factors_applied and eager. This is the reduced/faster version of the test
    using hypothesis.
    """
    xdim = fa.dim("x", n=4, d_pos=0.1, pos_min=-0.2, freq_min=-2.1)
    ydim = fa.dim("y", n=8, d_pos=0.03, pos_min=-0.5, freq_min=-4.7)
    dtype = getattr(xp, precision)
    arr = fa.coords_from_dim(xdim, space, xp=xp, dtype=dtype).into_eager(eager) + fa.coords_from_dim(ydim, space, xp=xp, dtype=dtype).into_eager(eager)
    # TODO: This tests either float without factors or complex with factors.
    if factors_applied:
        arr=arr.into_factors_applied(factors_applied)
    assert_basic_lazy_logic(arr, print)
    assert_dual_operand_fun_equivalence(arr, all(arr._factors_applied), print)
    assert_array_eager_factors_applied(arr, print)

@pytest.mark.parametrize("xp", XPS)
def test_immutability(xp) -> None:
    xdim = fa.dim("x", n=4, d_pos=0.1, pos_min=-0.2, freq_min=-2.1)
    arr = fa.coords_from_dim(xdim, "pos", xp=xp, dtype=xp.float64)
    values = arr.values("pos")
    assert arr.values("pos")[0] == -0.2
    try:
        # For array libraries with immutable arrays (e.g. jax), we assume this fails.
        # In these cases, we skip testing immutability ourself.
        values[0] = 10
    except(TypeError):
        pass

    assert arr.values("pos")[0] == -0.2
    arr_2 = arr.into_space("freq").into_space("pos")
    values_2 = arr_2.values("pos")
    try:
        values_2[0] = 10
    except(TypeError):
        pass
    assert arr_2.values("pos")[0] == -0.2

def is_precision(arr, precision: Literal["float32", "float64"]) -> bool:
    if isinstance(arr, fa.Array):
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
    """Tests whether Array.values() is equal to the internal _values for the
    special cases where factors_applied=True, spaces="pos" and comparing the
    absolute values, and where spaces="freq" and comparing values to
    _values/(n*d_freq).
    """
    if all(arr._factors_applied):
        # Array must be handled the same way as applying the operations to the values numpy array
        log("factors_applied=True -> x.values(x.spaces) == x._values(spaces=x.spaces)")
        np.testing.assert_array_equal(arr.values(arr.spaces), arr._values, strict=True)

    log("spaces='pos' -> abs(x.values('pos')) == abs(x._values)")
    log("spaces='freq' -> abs(x.values('freq')) == abs(x._values)/(n*d_freq)")
    scale = 1
    for dim, space, factors_applied in zip(arr.dims, arr.spaces, arr._factors_applied, strict=True):
        if space == "freq" and not factors_applied:
            scale *= 1/(dim.n*dim.d_freq)
    rtol = 1e-6 if is_precision(arr, "float32") else 1e-12
    np.testing.assert_allclose(np.abs(arr.values(arr.spaces)), np.abs(arr._values)*scale, rtol=rtol)

def is_inf_or_nan(x):
    """Check if (real or imag of) x is inf or nan"""
    xp = array_api_compat.array_namespace(x)
    return (xp.any(xp.isinf(x)) or xp.any(xp.isnan(x)))

def internal_and_public_values_should_differ(arr: fa.Array) -> bool:
    """Returns boolean, whether `arr.values(arr.spaces)` should differ from
    `arr._values`.
    This is the case if `factors_applied=False` and the values are non-zero
    (along at least one dimension).
    Note that the position space needs to be treated separately as the phase
    factor for the first coordinate is 1 (and thus does not change `_values`).
    """
    for factor, space, i, in zip(arr._factors_applied, arr.spaces, range(len(arr.dims)), strict=True):
        if not factor:
            # factor needs to be applied
            if space == "pos":
                # check if the values are non-zero
                # the phase factor in position space is 1 for the first
                # coordinate and, thus, is excluded from the check
                if arr.xp.any(arr.xp.take(arr._values, arr.xp.arange(1,arr.dims[i].n), axis=i)!=0):
                    return True
            else:
                # for spaces="freq", the factor includes scale unequal 1, so all
                # values along this dimension must be non-zero
                if arr.xp.any(arr._values!=0):
                    return True
    return False

def assert_equal_op(
        arr: fa.Array,
        values: Any,
        ops: Union[Callable[[Any],Any], Tuple[Callable[[Any],Any], Callable[[Any],Any]]],
        precise: bool,
        op_forces_factors_applied: bool,
        log
    ):
    """Helper function to test equality between an Array and a values array.
    `op` denotes the operation acting on the Array and on the values before
    comparison.
    `precise` denotes whether the comparison is performed using nulp (number of
    unit in the last place for tolerance) or using the less stringent
    `numpy.testing.allclose`.
    If `op_forces_factors_applied` is False, it will be tested whether
    op(Array)._values deviates from op(Array).values() (which is the case
    if the factors have not been applied after operation and if the values are
    non-zero). If it is True, it is tested if they are equal.
    """
    if isinstance(ops, tuple):
        np_op, fa_op = ops
    else:
        np_op = ops
        fa_op = ops

    f_arr_op = fa_op(arr)
    arr_op = f_arr_op.values(arr.spaces)
    values_op = np_op(values)

    xp = array_api_compat.array_namespace(arr_op, values_op)
    if arr_op.dtype != values_op.dtype:
        log(f"Changing type to {values_op.dtype}")
        arr_op = xp.astype(arr_op, values_op.dtype)
        # TODO: Why is this necessary?
        values_op = xp.astype(values_op, values_op.dtype)

    if is_inf_or_nan(values_op) or (not precise and is_inf_or_nan(arr_op)):
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

def assert_single_operand_fun_equivalence(arr: fa.Array, precise: bool, log):
    """Test whether applying operands to the Array (and then getting the
    values) is equivalent to applying the same operands to the values array:

        operand(Array).values() == operand(Array.values())

    """
    values = arr.values(arr.spaces)
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

def assert_dual_operand_fun_equivalence(arr: fa.Array, precise: bool, log):
    """Test whether a dual operation on an Array, e.g., the
    sum/multiplication of two, is equivalent to applying this operand to its
    values.

        operand(Array, Array).values() = operand(Array.values(), Array.values())

    """
    values = arr.values(arr.spaces)
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

def assert_array_eager_factors_applied(arr: fa.Array, log):
    """Tests whether the factors are only applied when necessary and whether
    the Array after performing an FFT has the correct properties. If the
    initial Array was eager, then the final Array also must be eager and
    have _factors_applied=True. If the initial Array was not eager, then the
    final Array should have eager=False and _factors_applied=False.
    """

    log("arr._factors_applied == (arr**2)._factors_applied")
    arr_sq = arr * arr
    np.testing.assert_array_equal(arr_sq.eager, arr.eager)
    np.testing.assert_array_equal(arr_sq._factors_applied, arr._factors_applied)

    log("abs(x)._factors_applied == True")
    arr_abs = fa.abs(arr)
    np.testing.assert_array_equal(arr_abs.eager, arr.eager)
    np.testing.assert_array_equal(arr_abs._factors_applied, True)

    log("(x*abs(x))._factors_applied == x._factors_applied")
    # if both _factors_applied=True, the resulting Array will also have it
    # True, otherwise False
    # given abs(x)._factors_applied=True, we test the patterns
    # True*True=True, False*True=False
    arr_abs_sq = arr * arr_abs
    np.testing.assert_array_equal(arr_abs_sq.eager, arr.eager)
    np.testing.assert_array_equal(arr_abs_sq._factors_applied, arr._factors_applied)

    log("(x+abs(x))._factors_applied == (x._factors_applied or x._eager)")
    arr_abs_sum = arr + arr_abs
    np.testing.assert_array_equal(arr_abs_sum.eager, arr.eager)
    for ea, ifa, ffa in zip(arr_abs_sum.eager, arr._factors_applied, arr_abs_sum._factors_applied, strict=True):
        # True+True=True
        # False+True=eager
        assert (ifa == ffa) or (ffa == ea)

    log("fft(x)._factors_applied ...")
    arr_fft = arr.into_space(get_other_space(arr.spaces))
    np.testing.assert_array_equal(arr.eager, arr_fft.eager)
    for ffapplied, feager in zip(arr_fft._factors_applied, arr_fft.eager, strict=True):
        assert (feager and ffapplied) or (not feager and not ffapplied)


@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("space", spaces)
def test_fft_ifft_invariance(xp, space: fa.Space):
    """Tests whether ifft(fft(*)) is an identity.

       ifft(fft(Array)) == Array

    """
    xdim = fa.dim("x", n=4, d_pos=0.1, pos_min=-0.2, freq_min=-2.1)
    ydim = fa.dim("y", n=8, d_pos=0.03, pos_min=-0.4, freq_min=-4.2)
    arr = fa.coords_from_dim(xdim, space, xp=xp) + fa.coords_from_dim(ydim, space, xp=xp)
    other_space = get_other_space(space)
    arr_fft = arr.into_space(other_space)
    arr_fft_ifft = arr_fft.into_space(space)
    if is_inf_or_nan(arr_fft_ifft.values(arr_fft_ifft.spaces)):
        # edge cases (very large numbers) result in inf after fft
        return
    rtol = 1e-5 if is_precision(arr, "float32") else 1e-6
    np.testing.assert_allclose(arr.values(arr.spaces), arr_fft_ifft.values(arr_fft_ifft.spaces), rtol=rtol, atol=1e-38)


@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("spaces", [("pos", "freq"), ("freq", "pos")])
@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_np_array(xp, spaces: Tuple[fa.Space, fa.Space], precision: PrecisionSpec):
    """Tests if `Array.np_array` returns the values as a NumPy array and if it has the correct precision.
    """
    xdim = fa.dim("x", n=4, d_pos=0.1, pos_min=-0.2, freq_min=-2.1)
    arr = fa.coords_from_dim(xdim, spaces[0], xp=xp, dtype=getattr(xp, precision))

    np_arr_same = arr.np_array(spaces[0])
    assert isinstance(np_arr_same, np.ndarray)
    if precision == "float32":
        assert np_arr_same.dtype == np.float32
    elif precision == "float64":
        assert np_arr_same.dtype == np.float64

    np_arr_other = arr.np_array(spaces[1])
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
    def test_grid_manipulation_in_jax_scan(space: fa.Space, dtc: bool, sel_method: str) -> None:
        """Tests Dimension's `dynamically_traced_coords` property on the level of
        an `Array`.

        Allowed by dynamic, error for static:
        - change Dimension properties of an Array inside a `jax.lax.scan` step
        function

        Allowed by static, error for dynamic:
        - selection by coordinate
        """

        xdim = fa.dim("x", n=4, d_pos=0.1, pos_min=-0.2, freq_min=-2.1, dynamically_traced_coords=dtc)
        ydim = fa.dim("y", n=8, d_pos=0.03, pos_min=-0.4, freq_min=-4.2, dynamically_traced_coords=dtc)
        arr = fa.coords_from_dim(xdim, space, xp=jnp) + fa.coords_from_dim(ydim, space, xp=jnp)

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
            carry.sel(x=xval, method=sel_method)
            return carry, None

        if dtc:
            jax.lax.scan(jax_scan_step_fun_dynamic, arr, jnp.arange(3))
            # internal logic in sel throws NotImplementedError for jitted index
            with pytest.raises(NotImplementedError):
                jax.lax.scan(jax_scan_step_fun_static, arr, jnp.arange(3))
        else:
            jax.lax.scan(jax_scan_step_fun_static, arr, jnp.arange(3))
            with pytest.raises(TypeError):
                jax.lax.scan(jax_scan_step_fun_dynamic, arr, jnp.arange(3))
except(ImportError):
    pass

def test_different_dimension_dynamic_prop() -> None:
    """Tests tracing of an Array whose dimensions have different
    `dynamically_traced_coords`.
    """
    x_dim = fa.dim(name="x", pos_min=0, freq_min=0, d_pos=1, n=8, dynamically_traced_coords=False)
    y_dim = fa.dim(name="y", pos_min=0, freq_min=0, d_pos=1, n=4, dynamically_traced_coords=True)
    arr = fa.coords_from_dim(x_dim, "pos", xp=jnp) + fa.coords_from_dim(y_dim, "pos", xp=jnp)

    def jax_scan_step_fun_valid(carry, *_):
        xval = carry._dims[0]._pos_min + carry._dims[0]._d_pos # static dimension
        new_dims = list(carry._dims)
        new_dims[1]._pos_min = 0.123 # dynamic dimension
        carry_sel = carry.sel(x=xval, method="nearest")
        carry._dims = tuple(new_dims)
        return carry, carry_sel

    jax.lax.scan(jax_scan_step_fun_valid, arr, jnp.arange(3))

    def jax_scan_step_fun_invalid_change(carry, *_):
        new_dims = list(carry._dims)
        new_dims[0]._pos_min = 0.123
        carry._dims = tuple(new_dims)
        return carry, None

    def jax_scan_step_fun_invalid_sel(carry, *_):
        carry._dims[1]._pos_min + carry._dims[1]._d_pos
        carry_sel = carry.sel(y=0, method="nearest")
        return carry, carry_sel

    with pytest.raises(TypeError):
        jax.lax.scan(jax_scan_step_fun_invalid_change, arr, jnp.arange(3))

    with pytest.raises(NotImplementedError):
        jax.lax.scan(jax_scan_step_fun_invalid_sel, arr, jnp.arange(3))

@st.composite
def array_strategy_int(draw) -> fa.Array:
    """Initializes an Array using hypothesis."""
    ndims = draw(st.integers(min_value=1, max_value=4))
    value = st.integers(min_value=np.iinfo(np.int32).min, max_value=np.iinfo(np.int32).max)
    eager = draw(st.lists(st.booleans(), min_size=ndims, max_size=ndims))
    note(f"eager={eager}") # TODO: remove when Array.__repr__ is implemented
    init_space = draw(st.sampled_from(["pos", "freq"]))
    note(f"spaces={init_space}") # TODO: remove when Array.__repr__ is implemented
    # xp = draw(st.sampled_from(XPS))
    xp = XPS[0] # this is array_api_strict
    dtype = getattr(xp, draw(st.sampled_from(precisions)))

    note(xp)
    note(dtype)
    dims = [
        fa.dim(f"{ndim}", n=draw(st.integers(min_value=2, max_value=8)), d_pos=0.1, pos_min=-0.2, freq_min=-2.1)
    for ndim in range(ndims)]
    note(dims)
    arr_values = xp.asarray(np.array(draw_hypothesis_array_values(draw, value, [dim.n for dim in dims])))
    note(arr_values.dtype)
    note(arr_values)

    return fa.array(arr_values, dims, init_space).into_eager(eager)


@pytest.mark.slow
@settings(max_examples=1000, deadline=None, print_blob=True)
@given(array_strategy_int())
def test_array_lazyness_int(arr):
    """Tests the lazyness of an Array, i.e., the correct behavior of
    factors_applied and eager.
    """
    note(arr)
    # -- basic tests
    assert_basic_lazy_logic(arr, note)
    # -- test operands
    assert_single_operand_fun_equivalence_int(arr, all(arr._factors_applied), note)
    assert_dual_operand_fun_equivalence_int(arr, all(arr._factors_applied), note)
    # Jax only supports FFT for dim<4
    # TODO: Block this off more generally? Array API does not seem to define
    # an upper limit to the number of dimensions (nor do the JAX docs for that matter).
    if len(arr.dims) < 4 or not arr.xp==jnp:
        # -- test eager, factors_applied logic
        assert_array_eager_factors_applied_int(arr, note)

def assert_dual_operand_fun_equivalence_int(arr: fa.Array, precise: bool, log):
    """Test whether a dual operation on an Array, e.g., the
    sum/multiplication of two, is equivalent to applying this operand to its
    values.

        operand(Array, Array).values() = operand(Array.values(), Array.values())

    """
    values = arr.values(arr.spaces)
    xp = array_api_compat.array_namespace(values)

    log("f(x,y) = x+y")
    assert_equal_op(arr, values, lambda x: x+x, precise, False, log)
    log("f(x,y) = x-2*y")
    assert_equal_op(arr, values, lambda x: x-2*x, precise, False, log)
    log("f(x,y) = x*y")
    assert_equal_op(arr, values, lambda x: x*x, precise, False, log)
    # log("f(x,y) = x/y")
    # assert_equal_op(arr, values, lambda x: x/x, precise, False, log)
    log("f(x,y) = x**y")
    # integers to negative integer powers are not allowed
    if "int" in str(values.dtype):
        assert_equal_op(arr, values, (lambda x: x**xp.abs(x), lambda x: x**fa.abs(x)), precise, True, log)
    else:
        assert_equal_op(arr, values, lambda x: x**x, precise, True, log)

def assert_array_eager_factors_applied_int(arr: fa.Array, log):
    """Tests whether the factors are only applied when necessary and whether
    the Array after performing an FFT has the correct properties. If the
    initial Array was eager, then the final Array also must be eager and
    have _factors_applied=True. If the initial Array was not eager, then the
    final Array should have eager=False and _factors_applied=False.
    """

    log("arr._factors_applied == (arr**2)._factors_applied")
    arr_sq = arr * arr
    np.testing.assert_array_equal(arr_sq.eager, arr.eager)
    np.testing.assert_array_equal(arr_sq._factors_applied, arr._factors_applied)

    log("abs(x)._factors_applied == True")
    arr_abs = fa.abs(arr)
    np.testing.assert_array_equal(arr_abs.eager, arr.eager)
    np.testing.assert_array_equal(arr_abs._factors_applied, True)

    log("(x*abs(x))._factors_applied == x._factors_applied")
    # if both _factors_applied=True, the resulting Array will also have it
    # True, otherwise False
    # given abs(x)._factors_applied=True, we test the patterns
    # True*True=True, False*True=False
    arr_abs_sq = arr * arr_abs
    np.testing.assert_array_equal(arr_abs_sq.eager, arr.eager)
    np.testing.assert_array_equal(arr_abs_sq._factors_applied, arr._factors_applied)

    log("(x+abs(x))._factors_applied == (x._factors_applied or x._eager)")
    arr_abs_sum = arr + arr_abs
    np.testing.assert_array_equal(arr_abs_sum.eager, arr.eager)
    for ea, ifa, ffa in zip(arr_abs_sum.eager, arr._factors_applied, arr_abs_sum._factors_applied, strict=True):
        # True+True=True
        # False+True=eager
        assert (ifa == ffa) or (ffa == ea)

    log("fft(x)._factors_applied ...")
    # arr_fft = arr.into_space(get_other_space(arr.spaces))
    # np.testing.assert_array_equal(arr.eager, arr_fft.eager)
    # for ffapplied, feager in zip(arr_fft._factors_applied, arr_fft.eager):
    #     assert (feager and ffapplied) or (not feager and not ffapplied)

def assert_single_operand_fun_equivalence_int(arr: fa.Array, precise: bool, log):
    """Test whether applying operands to the Array (and then getting the
    values) is equivalent to applying the same operands to the values array:

        operand(Array).values() == operand(Array.values())

    """
    values = arr.values(arr.spaces)
    xp = array_api_compat.array_namespace(values)
    log("f(x) = x")
    assert_equal_op(arr, values, lambda x: x, precise, False, log)
    log("f(x) = abs(x)")
    assert_equal_op(arr, values, (xp.abs, fa.abs), precise, True, log)
    log("f(x) = x**2")
    assert_equal_op(arr, values, lambda x: x**2, precise, True, log)
    log("f(x) = x**3")
    assert_equal_op(arr, values, lambda x: x**3, precise, True, log)
    # log("f(x) = exp(x)")
    # assert_equal_op(arr, values, (xp.exp, fa.exp), False, True, log) # precise comparison fails
    # log("f(x) = sqrt(x)")
    # assert_equal_op(arr, values, (xp.sqrt, fa.sqrt), False, True, log) # precise comparison fails
