from typing import List, Literal

import array_api_compat
import pytest
from hypothesis import given, strategies as st, note, settings
import numpy as np

import fftarray as fa
from fftarray.array import Array, Space

from fftarray.tests.helpers import XPS
from fftarray.tests.test_fft_array import draw_hypothesis_fft_array_values, assert_equal_op, assert_basic_lazy_logic, jnp


PrecisionSpec = Literal["float32", "float64"]

def assert_scalars_almost_equal_nulp(x, y, nulp = 1):
    np.testing.assert_array_almost_equal_nulp(np.array([x]), np.array([y]), nulp = nulp)

precisions: List[PrecisionSpec] = ["float32", "float64"]
spaces: List[Space] = ["pos", "freq"]


@st.composite
def fftarray_strategy_int(draw) -> Array:
    """Initializes an Array using hypothesis."""
    ndims = draw(st.integers(min_value=1, max_value=4))
    value = st.integers(min_value=np.iinfo(np.int32).min, max_value=np.iinfo(np.int32).max)
    eager = draw(st.lists(st.booleans(), min_size=ndims, max_size=ndims))
    note(f"eager={eager}") # TODO: remove when Array.__repr__ is implemented
    init_space = draw(st.sampled_from(["pos", "freq"]))
    note(f"space={init_space}") # TODO: remove when Array.__repr__ is implemented
    # xp = draw(st.sampled_from(XPS))
    xp = XPS[0] # this is array_api_strict
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

    return fa.array(fftarr_values, dims, init_space).into_eager(eager)


@pytest.mark.slow
@settings(max_examples=1000, deadline=None, print_blob=True)
@given(fftarray_strategy_int())
def test_fftarray_lazyness_int(fftarr):
    """Tests the lazyness of an Array, i.e., the correct behavior of
    factors_applied and eager.
    """
    note(fftarr)
    # -- basic tests
    assert_basic_lazy_logic(fftarr, note)
    # -- test operands
    assert_single_operand_fun_equivalence_int(fftarr, all(fftarr._factors_applied), note)
    assert_dual_operand_fun_equivalence_int(fftarr, all(fftarr._factors_applied), note)
    # Jax only supports FFT for dim<4
    # TODO: Block this off more generally? Array API does not seem to define
    # an upper limit to the number of dimensions (nor do the JAX docs for that matter).
    if len(fftarr.dims) < 4 or not fftarr.xp==jnp:
        # -- test eager, factors_applied logic
        assert_fftarray_eager_factors_applied_int(fftarr, note)

def assert_dual_operand_fun_equivalence_int(arr: Array, precise: bool, log):
    """Test whether a dual operation on an Array, e.g., the
    sum/multiplication of two, is equivalent to applying this operand to its
    values.

        operand(Array, Array).values() = operand(Array.values(), Array.values())

    """
    values = arr.values(arr.space)
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

def assert_fftarray_eager_factors_applied_int(arr: Array, log):
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
    # arr_fft = arr.into_space(get_other_space(arr.space))
    # np.testing.assert_array_equal(arr.eager, arr_fft.eager)
    # for ffapplied, feager in zip(arr_fft._factors_applied, arr_fft.eager):
    #     assert (feager and ffapplied) or (not feager and not ffapplied)

def assert_single_operand_fun_equivalence_int(arr: Array, precise: bool, log):
    """Test whether applying operands to the Array (and then getting the
    values) is equivalent to applying the same operands to the values array:

        operand(Array).values() == operand(Array.values())

    """
    values = arr.values(arr.space)
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