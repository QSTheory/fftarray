from typing import Any, List, Tuple, Literal, get_args

import array_api_strict
import pytest
import numpy as np

import fftarray as fa

from fftarray.tests.helpers import XPS
from tests.helpers  import assert_fa_array_exact_equal



# List all element-wise ops with the the data types which they support at minimum
# as mandated by the Pthon Array API Standard.
elementwise_ops_single_arg = {
    "abs": ("integral", "real floating", "complex floating"),
    "acos": ("real floating", "complex floating"),
    "acosh": ("real floating", "complex floating"),
    "asin": ("real floating", "complex floating"),
    "asinh": ("real floating", "complex floating"),
    "atan": ("real floating", "complex floating"),
    "atanh": ("real floating", "complex floating"),
    "bitwise_invert": ("bool", "integral"),
    "ceil": ("integral", "real floating"),
    "conj": ("complex floating"),
    "cos": ("real floating", "complex floating"),
    "cosh": ("real floating", "complex floating"),
    "exp": ("real floating", "complex floating"),
    "expm1": ("real floating", "complex floating"),
    "floor": ("integral", "real floating"),
    "imag": ("complex floating"),
    "isfinite": ("integral", "real floating", "complex floating"),
    "isinf": ("integral", "real floating", "complex floating"),
    "isnan": ("integral", "real floating", "complex floating"),
    "log": ("real floating", "complex floating"),
    "log1p": ("real floating", "complex floating"),
    "log2": ("real floating", "complex floating"),
    "log10": ("real floating", "complex floating"),
    "logical_not": ("bool"),
    "negative": ("integral", "real floating", "complex floating"),
    "positive": ("integral", "real floating", "complex floating"),
    "real": ("complex floating"),
    "round": ("integral", "real floating", "complex floating"),
    "sign": ("integral", "real floating", "complex floating"),
    "signbit": ("real floating"),
    "sin": ("real floating", "complex floating"),
    "sinh": ("real floating", "complex floating"),
    "sqrt": ("real floating", "complex floating"),
    "square": ("integral", "real floating", "complex floating"),
    "tan": ("real floating", "complex floating"),
    "tanh": ("real floating", "complex floating"),
    "trunc": ("integral", "real floating"),
}

# List all functions with a special path which get the
# full matrix of tests.
elementwise_ops_single_arg_full = [
    "abs",

    # representative for default path.
    "positive"
]

# The functions which get a more basic version of tests.
elementwise_ops_single_arg_sparse = [
    op_name for op_name in elementwise_ops_single_arg.keys() if op_name not in elementwise_ops_single_arg_full
]

# List all dunder methods with a single operand to be able to also test those.
single_operand_lambdas = {
    "abs": lambda x: abs(x),
    "positive": lambda x: +x,
    "negative": lambda x: -x,
    "bitwise_invert": lambda x: ~x,
}


@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("op_name", elementwise_ops_single_arg_full)
@pytest.mark.parametrize("eager", [True, False])
@pytest.mark.parametrize("space", get_args(fa.Space))
@pytest.mark.parametrize("factors_applied", [True, False])
def test_elementwise_single_arg_full_complex(
        xp,
        op_name: str,
        eager: bool,
        factors_applied: bool,
        space: fa.Space,
    ) -> None:
    """
        Doing all permutations of possible inputs for all functions creates unnecessarily many tests.
        Therefore this is only done for functions which have a special implementation path.
        Only complex values can be combined with ``factors_applied=False``
    """
    elementwise_single_arg(
        xp=xp,
        space=space,
        eager=eager,
        factors_applied=factors_applied,
        op_name=op_name,
        dtype_name="complex128",
    )


@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("op_name", elementwise_ops_single_arg_full)
@pytest.mark.parametrize("space", get_args(fa.Space))
@pytest.mark.parametrize("dtype_name", ["bool", "int64", "float64"])
def test_elementwise_single_arg_full(
        xp,
        op_name: str,
        space: fa.Space,
        dtype_name: str,
    ) -> None:
    """
        Doing all permutations of possible inputs for all functions creates unnecessarily many tests.
        Therefore this is only done for functions which have a special implementation path.
    """
    elementwise_single_arg(
        xp=xp,
        space=space,
        # Since factors_applied=True, it does not matter, which value eager has.
        eager=False,
        factors_applied=True,
        op_name=op_name,
        dtype_name=dtype_name,
    )

# List all element-wise ops with the the data types which they support at minimum
# for both operands as mandated by the Pthon Array API Standard.
elementwise_ops_double_arg = {
    "add": ("integral", "real floating", "complex floating"),
    "atan2": ("real floating"),
    "bitwise_and": ("bool", "integral"),
    "bitwise_left_shift": ("integral"),
    "bitwise_or": ("bool", "integral"),
    "bitwise_right_shift": ("integral"),
    "bitwise_xor": ("bool", "integral"),
    "copysign": ("real floating"),
    "divide": ("real floating", "complex floating"),
    "equal": ("bool", "integral", "real floating", "complex floating"),
    "floor_divide": ("integral", "real floating"),
    "greater": ("integral", "real floating"),
    "greater_equal": ("integral", "real floating"),
    "hypot": ("real floating"),
    "less": ("integral", "real floating"),
    "less_equal": ("integral", "real floating"),
    "logaddexp": ("real floating"),
    "logical_and": ("bool"),
    "logical_or": ("bool"),
    "logical_xor": ("bool"),
    "maximum": ("integral", "real floating"),
    "minimum": ("integral", "real floating"),
    "multiply": ("integral", "real floating", "complex floating"),
    "not_equal": ("bool", "integral", "real floating", "complex floating"),
    "pow": ("integral", "real floating", "complex floating"),
    "remainder": ("integral", "real floating"),
    "subtract": ("integral", "real floating", "complex floating"),
}

# These operations have special paths and therefore get more extensive tests.
elementwise_ops_double_arg_full = [
    "add",
    "divide",
    "floor_divide",
    "multiply",
    "pow",
    "subtract",

    # The representative for the default path
    "equal",
]

# These operations don't have a special path and therefore get more basic tests.
elementwise_ops_double_arg_sparse = [
    op_name for op_name in elementwise_ops_double_arg.keys() if op_name not in elementwise_ops_double_arg_full
]

# List all dunder methods with two operands to be able to also test those.
two_operand_lambdas = {
    "add": lambda x1, x2: x1+x2,
    "subtract": lambda x1, x2: x1-x2,
    "multiply": lambda x1, x2: x1*x2,
    "divide": lambda x1, x2: x1/x2,
    "floor_divide": lambda x1, x2: x1//x2,
    "remainder": lambda x1, x2: x1%x2,
    "pow": lambda x1, x2: x1**x2,
    "bitwise_and": lambda x1, x2: x1&x2,
    "bitwise_or": lambda x1, x2: x1|x2,
    "bitwise_xor": lambda x1, x2: x1^x2,
    "bitwise_left_shift": lambda x1, x2: x1<<x2,
    "bitwise_right_shift": lambda x1, x2: x1>>x2,
    "less": lambda x1, x2: x1<x2,
    "less_equal": lambda x1, x2: x1<=x2,
    "greater": lambda x1, x2: x1>x2,
    "greater_equal": lambda x1, x2: x1>=x2,
    "equal": lambda x1, x2: x1==x2,
    "not_equal": lambda x1, x2: x1!=x2,
}

# All two operarand dunder methods which have a special path
# and therefore get more extensive tests.
elementwise_ops_double_arg_full_dunder = [
    op_name for op_name in elementwise_ops_double_arg_full if op_name in two_operand_lambdas
]

# All two operarand dunder methods which don't have a special path
# and therefore get more basic tests.
elementwise_ops_double_arg_sparse_dunder = [
    op_name for op_name in elementwise_ops_double_arg_sparse if op_name in two_operand_lambdas
]

def get_two_operand_separate_dim_transform_factors(
        op_name: str,
        eager: bool,
        factors_applied_1: bool,
        factors_applied_2: bool,
    ) -> Tuple[bool, bool]:

    match op_name:
        case "add" | "subtract":
            if eager:
                return (True, True)
            else:
                return (factors_applied_1, factors_applied_2)

        case "multiply":
            return (factors_applied_1, factors_applied_2)

        case "divide":
            return (factors_applied_1, True)

        case _:
            return (True, True)


def get_two_operand_same_dim_transform_factors(
        op_name: str,
        eager: bool,
        factors_applied_1: bool,
        factors_applied_2: bool,
    ) -> Tuple[bool]:

    match op_name:
        case "add" | "subtract":
            if factors_applied_1 == factors_applied_2:
                return (factors_applied_1,)
            else:
                return (eager,)

        case "multiply":
            if factors_applied_1 and factors_applied_2:
                return (True,)
            else:
                return (False,)

        case "divide":
            return (factors_applied_1,)

        case _:
            return (True,)


@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("op_name", elementwise_ops_single_arg_sparse)
@pytest.mark.parametrize("dtype_name", ["bool", "int64", "float64", "complex128"])
def test_elementwise_single_arg_sparse(
        xp,
        op_name: str,
        dtype_name: str,
    ) -> None:
    """
        Limited permutations for all other functions.
    """
    elementwise_single_arg(
        xp=xp,
        space="pos",
        eager=True,
        factors_applied=True,
        op_name=op_name,
        dtype_name=dtype_name,
    )

def elementwise_single_arg(
        xp,
        op_name: str,
        eager: bool,
        factors_applied: bool,
        space: fa.Space,
        dtype_name: str,
    ) -> None:
    """
        Test all single operand functions and compare them with their direct
        "bare" counter-part of the underlying array API.
        This tests that there are no unnecessary dtype promotions and that all
        logic regarding lazy state works properly to give the same results
        (up to fp-accuracy) than just doing it directly on the correct values.
    """

    dtype = getattr(xp, dtype_name)
    x_dim = fa.dim("x", n=5, d_pos=0.1, pos_min=0, freq_min=0)

    arr1 = (
        fa.coords_from_dim(x_dim, space, xp=xp)
        .into_eager(eager=eager)
        .into_factors_applied(factors_applied)
    )

    if not xp.isdtype(dtype, "complex floating") and not factors_applied:
        with pytest.raises(ValueError):
                arr1.into_dtype(dtype)
        return

    arr1 = arr1.into_dtype(dtype)
    arr1_xp = arr1.values(space=space)

    if not xp.isdtype(dtype, elementwise_ops_single_arg[op_name]):
        # Other Array API implementations often allow more types.
        if xp == array_api_strict:
            with pytest.raises(TypeError):
                getattr(fa, op_name)(arr1)
        return

    fa_res = getattr(fa, op_name)(arr1)
    if op_name in single_operand_lambdas:
        op_lambda = single_operand_lambdas[op_name]
        fa_dunder_res =  op_lambda(arr1)
        assert_fa_array_exact_equal(fa_res, fa_dunder_res)

    xp_res = getattr(xp, op_name)(arr1_xp)
    assert fa_res.factors_applied == (True,)

    np.testing.assert_equal(
        np.array(fa_res._values),
        np.array(xp_res),
    )

    np.testing.assert_equal(
        np.array(fa_res.values(space=space)),
        np.array(xp_res),
    )

@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("space", get_args(fa.Space))
@pytest.mark.parametrize("eager", [True, False])
@pytest.mark.parametrize("factors_applied_1", [True, False])
@pytest.mark.parametrize("factors_applied_2", [True, False])
@pytest.mark.parametrize("op_name", elementwise_ops_double_arg_full)
def test_elementwise_two_arrs_full_complex(
        xp,
        space: fa.Space,
        eager: bool,
        factors_applied_1: bool,
        factors_applied_2: bool,
        op_name: str,
    ) -> None:
    elementwise_two_arrs(
        xp=xp,
        space=space,
        eager=eager,
        factors_applied_1=factors_applied_1,
        factors_applied_2=factors_applied_2,
        op_name=op_name,
        dtype_name="complex128",
    )

@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("space", get_args(fa.Space))
@pytest.mark.parametrize("op_name", elementwise_ops_double_arg_full)
@pytest.mark.parametrize("dtype_name", ["bool", "int64", "float64"])
def test_elementwise_two_arrs_full(
        xp,
        space: fa.Space,
        op_name: str,
        dtype_name: str,
    ) -> None:
    """
        Non-complex dtypes require factors_applied=True.
    """
    elementwise_two_arrs(
        xp=xp,
        space=space,
        # Since factors_applied=True, it does not matter, which value eager has.
        eager=False,
        factors_applied_1=True,
        factors_applied_2=True,
        op_name=op_name,
        dtype_name=dtype_name,
    )


@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("op_name", elementwise_ops_double_arg_sparse)
@pytest.mark.parametrize("dtype_name", ["bool", "int64", "float64", "complex128"])
def test_elementwise_two_arrs_sparse(
        xp,
        op_name: str,
        dtype_name: str,
    ) -> None:

    elementwise_two_arrs(
        xp=xp,
        space="pos",
        eager=True,
        factors_applied_1=True,
        factors_applied_2=True,
        op_name=op_name,
        dtype_name=dtype_name,
    )

def elementwise_two_arrs(
        xp,
        space: fa.Space,
        eager: bool,
        factors_applied_1: bool,
        factors_applied_2: bool,
        op_name: str,
        dtype_name: str,
    ) -> None:
    """
    Tests whether `factors_applied` is correctly handled in all two operand
    methods for a single dimension and for two different dimensions.
    This also tests the dunder-methods if they exist.
    """
    dtype = getattr(xp, dtype_name)
    is_op_valid_for_dtype = xp.isdtype(dtype, elementwise_ops_double_arg[op_name])

    x_dim = fa.dim("x", n=5, d_pos=0.1, pos_min=0, freq_min=0)
    y_dim = fa.dim("y", n=8, d_pos=0.1, pos_min=0, freq_min=0)

    x_arr = (
        fa.coords_from_dim(x_dim, space, xp=xp)
        .into_eager(eager=eager)
        .into_factors_applied(factors_applied_1)
        .into_dtype(dtype)

    )
    x2_arr = (
        fa.coords_from_dim(x_dim, space, xp=xp)
        .into_eager(eager=eager)
        .into_factors_applied(factors_applied_2)
        .into_dtype(dtype)
    )

    y_arr = (
        fa.coords_from_dim(y_dim, space, xp=xp)
        .into_eager(eager=eager)
        .into_factors_applied(factors_applied_2)
        .into_dtype(dtype)
    )

    if not is_op_valid_for_dtype:
        # Other Array API implementations often allow more types.
        if xp == array_api_strict:
            with pytest.raises(TypeError):
                getattr(fa, op_name)(x_arr, x2_arr)
        return

    ref_x2_values = getattr(xp, op_name)(x_arr.values(space), x2_arr.values(space))
    ref_xy_values = getattr(xp, op_name)(
        xp.reshape(x_arr.values(space), shape=(-1,1)),
        xp.reshape(y_arr.values(space), shape=(1,-1))
    )
    fa_x2 = getattr(fa, op_name)(x_arr, x2_arr)
    fa_xy = getattr(fa, op_name)(x_arr, y_arr)
    if op_name in two_operand_lambdas:
        op_lambda = two_operand_lambdas[op_name]
        fa_dunder_x2 =  op_lambda(x_arr, x2_arr)
        fa_dunder_xy = op_lambda(x_arr, y_arr)
        assert_fa_array_exact_equal(fa_x2, fa_dunder_x2)
        assert_fa_array_exact_equal(fa_xy, fa_dunder_xy)


    x2_factors = get_two_operand_same_dim_transform_factors(
        op_name=op_name,
        eager=eager,
        factors_applied_1=factors_applied_1,
        factors_applied_2=factors_applied_2,
    )
    xy_factors = get_two_operand_separate_dim_transform_factors(
        op_name=op_name,
        eager=eager,
        factors_applied_1=factors_applied_1,
        factors_applied_2=factors_applied_2,
    )
    assert fa_x2.factors_applied == x2_factors
    assert fa_xy.factors_applied == xy_factors

    np.testing.assert_allclose(
        fa_x2.values(space),
        ref_x2_values
    )
    np.testing.assert_allclose(
        fa_xy.values(space),
        ref_xy_values,
    )

# This limit the number of combinations where an upcasting
# is in principle possible according the array API standard.
dtype_scalar_combinations = [
    pytest.param("bool", True),
    pytest.param("uint32", 5),
    pytest.param("uint64", 5),
    pytest.param("int32", 5),
    pytest.param("int64", 5),
    pytest.param("float32", 5),
    pytest.param("float32", 5.),
    pytest.param("float64", 5),
    pytest.param("float64", 5.),
]

dtype_scalar_combinations_complex = [
    pytest.param("complex64", 5),
    pytest.param("complex64", 5.),
    pytest.param("complex64", 5.+1.j),
    pytest.param("complex128", 5),
    pytest.param("complex128", 5.),
    pytest.param("complex128", 5.+1.j),
]

@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("space", get_args(fa.Space))
@pytest.mark.parametrize("op_idxs", [(0,1), (1,0)])
@pytest.mark.parametrize("op_name", elementwise_ops_double_arg_full_dunder)
@pytest.mark.parametrize("dtype_name, scalar_value", dtype_scalar_combinations)
def test_elementwise_arr_scalar_full(
        xp,
        space: fa.Space,
        scalar_value,
        op_idxs: Tuple[Literal[0,1], Literal[0,1]],
        op_name: str,
        dtype_name: str,
    ) -> None:

    elementwise_arr_scalar(
        xp=xp,
        space=space,
        # Since factors_applied=True, it does not matter, which value eager has.
        eager=False,
        factors_applied=True,
        scalar_value=scalar_value,
        op_idxs=op_idxs,
        op_name=op_name,
        dtype_name=dtype_name,
    )

@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("space", get_args(fa.Space))
@pytest.mark.parametrize("eager", [True, False])
@pytest.mark.parametrize("factors_applied", [True, False])
@pytest.mark.parametrize("op_idxs", [(0,1), (1,0)])
@pytest.mark.parametrize("op_name", elementwise_ops_double_arg_full_dunder)
@pytest.mark.parametrize("dtype_name, scalar_value", dtype_scalar_combinations_complex)
def test_elementwise_arr_scalar_full_complex(
        xp,
        space: fa.Space,
        eager: bool,
        factors_applied: bool,
        scalar_value,
        op_idxs: Tuple[Literal[0,1], Literal[0,1]],
        op_name: str,
        dtype_name: str,
    ) -> None:

    elementwise_arr_scalar(
        xp=xp,
        space=space,
        eager=eager,
        factors_applied=factors_applied,
        scalar_value=scalar_value,
        op_idxs=op_idxs,
        op_name=op_name,
        dtype_name=dtype_name,
    )

@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("op_idxs", [(0,1), (1,0)])
@pytest.mark.parametrize("op_name", elementwise_ops_double_arg_sparse_dunder)
@pytest.mark.parametrize("dtype_name, scalar_value", [
    *dtype_scalar_combinations,
    *dtype_scalar_combinations_complex,
])
def test_elementwise_arr_scalar_sparse(
        xp,
        scalar_value,
        op_idxs: Tuple[Literal[0,1], Literal[0,1]],
        op_name: str,
        dtype_name: str,
    ) -> None:

    elementwise_arr_scalar(
        xp=xp,
        space="pos",
        # Since factors_applied=True, it does not matter, which value eager has.
        eager=False,
        factors_applied=True,
        scalar_value=scalar_value,
        op_idxs=op_idxs,
        op_name=op_name,
        dtype_name=dtype_name,
    )

def elementwise_arr_scalar(
        xp,
        space: fa.Space,
        eager: bool,
        factors_applied: bool,
        scalar_value,
        op_idxs: Tuple[Literal[0,1], Literal[0,1]],
        op_name: str,
        dtype_name: str,
    ) -> None:
    """
    Tests whether `factors_applied` is correctly handled in dunder-methods
    with a Scalar (both from the left and the right).
    Because the Array API standard currently does not allow scalars in
    the free standing methods this only tests dunder methods.
    """
    dtype = getattr(xp, dtype_name)
    is_op_valid_for_dtype = xp.isdtype(dtype, elementwise_ops_double_arg[op_name])

    x_dim = fa.dim("x", n=5, d_pos=0.1, pos_min=0, freq_min=0)

    x_arr = (
        fa.coords_from_dim(x_dim, space, xp=xp)
        .into_eager(eager=eager)
        .into_factors_applied(factors_applied)
        .into_dtype(dtype)
    )

    dtype = getattr(xp, dtype_name)

    op_lambda = two_operand_lambdas[op_name]

    input_raw_arr: List[Any] = [None, None]
    input_raw_arr[op_idxs[0]] = x_arr.values(space)
    input_raw_arr[op_idxs[1]] = scalar_value

    input_fa_arr: List[Any] = [None, None]
    input_fa_arr[op_idxs[0]] = x_arr
    input_fa_arr[op_idxs[1]] = scalar_value

    if not is_op_valid_for_dtype:
        # Other Array API implementations often allow more types.
        if xp == array_api_strict:
            with pytest.raises(TypeError):
                op_lambda(*input_fa_arr)
        return

    ref_x2_values = op_lambda(*input_raw_arr)
    fa_x2 = op_lambda(*input_fa_arr)

    if op_idxs[0] == 0:
        factors_applied_1 = factors_applied
        factors_applied_2 = True
    else:
        factors_applied_1 = True
        factors_applied_2 = factors_applied

    x2_factors = get_two_operand_same_dim_transform_factors(
        op_name=op_name,
        eager=eager,
        factors_applied_1=factors_applied_1,
        factors_applied_2=factors_applied_2,
    )
    assert fa_x2.factors_applied == x2_factors

    np.testing.assert_allclose(
        np.array(fa_x2.values(space=space)),
        np.array(ref_x2_values),
    )


@pytest.mark.parametrize("xp", XPS)
def test_clip(xp) -> None:
    dim1 = fa.dim("x", 4, 0.1, 0., 0.)
    vals = xp.asarray([1,2,3,4])
    arr1 = fa.array(vals, dims=[dim1], space="pos")
    assert xp.all(fa.clip(arr1, min=2, max=3).values("pos") == xp.clip(vals, min=2, max=3))
    assert xp.all(fa.clip(arr1, min=None, max=3).values("pos") == xp.clip(vals, min=None, max=3))
    assert xp.all(fa.clip(arr1).values("pos") == xp.clip(vals))

