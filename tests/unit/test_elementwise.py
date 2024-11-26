
import array_api_strict
import pytest
import fftarray as fa


from fftarray.tests.helpers import XPS
from tests.helpers  import get_dims, get_arr_from_dims


# List all element-wise ops with the the data types which they have to support.
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


@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("op_name", elementwise_ops_single_arg.keys())
@pytest.mark.parametrize("dtype_name", ["bool", "int64", "float64", "complex128"])
def test_elementwise_single_arg(xp, op_name: str, dtype_name: str) -> None:
    dtype = getattr(xp, dtype_name)

    dims = get_dims(1)
    arr1 = get_arr_from_dims(xp=xp, dims=dims).astype(dtype)
    arr1_xp = arr1.values(space="pos")

    if xp.isdtype(dtype, elementwise_ops_single_arg[op_name]):
        fa_res = getattr(fa, op_name)(arr1).values(space="pos")
        xp_res = getattr(xp, op_name)(arr1_xp)
        res = fa_res == xp_res
        assert fa_res.dtype == xp_res.dtype
        # We also want to have 'nan' count as equal
        if xp.isdtype(fa_res.dtype, ("real floating", "complex floating")):
            res = xp.logical_or(res, xp.isnan(fa_res) == xp.isnan(xp_res))
        assert xp.all(res)
    # Other Array API implementations often allow more types.
    elif xp == array_api_strict:
        with pytest.raises(TypeError):
            getattr(fa, op_name)(arr1)



@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("op_name", elementwise_ops_double_arg.keys())
@pytest.mark.parametrize("dtype_name", ["bool", "int64", "float64", "complex128"])
def test_elementwise_double_arg(xp, op_name: str, dtype_name: str) -> None:
    dtype = getattr(xp, dtype_name)

    dims = get_dims(2)
    arr1 = get_arr_from_dims(xp=xp, dims=dims[:1]).astype(dtype)
    arr2 = get_arr_from_dims(xp=xp, dims=dims).astype(dtype)
    arr1_xp = xp.reshape(arr1.values(space="pos"), (-1,1))
    arr2_xp = arr2.values(space="pos")

    if xp.isdtype(dtype, elementwise_ops_double_arg[op_name]):
        fa_res = getattr(fa, op_name)(arr1, arr2).values(space="pos")
        xp_res = getattr(xp, op_name)(arr1_xp, arr2_xp)
        assert xp.all(fa_res == xp_res)
    # Other Array API implementations often allow more types.
    elif xp == array_api_strict:
        with pytest.raises(TypeError):
            getattr(fa, op_name)(arr1, arr2)


def test_dunder() -> None:
    pass


@pytest.mark.parametrize("xp", XPS)
def test_clip(xp) -> None:
    dim1 = fa.dim("x", 4, 0.1, 0., 0.)
    vals = xp.asarray([1,2,3,4])
    arr1 = fa.array(vals, dims=[dim1], space="pos")
    assert xp.all(fa.clip(arr1, min=2, max=3).values("pos") == xp.clip(vals, min=2, max=3))
    assert xp.all(fa.clip(arr1, min=None, max=3).values("pos") == xp.clip(vals, min=None, max=3))
    assert xp.all(fa.clip(arr1).values("pos") == xp.clip(vals))

