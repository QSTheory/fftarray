from functools import reduce
from typing import List, Optional, Any, Union, Tuple, get_args, Iterable

import array_api_strict
import numpy as np
import pytest
import fftarray as fa
from fftarray._src.helpers import norm_space

from tests.helpers import XPS, get_dims, get_arr_from_dims, dtype_names_numeric_core, DTYPE_NAME

function_dtypes = {
    "max": ("integral", "real floating"),
    "min": ("integral", "real floating"),
    "mean": ("real floating"),
}

reduction_dims_combinations = [
    pytest.param(0, tuple([])),
    pytest.param(0, None),
    pytest.param(1, tuple([])),
    pytest.param(1, None),
    pytest.param(1, tuple([0])),
    pytest.param(1, 0),
    pytest.param(2, tuple([])),
    pytest.param(2, None),
    pytest.param(2, 1),
    pytest.param(2, tuple([0])),
    pytest.param(2, tuple([1,0])),
]

source_and_res_dtype_name_pairs = [
    pytest.param("float32", "float64", None),
    pytest.param("float32", None, None),
    pytest.param("float64", "float32", None),
    pytest.param("bool", "float32", TypeError),
    pytest.param("complex128", "complex64", None),
]

def _get_dim_names(indices: Optional[Union[int, Tuple[int]]]) -> Optional[Union[str, List[str]]]:
    if indices is None:
        return None

    if isinstance(indices, int):
        return str(indices)

    return [str(i) for i in indices]

def _to_integers(indices: Optional[Union[int, Tuple[int]]], ndims: int) -> Tuple[int, ...]:
    if indices is None:
        return tuple(range(ndims))

    if isinstance(indices, int):
        return (indices,)

    return tuple(indices)

@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("op_name", ["sum", "prod"])
@pytest.mark.parametrize(
    "source_dtype_name, acc_dtype_name, should_raise",
    source_and_res_dtype_name_pairs
)
@pytest.mark.parametrize("ndims, reduction_dims", reduction_dims_combinations)
@pytest.mark.parametrize("attribute_inversion", [False, True])
def test_sum_prod(
        xp,
        op_name,
        source_dtype_name: DTYPE_NAME,
        acc_dtype_name: Optional[DTYPE_NAME],
        should_raise: Optional[Any],
        ndims: int,
        reduction_dims: Optional[Union[int, Tuple[int]]],
        attribute_inversion: bool,
    ) -> None:
    source_dtype = getattr(xp, source_dtype_name)
    if acc_dtype_name is None:
        acc_dtype = None
    else:
        acc_dtype = getattr(xp, acc_dtype_name)
    dims = get_dims(ndims)
    arr = get_arr_from_dims(xp, dims, spaces="pos").into_dtype(source_dtype)

    # Just set alternating attributes in order to save on iterations.
    arr_attributes = [((i%2)==0)^attribute_inversion for i in range(ndims)]
    arr = arr.into_eager(arr_attributes)
    if xp.isdtype(source_dtype, "complex floating"):
        arr = arr.into_factors_applied(arr_attributes)
    arr = fa.permute_dims(arr, tuple(dim.name for dim in dims))

    if should_raise is not None:
        if xp == array_api_strict:
            with pytest.raises(should_raise):
                getattr(fa, op_name)(arr, dim_name=_get_dim_names(reduction_dims), dtype=acc_dtype)
        return

    fa_res = getattr(fa, op_name)(arr, dim_name=_get_dim_names(reduction_dims), dtype=acc_dtype)
    fa_res_values = fa_res.values("pos")

    ref_input = arr.values("pos")
    ref_res = getattr(xp, op_name)(ref_input, axis=reduction_dims, dtype=acc_dtype)

    ref_attributes = tuple(
        x for (i, x) in enumerate(arr_attributes)
        if i not in _to_integers(indices=reduction_dims, ndims=ndims)
    )
    assert fa_res.eager == ref_attributes
    # This will be changed at a later point to actually be conserved.
    assert fa_res.factors_applied == (True,)*len(ref_attributes)

    if len(fa_res_values.shape) > 0:
        assert type(ref_res) is type(fa_res_values)

    np.testing.assert_equal(
        np.array(ref_res),
        np.array(fa_res_values),
    )

@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("op_name", ["max", "min", "mean"])
@pytest.mark.parametrize("dtype_name", dtype_names_numeric_core)
@pytest.mark.parametrize("ndims, reduction_dims", reduction_dims_combinations)
@pytest.mark.parametrize("attribute_inversion", [False, True])
def test_max_min_mean(
        xp,
        op_name: str,
        dtype_name: DTYPE_NAME,
        ndims: int,
        reduction_dims: Optional[Union[int, Tuple[int]]],
        attribute_inversion: bool,
    ) -> None:
    dtype = getattr(xp, dtype_name)
    dims = get_dims(ndims)
    arr = get_arr_from_dims(xp, dims, spaces="pos").into_dtype(dtype)
    # Just set alternating attributes in order to save on iterations.
    arr_attributes = [((i%2)==0)^attribute_inversion for i in range(ndims)]
    arr = arr.into_eager(arr_attributes)
    if xp.isdtype(dtype, "complex floating"):
        arr = arr.into_factors_applied(arr_attributes)
    arr = fa.permute_dims(arr, tuple(dim.name for dim in dims))

    if not xp.isdtype(dtype, function_dtypes[op_name]):
        if xp == array_api_strict:
            with pytest.raises(TypeError):
                getattr(fa, op_name)(arr, dim=_get_dim_names(reduction_dims))
        return

    fa_res = getattr(fa, op_name)(arr, dim_name=_get_dim_names(reduction_dims))
    fa_res_values = fa_res.values("pos")

    ref_input = arr.values("pos")
    ref_res = getattr(xp, op_name)(ref_input, axis=reduction_dims)

    ref_attributes = tuple(
        x for (i, x) in enumerate(arr_attributes)
        if i not in _to_integers(indices=reduction_dims, ndims=ndims)
    )
    assert fa_res.eager == ref_attributes
    # This will be changed at a later point to actually be conserved.
    assert fa_res.factors_applied == (True,)*len(ref_attributes)

    if len(fa_res_values.shape) > 0:
        assert type(ref_res) is type(fa_res_values)

    np.testing.assert_equal(
        np.array(ref_res),
        np.array(fa_res_values),
    )

@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize(
    "source_dtype_name, acc_dtype_name, should_raise",
    source_and_res_dtype_name_pairs
)
@pytest.mark.parametrize("space", get_args(fa.Space))
@pytest.mark.parametrize("ndims, reduction_dims", reduction_dims_combinations)
@pytest.mark.parametrize("attribute_inversion", [False, True])
def test_integrate_base(
        xp,
        source_dtype_name: DTYPE_NAME,
        acc_dtype_name: Optional[DTYPE_NAME],
        should_raise: Optional[Any],
        space: Union[fa.Space, Iterable[fa.Space]],
        ndims: int,
        reduction_dims: Optional[Union[int, Tuple[int]]],
        attribute_inversion: bool,
    ) -> None:

    check_integrate(
        xp=xp,
        source_dtype_name=source_dtype_name,
        acc_dtype_name=acc_dtype_name,
        should_raise=should_raise,
        space=space,
        ndims=ndims,
        reduction_dims=reduction_dims,
        attribute_inversion=attribute_inversion,
    )

@pytest.mark.parametrize(
    "source_dtype_name, acc_dtype_name, should_raise",
    source_and_res_dtype_name_pairs
)
@pytest.mark.parametrize("ndims, reduction_dims, space",
    [
        pytest.param(2, tuple([0]), ("pos", "freq")),
        pytest.param(2, tuple([1,0]), ("pos", "freq")),
        pytest.param(3, tuple([0,2]), ("freq", "freq", "pos")),
    ]
)
@pytest.mark.parametrize("attribute_inversion", [False, True])
def test_integrate_mixed_space(
        source_dtype_name: DTYPE_NAME,
        acc_dtype_name: Optional[DTYPE_NAME],
        should_raise: Optional[Any],
        space: Union[fa.Space, Iterable[fa.Space]],
        ndims: int,
        reduction_dims: Optional[Union[int, Tuple[int]]],
        attribute_inversion: bool,
    ) -> None:

    check_integrate(
        xp=array_api_strict,
        source_dtype_name=source_dtype_name,
        acc_dtype_name=acc_dtype_name,
        should_raise=should_raise,
        space=space,
        ndims=ndims,
        reduction_dims=reduction_dims,
        attribute_inversion=attribute_inversion,
    )


def check_integrate(
        xp,
        source_dtype_name: DTYPE_NAME,
        acc_dtype_name: Optional[DTYPE_NAME],
        should_raise: Optional[Any],
        space: Union[fa.Space, Iterable[fa.Space]],
        ndims: int,
        reduction_dims: Optional[Union[int, Tuple[int]]],
        attribute_inversion: bool,
    ) -> None:

    space_norm = norm_space(val=space, n=ndims)
    source_dtype = getattr(xp, source_dtype_name)
    if acc_dtype_name is None:
        acc_dtype = None
    else:
        acc_dtype = getattr(xp, acc_dtype_name)
    dims = get_dims(ndims)
    arr = get_arr_from_dims(xp, dims, spaces=space_norm).into_dtype(source_dtype)
    # Just set alternating attributes in order to save on iterations.
    arr_attributes = [((i%2)==0)^attribute_inversion for i in range(ndims)]
    arr = arr.into_eager(arr_attributes)
    if xp.isdtype(source_dtype, "complex floating"):
        arr = arr.into_factors_applied(arr_attributes)
    arr = fa.permute_dims(arr, tuple(dim.name for dim in dims))

    if should_raise is not None:
        if xp == array_api_strict:
            with pytest.raises(should_raise):
                fa.integrate(arr, dim_name=_get_dim_names(reduction_dims), dtype=acc_dtype)
        return

    fa_res = fa.integrate(arr, dim_name=_get_dim_names(reduction_dims), dtype=acc_dtype)
    res_space = tuple(
        x for (i, x) in enumerate(space_norm)
        if i not in _to_integers(indices=reduction_dims, ndims=ndims)
    )
    fa_res_values = fa_res.values(res_space)

    ref_input = arr.values(space)
    ref_res = xp.sum(ref_input, axis=reduction_dims, dtype=acc_dtype)
    if reduction_dims is None:
        integration_element = reduce(
            lambda a,b: a*b,
            [getattr(dim, f"d_{dim_space}") for dim, dim_space in zip(dims, space_norm, strict=True)],
            1.
        )
    elif isinstance(reduction_dims, int):
        integration_element = getattr(dims[reduction_dims], f"d_{space_norm[reduction_dims]}")
    else:
        integration_element = reduce(
            lambda a,b: a*b,
            [
                getattr(dim, f"d_{space_norm[dim_idx]}")
                for dim_idx, dim in enumerate(dims)
                if dim_idx in reduction_dims
            ],
            1.,
        )
    ref_res = ref_res * xp.asarray(integration_element, dtype=ref_res.dtype)

    ref_attributes = tuple(
        x for (i, x) in enumerate(arr_attributes)
        if i not in _to_integers(indices=reduction_dims, ndims=ndims)
    )
    assert fa_res.spaces == res_space
    assert fa_res.eager == ref_attributes
    # This will be changed at a later point to actually be conserved.
    assert fa_res.factors_applied == (True,)*len(ref_attributes)

    if len(fa_res_values.shape) > 0:
        assert type(ref_res) is type(fa_res_values)
        np.testing.assert_equal(
            np.array(ref_res),
            np.array(fa_res_values),
        )
    else:
        # fa_res_values is a python scalar and does not have a dtype, so the
        # test failes in numpy 1.25 without explicitly setting the dtype
        np.testing.assert_equal(
            np.array(ref_res),
            np.array(fa_res_values, dtype=type(ref_res)),
        )
