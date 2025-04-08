from typing import List, get_args
import itertools

import array_api_strict
import pytest
import numpy as np

import fftarray as fa
from tests.helpers import XPS, get_dims

@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize(("init_dtype_name, target_dtype_name"), [
    pytest.param("complex64", "complex128"),
    pytest.param("int64", "float32"),
    pytest.param("int64", "bool"),
    pytest.param("int32", "int64"),
    pytest.param("bool", "int64"),
    pytest.param("bool", "float32"),
])
def test_into_dtype(xp, init_dtype_name, target_dtype_name) -> None:
    dim = fa.dim("x", 4, 0.1, 0., 0.)
    arr1 = fa.array(
        xp.asarray([0, 1,2,3]),
        [dim],
        "pos",
        dtype=getattr(xp, init_dtype_name),
    )

    target_dtype = getattr(xp, target_dtype_name)
    arr2 = arr1.into_dtype(target_dtype)
    assert arr2.dtype == target_dtype



@pytest.mark.parametrize("xp", [array_api_strict])
@pytest.mark.parametrize("ndims, permutation",
    [
        pytest.param(ndims, permutation)
        for ndims in [0,1,2]
        for permutation in itertools.permutations(range(ndims))
    ]
)
@pytest.mark.parametrize("space", get_args(fa.Space))
def test_transpose(xp, ndims: int, permutation: List[int], space: fa.Space) -> None:
    dims = get_dims(ndims)
    shape = tuple(dim.n for dim in dims)
    size = int(xp.prod(xp.asarray(shape)))
    input_values = xp.reshape(xp.arange(size), shape=shape)

    arr = fa.array(input_values, dims, space)

    ref_res = xp.permute_dims(input_values, axes=tuple(permutation))

    fa_res = fa.permute_dims(arr, tuple(str(i) for i in permutation))
    np.testing.assert_equal(
        np.array(fa_res.values(space)),
        np.array(ref_res),
    )
