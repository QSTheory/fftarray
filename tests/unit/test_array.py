from typing import List, get_args
import itertools

import pytest
import numpy as np

import fftarray as fa
from fftarray.tests.helpers import XPS
from tests.helpers  import get_dims, dtypes_names_all

@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("init_dtype_name", dtypes_names_all)
@pytest.mark.parametrize("target_dtype_name", dtypes_names_all)
def test_astype(xp, init_dtype_name, target_dtype_name) -> None:
    dim = fa.dim("x", 4, 0.1, 0., 0.)
    arr1 = fa.array(
        xp.asarray([0, 1,2,3]),
        [dim],
        "pos",
        dtype=getattr(xp, init_dtype_name),
    )
    arr2 = arr1.into_dtype(getattr(xp, target_dtype_name))
    assert arr2.dtype == getattr(xp, target_dtype_name)



@pytest.mark.parametrize("xp", XPS)
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

    fa_res = arr.transpose(*[str(i) for i in permutation])
    np.testing.assert_equal(
        np.array(fa_res.values(space)),
        np.array(ref_res),
    )
