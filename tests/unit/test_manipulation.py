from typing import Any, Tuple
from itertools import permutations

import array_api_strict
import pytest
import numpy as np

import fftarray as fa
from tests.helpers import get_dims, get_arr_from_dims

@pytest.mark.parametrize("xp", [array_api_strict])
@pytest.mark.parametrize("ndims, permuted_axes",
    [pytest.param(nd, perm) for nd in [0,1,2,3] for perm in permutations(range(nd))]
)
def test_permute_dims(xp: Any, ndims: int, permuted_axes: Tuple[int]) -> None:
    """Tests fa.permute_dims. Creates an Array with 0,1,2,3 Dimensions and
    tests all possible permutations, e.g., with 2 Dimensions it tests to permute
    the axes from [0,1] to [0,1], [1,0].
    """
    permuted_dim_names = tuple(str(p) for p in permuted_axes)
    dims = get_dims(ndims)
    arr_before = get_arr_from_dims(xp, dims, "pos")
    # give eager and factors_applied different values for each dimension
    eager_before = [i%2!=0 for i in range(ndims)]
    factors_applied_before = [i%2==0 for i in range(ndims)]
    arr_before = arr_before.into_factors_applied(factors_applied_before)
    arr_before = arr_before.into_eager(eager_before)
    dim_names_before = tuple(dim.name for dim in arr_before.dims)
    values_before = arr_before.values("pos")

    # perform equivalent operation using xp
    values_ref = xp.permute_dims(values_before, permuted_axes)

    # fftarray operation
    arr_permuted = fa.permute_dims(arr_before, permuted_dim_names)
    values_permuted = arr_permuted.values("pos")
    dim_names_after = tuple(dim.name for dim in arr_permuted.dims)

    # test that everything is permuted correctly, checks might be redundant
    assert permuted_dim_names == dim_names_after
    assert dim_names_after == tuple(dim_names_before[a] for a in permuted_axes)
    assert arr_permuted.xp == arr_before.xp
    assert arr_permuted.spaces == arr_before.spaces
    assert arr_permuted.eager == tuple(eager_before[a] for a in permuted_axes)
    assert arr_permuted.factors_applied == tuple(factors_applied_before[a] for a in permuted_axes)
    assert values_ref.shape == values_permuted.shape
    assert arr_permuted.shape == tuple(arr_before.shape[a] for a in permuted_axes)
    np.testing.assert_equal(
        np.array(values_permuted),
        np.array(values_ref),
    )
