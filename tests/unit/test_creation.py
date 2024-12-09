from typing import Iterable

import numpy as np
import pytest
import fftarray as fa

from fftarray.fft_dimension import FFTDimension
from fftarray.tests.helpers import XPS
from tests.helpers  import get_dims, dtypes_names_all, DTYPE_NAME


@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("dtype_name", dtypes_names_all)
@pytest.mark.parametrize("ndims", [0,1,2])
@pytest.mark.parametrize("copy", [False, True])
def test_array(xp, dtype_name: DTYPE_NAME, ndims: int, copy: bool) -> None:
    dims = get_dims(ndims)
    shape = tuple(dim.n for dim in dims)
    dtype = getattr(xp, dtype_name)
    values = xp.full(shape, 1., dtype=dtype)
    values_ref = xp.asarray(values, copy=True)

    arr = fa.array(
        values=values,
        dims=dims,
        space="pos",
        copy=copy,
    )
    try:
        # For array libraries with immutable arrays (e.g. jax), we assume this fails.
        # In these cases, we skip testing immutability ourself.
        values += 2
    except(TypeError):
        pass

    assert arr.xp == xp
    assert arr.dtype == dtype
    assert arr.shape == shape
    if copy:
        assert xp.all(arr.values(space="pos") == values_ref)
    # If not copy, we cannot test for inequality because aliasing behavior
    # is not defined and for jax for example an inequality check would fail.

    if ndims > 0:
        wrong_shape = list(shape)
        wrong_shape[0] = 10
        values = xp.full(tuple(wrong_shape), 1., dtype=dtype)
        with pytest.raises(ValueError):
            arr = fa.array(
                values=values,
                dims=dims,
                space="pos",
            )

@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("copy", [False, True])
def test_array_from_list(xp, copy: bool) -> None:
    with fa.default_xp(xp):
        x_dim = fa.dim("x", n=3, d_pos=0.1, pos_min=0, freq_min=0)
        y_dim = fa.dim("y", n=2, d_pos=0.1, pos_min=0, freq_min=0)


        check_array_from_list(
            xp=xp,
            dims=[x_dim],
            vals_list = [1,2,3],
            copy=copy,
        )
        check_array_from_list(
            xp=xp,
            dims=[x_dim, y_dim],
            vals_list = [[1,4],[2,5],[3,6]],
            copy=copy,
        )

        # Test that inhomogeneous list triggers the correct error.
        with pytest.raises(ValueError):
            fa.array(
                values=[1,[2]],
                dims=[x_dim],
                space="pos",
                copy=copy,
            )

def check_array_from_list(xp, dims: Iterable[FFTDimension], vals_list, copy: bool):
    ref_vals = xp.asarray(vals_list)

    arr = fa.array(
        values=vals_list,
        dims=dims,
        space="pos",
        copy=copy,
    )
    arr_vals = arr.values(space="pos")

    assert arr.xp == xp
    assert arr.shape == ref_vals.shape
    assert arr.dtype == ref_vals.dtype
    assert type(arr_vals) is type(ref_vals)
    np.testing.assert_equal(
        np.array(arr_vals),
        np.array(vals_list),
    )
