from typing import get_args

import pytest
import numpy as np

import fftarray as fa

from tests.helpers import XPS, get_other_space
from fftarray._src.transform_application import complex_type

@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("eager", [True, False])
@pytest.mark.parametrize("space", get_args(fa.Space))
@pytest.mark.parametrize("init_dtype_name, atol", [
    pytest.param("float32", 1e-4),
    pytest.param("float64", 1e-14),
    pytest.param("complex64", 1e-4),
    pytest.param("complex128", 1e-14),
])
def test_shift(
        xp,
        eager: bool,
        space: fa.Space,
        init_dtype_name: str,
        atol: float,
    ) -> None:

    init_dtype = getattr(xp, init_dtype_name)
    other_space= get_other_space(space)
    # It is important to place both spaces symmetrically around zero to prevent aliasing of the test function in
    # both spaces.
    dim = fa.dim_from_constraints(name="x", n=128, d_pos=0.01, pos_middle=0., freq_middle=0.)
    arr = fa.coords_from_dim(
        dim, space, xp=xp, dtype=init_dtype,
    ).into_eager(eager)

    # Use a frequency which fits exactly into the domain to allow periodic shifts
    test_frequency = 5*2*np.pi*getattr(dim, f"d_{other_space}")
    orig = fa.sin(test_frequency*arr)
    shift_amount = 8.1*getattr(dim, f"d_{space}")

    ref_shifted = fa.sin(test_frequency*(arr-shift_amount)).into_dtype("complex")

    shift_fun = getattr(fa, f"shift_{space}")
    fft_shifted = shift_fun(orig, {"x": shift_amount})

    assert fft_shifted.eager == orig.eager
    assert fft_shifted.dtype == complex_type(xp, init_dtype)
    assert fft_shifted.xp == orig.xp
    assert fft_shifted.dims == orig.dims

    np.testing.assert_allclose(
        np.array(ref_shifted.values(space)),
        np.array(fft_shifted.values(space)),
        atol=atol,
    )

@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("init_dtype_name", ["bool", "int64", "uint64"])
@pytest.mark.parametrize("space", get_args(fa.Space))
def test_shift_int(
        xp,
        init_dtype_name: str,
        space: fa.Space,
    ) -> None:
    """
        Simply tests that the functions properly raise a ``ValueError``.
    """
    dim = fa.dim(name="x", n=4, d_pos=0.1, pos_min=0., freq_min=0.)
    arr = fa.array(
        xp.asarray([0, 1, 2, 4]),
        [dim],
        space,
        dtype=getattr(xp, init_dtype_name),
    )

    shift_fun = getattr(fa, f"shift_{space}")
    with pytest.raises(ValueError):
        shift_fun(arr, {"x": 0.1})
