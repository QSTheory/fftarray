import pytest
import fftarray as fa

from fftarray.tests.helpers import XPS
from tests.helpers  import get_dims, dtypes_names_all, DTYPE_NAME


@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("init_dtype_name", dtypes_names_all)
@pytest.mark.parametrize("target_dtype_name", dtypes_names_all)
def test_astype(xp, init_dtype_name, target_dtype_name) -> None:
    dim = fa.dim("x", 4, 0.1, 0., 0.)
    arr1 = fa.array(
        xp.asarray([0, 1,2,3], dtype=getattr(xp, init_dtype_name)),
        dims=[dim],
        space="pos",
    )
    arr2 = arr1.astype(getattr(xp, target_dtype_name))
    assert arr2.dtype == getattr(xp, target_dtype_name)

@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("init_dtype_name", ["complex64", "complex128"])
@pytest.mark.parametrize("target_dtype_name", dtypes_names_all)
def test_astype_no_factors(xp, init_dtype_name, target_dtype_name) -> None:
    dim = fa.dim("x", 4, 0.1, 0., 0.)
    arr1 = fa.array(
        xp.asarray([0, 1,2,3], dtype=getattr(xp, init_dtype_name)),
        dims=[dim],
        space="pos",
        factors_applied=False,
    )

    target_dtype = getattr(xp, target_dtype_name)
    if xp.isdtype(target_dtype, "complex floating"):
        arr2 = arr1.astype(target_dtype)
        assert arr2.dtype == target_dtype
    else:
        with pytest.raises(ValueError):
            arr2 = arr1.astype(target_dtype)


@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("dtype_name", dtypes_names_all)
@pytest.mark.parametrize("ndims", [0,1,2])
@pytest.mark.parametrize("copy", [False, True])
def test_array_creation(xp, dtype_name: DTYPE_NAME, ndims: int, copy: bool) -> None:
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
    except:
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
