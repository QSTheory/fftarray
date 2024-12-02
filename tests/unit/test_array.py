import pytest
import fftarray as fa

from fftarray.tests.helpers import XPS
from tests.helpers  import dtypes_names_all


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
