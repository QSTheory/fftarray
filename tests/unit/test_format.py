import pytest

from fftarray.dimension import Dimension
from fftarray._utils.formatting import format_bytes, format_n
from fftarray.tests.helpers import XPS
from tests.helpers import get_dims, get_arr_from_dims

@pytest.mark.parametrize("xp", XPS)
def test_format(xp) -> None:
    """
        Tests that `__str__` and `__repr__` of `Dimension`
        and `Array` at least do not crash and return a string.
    """
    dims = get_dims(1)
    dim1 = Dimension(
        name="veryLongName",
        n=1024,
        d_pos=0.1,
        pos_min=0.,
        freq_min=0.,
        dynamically_traced_coords=True,
    )
    dims = [*dims, dim1]
    for dim in dims:
        assert isinstance(str(dim), str)
        assert isinstance(repr(dim), str)

    arr1 = get_arr_from_dims(xp=xp, dims=dims[:1])
    assert isinstance(str(arr1), str)
    assert isinstance(repr(arr1), str)
    arr2 = get_arr_from_dims(xp=xp, dims=dims)
    assert isinstance(str(arr2), str)
    assert isinstance(repr(arr2), str)
    arr3 = get_arr_from_dims(xp=xp, dims=dims, spaces="freq")
    assert isinstance(str(arr3), str)
    assert isinstance(repr(arr3), str)


def test_format_bytes() -> None:
    assert format_bytes(12) == "12 bytes"
    assert format_bytes(2**10) == "1.0 KiB"
    assert format_bytes(1.2*2**10) == "1.2 KiB"
    assert format_bytes(2**20) == "1.0 MiB"
    assert format_bytes(2**40) == "1.0 TiB"
    assert format_bytes(1.53*2**40) == "1.5 TiB"


def test_format_n() -> None:
    assert format_n(12) == "12"
    assert format_n(8) == "2^3"
    assert format_n(12000) == "1.20e+04"
    assert format_n(5000) == "5000"
