import pytest

import fftarray as fa
from fftarray._src.dimension import Dimension
from fftarray._src.helpers import _check_space, _check_bool, _norm_param, norm_space, norm_bool


@pytest.fixture
def tuple_of_dims_tuple() -> tuple[tuple[Dimension, ...], ...]:
    dim_x = fa.dim("x", 1, 1., 0., 0.)
    dim_y = fa.dim("y", 1, 1., 0., 0.)
    dims0: tuple[fa.Dimension, ...] = tuple()
    dims1: tuple[fa.Dimension, ...] = (dim_x,)
    dims2: tuple[fa.Dimension, ...] = (dim_x,dim_y,)

    return dims0, dims1, dims2

def test_check_space() -> None:
    assert _check_space("pos", "space") == "pos"
    assert _check_space("freq", "space") == "freq"
    with pytest.raises(ValueError):
        _check_space(5, "space") # type: ignore
    with pytest.raises(ValueError):
        _check_space("pos2", "space")

def test_check_bool() -> None:
    assert not _check_bool(False, "eager")
    assert _check_bool(True, "eager")
    with pytest.raises(ValueError):
        _check_bool(5, "eager") # type: ignore
    with pytest.raises(ValueError):
        _check_bool("pos2", "eager") # type: ignore

def test_norm_param(tuple_of_dims_tuple) -> None:

    assert _norm_param(["pos"], tuple_of_dims_tuple[1], ("freq",), _check_space, "space") == ("pos",)
    assert _norm_param(["freq"], tuple_of_dims_tuple[1], ("freq",), _check_space, "space") == ("freq",)
    assert _norm_param(["pos", "freq"], tuple_of_dims_tuple[2], ("pos", "freq",), _check_space, "space") == ("pos", "freq")
    assert _norm_param(["freq", "freq"], tuple_of_dims_tuple[2], ("pos", "freq",), _check_space, "space") == ("freq", "freq")
    assert _norm_param({"y": "freq"}, tuple_of_dims_tuple[2], ("pos", "pos",), _check_space, "space") == ("pos", "freq")
    assert _norm_param({"x": "freq"}, tuple_of_dims_tuple[2], ("pos", "pos",), _check_space, "space") == ("freq", "pos")

    assert _norm_param({"y": True}, tuple_of_dims_tuple[2], (False, False,), _check_bool, "eager") == (False, True)
    assert _norm_param({"x": True}, tuple_of_dims_tuple[2], (False, False,), _check_bool, "eager") == (True, False)
    assert _norm_param({"x": True, "y": False}, tuple_of_dims_tuple[2], None, _check_bool, "eager") == (True, False)

    assert _norm_param({"x": True}, tuple_of_dims_tuple[1], None, _check_bool, "eager") == (True,)

    with pytest.raises(ValueError):
        _norm_param({"y": True}, tuple_of_dims_tuple[2], None, _check_bool, "eager")

    with pytest.raises(ValueError):
        _norm_param({"z": True}, tuple_of_dims_tuple[2], None, _check_bool, "eager")

    with pytest.raises(ValueError):
        _norm_param({}, tuple_of_dims_tuple[2], None, _check_bool, "eager")


def test_norm_space(tuple_of_dims_tuple) -> None:

    assert norm_space("freq", tuple_of_dims_tuple[0], tuple()) == tuple()
    assert norm_space("pos", tuple_of_dims_tuple[0], tuple()) == tuple()
    assert norm_space("pos", tuple_of_dims_tuple[1], ("freq",)) == ("pos",)
    assert norm_space("freq", tuple_of_dims_tuple[1], ("freq",)) == ("freq",)
    assert norm_space("pos", tuple_of_dims_tuple[2], ("pos", "freq",)) == ("pos",)*2
    assert norm_space("freq", tuple_of_dims_tuple[2], ("pos", "freq",)) == ("freq",)*2

    assert norm_space(["pos"], tuple_of_dims_tuple[1], ("freq",)) == ("pos",)
    assert norm_space(["freq"], tuple_of_dims_tuple[1], ("freq",)) == ("freq",)
    assert norm_space(["pos", "freq"], tuple_of_dims_tuple[2], ("pos", "freq",)) == ("pos", "freq")
    assert norm_space(["freq", "freq"], tuple_of_dims_tuple[2], ("pos", "freq",)) == ("freq", "freq")
    # Under CPython 3.12 mypy does not correctly infer the type of the passed in dict.
    assert norm_space({"x": "freq"}, tuple_of_dims_tuple[2], ("pos", "pos",)) == ("freq", "pos") # type: ignore

    with pytest.raises(TypeError):
        norm_space(5, tuple_of_dims_tuple[0], tuple()) # type: ignore
    with pytest.raises(TypeError):
        norm_space(5, tuple_of_dims_tuple[1], tuple()) # type: ignore
    with pytest.raises(ValueError):
        norm_space("pos2", tuple_of_dims_tuple[0], ("freq",)) # type: ignore
    with pytest.raises(ValueError):
        norm_space("pos2", tuple_of_dims_tuple[1], ("freq",)) # type: ignore
    with pytest.raises(ValueError):
        norm_space(["pos"], tuple_of_dims_tuple[2], ("pos", "freq",))
    with pytest.raises(ValueError):
        norm_space(["pos", "freq"], tuple_of_dims_tuple[1], ("freq",))
    with pytest.raises(ValueError):
        # Under CPython 3.12 mypy does not correctly infer the type of the passed in dict.
        norm_space({"y": "pos"}, tuple_of_dims_tuple[2], None) # type: ignore

def test_norm_bool(tuple_of_dims_tuple) -> None:

    assert norm_bool(True, tuple_of_dims_tuple[0], tuple(), "arg") == tuple()
    assert norm_bool(False, tuple_of_dims_tuple[0], tuple(), "arg") == tuple()
    assert norm_bool(False, tuple_of_dims_tuple[1], (True,), "arg") == (False,)
    assert norm_bool(True, tuple_of_dims_tuple[1], (True,), "arg") == (True,)
    assert norm_bool(False, tuple_of_dims_tuple[2], (False, True,), "arg") == (False,)*2
    assert norm_bool(True, tuple_of_dims_tuple[2], (False, True,), "arg") == (True,)*2

    assert norm_bool([False], tuple_of_dims_tuple[1], (True,), "arg") == (False,)
    assert norm_bool([True], tuple_of_dims_tuple[1], (True,), "arg") == (True,)
    assert norm_bool([False, True], tuple_of_dims_tuple[2], (False, True,), "arg") == (False, True)
    assert norm_bool([True, True], tuple_of_dims_tuple[2], (False, True,), "arg") == (True, True)
    assert norm_bool({"x": True}, tuple_of_dims_tuple[2], (False, False,), "arg") == (True, False)

    with pytest.raises(TypeError):
        norm_bool(5, tuple_of_dims_tuple[0], tuple(), "arg") # type: ignore
    with pytest.raises(TypeError):
        norm_bool(5, tuple_of_dims_tuple[1], tuple(), "arg") # type: ignore
    with pytest.raises(ValueError):
        norm_bool("pos2", tuple_of_dims_tuple[0], (True,), "arg") # type: ignore
    with pytest.raises(ValueError):
        norm_bool("pos2", tuple_of_dims_tuple[1], (True,), "arg") # type: ignore
    with pytest.raises(ValueError):
        norm_bool([False], tuple_of_dims_tuple[2], (False, True,), "arg")
    with pytest.raises(ValueError):
        norm_bool([False, True], tuple_of_dims_tuple[1], (True,), "arg")

