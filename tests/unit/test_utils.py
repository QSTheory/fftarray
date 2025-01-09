import pytest

from fftarray._src.helpers import _check_space, norm_space

def test_check_space() -> None:
    assert _check_space("pos") == "pos"
    assert _check_space("freq") == "freq"
    with pytest.raises(ValueError):
        _check_space(5) # type: ignore
    with pytest.raises(ValueError):
        _check_space("pos2")

def test_norm_space() -> None:

    assert norm_space("freq", 0) == tuple()
    assert norm_space("pos", 0) == tuple()
    assert norm_space("pos", 1) == ("pos",)
    assert norm_space("freq", 1) == ("freq",)
    assert norm_space("pos", 2) == ("pos",)*2
    assert norm_space("freq", 2) == ("freq",)*2

    assert norm_space(["pos"], 1) == ("pos",)
    assert norm_space(["freq"], 1) == ("freq",)
    assert norm_space(["pos", "freq"], 2) == ("pos", "freq")
    assert norm_space(["freq", "freq"], 2) == ("freq", "freq")

    with pytest.raises(TypeError):
        norm_space(5, 0) # type: ignore
    with pytest.raises(TypeError):
        norm_space(5, 1) # type: ignore
    with pytest.raises(ValueError):
        norm_space("pos2", 0) # type: ignore
    with pytest.raises(ValueError):
        norm_space("pos2", 1) # type: ignore
    with pytest.raises(ValueError):
        norm_space(["pos"], 2)
    with pytest.raises(ValueError):
        norm_space(["pos", "freq"], 1)
