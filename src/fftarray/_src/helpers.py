from typing import TypeVar, Union, Iterable, Tuple, get_args, cast

from .space import Space

T = TypeVar("T")


space_args = get_args(Space)
def _check_space(x: str) -> Space:
    if x not in space_args:
        raise ValueError(f"Only valid values for space are: {space_args}. Got passed '{x}'.")
    return cast(Space, x)

def norm_space(val: Union[Space, Iterable[Space]], n: int) -> Tuple[Space, ...]:
    """
       `val` has to be immutable.
    """

    if isinstance(val, str):
        return (_check_space(val),)*n

    try:
        input_list = list(val)
    except(TypeError) as e:
        raise TypeError(
            f"Got passed '{val}' as space which raised an error on iteration."
        ) from e

    res_tuple = tuple(_check_space(x) for x in input_list)

    if len(res_tuple) != n:
        raise ValueError(
            f"Got passed '{val}' as space which has length {len(res_tuple)} "
            + f"but there are {n} dimensions."
        )
    return res_tuple

def norm_param(val: Union[T, Iterable[T]], n: int, types) -> Tuple[T, ...]:
    """
       `val` has to be immutable.
    """
    if isinstance(val, types):
        return (val,)*n

    # TODO: Can we make this type check work?
    return tuple(val) # type: ignore
