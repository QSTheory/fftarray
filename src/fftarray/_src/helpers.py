from typing import TypeVar, Union, Optional, Any, Iterable, Tuple, get_args, cast

from .compat_namespace import get_compat_namespace, get_array_compat_namespace
from .defaults import get_default_xp
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



def norm_xp_with_values(
            arg_xp: Optional[Any],
            values,
        ) -> Tuple[Any, bool]:
    """
        Determine xp from passed in values and explicit xp argument.
        An implied xp conversion raises a ``ValueError``.
    """
    used_default_xp = False

    if arg_xp is not None:
        arg_xp = get_compat_namespace(arg_xp)

    try:
        derived_xp = get_array_compat_namespace(values)
    except(TypeError):
        derived_xp = None

    match (arg_xp, derived_xp):
        case (None, None):
            xp = get_default_xp()
            used_default_xp = True
        case (None, _):
            xp = derived_xp
        case (_, None):
            xp = arg_xp
        case (_,_):
            if derived_xp != arg_xp:
                raise ValueError("Got passed different explicit xp than the xp of the array." \
                    "Cross-library conversion is not supported as it is not mandated to work properly by the Python Array API standard."
                )
            xp = derived_xp

    return xp, used_default_xp

def norm_xp(
            xp_arg: Optional[Any],
        ) -> Any:
    """
        Normalize xp_arg with potentially using the default_xp
    """
    if xp_arg is None:
        xp = get_default_xp()
    else:
        xp = get_compat_namespace(xp_arg)

    return xp
