from typing import TypeVar, Union, Optional, Any, Iterable, get_args, cast, Callable, TYPE_CHECKING

from .compat_namespace import get_compat_namespace, get_array_compat_namespace
from .defaults import get_default_xp
from .space import Space

T = TypeVar("T")

if TYPE_CHECKING:
    from .dimension import Dimension

space_args = get_args(Space)
def _check_space(x: str, arg_name: str) -> Space:
    if x not in space_args:
        raise ValueError(f"Only valid values for {arg_name} are: {space_args}. Got passed '{x}'.")
    return cast(Space, x)

def _check_bool(x: bool, arg_name: str) -> bool:
    if not isinstance(x, bool):
        raise ValueError(f"Only valid values for {arg_name} are booleans. Got passed '{x}'.")
    return cast(bool, x)

def norm_space(
        val: Union[Space, Iterable[Space], dict[str, Space]],
        dims: tuple["Dimension", ...],
        old_val: Optional[tuple[Space, ...]],
    ) -> tuple[Space, ...]:

    # First check for the scalar case.
    if isinstance(val, str):
        return (_check_space(val, "space"),)*len(dims)

    return _norm_param(
        val=val,
        dims=dims,
        old_val=old_val,
        check_fun=_check_space,
        arg_name="space",
    )


def norm_bool(
        val: Union[bool, Iterable[bool], dict[str, bool]],
        dims: tuple["Dimension", ...],
        old_val: tuple[bool, ...],
        arg_name: str,
    ) -> tuple[bool, ...]:

    # First check for the scalar case.
    if isinstance(val, bool):
        return (_check_bool(val, arg_name),)*len(dims)

    return _norm_param(
        val=val,
        dims=dims,
        old_val=old_val,
        check_fun=_check_bool,
        arg_name=arg_name,
    )


def _norm_param(
        val: Union[Iterable[T], dict[str, T]],
        dims: tuple["Dimension", ...],
        old_val: Optional[tuple[T, ...]],
        check_fun: Callable[[T, str], T],
        arg_name: str,
    ) -> tuple[T, ...]:
    """
        Normalize one of the per dimension parameters (``space``, ``eager``, ``factors_applied``)
        from an ``Iterable`` or a ``dict`` into a tuple.

        Parameters
        ----------
        val:
            The user-specified ``Iterable`` or ``dict`` to set the new values
        dims:
            Dimensions of the Array which also determine the order of the values in the tuple.
        old_val:
            If applicable the old values of the parameter which is normalized.
            This allows the user to list in a dict only a subset of all dimensions
            which then only overrides those specific dimensions.
        check_fun:
            Function which checks whether the given value is valid for the passed in parameter type.
            It also takes the name of the parameter (``space``, ``eager``, ``factors_applied``) in order
            to throw an appropiate error message.
        arg_name:
            Name of the parameter (``space``, ``eager``, ``factors_applied``) which is normalized.

        Returns
        -------
        tuple[T, ...]
            The parameter values per dim as a tuple.
    """

    n = len(dims)

    if isinstance(val, dict):
        # Use a list and linear search since the overhead
        # should be smaller than with a dict as a LUT.
        names = [dim.name for dim in dims]

        if old_val is None:
            res: list[Optional[T]] = [None,] * n
            # Since we were not passed an old value,
            # all dimensions of the array need to be in the dict.
            missing_dims = [
                dim.name
                for dim in dims
                if dim.name not in val
            ]
            if len(missing_dims) > 0:
                raise ValueError(f"Missing {arg_name} value for dims {missing_dims}.")
        else:
            res = list(old_val)

        for dim_name, x in val.items():
            try:
                dim_idx = names.index(dim_name)
            except ValueError:
                # The index failure is not really the cause,
                # so we do not want to print it in the stack trace.
                raise ValueError(f"There is no dimension '{dim_name}', existing dimensions: {names}.") from None
            res[dim_idx] = x



        # Mypy does not understand the above check.
        return tuple(res) # type: ignore

    try:
        input_list = list(val)
    except(TypeError) as e:
        raise TypeError(
            f"Got passed '{val}' as {arg_name} which raised an error on iteration."
        ) from e

    res_tuple = tuple(check_fun(x, arg_name) for x in input_list)

    if len(res_tuple) != n:
        raise ValueError(
            f"Got passed '{val}' as {arg_name} which has length {len(res_tuple)} "
            + f"but there are {n} dimensions."
        )
    return res_tuple



def norm_xp_with_values(
            arg_xp: Optional[Any],
            values,
        ) -> tuple[Any, bool]:
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
