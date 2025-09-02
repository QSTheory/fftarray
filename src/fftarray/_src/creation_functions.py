from typing import Any, Optional, Union, Iterable, Tuple, Dict, get_args

from .dimension import Dimension
from .array import Array
from .space import Space
from .transform_application import real_type
from .defaults import get_default_eager
from .helpers import norm_space, norm_xp, norm_xp_with_values


def array(
        values,
        dim: Union[Dimension, Iterable[Dimension]],
        space: Union[Space, Iterable[Space], Dict[str, Space]],
        /,
        *,
        xp: Optional[Any] = None,
        dtype: Optional[Any] = None,
        device: Optional[Any] = None,
        defensive_copy: bool = True,
    ) -> Array:
    """
        Construct a new instance of `Array` from raw values.

        Parameters
        ----------
        values:
            The values to initialize the ``Array`` with.
            They can be of any Python Arrray API v2023.12 compatible library.
            By default they are copied to make sure an external alias cannot influence the created ``Array``.
        dim:
            The Dimension(s) for each dimension of the passed in values.
        space:
            Specify the space(s) of the values with which the returned ``Array`` intialized.
            If given as a dictionary it must contain all dimensions passed into ``dim``.
        xp:
            The Array API namespace to use for the created ``Array``.
            If it is None, ``array_api_compat.array_namespace(values)`` is used.
            If that fails the default namespace from ``get_default_xp()`` is used.
        dtype:
            Directly passed on to the ``xp.asarray`` of the determined xp.
            If None the ``dtype`` of values or the defaults for the passed in scalar of the underlying
            array library are used.
        defensive_copy:
            If ``True`` the values array is always copied in order to ensure no external alias to it exists.
            This ensures the immutability of the created ``Array``.
            If this is unnecessary, this defensive copy can be prevented by setting this argument to ``False``.
            In this case it has to be ensured that the passed in array is not used externally after creation.

        Returns
        -------
        Array

        See Also
        --------
        set_default_eager, get_default_eager
        Array
    """

    xp, used_default_xp = norm_xp_with_values(
        arg_xp=xp,
        values=values,
    )

    if isinstance(dim, Dimension):
        dims_tuple: Tuple[Dimension, ...] = (dim,)
    else:
        dims_tuple = tuple(dim)

    if defensive_copy:
        copy = True
    else:
        copy = None

    try:
        values = xp.asarray(values, copy=copy, dtype=dtype, device=device)
    except(Exception) as exc:
        if used_default_xp:
            raise type(exc)(
                "An Array API namespace could not be derived from "
                +f"'{values}' and therefore the default '{xp}' was used. "
                +"Calling 'asarray' on that namespace resulted in the following error: "
                +str(exc)
            ) from exc
        else:
            raise exc

    spaces_normalized: Tuple[Space, ...] = norm_space(space, dims_tuple, None)
    for sub_space in spaces_normalized:
        assert sub_space in get_args(Space)

    for i, (length, dim) in enumerate(zip(values.shape, dims_tuple, strict=True)):
        if length != dim.n:
            raise ValueError(f"The dimension `{dim.name}' has length {dim.n} but axis {i} of the passed in `values` array has length {length}.")

    n_dims = len(dims_tuple)

    arr = Array(
        dims=dims_tuple,
        values=values,
        spaces=spaces_normalized,
        eager=(get_default_eager(),)*n_dims,
        factors_applied=(True,)*n_dims,
        xp=xp,
    )
    arr._check_consistency()
    return arr

def coords_from_dim(
        dim: Dimension,
        space: Space,
        /,
        *,
        xp: Optional[Any] = None,
        dtype: Optional[Any] = None,
        device: Optional[Any] = None,
    ) -> Array:
    """..

    Parameters
    ----------
    dim : Dimension
        The dimension from which the coordinate grid should be created.
    space : Space
        Specify the space of the coordinates and in which space the returned ``Array`` is intialized.
    xp : Optional[Any]
        The array namespace to use for the returned ``Array``. `None` uses default ``numpy`` which can be globally changed.
    dtype : Optional[Any]
        The dtype to use for the returned ``Array``. `None` uses the default floating point type of the chosen ``xp``.

    Returns
    -------
    ``Array``
        The grid coordinates of the chosen space packed into an ``Array`` with self as only dimension.

    See Also
    --------
        set_default_eager, get_default_eager
    """

    xp = norm_xp(xp)

    values = dim.values(
        space,
        xp=xp,
        dtype=dtype,
        device=device,
    )

    return Array(
        values=values,
        dims=(dim,),
        eager=(get_default_eager(),),
        factors_applied=(True,),
        spaces=(space,),
        xp=xp,
    )


def coords_from_arr(
        x: Array,
        dim_name: str,
        space: Space,
        /,
        *,
        xp: Optional[Any] = None,
        dtype: Optional[Any] = None,
        device: Optional[Any] = None,
	) -> Array:
    """

    Constructs an array filled with the coordinates of the specified dimension
    while keeping all other attributes (Array API namespace, eager) of the
    specified array.

    Parameters
    ----------
    x : Array
        The array from which to construct the coordinate array.
    space : Space
        Specify the space of the returned Array is intialized.
    dim_name : str
        The name of the dimension from which to construct the coordinate array.
    xp : Optional[Any]
        The array namespace to use for the returned Array. ``None`` uses the array namespace of ``x``.
    dtype : Optional[Any]
        The dtype to use for the created coordinate array.
        ``None`` uses a real floating point type with the same precision as ``x``.

    Returns
    -------
    Array
        The grid coordinates of the chosen space packed into an Array with the dimension of name ``dim_name``.
        ``eager`` of the created array is the same as eager in the selected dimension of ``x``.

    See Also
    --------
    """

    xp, _ = norm_xp_with_values(
        arg_xp=xp,
        values=x._values,
    )

    if dtype is None:
        dtype = x.dtype

        if not xp.isdtype(dtype, ("real floating", "complex floating")):
            raise ValueError(
                "Coordinates can only have a real-valued floating point dtype. " \
                f"Unable to infer data type from {dtype}."
            )

        dtype = real_type(x.xp, dtype)

    if device is None:
        device = x.device

    for dim_idx, dim in enumerate(x.dims):
        if dim.name == dim_name:
            return coords_from_dim(
                dim, space, xp=xp, dtype=dtype, device=device,
            ).into_eager(x.eager[dim_idx])
    raise ValueError("Specified dim_name not part of the Array's dimensions.")

def full(
        dim: Union[Dimension, Iterable[Dimension]],
        space: Union[Space, Iterable[Space], Dict[str, Space]],
        fill_value: Union[bool, int, float, complex, Any],
        /,
        *,
        xp: Optional[Any] = None,
        dtype: Optional[Any] = None,
        device: Optional[Any] = None,
    ) -> Array:
    """..

    Parameters
    ----------
    dim:
        The dimension(s) of the created array. They also imply the shape.
    space:
        Specify the space(s) in which the returned Array is intialized.
        If given as a dictionary it must contain all dimensions passed into ``dim``.
    xp:
        The Array API namespace to use for the created ``Array``.
        If it is None, ``array_api_compat.array_namespace(fill_value)`` is used.
        If that fails the default namespace from ``get_default_xp()`` is used.
    dtype : Optional[Any]
        The dtype to use for the returned Array.
        If the value is `None`, the dtype is inferred from ``fill_value``
        according to the rules of the underlying Array API.


    Returns
    -------
    Array
        The grid coordinates of the chosen space packed into an Array with self as only dimension.

    See Also
    --------
        set_default_eager, get_default_eager
    """

    xp, _ = norm_xp_with_values(
        arg_xp=xp,
        values=fill_value,
    )

    if isinstance(dim, Dimension):
        dims: Tuple[Dimension, ...] = (dim,)
    else:
        dims = tuple(dim)

    n_dims = len(dims)
    shape = tuple(dim.n for dim in dims)
    values = xp.full(shape, fill_value, dtype=dtype, device=device)

    arr = Array(
        values=values,
        dims=dims,
        spaces=norm_space(space, dims, None),
        eager=(get_default_eager(),)*n_dims,
        factors_applied=(True,)*n_dims,
        xp=xp,
    )
    arr._check_consistency()

    return arr


