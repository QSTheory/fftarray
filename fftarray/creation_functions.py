from typing import Any, Optional, Union, Iterable, Tuple, get_args

import array_api_compat

from .fft_dimension import FFTDimension
from .fft_array import FFTArray
from .space import Space
from .transform_application import real_type
from ._utils.defaults import get_default_eager, get_default_dtype_name, get_default_xp
from ._utils.helpers import norm_space

def _get_xp(xp: Optional[Any], values) -> Tuple[Any, bool]:
    used_default_xp = False
    if xp is None:
        try:
            xp = array_api_compat.array_namespace(values)
        except(TypeError):
            xp = get_default_xp()
            used_default_xp = True
    else:
        xp = array_api_compat.array_namespace(xp.asarray(1))

    return xp, used_default_xp

def array(
        values,
        dims: Union[FFTDimension, Iterable[FFTDimension]],
        space: Union[Space, Iterable[Space]],
        *,
        xp: Optional[Any] = None,
        dtype: Optional[Any] = None,
        defensive_copy: bool = True,
    ) -> FFTArray:
    """
        Construct a new instance of FFTArray from raw values.

        Parameters
        ----------
        values :
            The values to initialize the `FFTArray` with.
            They can be of any Python Arrray API v2023.12 compatible library.
            By default they are copied to make sure an external alias cannot influence the created ``FFTArray``.
        dims : Iterable[FFTDimension]
            The FFTDimensions for each dimension of the passed in values.
        space: Union[Space, Iterable[Space]]
            Specify the space of the values with which the returned FFTArray is intialized.
        xp: Optional[Any]
            The Array API namespace to use for the created ``FFTArray``.
            If it is None, ``array_api_compat.array_namespace(values)`` is used.
            If that fails the default namespace from ``get_default_xp()`` is used.
        dtype: Optional[Any]
            Directly passed on to the ``xp.asarray`` of the determined xp.
            If None the ``dtype`` of values or the defaults for the passed in scalar of the underlying
            array library are used.
        defensive_copy:  bool
            If ``True`` the values array is always copied in order to ensure no external alias to it exists.
            This ensures the immutability of the created ``FFTArray``.
            If this is unnecessary, this defensive copy can be prevented by setting this argument to ``False``.
            In this case it has to be ensured that the passed in array is not used externally after creation.

        Returns
        -------
        FFTArray

        See Also
        --------
        set_default_eager, get_default_eager
        fft_array
    """

    if isinstance(dims, FFTDimension):
        dims_tuple: Tuple[FFTDimension, ...] = (dims,)
    else:
        dims_tuple = tuple(dims)

    xp, used_default_xp = _get_xp(xp, values)

    if defensive_copy:
        copy = True
    else:
        copy = None

    try:
        values = xp.asarray(values, copy=copy, dtype=dtype)
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

    n_dims = len(dims_tuple)
    spaces_normalized: Tuple[Space, ...] = norm_space(space, n_dims)
    for sub_space in spaces_normalized:
        assert sub_space in get_args(Space)

    for i, (length, dim) in enumerate(zip(values.shape, dims_tuple, strict=True)):
        if length != dim.n:
            raise ValueError(f"The dimension `{dim.name}' has length {dim.n} but axis {i} of the passed in `values` array has length {length}.")

    arr = FFTArray(
        dims=dims_tuple,
        values=values,
        space=spaces_normalized,
        eager=(get_default_eager(),)*n_dims,
        factors_applied=(True,)*n_dims,
        xp=xp,
    )
    arr._check_consistency()
    return arr

def coords_from_dim(
        dim: FFTDimension,
        space: Space,
        *,
        xp: Optional[Any] = None,
        dtype: Optional[Any] = None,
    ) -> FFTArray:
    """..

    Parameters
    ----------
    dim : FFTDimension
        The dimension from which the coordinate grid should be created.
    space : Space
        Specify the space of the coordinates and in which space the returned FFTArray is intialized.
    xp : Optional[Any]
        The array namespace to use for the returned FFTArray. `None` uses default ``numpy`` which can be globally changed.
    dtype : Optional[Any]
        The dtype to use for the returned FFTArray. `None` uses default ``float64`` which can be globally changed.

    Returns
    -------
    FFTArray
        The grid coordinates of the chosen space packed into an FFTArray with self as only dimension.

    See Also
    --------
        set_default_eager, get_default_eager
    """

    if xp is None:
        xp = get_default_xp()
    else:
        xp = array_api_compat.array_namespace(xp.asarray(0))

    if dtype is None:
        dtype = getattr(xp, get_default_dtype_name())

    if not xp.isdtype(dtype, ("real floating", "complex floating")):
        raise ValueError(f"Coordinates can only be initialized as real or complex numbers but got passed dtype '{dtype}'")

    values = dim._raw_coord_array(
        xp=xp,
        dtype=dtype,
        space=space,
    )

    return FFTArray(
        values=values,
        dims=(dim,),
        eager=(get_default_eager(),),
        factors_applied=(True,),
        space=(space,),
        xp=xp,
    )


def coords_from_arr(
        x: FFTArray,
        dim_name: str,
        space: Union[Space],
        *,
        xp: Optional[Any] = None,
        dtype: Optional[Any] = None,
	) -> FFTArray:
    """

    Constructs an array filled with the coordinates of the specified dimension
    while keeping all other attributes (Array API namespace, eager) of the
    specified array.

    Parameters
    ----------
    x : FFTArray
        The dimensions of the created array. They also imply the shape.
    space : Space
        Specify the space of the returned FFTArray is intialized.
    dim_name : str
        The name of the dimension from which to construct the coordinate array.
    xp : Optional[Any]
        The array namespace to use for the returned FFTArray. ``None`` uses the array namespace of ``x``.
    dtype : Optional[Any]
        The dtype to use for the created coordinate array.
        ``None`` uses a real floating point type with the same precision as ``x``.

    Returns
    -------
    FFTArray
        The grid coordinates of the chosen space packed into an FFTArray with the dimension of name ``dim_name``.
        ``eager`` of the created array is the same as eager in the selected dimension of ``x``.

    See Also
    --------
    """

    if dtype is None:
        dtype = real_type(x.xp, x.dtype)

    for dim_idx, dim in enumerate(x.dims):
        if dim.name == dim_name:
            if xp is None:
                xp_norm = x.xp
            else:
                xp_norm = array_api_compat.array_namespace(xp.array(1.))

            return coords_from_dim(
                dim=dim,
                space=space,
                xp=xp_norm,
                dtype=dtype,
            ).as_eager(eager=x.eager[dim_idx])
    raise ValueError("Specified dim_name not part of the FFTArray's dimensions.")

def full(
        dim: Union[FFTDimension, Iterable[FFTDimension]],
        space: Union[Space, Iterable[Space]],
        fill_value: Union[bool, int, float, complex, Any],
        *,
        xp: Optional[Any] = None,
        dtype: Optional[Any] = None,
    ) -> FFTArray:
    """..

    Parameters
    ----------
    dim : Union[FFTDimension, Iterable[FFTDimension]]
        The dimensions of the created array. They also imply the shape.
    space : Space
        Specify the space of the returned FFTArray is intialized.
    xp:
        The Array API namespace to use for the created ``FFTArray``.
        If it is None, ``array_api_compat.array_namespace(fill_value)`` is used.
        If that fails the default namespace from ``get_default_xp()`` is used.
    dtype : Optional[Any]
        The dtype to use for the returned FFTArray.
        If the value is `None`, the dtype is inferred from ``fill_value``
        according to the rules of the underlying Array API.


    Returns
    -------
    FFTArray
        The grid coordinates of the chosen space packed into an FFTArray with self as only dimension.

    See Also
    --------
        set_default_eager, get_default_eager
    """

    xp, _ = _get_xp(xp, fill_value)

    if isinstance(dim, FFTDimension):
        dims: Tuple[FFTDimension, ...] = (dim,)
    else:
        dims = tuple(dim)

    n_dims = len(dims)
    shape = tuple(dim.n for dim in dims)
    # Convert the fill_value from a potentially different array API implementation.
    fill_value = xp.asarray(fill_value, dtype=dtype)
    values = xp.full(shape, fill_value)

    arr = FFTArray(
        values=values,
        dims=dims,
        space=norm_space(space, n_dims),
        eager=(get_default_eager(),)*n_dims,
        factors_applied=(True,)*n_dims,
        xp=xp,
    )
    arr._check_consistency()

    return arr


