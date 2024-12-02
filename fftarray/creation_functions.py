from typing import Any, Optional, Union, Iterable, Tuple, get_args

import array_api_compat

from .fft_dimension import FFTDimension
from .fft_array import FFTArray
from .space import Space
from .transform_application import real_type
from ._utils.defaults import get_default_eager, get_default_dtype_name, get_default_xp
from ._utils.helpers import norm_param

def array(
        values,
        dims: Union[FFTDimension, Iterable[FFTDimension]],
        space: Union[Space, Iterable[Space]],
        *,
        eager: Optional[Union[bool, Iterable[bool]]] = None,
        factors_applied: Union[bool, Iterable[bool]] = True,
        copy: bool = True,
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
            Specify the space of the coordinates and in which space the returned FFTArray is intialized.
        eager: Union[bool, Iterable[bool]]
            The eager-mode to use for the returned FFTArray.  `None` uses default `False` which can be globally changed.
        factors_applied: Union[bool, Iterable[bool]]
            Whether the fft-factors are applied are already applied for the various dimensions.
            For external values this is usually `True` since `False` assumes the internal (and unstable)
            factors-format.
        copy:  bool
            If ``True`` the values array is always copied in order to ensure no external alias to it exists.
            This ensures the immutability of the created ``FFTArray``.
            If this is unnecessary, this defensive copy can be prevented by setting this argument to ``False``.
            In this case it has to be ensured that the passed in array is not used externally after creation.

        Returns
        -------
        FFTArray
            The grid coordinates of the chosen space packed into an FFTArray with self as only dimension.

        See Also
        --------
        set_default_eager, get_default_eager
        fft_array
    """


    if eager is None:
        eager = get_default_eager()

    if isinstance(dims, FFTDimension):
        dims_tuple: Tuple[FFTDimension, ...] = (dims,)
    else:
        dims_tuple = tuple(dims)

    xp = array_api_compat.array_namespace(values)
    if copy:
        values = xp.asarray(values, copy=True)
    n_dims = len(dims_tuple)
    inner_values = xp.asarray(values)
    spaces_normalized: Tuple[Space, ...] = norm_param(space, n_dims, str)
    for sub_space in spaces_normalized:
        assert sub_space in get_args(Space)

    for i, (length, dim) in enumerate(zip(values.shape, dims_tuple, strict=True)):
        if length != dim.n:
            raise ValueError(f"The dimension `{dim.name}' has length {dim.n} but the dim {i} of the passed in `values` array has length {length}.")

    factors_applied_norm = norm_param(factors_applied, n_dims, bool)
    if not all(factors_applied_norm) and not xp.isdtype(inner_values.dtype, "complex floating"):
        raise ValueError(
            "If any `factors_applied' is False, the values have to be of dtype 'complex floating'"
            + " since the not applied phase factor implies a complex value."
        )

    arr = FFTArray(
        dims=dims_tuple,
        values=inner_values,
        space=spaces_normalized,
        eager=norm_param(eager, n_dims, bool),
        factors_applied=factors_applied_norm,
        xp=xp,
    )
    arr._check_consistency()
    return arr

def array_from_dim(
        dim: FFTDimension,
        space: Space,
        *,
        xp: Optional[Any] = None,
        dtype: Optional[Any] = None,
        eager: Optional[bool] = None,
    ) -> FFTArray:
    """..

    Parameters
    ----------
    dim : FFTDimension
        The dimension from which the coordinate grid should be created.
    space : Space
        Specify the space of the coordinates and in which space the returned FFTArray is intialized.
    xp : Optional[Any]
        The array namespace to use for the returned FFTArray. `None` uses default `numpy` which can be globally changed.
    dtype : Optional[Any]
        The dtype to use for the returned FFTArray. `None` uses default `numpy.float64` which can be globally changed.
    eager :  Optional[bool]
        The eager-mode to use for the returned FFTArray. `None` uses default `False` which can be globally changed.

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

    if eager is None:
        eager = get_default_eager()

    values = dim._raw_coord_array(
        xp=xp,
        dtype=dtype,
        space=space,
    )

    return FFTArray(
        values=values,
        dims=(dim,),
        eager=(eager,),
        factors_applied=(True,),
        space=(space,),
        xp=xp,
    )


def coords_array(
        x: FFTArray,
        dim_name: str,
        space: Union[Space],
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

    Returns
    -------
    FFTArray
        The grid coordinates of the chosen space packed into an FFTArray with self as only dimension.

    See Also
    --------
    """

    for dim_idx, dim in enumerate(x.dims):
        if dim.name == dim_name:
            return array_from_dim(
                dim=dim,
                space=space,
                xp=x.xp,
                eager=x.eager[dim_idx],
                dtype=real_type(x.xp, x.dtype),
            )
    raise ValueError("Specified dim_name not part of the FFTArray's dimensions.")

def full(
        dim: Union[FFTDimension, Iterable[FFTDimension]],
        space: Union[Space, Iterable[Space]],
        fill_value: Union[bool, int, float, complex],
        *,
        xp: Optional[Any] = None,
        dtype: Optional[Any] = None,
        eager: Optional[bool] = None,
    ) -> FFTArray:
    """..

    Parameters
    ----------
    dim : Union[FFTDimension, Iterable[FFTDimension]]
        The dimensions of the created array. They also imply the shape.
    space : Space
        Specify the space of the returned FFTArray is intialized.
    xp : Optional[Any]
        The array namespace to use for the returned FFTArray. `None` uses default `numpy` which can be globally changed.
    dtype : Optional[Any]
        The dtype to use for the returned FFTArray. `None` uses default `numpy.float64` which can be globally changed.
    eager :  Optional[bool]
        The eager-mode to use for the returned FFTArray. `None` uses default `False` which can be globally changed.

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

    if eager is None:
        eager = get_default_eager()

    if isinstance(dim, FFTDimension):
        dims: Tuple[FFTDimension, ...] = (dim,)
    else:
        dims = tuple(dim)

    n_dims = len(dims)
    shape = tuple(dim.n for dim in dims)
    values = xp.full(shape, fill_value, dtype=dtype)

    arr = FFTArray(
        values=values,
        dims=dims,
        space=norm_param(space, n_dims, str),
        eager=norm_param(eager, n_dims, bool),
        factors_applied=(True,)*n_dims,
        xp=xp,
    )
    arr._check_consistency()

    return arr


