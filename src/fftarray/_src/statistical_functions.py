from dataclasses import dataclass
from typing import Iterable, Optional, Union, List, Tuple
from typing_extensions import assert_never

from .dimension import Dimension
from .array import Array
from .space import Space

@dataclass
class SplitArrayMeta:
    """Internal helper class for the metadata after a reduction operation.

    :meta private:
    """
    axis: List[int]
    eager: Tuple[bool, ...]
    spaces: Tuple[Space, ...]
    dims: Tuple[Dimension, ...]

def _named_dims_to_axis(
        x: Array,
        dim_name: Optional[Union[str, Iterable[str]]],
        /,
    ) -> SplitArrayMeta:
    """
    Transform dimension names into axis indices and extract all metadata that is
    kept after the reduction operation.

    The order of `dim_name` is kept to allow precise control in case the
    underlying implementation is not commutative in axis-order.
    """
    if dim_name is None:
        return SplitArrayMeta(
            axis=list(range(len(x.shape))),
            spaces=tuple([]),
            dims=tuple([]),
            eager=tuple([]),
        )

    if isinstance(dim_name, str):
        dim_name = [dim_name]

    dim_names = [dim.name for dim in x.dims]
    axis = []
    for dim_ident in dim_name:
        dim_idx = dim_names.index(dim_ident)
        axis.append(dim_idx)

    dims = []
    spaces = []
    eagers = []
    for dim, space, eager in zip(x.dims, x.spaces, x.eager, strict=True):
        if dim.name not in dim_name:
            dims.append(dim)
            spaces.append(space)
            eagers.append(eager)

    return SplitArrayMeta(
        axis=axis,
        spaces=tuple(spaces),
        dims=tuple(dims),
        eager=tuple(eagers),
    )

def sum(
        x: Array,
        /,
        *,
        dim_name: Optional[Union[str, Iterable[str]]] = None,
        dtype = None,
    ) -> Array:
    """Computes the sum of all values over the specified Dimension(s).
    This is a thin wrapper around [7]_, refer to its documentation and
    the documentation of the used array library for detailed
    information about its semantics.

    Parameters
    ----------
    x : Array
        The input Array.
    dim_name : Optional[Union[str, Iterable[str]]], optional
        Dimension name(s) over which the sum is performed. The default,
        ``dim_name=None``, will reduce over all Dimensions.
    dtype : Any, optional
        The dtype of the returned Array, by default None.

    Returns
    -------
    Array
        New Array with the specified dimensions reduced by computing the sum of the elements along those dimensions.

    See Also
    --------
    numpy.sum
    xarray.DataArray.sum

    References
    ----------
    .. [7] `Array API Standard - sum <https://data-apis.org/array-api/latest/API_specification/generated/array_api.sum>`_
    """

    res_meta = _named_dims_to_axis(x, dim_name)

    reduced_values = x.xp.sum(x.values(x.spaces), axis=tuple(res_meta.axis), dtype=dtype)

    return Array(
        values=reduced_values,
        spaces=res_meta.spaces,
        dims=res_meta.dims,
        eager=res_meta.eager,
        factors_applied=(True,)*len(res_meta.dims),
        xp=x.xp,
    )

def prod(
        x: Array,
        /,
        *,
        dim_name: Optional[Union[str, Iterable[str]]] = None,
        dtype = None,
    ) -> Array:
    """Computes the product of all values over the specified Dimension(s).
    This is a thin wrapper around [6]_, refer to its documentation and
    the documentation of the used array library for detailed
    information about its semantics.

    Parameters
    ----------
    x : Array
        The input Array.
    dim_name : Optional[Union[str, Iterable[str]]], optional
        Dimension name(s) along which the products are computed. The default,
        ``dim_name=None``, will reduce over all Dimensions.
    dtype : Any, optional
        The dtype of the returned Array, by default None.

    Returns
    -------
    Array
        New Array with the specified dimensions reduced by computing the product of the elements along those dimensions.

    See Also
    --------
    numpy.prod
    xarray.DataArray.prod

    References
    ----------
    .. [6] `Array API Standard - prod <https://data-apis.org/array-api/latest/API_specification/generated/array_api.prod>`_
    """

    res_meta = _named_dims_to_axis(x, dim_name)

    reduced_values = x.xp.prod(x.values(x.spaces), axis=tuple(res_meta.axis), dtype=dtype)

    return Array(
        values=reduced_values,
        spaces=res_meta.spaces,
        dims=res_meta.dims,
        eager=res_meta.eager,
        factors_applied=(True,)*len(res_meta.dims),
        xp=x.xp,
    )

def max(
        x: Array,
        /,
        *,
        dim_name: Optional[Union[str, Iterable[str]]] = None,
    ) -> Array:
    """Computes the maximum of all values along the specified Dimension(s).
    This is a thin wrapper around [3]_, refer to its documentation and
    the documentation of the used array library for detailed
    information about its semantics.

    Parameters
    ----------
    x : Array
        The input Array.
    dim_name : Optional[Union[str, Iterable[str]]], optional
        Dimension name(s) along which the maximum is computed. The default,
        ``dim_name=None``, will reduce over all Dimensions.

    Returns
    -------
    Array
        New Array with the specified dimensions reduced by computing the maximum of the elements along those dimensions.

    See Also
    --------
    min
    numpy.max
    xarray.DataArray.max

    References
    ----------
    .. [3] `Array API Standard - max <https://data-apis.org/array-api/latest/API_specification/generated/array_api.max>`_
    """

    res_meta = _named_dims_to_axis(x, dim_name)

    reduced_values = x.xp.max(x.values(x.spaces), axis=tuple(res_meta.axis))

    return Array(
        values=reduced_values,
        spaces=res_meta.spaces,
        dims=res_meta.dims,
        eager=res_meta.eager,
        factors_applied=(True,)*len(res_meta.dims),
        xp=x.xp,
    )

def min(
        x: Array,
        /,
        *,
        dim_name: Optional[Union[str, Iterable[str]]] = None,
    ) -> Array:
    """Computes the minimum of all values along the specified Dimension(s).
    This is a thin wrapper around [5]_, refer to its documentation and
    the documentation of the used array library for detailed
    information about its semantics.

    Parameters
    ----------
    x : Array
        The input Array.
    dim_name : Optional[Union[str, Iterable[str]]], optional
        Dimension name(s) along which the minimum is computed. The default,
        ``dim_name=None``, will reduce over all Dimensions.

    Returns
    -------
    Array
        New Array with the specified dimensions reduced by computing the minimum of the elements along those dimensions.

    See Also
    --------
    max
    numpy.min
    xarray.DataArray.min

    References
    ----------
    .. [5] `Array API Standard - min <https://data-apis.org/array-api/latest/API_specification/generated/array_api.min>`_
    """

    res_meta = _named_dims_to_axis(x, dim_name)

    reduced_values = x.xp.min(x.values(x.spaces), axis=tuple(res_meta.axis))

    return Array(
        values=reduced_values,
        spaces=res_meta.spaces,
        dims=res_meta.dims,
        eager=res_meta.eager,
        factors_applied=(True,)*len(res_meta.dims),
        xp=x.xp,
    )

def mean(
        x: Array,
        /,
        *,
        dim_name: Optional[Union[str, Iterable[str]]] = None,
    ) -> Array:
    """Computes the mean of all values along the specified Dimension(s).
    This is a thin wrapper around [4]_, refer to its documentation and
    the documentation of the used array library for detailed
    information about its semantics.

    Parameters
    ----------
    x : Array
        The input Array.
    dim_name : Optional[Union[str, Iterable[str]]], optional
        Dimension name(s) along which the mean is computed. The default,
        ``dim_name=None``, will reduce over all Dimensions.

    Returns
    -------
    Array
        New Array with the specified dimensions reduced by computing the mean of the elements along those dimensions.

    See Also
    --------
    numpy.mean
    xarray.DataArray.mean

    References
    ----------
    .. [4] `Array API Standard - mean <https://data-apis.org/array-api/latest/API_specification/generated/array_api.mean>`_
    """

    res_meta = _named_dims_to_axis(x, dim_name)

    reduced_values = x.xp.mean(x.values(x.spaces), axis=tuple(res_meta.axis))

    return Array(
        values=reduced_values,
        spaces=res_meta.spaces,
        dims=res_meta.dims,
        eager=res_meta.eager,
        factors_applied=(True,)*len(res_meta.dims),
        xp=x.xp,
    )

def integrate(
        x: Array,
        /,
        *,
        dim_name: Optional[Union[str, Iterable[str]]] = None,
        dtype = None,
    ) -> Array:
    """Computes the integral along the specified Dimension(s). The integration
    is performed using simple rectangle rule integration, also known as Riemann
    summation [1]_. The integration is equivalent to summing up the values along
    the specified Dimension(s) and multipling them with the integration element
    given by ``d_pos`` or ``d_freq`` of the respective Dimension. The
    integration for each Dimension is performed in the space of the input Array.
    The actual summation is forwarded to [2]_.

    Parameters
    ----------
    x : Array
        The input Array.
    dim_name : Optional[Union[str, Iterable[str]]], optional
        Dimension name(s) along which the integration is performed. The default,
        ``dim_name=None``, will reduce over all Dimensions.
    dtype : Any, optional
        The dtype of the returned Array, by default None.

    Returns
    -------
    Array
        New Array with the specified dimensions reduced by computing the integration of the elements along those dimensions.

    See Also
    --------
    sum

    References
    ----------
    .. [1] Wikipedia, "Riemann sum", https://en.wikipedia.org/wiki/Riemann_sum
    .. [2] `Array API Standard - sum <https://data-apis.org/array-api/latest/API_specification/generated/array_api.sum>`_
    """
    res_meta = _named_dims_to_axis(x, dim_name)

    integration_element = 1.
    for i in res_meta.axis:
        space = x.spaces[i]
        match space:
            case "pos":
                integration_element *= x.dims[i].d_pos
            case "freq":
                integration_element *= x.dims[i].d_freq
            case _:
                assert_never(space)

    if dtype is None:
        dtype = x.dtype

    reduced_values = x.xp.sum(x.values(x.spaces), axis=tuple(res_meta.axis), dtype=dtype)
    reduced_values *= x.xp.asarray(integration_element, dtype=dtype)

    return Array(
        values=reduced_values,
        spaces=res_meta.spaces,
        dims=res_meta.dims,
        eager=res_meta.eager,
        factors_applied=(True,)*len(res_meta.dims),
        xp=x.xp,
    )
