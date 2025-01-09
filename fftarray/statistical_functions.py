from dataclasses import dataclass
from typing import Iterable, Optional, Union, List, Tuple
from typing_extensions import assert_never

from .dimension import Dimension
from .array import Array
from .space import Space

@dataclass
class SplitArrayMeta:
    """
        Internal helper class for the metadata after a reduction operation.
    """
    axis: List[int]
    eager: Tuple[bool, ...]
    space: Tuple[Space, ...]
    dims: Tuple[Dimension, ...]

def _named_dims_to_axis(x: Array, dim_name: Optional[Union[str, Iterable[str]]], /) -> SplitArrayMeta:
    """
        Transform dimension names into axis indices and extract
        all metadata that is kept after the reduction operation.

        The order of `dim_name` is kept to allow precise control in case the underlying implementation
        is not commutative in axis-order.
    """
    if dim_name is None:
        return SplitArrayMeta(
            axis=list(range(len(x.shape))),
            space=tuple([]),
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
    for dim, space, eager in zip(x.dims, x.space, x.eager, strict=True):
        if dim.name not in dim_name:
            dims.append(dim)
            spaces.append(space)
            eagers.append(eager)

    return SplitArrayMeta(
        axis=axis,
        space=tuple(spaces),
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

    res_meta = _named_dims_to_axis(x, dim_name)

    reduced_values = x.xp.sum(x.values(x.space), axis=tuple(res_meta.axis), dtype=dtype)

    return Array(
        values=reduced_values,
        space=res_meta.space,
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

    res_meta = _named_dims_to_axis(x, dim_name)

    reduced_values = x.xp.max(x.values(x.space), axis=tuple(res_meta.axis))

    return Array(
        values=reduced_values,
        space=res_meta.space,
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
    """
        Does a simple rectangle rule integration.
        Automatically uses the `d_pos` or `d_freq` of the integrated dimension
        depending on the space the dimension is in at the time of integration.
    """
    res_meta = _named_dims_to_axis(x, dim_name)

    integration_element = 1.
    for i in res_meta.axis:
        space = x.space[i]
        match space:
            case "pos":
                integration_element *= x.dims[i].d_pos
            case "freq":
                integration_element *= x.dims[i].d_freq
            case _:
                assert_never(space)

    reduced_values = x.xp.sum(x.values(x.space), axis=tuple(res_meta.axis), dtype=dtype)
    reduced_values *= integration_element

    return Array(
        values=reduced_values,
        space=res_meta.space,
        dims=res_meta.dims,
        eager=res_meta.eager,
        factors_applied=(True,)*len(res_meta.dims),
        xp=x.xp,
    )
