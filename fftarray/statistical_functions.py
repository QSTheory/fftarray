from dataclasses import dataclass
from typing import Iterable, Optional, Union, List, Tuple, Dict

from fftarray.fft_dimension import FFTDimension

from .fft_array import FFTArray
from .space import Space

@dataclass
class SplitArrayMeta:
    """
        Internal helper class for the metadata after a reduction operation.
    """
    axis: List[int]
    eager: Tuple[bool, ...]
    space: Tuple[Space, ...]
    fft_dims: Tuple[FFTDimension, ...]

def _named_dims_to_axis(x: FFTArray, dim: Union[str, Iterable[str]]) -> SplitArrayMeta:
    """
        Transform dimension names into axis indices and extract
        all metadata that is kept after the reduction operation.
    """
    if isinstance(dim, str):
        dim = [dim]

    axis = []
    fft_dims = []
    spaces = []
    eagers = []
    for idx, (fft_dim, space, eager) in enumerate(zip(x.dims, x.space, x.eager)):
        if fft_dim.name in dim:
            axis.append(idx)
        else:
            fft_dims.append(fft_dim)
            spaces.append(space)
            eagers.append(eager)

    return SplitArrayMeta(
        axis=axis,
        space=tuple(spaces),
        fft_dims=tuple(fft_dims),
        eager=tuple(eagers),
    )

def sum(
        x: FFTArray,
        *,
        dim: Optional[Union[str, Iterable[str]]] = None,
        dtype = None,
    ):

    if dim is None:
        return x.xp.sum(x.values(space=x.space), dtype=dtype)

    res_meta = _named_dims_to_axis(x, dim)

    reduced_values = x.xp.sum(x.values(space=x.space), axis=tuple(res_meta.axis), dtype=dtype)

    return FFTArray(
        values=reduced_values,
        space=res_meta.space,
        dims=res_meta.fft_dims,
        eager=res_meta.eager,
        factors_applied=(True,)*len(res_meta.fft_dims),
        xp=x.xp,
    )

def max(
        x: FFTArray,
        *,
        dim: Optional[Union[str, Iterable[str]]] = None,
    ):

    if dim is None:
        return x.xp.max(x.values(space=x.space))

    res_meta = _named_dims_to_axis(x, dim)

    reduced_values = x.xp.max(x.values(space=x.space), axis=tuple(res_meta.axis))

    return FFTArray(
        values=reduced_values,
        space=res_meta.space,
        dims=res_meta.fft_dims,
        eager=res_meta.eager,
        factors_applied=(True,)*len(res_meta.fft_dims),
        xp=x.xp,
    )

# def integrate(
#         x: FFTArray,
#         *,
#         dim: Optional[Union[str, Iterable[str]]] = None,
#         dtype = None,
#     ):
#     """
#         Does a simple rectangle rule integration.
#         Automatically uses the `d_pos` or `d_freq` of the integrated dimension
#         depending on the space the dimension is in at the time of integration.
#     """
#     if dim is None:
#         return x.xp.sum(x.values(space=x.space), dtype=dtype)

#     res_meta = _named_dims_to_axis(x, dim)

#     integration_element = 1.
#     for i in res_meta.axis:
#         space = x.space[i]
#         if space == "pos":
#             integration_element *= x.dims[i].d_pos
#         else:
#             assert space == "freq"
#             integration_element *= x.dims[i].d_freq

    # reduced_values = x.xp.sum(x.values(space=x.space), axis=tuple(res_meta.axis), dtype=dtype)
    # reduced_values *= integration_element

    # return FFTArray(
    #     values=reduced_values,
    #     space=res_meta.space,
    #     dims=res_meta.fft_dims,
    #     eager=res_meta.eager,
    #     factors_applied=(True,)*len(res_meta.fft_dims),
    #     xp=x.xp,
    # )
