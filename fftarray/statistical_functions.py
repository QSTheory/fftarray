from typing import Iterable, Optional, Union, Tuple, List

from fftarray.fft_dimension import FFTDimension

from .fft_array import FFTArray
from .space import Space

def sum(
        x: FFTArray,
        *,
        dim: Optional[Union[str, Iterable[str]]] = None,
        dtype = None,
    ):

    if dim is None:
        return x.backend.numpy_ufuncs.sum(x.values(space=x.space), dtype=dtype)

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

    reduced_values = x.backend.numpy_ufuncs.sum(x.values(space=x.space), axis=tuple(axis), dtype=dtype)

    return FFTArray(
        values=reduced_values,
        space=spaces,
        dims=fft_dims,
        eager=eagers,
        factors_applied=True,
        backend=x.backend,
    )

