from typing import Optional

from .backends.backend import Backend
from .fft_dimension import FFTDimension
from .fft_array import FFTArray, get_default_backend, get_default_eager
from .space import Space

def array_from_dim(
        dim: FFTDimension,
        space: Space,
        *,
        backend: Optional[Backend] = None,
        eager: Optional[bool] = None,
    ) -> FFTArray:
    """..

    Parameters
    ----------
    dim : FFTDimension
        The dimension from which the coordinate grid should be created.
    space : Space
        Specify the space of the coordinates and in which space the returned FFTArray is intialized.
    backend : Optional[Backend]
        The backend to use for the returned FFTArray.  `None` uses default `NumpyBackend("default")` which can be globally changed.
    eager :  Optional[bool]
        The eager-mode to use for the returned FFTArray.  `None` uses default `False` which can be globally changed.

    Returns
    -------
    FFTArray
        The grid coordinates of the chosen space packed into an FFTArray with self as only dimension.

    See Also
    --------
        set_default_backend, get_default_backend
        set_default_eager, get_default_eager
    """

    if backend is None:
        backend = get_default_backend()

    if eager is None:
        eager = get_default_eager()

    values = dim._raw_coord_array(
        backend=backend,
        space=space,
    )

    return FFTArray(
        values=values,
        dims=[dim],
        eager=eager,
        factors_applied=True,
        space=space,
        backend=backend,
    )
