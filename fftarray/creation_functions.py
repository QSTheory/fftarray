from typing import Optional, Union, Iterable, Tuple, get_args

from .backends.backend import Backend
from .fft_dimension import FFTDimension
from .fft_array import FFTArray
from ._utils.defaults import get_default_backend, get_default_eager
from .space import Space
from ._utils.helpers import norm_param

def array(
        values,
        dims: Union[FFTDimension, Iterable[FFTDimension]],
        space: Union[Space, Iterable[Space]],
        *,
        backend: Optional[Backend] = None,
        eager: Optional[Union[bool, Iterable[bool]]] = None,
        factors_applied: Union[bool, Iterable[bool]] = True,
    ) -> FFTArray:
    """
        Construct a new instance of FFTArray from raw values.

        Parameters
        ----------
        values :
            The values to initialize the `FFTArray` with.
            They are converted and copied into an array of the backend.
        dims : Iterable[FFTDimension]
            The FFTDimensions for each dimension of the passed in values.
        space: Union[Space, Iterable[Space]]
            Specify the space of the coordinates and in which space the returned FFTArray is intialized.
        backend: Optional[Backend]
            The backend to use for the returned FFTArray.  `None` uses default `NumpyBackend("default")` which can be globally changed.
            The values are transformed into the appropiate type defined by the backend.
        eager: Union[bool, Iterable[bool]]
            The eager-mode to use for the returned FFTArray.  `None` uses default `False` which can be globally changed.
        factors_applied: Union[bool, Iterable[bool]]
            Whether the fft-factors are applied are already applied for the various dimensions.
            For external values this is usually `True` since `False` assumes the internal (and unstable)
            factors-format.

        Returns
        -------
        FFTArray
            The grid coordinates of the chosen space packed into an FFTArray with self as only dimension.

        See Also
        --------
        set_default_backend, get_default_backend
        set_default_eager, get_default_eager
        fft_array
    """

    if backend is None:
        backend = get_default_backend()

    if eager is None:
        eager = get_default_eager()

    if isinstance(dims, FFTDimension):
        dims_tuple: Tuple[FFTDimension, ...] = (dims,)
    else:
        dims_tuple = tuple(dims)

    n_dims = len(dims_tuple)
    inner_values = backend.array(values)
    spaces_normalized: Tuple[Space, ...] = norm_param(space, n_dims, str)
    for sub_space in spaces_normalized:
        assert sub_space in get_args(Space)

    arr = FFTArray(
        dims=dims_tuple,
        values=inner_values,
        space=spaces_normalized,
        eager=norm_param(eager, n_dims, bool),
        factors_applied=norm_param(factors_applied, n_dims, bool),
        backend=backend,
    )
    arr._check_consistency()
    return arr

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
        dims=(dim,),
        eager=(eager,),
        factors_applied=(True,),
        space=(space,),
        backend=backend,
    )
