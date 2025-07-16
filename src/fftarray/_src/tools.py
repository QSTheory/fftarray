from typing import Dict

import numpy as np

from fftarray import Array
import fftarray as fa

def shift_freq(x: Array, offsets: Dict[str, float], /) -> Array:
    """Cyclically shift the Array by a frequency offset via multiplication with a phase factor in position space:
    :math:`f \\mapsto f - \\text{offsets}`.

    This operation does not change the domain, it only shifts the values.


    Parameters
    ----------
    x:
        The initial Array.
    offsets:
        The frequency shift for each shifted dimension by name.

    Returns
    -------
    Array
        The Array with its contents shifted in frequency space.
    """
    if not x.xp.isdtype(x.dtype, ("real floating", "complex floating")):
        raise ValueError(
            f"'shift_freq' requires an Array with a float or complex dtype, but got passed array of type '{x.dtype}'. "
            + "The float or complex dtype is required because the values are shifted by multiplication with a complex phase "
            + "which only makes sense with float values."
        )
    phase_shift = fa.full([], [], 1., xp=x.xp, dtype=x.dtype)
    for dim_name, offset in offsets.items():
        x_arr = fa.coords_from_arr(x, dim_name, "pos").into_dtype("complex")
        phase_shift = phase_shift * fa.exp(1.j * offset * 2*np.pi * x_arr)
    return x.into_space("pos") * phase_shift

def shift_pos(x: Array, offsets: Dict[str, float]) -> Array:
    """Cyclically shift the Array by a position offset via multiplication with a phase factor in frequency space:
    :math:`x \\mapsto x - \\text{offsets}`.


    This operation does not change the domain, it only shifts the values.

    Parameters
    ----------
    x : Array
        The initial Array.
    offsets:
        The position shift for each shifted dimension by name.

    Returns
    -------
    Array
        The Array with its contents shifted in position space.
    """
    if not x.xp.isdtype(x.dtype, ("real floating", "complex floating")):
        raise ValueError(
            f"'shift_pos' requires an Array with a float or complex dtype, but got passed array of type '{x.dtype}'. "
            + "The float or complex dtype is required because the values are shifted by multiplication with a complex phase "
            + "which only makes sense with float values."
        )

    phase_shift = fa.full([], [], 1., xp=x.xp, dtype=x.dtype)
    for dim_name, offset in offsets.items():
        f_arr = fa.coords_from_arr(x, dim_name, "freq").into_dtype("complex")
        phase_shift = phase_shift * fa.exp(-1.j * offset * 2*np.pi * f_arr)
    return x.into_space("freq") * phase_shift

