from typing import Dict

import numpy as np

from fftarray import Array
import fftarray as fa

def shift_freq(x: Array, offsets: Dict[str, float]) -> Array:
    """Cyclically shift the Array in frequency space:
    :math:`k_{x,y,z} \\mapsto k_{x,y,z} - \\Delta k_{x,y,z}`.
    The Array is transformed according to:

    .. math::

        \\Psi \\mapsto \\Psi e^{i (x*\\Delta k_x + y*\\Delta k_y + z*\\Delta k_z)}

    This operation does not change the domain, it only shifts the values.


    Parameters
    ----------
    x : Array
        The initial Array.
    delta_kx : float, optional
        The frequency shift in x direction, by default 0.
    delta_ky : float, optional
        The frequency shift in y direction, by default 0.
    delta_kz : float, optional
        The frequency shift in z direction, by default 0.

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
    """Cyclically shift the Array in position space:
    :math:`x \\mapsto x - \\Delta x`. :math:`y` and :math:`z` analogously.
    The Array is transformed according to:

    .. math::

        \\Psi \\mapsto e^{-i (k_x*\\Delta x + k_y*\\Delta y + k_z*\\Delta z)} \\Psi

    This operation does not change the domain, it only shifts the values.

    Parameters
    ----------
    x : Array
        The initial Array.
    delta_kx : float, optional
        The position shift in x direction, by default 0.
    delta_ky : float, optional
        The position shift in y direction, by default 0.
    delta_kz : float, optional
        The position shift in z direction, by default 0.

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

