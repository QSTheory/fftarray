from typing import Dict

import numpy as np

from fftarray import FFTArray
import fftarray as fa

# TODO: change names of FFTArray argument here
def shift_frequency(wf: FFTArray, offsets: Dict[str, float]) -> FFTArray:
    """Shift the wavefunction in frequency space:
    :math:`k_{x,y,z} \mapsto k_{x,y,z} - \Delta k_{x,y,z}`.
    The wavefunction is transformed according to:

    .. math::

        \Psi \\mapsto \Psi e^{i (x*\Delta k_x + y*\Delta k_y + z*\Delta k_z)}

    Parameters
    ----------
    wf : FFTWave
        The initial wavefunction.
    delta_kx : float, optional
        The frequency shift in x direction, by default 0.
    delta_ky : float, optional
        The frequency shift in y direction, by default 0.
    delta_kz : float, optional
        The frequency shift in z direction, by default 0.

    Returns
    -------
    FFTWave
        The wavefunction with shifted frequency space.
    """
    phase_shift = 1.
    dim_names = [dim.name for dim in wf.dims]
    for dim_name, offset in offsets.items():
        dim_idx = dim_names.index(dim_name)
        x = fa.array_from_dim(
            dim=wf.dims[dim_idx],
            space="pos",
            backend=wf.backend,
            eager=wf.eager[dim_idx],
        )
        phase_shift *= np.exp(1.j * 2.*np.pi * offset * x)
    return wf.into(space="pos") * phase_shift

def shift_position(wf: FFTArray, offsets: Dict[str, float]) -> FFTArray:
    """Shift the wavefunction in position space:
    :math:`x \mapsto x - \Delta x`. :math:`y` and :math:`z` analogously.
    The wavefunction is transformed according to:

    .. math::

        \Psi \\mapsto e^{-i (k_x*\Delta x + k_y*\Delta y + k_z*\Delta z)} \Psi

    Parameters
    ----------
    wf : FFTWave
        The initial wavefunction.
    delta_kx : float, optional
        The position shift in x direction, by default 0.
    delta_ky : float, optional
        The position shift in y direction, by default 0.
    delta_kz : float, optional
        The position shift in z direction, by default 0.

    Returns
    -------
    FFTWave
        The wavefunction with shifted position space.
    """
    phase_shift = 1.
    dim_names = [dim.name for dim in wf.dims]
    for dim_name, offset in offsets.items():
        dim_idx = dim_names.index(dim_name)
        f = fa.array_from_dim(
            dim=wf.dims[dim_idx],
            space="freq",
            backend=wf.backend,
            eager=wf.eager[dim_idx],
        )
        phase_shift *= np.exp(-1.j * offset * 2*np.pi * f)
    return wf.into(space="freq") * phase_shift

