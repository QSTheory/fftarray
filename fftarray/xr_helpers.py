import numpy as np
import xarray as xr

from .fft_array import FFTArray

#--------------------
# XArray interoperability
#--------------------

def as_xr_pos(arr: FFTArray) -> xr.DataArray:
    return xr.DataArray(
        np.array(arr.into(space="pos")),
        coords = {dim.name: dim.np_array(space="pos") for dim in arr.dims},
        # TODO These in the attributes somehow crash where with a pickle error.
        # attrs = _xr_attribs(arr),
    )

def as_xr_freq(arr: FFTArray) -> xr.DataArray:
    return xr.DataArray(
        np.array(arr.into(space="freq")),
        coords = {dim.name: dim.np_array(space="freq") for dim in arr.dims},
        # attrs = _xr_attribs(arr),
    )

def as_xr_dataset(arr: FFTArray) -> xr.Dataset:
    return xr.Dataset({
            "pos": xr.DataArray(
                np.array(arr.into(space="pos")),
                coords = {
                    f"{dim.name}_pos": dim.np_array(space="pos")
                    for dim in arr.dims
                }
            ),
            "freq":  xr.DataArray(
                np.array(arr.into(space="freq")),
                coords = {
                    f"{dim.name}_freq": dim.np_array(space="freq")
                    for dim in arr.dims
                }
            ),
        },
        # attrs = _xr_attribs(arr),
    )

def from_xr(arr: xr.DataArray):
    raise NotImplementedError

def _xr_attribs(arr):
    return {f"{dim.name}_fft_dim": dim for dim in arr.dims}
