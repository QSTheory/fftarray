from .fft_array import FFTArray

import numpy as np
import xarray as xr

#--------------------
# XArray interoperability
#--------------------
def as_xr_pos(arr: FFTArray) -> xr.DataArray:
    return xr.DataArray(
        np.array(arr.pos_array()),
        coords = {dim.name: np.array(dim.pos_array()) for dim in arr.dims},
        # TODO These in the attributes somehow crash where with a pickle error.
        # attrs = _xr_attribs(arr),
    )

def as_xr_freq(arr: FFTArray) -> xr.DataArray:
    return xr.DataArray(
        np.array(arr.freq_array()),
        coords = {dim.name: np.array(dim.freq_array()) for dim in arr.dims},
        # attrs = _xr_attribs(arr),
    )

def as_xr_dataset(arr: FFTArray) -> xr.Dataset:
    return xr.Dataset({
            "pos": xr.DataArray(
                np.array(arr.pos_array()),
                coords = {f"{dim.name}_pos": np.array(dim.pos_array()) for dim in arr.dims}
            ),
            "freq":  xr.DataArray(
                np.array(arr.freq_array()),
                coords = {f"{dim.name}_freq": np.array(dim.freq_array()) for dim in arr.dims}
            ),
        },
        # attrs = _xr_attribs(arr),
    )

def from_xr(arr: xr.DataArray):
    raise NotImplementedError

def _xr_attribs(arr):
    return {f"{dim.name}_fft_dim": dim for dim in arr.dims}
