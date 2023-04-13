import pytest
import numpy as np
import jax

from fftarray.fft_array import FFTDimension, LazyState, PhaseFactors
from fftarray.backends.jax_backend import JaxTensorLib
from fftarray.backends.np_backend import NumpyTensorLib
from fftarray.backends.pyfftw_backend import PyFFTWTensorLib
from fftarray.xr_helpers import as_xr_pos

jax.config.update("jax_enable_x64", True)

def assert_scalars_almost_equal_nulp(x, y, nulp = 1):
    np.testing.assert_array_almost_equal_nulp(np.array([x]), np.array([y]), nulp = nulp)

tensor_libs = [NumpyTensorLib, JaxTensorLib, PyFFTWTensorLib]

@pytest.mark.parametrize("tlib", tensor_libs)
@pytest.mark.parametrize("do_jit", [False, True])
def test_indexing(tlib, do_jit: bool):
    if do_jit and type(tlib) != JaxTensorLib:
        return

    x_dim = FFTDimension("x",
        n=4,
        d_pos=1,
        pos_min=0.5,
        freq_middle=0.,
        default_tlib=tlib(precision="default"),
    )
    y_dim = FFTDimension("y",
        n=4,
        d_pos=2,
        pos_min=-2,
        freq_middle=0.,
        default_tlib=tlib(precision="default"),
    )

    arr_2d = x_dim.pos_array() + y_dim.pos_array()**2
    xr_arr = as_xr_pos(arr_2d)

    assert x_dim._index_from_coord(0.5, method = None, space="pos") == 0
    assert x_dim._index_from_coord(2.5, method = None, space="pos") == 2
    assert x_dim._index_from_coord(0.4, method = "nearest", space="pos") == 0
    assert x_dim._index_from_coord(2.6, method = "nearest", space="pos") == 2


    assert np.array_equal(arr_2d.values[0:3:2,:], xr_arr.values[0:3:2,:])
    assert np.array_equal(
        arr_2d.isel(x=1,y=slice(0,2,None)).transpose("x", "y"),
        xr_arr.isel(x=1,y=slice(0,2,None)).expand_dims({"x": 1}).transpose("x", "y")
    )
    assert np.array_equal(
        arr_2d.sel(x=(1,3),y=3.4, method="nearest").transpose("x", "y"),
        xr_arr.sel(y=3.4, method="nearest")
            .where(xr_arr.x > 1, drop=True)
            .where(xr_arr.x < 3, drop=True)
            .expand_dims({"y": 1}).transpose("x", "y")
    )

    assert np.array_equal(
        arr_2d.loc[:, 0].transpose("x", "y").values,
        xr_arr.loc[:,0].expand_dims({"y": 1}).transpose("x", "y")
    )
    assert np.array_equal(
        arr_2d.loc[(1,3), 2].transpose("x", "y").values,
        xr_arr.sel(y=2)
            .where(xr_arr.x > 1, drop=True)
            .where(xr_arr.x < 3, drop=True)
            .expand_dims({"y": 1}).transpose("x", "y")
    )

    def test_jittable(x_dim, arr_2d):
        return (
            x_dim._index_from_coord(0.4, method = "nearest", space="pos"),
            x_dim._index_from_coord(2.6, method = "nearest", space="pos"),
            arr_2d.sel(x=1,y=3.4, method="nearest"),
            arr_2d.sel(x=-100,y=3.4, method="nearest"),
            arr_2d.loc[:],
            arr_2d.isel(x=3, y=2),
        )
    if do_jit:
        test_jittable = jax.jit(test_jittable)

    jit_res = test_jittable(x_dim=x_dim, arr_2d=arr_2d)
    assert jit_res[0] == 0
    assert jit_res[1] == 2
    assert np.array_equal(jit_res[2], xr_arr.sel(x=1,y=3.4, method="nearest").expand_dims({"x": 1, "y": 1}))
    assert np.array_equal(jit_res[3], xr_arr.sel(x=-100,y=3.4, method="nearest").expand_dims({"x": 1, "y": 1}))
    assert np.array_equal(jit_res[4], xr_arr.loc[:])
    assert np.array_equal(jit_res[5], xr_arr.isel(x=3, y=2).expand_dims({"x": 1, "y": 1}))


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("tensor_lib", tensor_libs)
@pytest.mark.parametrize("precision", ("fp32", "fp64", "default"))
@pytest.mark.parametrize("override", (None, "fp32", "fp64", "default"))
@pytest.mark.parametrize("eager", [False, True])
def test_dtype(tensor_lib, precision, override, eager: bool):
    tlib = tensor_lib(precision=precision)
    tlib_override = tensor_lib(precision=override)
    x_dim = FFTDimension("x",
        n=4,
        d_pos=1,
        pos_min=0.,
        freq_middle=0.,
        default_tlib=tlib,
        default_eager=eager,
    )

    if override is None:
        assert x_dim.pos_array().values.dtype == tlib.real_type
    else:
        assert x_dim.pos_array(tlib_override).values.dtype == tlib_override.real_type
        assert x_dim.pos_array().with_tlib(tlib_override).values.dtype == tlib_override.real_type


    if override is None:
        assert x_dim.freq_array().values.dtype == tlib.real_type
    else:
        assert x_dim.freq_array(tlib=tlib_override).values.dtype == tlib_override.real_type
        assert x_dim.freq_array().with_tlib(tlib=tlib_override).values.dtype == tlib_override.real_type

    assert x_dim.pos_array().freq_array().values.dtype == tlib.complex_type
    assert x_dim.freq_array().pos_array().values.dtype == tlib.complex_type

    assert np.abs(x_dim.pos_array().freq_array()).values.dtype == tlib.real_type # type: ignore
    assert np.abs(x_dim.freq_array().pos_array()).values.dtype == tlib.real_type # type: ignore

    if override is not None:
        assert x_dim.pos_array().freq_array(tlib=tlib_override).values.dtype == tlib_override.complex_type
        assert x_dim.freq_array().pos_array(tlib=tlib_override).values.dtype == tlib_override.complex_type


@pytest.mark.parametrize("tensor_lib", tensor_libs)
@pytest.mark.parametrize("override", tensor_libs)
def test_backend_override(tensor_lib, override):
    x_dim = FFTDimension("x",
        n=4,
        d_pos=1,
        pos_min=0.,
        freq_middle=0.,
        default_tlib=tensor_lib(),
    )

    x_dim_override = FFTDimension("x",
        n=4,
        d_pos=1,
        pos_min=0.,
        freq_middle=0.,
        default_tlib=override(),
    )

    assert type(x_dim.pos_array(tlib=override()).values) == type(x_dim_override.pos_array().values)
    assert type(x_dim.freq_array(tlib=override()).values) == type(x_dim_override.freq_array().values)
    assert type(x_dim.pos_array(tlib=override()).freq_array().values) == type(x_dim_override.freq_array().values)
    assert type(x_dim.freq_array(tlib=override()).pos_array().values) == type(x_dim_override.freq_array().values)

    assert type(x_dim.pos_array().with_tlib(tlib=override()).values) == type(x_dim_override.pos_array().values)
    assert type(x_dim.freq_array().with_tlib(tlib=override()).values) == type(x_dim_override.freq_array().values)
    assert type(x_dim.pos_array().with_tlib(tlib=override()).freq_array().values) == type(x_dim_override.freq_array().values)
    assert type(x_dim.freq_array().with_tlib(tlib=override()).pos_array().values) == type(x_dim_override.freq_array().values)

    assert type(x_dim.pos_array().freq_array(tlib=override()).values) == type(x_dim_override.freq_array().values)
    assert type(x_dim.freq_array().pos_array(tlib=override()).values) == type(x_dim_override.freq_array().values)


def test_broadcasting(nulp: int = 1) -> None:
    x_dim = FFTDimension("x", n=4, d_pos=1, pos_min=0., freq_middle=0.)
    y_dim = FFTDimension("y", n=8, d_pos=1, pos_min=0., freq_middle=0.)

    x_ref = np.arange(0., 4.)
    y_ref = np.arange(0., 8.)
    np.testing.assert_array_almost_equal_nulp(np.array(x_dim.pos_array()), x_ref, nulp = 0)
    np.testing.assert_array_almost_equal_nulp(np.array(y_dim.pos_array()), y_ref, nulp = 0)

    x_ref_broadcast = x_ref.reshape(1,-1)
    y_ref_broadcast = y_ref.reshape(-1,1)
    np.testing.assert_array_almost_equal_nulp((x_dim.pos_array() + y_dim.pos_array()).transpose("x", "y").values, (x_ref_broadcast+y_ref_broadcast).transpose(), nulp = 0)
    np.testing.assert_array_almost_equal_nulp((x_dim.pos_array() + y_dim.pos_array()).transpose("y", "x").values, x_ref_broadcast+y_ref_broadcast, nulp = 0)

def assert_special_fun_equivalence(arr_lazy, arr_ref, eager: bool):
    np.testing.assert_array_almost_equal(arr_lazy.values, arr_ref)
    if not eager:
        np.testing.assert_array_almost_equal(arr_lazy._set_lazy_state(LazyState()), arr_ref)
    np.testing.assert_array_almost_equal(np.abs(arr_lazy).values, np.abs(arr_ref))

    np.testing.assert_array_almost_equal((arr_lazy*arr_lazy).values, arr_ref*arr_ref)
    if not eager:
        np.testing.assert_array_almost_equal((arr_lazy*arr_lazy)._set_lazy_state(LazyState()), arr_ref*arr_ref)
    np.testing.assert_array_almost_equal(np.abs(arr_lazy*arr_lazy).values, np.abs(arr_ref*arr_ref))

@pytest.mark.parametrize("eager", [False, True])
def test_lazy(eager: bool) -> None:
    dim_pos_x = FFTDimension("x", n = 4, d_pos = 1., pos_min = 0.3, freq_min = 0.7, default_eager=eager)
    dim_pos_y = FFTDimension("y", n = 4, d_pos = 1., pos_min = 1.3, freq_min = 1.7, default_eager=eager)
    dim_freq_x = FFTDimension("x", n = 4, d_freq = 1., pos_min = 0.7, freq_min = 0.3, default_eager=eager)
    dim_freq_y = FFTDimension("y", n = 4, d_freq = 1., pos_min = 1.7, freq_min = 1.3, default_eager=eager)

    ref_values = np.arange(4).reshape(4,1)+0.3 + np.arange(4).reshape(1,4)+1.3
    arrs = [
        (dim_pos_x.pos_array() + dim_pos_y.pos_array()).transpose("x", "y"),
        (dim_freq_x.freq_array() + dim_freq_y.freq_array()).transpose("x", "y"),
    ]
    for arr in arrs:
        np.testing.assert_array_almost_equal(arr.freq_array().pos_array().freq_array().values, arr.freq_array().values)
        np.testing.assert_array_almost_equal(arr.values, ref_values)

        ref_scaled = 2*ref_values

        arr_lazy = arr.add_scale(2.)
        if not eager:
            assert arr_lazy._lazy_state == LazyState(scale = 2.)
        assert_special_fun_equivalence(arr_lazy, ref_scaled, eager)

        # This feature is currently commented out due to some problems with jax tracing
        # and it is also questionable whether it is a good design at all.
        # arr_lazy = 2. * arr
        # assert arr_lazy._lazy_state == LazyState(scale = 2.)
        # assert_special_fun_equivalence(arr_lazy, ref_scaled)

        for order in [0,1,2,3]:
            phases_x = np.zeros(4)
            phases_x[order] = 0.3
            phases_y = np.zeros(4)
            phases_y[order] = 0.9
            arr_lazy = arr
            arr_lazy = arr_lazy.add_phase_factor("x", "a", PhaseFactors({i: phase for i, phase in enumerate(list(phases_x))}))
            arr_lazy = arr_lazy.add_phase_factor("y", "a", PhaseFactors({i: phase for i, phase in enumerate(list(phases_y))}))

            ref_shifted = ref_values
            ref_shifted = ref_shifted * np.exp(1.j * 0.3 * np.arange(4).reshape(-1,1)**order)
            ref_shifted = ref_shifted * np.exp(1.j * 0.9 * np.arange(4).reshape(1,-1)**order)

            assert_special_fun_equivalence(arr_lazy, ref_shifted, eager)
