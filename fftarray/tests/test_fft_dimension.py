import copy

import numpy as np
import pytest
import jax

from fftarray import FFTDimension
from fftarray.backends.jax_backend import JaxTensorLib
from fftarray.backends.np_backend import NumpyTensorLib
from fftarray.backends.pyfftw_backend import PyFFTWTensorLib

jax.config.update("jax_enable_x64", True)

def assert_scalars_almost_equal_nulp(x, y, nulp = 1):
    np.testing.assert_array_almost_equal_nulp(np.array([x]), np.array([y]), nulp = nulp)

tensor_libs = [NumpyTensorLib(precision="fp64"), JaxTensorLib(precision="fp64"), PyFFTWTensorLib(precision="fp64")]


def test_fftdim_accessors():
    """
    Test if the accessors of FFTDimension are defined and do not result in a
    contradiction.
    """
    sol = FFTDimension("x",
        pos_min = 3e-6,
        d_pos = 1e-5,
        freq_min = 0.,
        n = 16,
    )
    assert np.isclose(sol.d_pos * sol.d_freq * sol.n, 1.)
    assert np.isclose(sol.pos_middle, sol.pos_min + sol.d_pos * sol.n/2)
    assert np.isclose(sol.pos_extent, sol.pos_max - sol.pos_min)
    assert np.isclose(sol.pos_extent, sol.d_pos * (sol.n - 1.))
    assert np.isclose(sol.freq_middle, sol.freq_min + sol.d_freq * sol.n/2)
    assert np.isclose(sol.freq_extent, sol.freq_max - sol.freq_min)
    assert np.isclose(sol.freq_extent, sol.d_freq * (sol.n - 1.))

def test_fftdim_jax():
    """
    Test if the pytree of FFTDimension is correctly defined, i.e., if the
    flattening and unflattening works accordingly.
    """
    @jax.jit
    def jax_func(fftdim: FFTDimension):
        return fftdim

    fftdim = FFTDimension("x",
        pos_min = 3e-6,
        d_pos = 1e-5,
        freq_min = 0.,
        n = 16,
    )
    assert jax_func(fftdim) == fftdim


@pytest.mark.parametrize("tensor_lib", tensor_libs)
def test_arrays(tensor_lib) -> None:
    """
    Test that the manual arrays and the performance-optimized kernels create the same values in the supplied direction.
    """

    n = 16

    fftdim = FFTDimension("x",
        pos_min = 3e-6,
        d_pos = 1e-5,
        freq_min = 0.,
        n = n,
    )

    pos_grid = np.array(fftdim.fft_array(tensor_lib, space="pos"))
    assert_scalars_almost_equal_nulp(fftdim.pos_min, np.min(pos_grid))
    assert_scalars_almost_equal_nulp(fftdim.pos_min, pos_grid[0])
    assert_scalars_almost_equal_nulp(fftdim.pos_max, np.max(pos_grid))
    assert_scalars_almost_equal_nulp(fftdim.pos_max, pos_grid[-1])
    assert_scalars_almost_equal_nulp(fftdim.pos_middle, pos_grid[int(n/2)])

    freq_grid = np.array(fftdim.fft_array(tensor_lib, space="freq"))
    assert_scalars_almost_equal_nulp(fftdim.freq_min, np.min(freq_grid))
    assert_scalars_almost_equal_nulp(fftdim.freq_min, freq_grid[0])
    assert_scalars_almost_equal_nulp(fftdim.freq_max, np.max(freq_grid))
    assert_scalars_almost_equal_nulp(fftdim.freq_max, freq_grid[-1])

def test_equality() -> None:
    dim_1 = FFTDimension("x",
        pos_min = 3e-6,
        d_pos = 1e-5,
        freq_min = 0.,
        n = 8,
    )
    dim_2 = FFTDimension("x",
        pos_min = 2e-6,
        d_pos = 1e-5,
        freq_min = 0.,
        n = 8,
    )
    assert dim_1 != dim_2
    assert dim_1 == dim_1
    assert dim_1 == copy.copy(dim_1)

@pytest.mark.parametrize("dtc", [True, False])
def test_dynamically_traced_coords(dtc: bool) -> None:
    """
    Test the tracing of an FFTDimension. The tracing behavior (dynamic/static)
    is determined by its property `dynamically_traced_coords` (False/True).

    If `dynamically_traced_coords=True`, `d_pos`, `pos_min` and `freq_min`
    should be jax-leaves.
    If `dynamically_traced_coords=True`, all properties should be static.

    Here, only the basics are tested, whether the FFTDimension properties can be
    change within a jax.lax.scan step function.
    """

    fftdim_test = FFTDimension("x",
        pos_min = 3e-6,
        d_pos = 1e-5,
        freq_min = 0.,
        n = 16,
        dynamically_traced_coords = dtc
    )

    def jax_step_func_static(fftdim: FFTDimension, a):
        o = fftdim._n * fftdim._d_pos + a * fftdim._freq_min
        return fftdim, o

    def jax_step_func_dynamic(fftdim: FFTDimension, a):
        fftdim._pos_min = fftdim._pos_min - a
        fftdim._d_pos = a*fftdim._d_pos
        fftdim._freq_min = fftdim._freq_min/a
        return fftdim, None

    def jax_step_func_forbidden(fftdim: FFTDimension, a):
        fftdim._name = f"new{fftdim._name}"
        fftdim._n = fftdim._n + 1
        fftdim._dynamically_traced_coords = not fftdim._dynamically_traced_coords
        return fftdim, None

    # both (static and dynamic) should support this
    jax.lax.scan(jax_step_func_static, fftdim_test, jax.numpy.arange(3))

    if dtc:
        # dynamic
        jax.lax.scan(jax_step_func_dynamic, fftdim_test, jax.numpy.arange(3))
    else:
        # static
        with pytest.raises(jax.errors.UnexpectedTracerError):
            jax.lax.scan(jax_step_func_dynamic, fftdim_test, jax.numpy.arange(3))

    with pytest.raises(TypeError):
        jax.lax.scan(jax_step_func_forbidden, fftdim_test, jax.numpy.arange(3))
