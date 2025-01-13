import copy

import numpy as np
import pytest
import jax

import fftarray as fa
from tests.helpers import XPS

jax.config.update("jax_enable_x64", True)

def assert_scalars_almost_equal_nulp(x, y, nulp = 1):
    np.testing.assert_array_almost_equal_nulp(np.array([x]), np.array([y]), nulp = nulp)



def test_dim_accessors():
    """
    Test if the accessors of Dimension are defined and do not result in a
    contradiction.
    """
    sol = fa.dim("x",
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

def test_dim_jax():
    """
    Test if the pytree of Dimension is correctly defined, i.e., if the
    flattening and unflattening works accordingly.
    """
    @jax.jit
    def jax_func(dim: fa.Dimension):
        return dim

    dim = fa.dim("x",
        pos_min = 3e-6,
        d_pos = 1e-5,
        freq_min = 0.,
        n = 16,
    )
    assert jax_func(dim) == dim


@pytest.mark.parametrize("xp", XPS)
def test_arrays(xp) -> None:
    """
    Test that the manual arrays and the performance-optimized kernels create the same values in the supplied direction.
    """

    n = 16

    dim = fa.dim("x",
        pos_min = 3e-6,
        d_pos = 1e-5,
        freq_min = 0.,
        n = n,
    )

    pos_grid = fa.coords_from_dim(dim, "pos", xp=xp).np_array("pos")
    assert_scalars_almost_equal_nulp(dim.pos_min, np.min(pos_grid))
    assert_scalars_almost_equal_nulp(dim.pos_min, pos_grid[0])
    assert_scalars_almost_equal_nulp(dim.pos_max, np.max(pos_grid))
    assert_scalars_almost_equal_nulp(dim.pos_max, pos_grid[-1])
    assert_scalars_almost_equal_nulp(dim.pos_middle, pos_grid[int(n/2)])

    freq_grid = fa.coords_from_dim(dim, "freq", xp=xp).np_array("freq")
    assert_scalars_almost_equal_nulp(dim.freq_min, np.min(freq_grid))
    assert_scalars_almost_equal_nulp(dim.freq_min, freq_grid[0])
    assert_scalars_almost_equal_nulp(dim.freq_max, np.max(freq_grid))
    assert_scalars_almost_equal_nulp(dim.freq_max, freq_grid[-1])

def test_equality() -> None:
    dim_1 = fa.dim("x",
        pos_min = 3e-6,
        d_pos = 1e-5,
        freq_min = 0.,
        n = 8,
    )
    dim_2 = fa.dim("x",
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
    Test the tracing of a Dimension. The tracing behavior (dynamic/static)
    is determined by its property `dynamically_traced_coords` (False/True).

    If `dynamically_traced_coords=True`, `d_pos`, `pos_min` and `freq_min`
    should be jax-leaves.
    If `dynamically_traced_coords=False`, all properties should be static.

    Here, only the basics are tested, whether the Dimension properties can be
    changed within a jax.lax.scan step function.
    """

    dim_test = fa.dim("x",
        pos_min = 3e-6,
        d_pos = 1e-5,
        freq_min = 0.,
        n = 16,
        dynamically_traced_coords = dtc
    )

    def jax_step_func_static(dim: fa.Dimension, a):
        o = dim._n * dim._d_pos + a * dim._freq_min
        return dim, o

    def jax_step_func_dynamic(dim: fa.Dimension, a):
        dim._pos_min = dim._pos_min - a
        dim._d_pos = a*dim._d_pos
        dim._freq_min = dim._freq_min/a
        return dim, a

    def jax_step_func_forbidden(dim: fa.Dimension, a):
        dim._name = f"new{dim._name}"
        dim._n = dim._n + 1
        dim._dynamically_traced_coords = not dim._dynamically_traced_coords
        return dim, a

    # both (static and dynamic) should support this
    jax.lax.scan(jax_step_func_static, dim_test, jax.numpy.arange(3))

    if dtc:
        # dynamic
        jax.lax.scan(jax_step_func_dynamic, dim_test, jax.numpy.arange(3))
    else:
        # static
        with pytest.raises(jax.errors.UnexpectedTracerError):
            jax.lax.scan(jax_step_func_dynamic, dim_test, jax.numpy.arange(3))

    with pytest.raises(TypeError):
        jax.lax.scan(jax_step_func_forbidden, dim_test, jax.numpy.arange(3))
