from typing import Literal, Tuple, Optional, Any
from typing_extensions import assert_never
import sys

import numpy as np
import xarray as xr
import jax
import jax.numpy as jnp
import torch
from scipy.constants import hbar, Boltzmann
import pytest
import fftarray as fa

from tests.helpers import XPS, XPS_DEVICE_PAIRS


dt = 2.5e-3
mass_rb87: float = 86.909 * 1.66053906660e-27
omega_x = 0.5*2.*np.pi # angular freq. for desired ground state
n_steps: int = 10


@pytest.mark.parametrize(("xp, device_arg, device_res"), XPS_DEVICE_PAIRS)
@pytest.mark.parametrize("eager", [False, True])
def test_ground_state_device(
            xp,
            device_arg: Optional[Any],
            device_res: Optional[Any],
            eager: bool,
        ) -> None:
    V, k_sq = get_V_k2(
        precision="float64",
        eager=eager,
        xp=xp,
        device=device_arg,
    )
    psi = V*0.+1.
    for _ in range(n_steps):
        psi = split_step_imag(
            psi,
            dt=dt,
            mass=mass_rb87,
            V=V,
            k_sq=k_sq,
        )
    assert psi.device == device_res

@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_ground_state_direct_for(
            xp,
            eager: bool,
            precision: Literal["float32", "float64"],
        ) -> None:
    V, k_sq = get_V_k2(
        precision=precision,
        eager=eager,
        xp=xp,
    )
    psi = V*0.+1.
    for _ in range(n_steps):
        psi = split_step_imag(
            psi,
            dt=dt,
            mass=mass_rb87,
            V=V,
            k_sq=k_sq,
        )

    verify_result(
        psi=psi,
        precision=precision,
        eager=eager,
    )


# This test launches a lot of compiles which are by default guarded against.
# But in this case this is the intended behavior.
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.accumulated_cache_size_limit = 64

torch_devices = ["cpu"]
if torch.cuda.is_available():
    torch_devices.append("cuda")

@pytest.mark.slow
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("mode", [
    "default",
    "reduce-overhead",
    "max-autotune",
])
@pytest.mark.parametrize("device", torch_devices)
def test_ground_state_pytorch_compile_all(
            eager: bool,
            precision: Literal["float32", "float64"],
            mode: Literal["default", "reduce-overhead", "max-autotune"],
            device: Literal["cpu", "cuda"],
        ) -> None:
    check_ground_state_pytorch_compile(
        eager=eager,
        precision=precision,
        mode=mode,
        device=device,
    )

params = [
    (False, "float32", "default", "cpu"),
    (True, "float64", "max-autotune", "cuda"),
]
@pytest.mark.parametrize("eager, precision, mode, device", params)
def test_ground_state_pytorch_compile_base(
            eager: bool,
            precision: Literal["float32", "float64"],
            mode: Literal["default", "reduce-overhead", "max-autotune"],
            device: Literal["cpu", "cuda"],
        ) -> None:
    # We still want to run the second config on non-cuda machines.
    if not torch.cuda.is_available():
        device = "cpu"
    check_ground_state_pytorch_compile(
        eager=eager,
        precision=precision,
        mode=mode,
        device=device,
    )

def check_ground_state_pytorch_compile(
            eager: bool,
            precision: Literal["float32", "float64"],
            mode: Literal["default", "reduce-overhead", "max-autotune"],
            device: Literal["cpu", "cuda"],
        ) -> None:


    assert sys.version_info.major >= 3
    # FIXME:
    # torch.compile is broken in the versions set up by pixi
    # for python 3.10, 3.11 on cpu and cuda devices
    if sys.version_info.minor in [10, 11]:
        return
    # It is also broken in 3.12 on cuda devices
    if sys.version_info.minor == 12 and device == "cuda":
        return

    V, k_sq = get_V_k2(
        precision=precision,
        eager=eager,
        xp=torch,
        device=device,
    )

    psi = V*0.+1.

    step_fun = torch.compile(
        split_step_imag,
        fullgraph=True,
        mode=mode,
    )
    for _ in range(n_steps):
        if mode in ["reduce-overhead", "max-autotune"] and device == "cuda":
            # We cannot just simply rerun the graph.
            # See https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
            # This does not work for some reason, so we need to go the "clone-route".
            # torch.compiler.cudagraph_mark_step_begin()

            # Clone the array due to CUDA Graph
            psi = fa.Array(
                values=torch.clone(psi._values),
                dims=psi.dims,
                spaces=psi.spaces,
                eager=psi.eager,
                factors_applied=psi.factors_applied,
                xp=psi.xp,
            )
        psi = step_fun(
            psi=psi,
            dt=dt,
            mass=mass_rb87,
            V=V,
            k_sq=k_sq,
        )

    verify_result(
        psi=psi,
        precision=precision,
        eager=eager,
    )


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_ground_state_jax_scan(
            eager: bool,
            precision: Literal["float32", "float64"],
        ) -> None:
    V, k_sq = get_V_k2(
        precision=precision,
        eager=eager,
        xp=jnp,
    )

    def step_fun(psi: fa.Array, per_step) -> Tuple[fa.Array, None]:
        psi = split_step_imag(
            psi,
            dt=dt,
            mass=mass_rb87,
            V=V,
            k_sq=k_sq,
        )
        return psi, None


    psi = V*0.+1.
    psi, _ = jax.lax.scan(
        step_fun,
        init=psi.into_space("freq"),
        length=n_steps,
    )

    verify_result(
        psi=psi,
        precision=precision,
        eager=eager,
    )

def normalize(psi: fa.Array) -> fa.Array:
    state_norm = fa.integrate(fa.abs(psi)**2)
    return psi * fa.sqrt(1./state_norm)

def expectation_value(psi: fa.Array, op: fa.Array) -> fa.Array:
    return fa.integrate((fa.abs(psi)**2)*op)

def split_step_imag(psi: fa.Array, *,
               dt: float,
               mass: float,
               V: fa.Array,
               k_sq: fa.Array
            ) -> fa.Array:

    psi = psi.into_space("freq") * fa.exp((-0.5 * dt * hbar / (2*mass)) * k_sq)
    psi = psi.into_space("pos") * fa.exp((-1. / hbar * dt) * V)
    psi = psi.into_space("freq") * fa.exp((-0.5 * dt * hbar / (2*mass)) * k_sq)

    psi = normalize(psi)

    return psi

def get_V_k2(
            xp,
            precision: Literal["float32", "float64"],
            eager: bool,
            device: Optional[Any] = None,
        ) -> Tuple[fa.Array, fa.Array]:
    dims = [
        fa.dim_from_constraints("x",
            pos_min=-100e-6,
            pos_max=100e-6,
            freq_middle=0.,
            n=256,
            dynamically_traced_coords=False,
        ),
        fa.dim_from_constraints("y",
            pos_min=-100e-6,
            pos_max=100e-6,
            freq_middle=0.,
            n=256,
            dynamically_traced_coords=False,
        ),
    ]

    V = fa.array(0., [], "pos", xp=xp, dtype=getattr(xp, precision), device=device).into_eager(eager)
    k_sq = fa.array(0., [], "pos", xp=xp, dtype=getattr(xp, precision), device=device).into_eager(eager)
    for dim in dims:
        x = fa.coords_from_dim(dim, "pos", xp=xp, dtype=getattr(xp, precision), device=device).into_eager(eager)
        f = fa.coords_from_dim(dim, "freq", xp=xp, dtype=getattr(xp, precision), device=device).into_eager(eager)
        V += 0.5 * mass_rb87 * omega_x**2. * x**2
        k_sq += (2*np.pi*f)**2.

    return V, k_sq


def get_ref() -> xr.Dataset:
    """
        Reference values from a tested implementation.
    """
    return xr.Dataset({
        "energy_diff_scaled": xr.DataArray(
            np.array(
                [
                    [4.9751377709625135e+00, 4.9751348504438990e+00],
                    [4.9751366970314050e+00, 4.9751348504438990e+00]
                ],
                dtype=np.float64,
            ),
            dims=('eager', 'precision'),
            coords={
                "eager": xr.DataArray([False, True], dims="eager"),
                "precision": xr.DataArray(["float32", "float64"], dims="precision"),
            }
        ),
        "infidelity": xr.DataArray(
            np.array(
                [
                    [4.7610294270187559e-01, 4.7610282020159933e-01],
                    [4.7610291551765760e-01, 4.7610282020159933e-01]
                ],
                dtype=np.float64,
            ),
            dims=('eager', 'precision'),
            coords={
                "eager": xr.DataArray([False, True], dims="eager"),
                "precision": xr.DataArray(["float32", "float64"], dims="precision"),
            }
        )
    })

def verify_result(
                psi: fa.Array,
                eager: bool,
                precision: Literal["float32", "float64"],
        ) -> None:
    """
        Test that the passed in psi has the expected energy and infidelity.
        This is a regression and change test.
        It does not matter whether it is better or worse compared to the analytical solution.
        Any deviation from the expected value is a fail.
        This test checks that the overall mathematics including tracing is sound
        for different modes and array API namespaces.
        Directly comparing with the analytical solution would require too many resources for CI.
        Therefore this test hard codes the total energy and infidelity relative to the ground state
        that should be reached after 10 time steps.
        These reference values come from a solution that was more extensively checked.
        And this is a regression test for all the moving parts.
    """
    ref_metrics = get_ref()

    # Note: computing the norm with jax in float64 causes the energy threshold below to narrowly fail.
    psi = normalize(psi.into_eager(False).into_dtype(psi.xp.complex128).into_xp(np))
    psi_pos = psi.into_space("pos")

    psi_ref = 1.
    for dim in psi.dims:
        coords = fa.coords_from_dim(dim, "pos", xp=psi.xp, dtype=psi.xp.float64)
        psi_ref *= (mass_rb87 * omega_x / (np.pi*hbar))**(1./4.) * fa.exp(-(mass_rb87 * omega_x * (coords**2.)/(2.*hbar)))

    V, k_sq = get_V_k2(xp=psi.xp, precision="float64", eager=False)

    post_factor = hbar**2/(2*mass_rb87)
    post_factor /= (Boltzmann * 1e-6)
    E_kin = expectation_value(psi, k_sq) * post_factor
    E_pot = expectation_value(psi_pos, V) / (Boltzmann * 1e-6)
    E_tot = float((E_kin + E_pot).values([]))
    E_tot_analytical = hbar*omega_x*(len(psi.dims)/2.) / (Boltzmann * 1e-6)
    E_diff_scaled = np.abs(E_tot-E_tot_analytical)/E_tot_analytical
    E_diff_scaled_ref = ref_metrics.energy_diff_scaled.sel(eager=eager, precision=precision).item()
    E_diff_from_ref = np.abs(E_diff_scaled-E_diff_scaled_ref)

    fidelity = fa.abs(fa.integrate(psi_ref*psi_pos))
    infidelity = float((1. - fidelity).values([]))
    infidelity_ref = ref_metrics.infidelity.sel(eager=eager, precision=precision).item()
    infidelity_diff_from_ref = np.abs(infidelity-infidelity_ref)

    match precision:
        case "float32":
            assert E_diff_from_ref < 5e-6
            assert infidelity_diff_from_ref < 5e-6
        case "float64":
            assert E_diff_from_ref < 1e-14
            assert infidelity_diff_from_ref < 1e-14
        case _:
            assert_never(precision)

