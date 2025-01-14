"""
This example demonstrates the calculation of the ground state of a dual-species Bose-Einstein condensate.
We use the Gross-Pitaevskii equation (GPE) to describe the dynamics of the two wavefunctions.
We perform an imaginary time evolution using the split-step method to find the ground state.
For further details and for the specific configuration used in this example,
refer to: Pichery et al., AVS Quantum Sci. 5, 044401 (2023); doi: 10.1116/5.0163850.

For displaying the results in your browser, run "pixi run -e examples bokeh serve --show two_species_groundstate.py".
On a linux-x64 system with a Nvidia GPU run "pixi run -e examplescuda bokeh serve --show two_species_groundstate.py"
"""

from functools import reduce
from typing import (
    Dict, Iterable, List, Literal, Optional, Tuple, TypedDict, get_args
)
import os

from scipy import constants # type: ignore
import numpy as np
import jax.numpy as jnp
from jax.lax import scan
from jax import config
import matplotlib.pyplot as plt
from bokeh.plotting import figure, curdoc, gridplot
from bokeh.models import PrintfTickFormatter
from bokeh.io import export_png
from PIL import Image

import fftarray as fa
from helpers import plt_integrated_2d_density

# Enable double float precision for jax
config.update("jax_enable_x64", True)

# Enable high resolution PNG export
Image.MAX_IMAGE_PIXELS = None

# Register fftarray pytree nodes for JAX
try:
    fa.jax_register_pytree_nodes()
except ValueError:
    print("JAX pytree nodes registration failed. Probably due to being already "
        + "registered and the reexecution being triggered by the bokeh server.")

COLORS = ["#CC6677", "#88CCEE", "#DDCC77", "#332288", "#117733"]

# --------------------
# Physical constants
# --------------------

hbar: float = constants.hbar
a_0: float = constants.physical_constants['Bohr radius'][0]
kb: float = constants.Boltzmann

# coupling constant (used in GPE)
def coupling_fun(m_red: float, a: float) -> float:
    return 2 * np.pi * hbar**2 * a / m_red

# Rubidium 87
m_rb87: float = 86.909 * constants.atomic_mass # The atom's mass in kg.
a_rb87: float = 98 * a_0 # s-wave scattering length
coupling_rb87: float = coupling_fun(0.5*m_rb87, a_rb87) # coupling constant (used in GPE)

# Potassium 41
m_k41: float = 40.962 * constants.atomic_mass # The atom's mass in kg.
a_k41: float = 60 * a_0 # s-wave scattering length
coupling_k41: float = coupling_fun(0.5*m_k41, a_k41) # coupling constant (used in GPE)

# Interspecies interaction
a_rb87_k41: float = 165.3 * a_0
reduced_mass_rb87_k41 = m_rb87 * m_k41 / (m_rb87 + m_k41)
coupling_rb87_k41: float = coupling_fun(reduced_mass_rb87_k41, a_rb87_k41) # coupling constant (used in GPE)

def ground_state_ho(
    mass: float,
    omegas: Iterable[float],
    pos_coords: Iterable[fa.Array],
) -> fa.Array:
    psi = fa.full([], "pos", 1.0, xp=jnp)
    for omega, pos_1d in zip(omegas, pos_coords, strict=True):
        psi = (
            psi * (omega / (np.pi*hbar))**(1/4)
            * fa.exp(-mass * omega * (pos_1d**2)/(2*hbar))
        )
    norm = fa.integrate(fa.abs(psi)**2)
    return psi / fa.sqrt(norm)

# Define dual species GPE potentials

def gpe_potential_two_species(
    psi_pos_sq_1: fa.Array,
    psi_pos_sq_2: fa.Array,
    coupling_constant_1: float,
    coupling_constant_12: float,
    trap_potential_1: fa.Array,
    num_atoms_1: float,
    num_atoms_2: float,
) -> fa.Array:
    """
    Calculate the 2-species GPE potential for species number 1.
    This does not include the energies that only depend on the other species.
    This does not include the kinetic energy.
    """
    self_interaction = num_atoms_1 * coupling_constant_1 * psi_pos_sq_1
    interaction_12 = num_atoms_2 * coupling_constant_12 * psi_pos_sq_2
    return self_interaction + interaction_12 + trap_potential_1

def split_step_imaginary_time(
    psi: fa.Array,
    V: fa.Array,
    dt: float,
    mass: float,
) -> fa.Array:
    """Perform an imaginary time split-step of second order in VPV configuration."""

    # Calculate half step imaginary time potential propagator
    V_prop = fa.exp((-0.5*dt / hbar) * V)
    # Calculate full step imaginary time kinetic propagator (ksq = kx^2 + ky^2 + kz^2)
    ksq = reduce(lambda a,b: a+b, [
        (2*np.pi * fa.coords_from_dim(dim, "freq", xp=jnp, dtype=jnp.float64))**2
        for dim in psi.dims
    ])
    T_prop = fa.exp(-dt * hbar * ksq / (2*mass))

    # Apply half potential propagator
    psi = V_prop * psi.into_space("pos")

    # Apply full kinetic propagator
    psi = T_prop * psi.into_space("freq")

    # Apply half potential propagator
    psi = V_prop * psi.into_space("pos")

    # Normalize after step
    state_norm = fa.integrate(fa.abs(psi)**2)
    psi = psi / fa.sqrt(state_norm)

    return psi

def get_e_kin(
    psi: fa.Array,
    mass: float,
) -> float:
    """Calculate the kinetic energy of the wavefunction."""
    # Calculate k^2 = (2πf)^2
    ksq = reduce(lambda a,b: a+b, [
        (2*np.pi * fa.coords_from_dim(dim, "freq", xp=jnp, dtype=jnp.float64))**2
        for dim in psi.dims
    ])

    post_factor = hbar**2 / (2*mass)

    # Calculate |ψ(f)|^2
    wf_abs_sq = fa.abs(psi.into_space("freq"))**2

    # Calculate E_kin = <ψ|(hbar*k)^2/2m|ψ> = ∫|ψ(f)|^2 * k^2 df * hbar^2 / (2m)
    return fa.integrate(wf_abs_sq * ksq).values("freq") * post_factor

class DualSpeciesProperties(TypedDict):
    psi_rb87: fa.Array
    psi_k41: fa.Array
    rb_potential: fa.Array
    k_potential: fa.Array
    num_atoms_rb87: float
    num_atoms_k41: float

def imaginary_time_step_dual_species(
    properties: DualSpeciesProperties,
    dt: float,
) -> Tuple[DualSpeciesProperties, Dict[str, float]]:
    """
    Perform a single imaginary time step for the dual species GPE.
    Additionally, calculate all relevant energies.
    The states are returned in position space.
    """

    psi_rb87 = properties["psi_rb87"]
    psi_k41 = properties["psi_k41"]
    rb_potential = properties["rb_potential"]
    k_potential = properties["k_potential"]
    num_atoms_rb87 = properties["num_atoms_rb87"]
    num_atoms_k41 = properties["num_atoms_k41"]

    ## Calculate the potential energy operators (used for split-step and plots)
    psi_rb87 = psi_rb87.into_space("pos")
    psi_k41 = psi_k41.into_space("pos")

    psi_pos_sq_rb87 = fa.abs(psi_rb87)**2
    psi_pos_sq_k41 = fa.abs(psi_k41)**2

    V_rb87 = gpe_potential_two_species(
        psi_pos_sq_1=psi_pos_sq_rb87,
        psi_pos_sq_2=psi_pos_sq_k41,
        coupling_constant_1=coupling_rb87,
        coupling_constant_12=coupling_rb87_k41,
        trap_potential_1=rb_potential,
        num_atoms_1=num_atoms_rb87,
        num_atoms_2=num_atoms_k41,
    )

    V_k41 = gpe_potential_two_species(
        psi_pos_sq_1=psi_pos_sq_k41,
        psi_pos_sq_2=psi_pos_sq_rb87,
        coupling_constant_1=coupling_k41,
        coupling_constant_12=coupling_rb87_k41,
        trap_potential_1=k_potential,
        num_atoms_1=num_atoms_k41,
        num_atoms_2=num_atoms_rb87,
    )

    ## Calculate energies for plotting and convergence check

    # Calculate potential energy (in µK)
    E_pot_rb87 = fa.integrate(
        psi_pos_sq_rb87 * V_rb87,
        dtype="float64"
    ).values("pos") / (kb * 1e-6)
    E_pot_k41 = fa.integrate(
        psi_pos_sq_k41 * V_k41,
        dtype="float64"
    ).values("pos") / (kb * 1e-6)

    # Calculate kinetic energy (in µK)
    E_kin_rb87: float = get_e_kin(psi_rb87, mass=m_rb87) / (kb * 1e-6)
    E_kin_k41: float = get_e_kin(psi_k41, mass=m_k41) / (kb * 1e-6)

    # Calculate the total energy
    E_tot = E_kin_rb87 + E_kin_k41 + E_pot_rb87 + E_pot_k41

    ## Imaginary time split step application

    psi_rb87 = split_step_imaginary_time(
        psi=psi_rb87,
        V=V_rb87,
        dt=dt,
        mass=m_rb87,
    )
    psi_k41 = split_step_imaginary_time(
        psi=psi_k41,
        V=V_k41,
        dt=dt,
        mass=m_k41,
    )

    return properties | {"psi_rb87": psi_rb87, "psi_k41": psi_k41}, {
        "E_kin_rb87": E_kin_rb87,
        "E_kin_k41": E_kin_k41,
        "E_pot_rb87": E_pot_rb87,
        "E_pot_k41": E_pot_k41,
        "E_tot": E_tot
    }

def plot_energies(energies: dict, filename: str) -> None:
    fig, ax1 = plt.subplots()

    ax1.plot(energies["E_kin_rb87"], label="E_kin_rb87", color=COLORS[0])
    ax1.plot(energies["E_kin_k41"], label="E_kin_k41", color=COLORS[1])
    ax1.plot(energies["E_pot_rb87"], label="E_pot_rb87", color=COLORS[2])
    ax1.plot(energies["E_pot_k41"], label="E_pot_k41", color=COLORS[3])
    ax1.plot(energies["E_tot"], label="E_tot", color=COLORS[4])

    ax1.set_title(f"Final energy of {energies['E_tot'][-1]:.2f} µK")
    ax1.set_xlabel("Iteration step m")
    ax1.set_ylabel("Energy (µK)")
    ax1.set_yscale("log")
    ax1.set_ylim(bottom=1e-3)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    relative_change = np.abs(np.diff(energies["E_tot"]) / energies["E_tot"][:-1])
    ax2.plot(np.arange(1, len(energies["E_tot"])), relative_change, color="gray", linestyle="--", label=r"Rel. E_tot change $|(E_{m+1}-E_m) \:/ \: E_{m}|$")
    ax2.set_ylabel("Relative change")
    ax2.set_yscale("log")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(filename)

def plt_integrated_1d_densities(
    arrs: Dict[str, fa.Array],
    red_dim_names: List[str],
    filename: Optional[str] = None,
) -> None:
    """
    Plot the 1D projection of ``fa.abs(arr)**2`` in both position and frequency space,
    obtained via integration from potentially higher-dimensional ``fa.Array`` objects.
    The integration is performed along the provided red_dim_names.

    If a filename is provided, the plot is saved as a png file with the given filename.
    Otherwise, when using "bokeh serve --show your_script.py" the plot is displayed in the browser.
    """

    plots = []
    for space in get_args(fa.Space):

        density_arrs = {
            data_name: fa.integrate(
                fa.abs(arr.into_xp(np).into_space(space))**2,
                dim_name=red_dim_names
            ) for data_name, arr in arrs.items()
        }

        dim_names = [arr.dims[0].name for arr in list(density_arrs.values())]
        # Check dim_names all same, raise ValueError if not
        if len(set(dim_names)) != 1:
            raise ValueError("All density arrays must have the same dimension name.")
        match space:
            case "pos":
                x_unit = "m"
                y_unit = "1/m"
                x_symbol = f"{dim_names[0]}"
            case "freq":
                x_unit = "1/m"
                y_unit = "m"
                x_symbol = f"f_{dim_names[0]}"

        fig = figure(
            width=450,
            height=360,
            x_axis_label = f"$${x_symbol} \\, [{x_unit}]$$",
            y_axis_label = f"$$N|\\Psi({x_symbol})|^2 \\, [{y_unit}]$$",
            min_border=50,
        )
        for i, (data_name, arr_density_1d) in enumerate(density_arrs.items()):
            density_values = arr_density_1d.values(space)
            assert len(arr_density_1d.dims) == 1, "Reduced array must have only one dimension."
            dim_values = arr_density_1d.dims[0].values(space, xp=np)
            fig.line(
                dim_values, density_values,
                line_width=2, legend_label=data_name, color=COLORS[i % len(COLORS)]
            )

        from bokeh.models import Range1d, FixedTicker
        fig.x_range = Range1d(-1e5, +1e5) if space == "freq" else Range1d(-4e-5, +4e-5)
        if space == "pos":
            fig.xaxis.ticker = FixedTicker(ticks=[-4e-5, -2e-5, 0, 2e-5, 4e-5], minor_ticks=np.arange(-4e-5, 4e-5, 0.5e-5))

        fig.xaxis[0].formatter = PrintfTickFormatter(format="%.1e")
        fig.yaxis[0].formatter = PrintfTickFormatter(format="%.1e")
        plots.append(fig)

    grid = gridplot(plots, ncols=2) # type: ignore

    curdoc().add_root(grid)

    if filename:
        grid.toolbar.logo = None # type: ignore
        grid.toolbar_location = None # type: ignore
        export_png(grid, filename=f"{filename}.png", scale_factor=2)

def calc_ground_state_two_species(
    N_iter: int,
    plot_dir: Optional[str] = None,
) -> Dict[Literal["psi_rb87", "psi_k41"], fa.Array]:

    if plot_dir and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    dt_list = np.full(N_iter, 5e-7)

    # Number of atoms
    num_atoms_rb87 = 43900
    num_atoms_k41 = 14400

    rb_omega_x = 2*np.pi * 24.8 # rad/s
    rb_omega_y = 2*np.pi * 378.3 # rad/s
    rb_omega_z = 2*np.pi * 384.0 # rad/s

    k_omega_x = rb_omega_x * np.sqrt(m_rb87/m_k41)
    k_omega_y = rb_omega_y * np.sqrt(m_rb87/m_k41)
    k_omega_z = rb_omega_z * np.sqrt(m_rb87/m_k41)

    # --------------------
    # fftarray definitions
    # --------------------

    # Define dimensions

    x_dim = fa.dim_from_constraints(
        "x",
        pos_extent=800e-6,
        n=2**11,
        freq_middle=0.,
        pos_middle=0.,
        dynamically_traced_coords=False,
    )

    y_dim = fa.dim_from_constraints(
        "y",
        pos_extent=50e-6,
        n=2**7,
        freq_middle=0.,
        pos_middle=0.,
        dynamically_traced_coords=False,
    )

    z_dim = fa.dim_from_constraints(
        "z",
        pos_extent=50e-6,
        n=2**7,
        freq_middle=0.,
        pos_middle=0.,
        dynamically_traced_coords=False,
    )

    # Define 1d arrays
    x: fa.Array = fa.coords_from_dim(x_dim, "pos", xp=jnp, dtype=jnp.float64)
    y: fa.Array = fa.coords_from_dim(y_dim, "pos", xp=jnp, dtype=jnp.float64)
    z: fa.Array = fa.coords_from_dim(z_dim, "pos", xp=jnp, dtype=jnp.float64)

    # Define 3d arrays for potential
    rb_potential = 0.5 * m_rb87 * (rb_omega_x**2 * x**2 + rb_omega_y**2 * y**2 + rb_omega_z**2 * z**2)
    k_potential = 0.5 * m_k41 * (k_omega_x**2 * x**2 + k_omega_y**2 * y**2 + k_omega_z**2 * z**2)

    # Define 3d arrays for initial wavefunction
    init_psi = (
        fa.full(x_dim, "pos", 1, xp=jnp, dtype=jnp.float64)
        * fa.full(y_dim, "pos", 1, xp=jnp, dtype=jnp.float64)
        * fa.full(z_dim, "pos", 1, xp=jnp, dtype=jnp.float64)
    )
    state_norm = fa.integrate(fa.abs(init_psi)**2)

    init_psi_rb = init_psi / fa.sqrt(state_norm)
    init_psi_k = init_psi / fa.sqrt(state_norm)

    # When using jax.lax.scan, the input fa.Array must have the same properties as
    # the output one. As the scanned method imaginary_time_step_dual_species
    # returns the fa.Array with space="pos" and factors_applied=False,
    # we transform the input state accordingly.
    init_psi_rb = init_psi_rb.into_space("pos").into_factors_applied(False)
    init_psi_k = init_psi_k.into_space("pos").into_factors_applied(False)

    init_properties = {
        "psi_rb87": init_psi_rb,
        "psi_k41": init_psi_k,
        "rb_potential": rb_potential,
        "k_potential": k_potential,
        "num_atoms_rb87": num_atoms_rb87,
        "num_atoms_k41": num_atoms_k41
    }

    final_properties, energies = scan(
        f=imaginary_time_step_dual_species, # type: ignore
        init=init_properties,
        xs=dt_list,
    )

    # --------------------
    # Plot and save energies vs iteration steps
    # --------------------

    if plot_dir:
        plot_energies(
            energies,
            filename=os.path.join(plot_dir, "energies_vs_iteration.png")
        )

    ### Plot ground states with bokeh ###

    # Normalize wavefunctions to respective number of atoms
    rb_ground_state = final_properties["psi_rb87"].into_space("pos") * np.sqrt(num_atoms_rb87)
    k_ground_state = final_properties["psi_k41"].into_space("pos") * np.sqrt(num_atoms_k41)

    plt_integrated_2d_density(
        rb_ground_state,
        red_dim_name="z",
        data_name="Rb87",
        filename=os.path.join(plot_dir, "rb_ground_state_xy") if plot_dir else None,
    )
    plt_integrated_2d_density(
        k_ground_state,
        red_dim_name="z",
        data_name="K41",
        filename=os.path.join(plot_dir, "k_ground_state_xy") if plot_dir else None,
    )

    # Plot 1d densities

    plt_integrated_1d_densities(
        arrs={"Rb87": rb_ground_state, "K41": k_ground_state},
        red_dim_names=["y", "z"],
        filename=os.path.join(plot_dir, "1d_densities_x") if plot_dir else None,
    )

    return {"psi_rb87": rb_ground_state, "psi_k41": k_ground_state}

if __name__ == "__main__" or "bokeh_app" in __name__:

    calc_ground_state_two_species(
        N_iter=5000,
        plot_dir="two_species_groundstate_figures"
    )
