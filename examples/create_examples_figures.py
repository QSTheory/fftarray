import os

from two_species_groundstate import calc_ground_state_two_species

plot_dir = os.path.join(os.path.dirname(__file__), "two_species_groundstate_figures")

calc_ground_state_two_species(N_iter=5000, plot_dir=plot_dir)
