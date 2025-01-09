import sys

import numpy as np
from numpy.testing import assert_array_almost_equal_nulp
from numpy import pi
import pytest
from hypothesis import given, strategies as st, settings, note
import jax.numpy as jnp
from jax import config

from fftarray.constraint_solver import _z3_constraint_solver
from fftarray.constraint_solver import ConstraintSolverError, NoSolutionFoundError, NoUniqueSolutionError, ConstraintValueError

config.update("jax_enable_x64", True)

"""Constant of Rubidium 87"""
k_L: float = 2 * pi /  780 * 1e-9

def assert_scalars_almost_equal_nulp(x, y, nulp = 4):
    assert_array_almost_equal_nulp(np.array([x]), np.array([y]), nulp = nulp)

def assert_constraints_are_equal(dict1, dict2):
    assert_scalars_almost_equal_nulp(dict1['n'], dict2['n'])
    assert_scalars_almost_equal_nulp(dict1['d_pos'], dict2['d_pos'])
    assert_scalars_almost_equal_nulp(dict1['d_freq'], dict2['d_freq'])
    assert_scalars_almost_equal_nulp(dict1['pos_min'], dict2['pos_min'])
    assert_scalars_almost_equal_nulp(dict1['freq_middle'], dict2['freq_middle'])

    assert_scalars_almost_equal_nulp(dict1['pos_max'], dict2['pos_max'])
    assert_scalars_almost_equal_nulp(dict1['pos_middle'], dict2['pos_middle'])
    assert_scalars_almost_equal_nulp(dict1['pos_extent'], dict2['pos_extent'])
    assert_scalars_almost_equal_nulp(dict1['freq_min'], dict2['freq_min'])
    assert_scalars_almost_equal_nulp(dict1['freq_max'], dict2['freq_max'])
    assert_scalars_almost_equal_nulp(dict1['freq_extent'], dict2['freq_extent'])

def test_symmetric_space_even_n_with_freq_middle() -> None:
    user_constraints = dict(
        n = 4,
        pos_min = -1.5*np.pi,
        pos_max = 1.5*np.pi,
        freq_middle = 0
    )

    pos_extent = user_constraints['pos_max'] - user_constraints['pos_min']
    d_pos = pos_extent / (user_constraints['n'] - 1)
    d_freq: float = 1. / (d_pos * user_constraints['n'])
    freq_extent = d_freq * (user_constraints['n'] - 1)
    freq_min = user_constraints['freq_middle'] - user_constraints['n'] / 2 * d_freq
    freq_max = user_constraints['freq_middle'] + (user_constraints['n'] / 2 - 1) * d_freq
    pos_middle = user_constraints["pos_min"] + user_constraints["n"]/2 * d_pos

    missing_expected_quantities = dict(
        pos_extent = pos_extent,
        d_pos = d_pos,
        d_freq = d_freq,
        pos_middle = pos_middle,
        freq_extent = freq_extent,
        freq_min = freq_min,
        freq_max = freq_max
    )

    expected_quantities = {**user_constraints, **missing_expected_quantities}
    solution_quantities = _z3_constraint_solver(
        constraints=user_constraints,
        loose_params=[],
        make_suggestions=False
    )

    print('Expected: ', expected_quantities)
    print('Solution: ', solution_quantities)

    assert_constraints_are_equal(expected_quantities, solution_quantities)

def test_symmetric_space_even_n_with_freq_middle_numerical_stability():
    user_constraints = dict(
        pos_min = 0.e-6,
        pos_max = 20.e-6,
        n = 8192,
        freq_middle=0.
    )

    pos_extent = user_constraints['pos_max'] - user_constraints['pos_min']
    d_pos = pos_extent / (user_constraints['n'] - 1)
    d_freq = 1. / (d_pos * user_constraints['n'])
    freq_extent = d_freq * (user_constraints['n'] - 1)
    freq_min = user_constraints['freq_middle'] - user_constraints['n'] / 2 * d_freq
    freq_max = user_constraints['freq_middle'] + (user_constraints['n'] / 2 - 1) * d_freq
    pos_middle = user_constraints["pos_min"] + user_constraints["n"]/2 * d_pos

    missing_expected_quantities = dict(
        pos_extent = pos_extent,
        d_pos = d_pos,
        d_freq = d_freq,
        pos_middle = pos_middle,
        freq_extent = freq_extent,
        freq_min = freq_min,
        freq_max = freq_max
    )

    expected_quantities = {**user_constraints, **missing_expected_quantities}
    solution_quantities = _z3_constraint_solver(
        constraints=user_constraints,
        loose_params=[],
        make_suggestions=False
    )

    print('Expected: ', expected_quantities)
    print('Solution: ', solution_quantities)

    assert_constraints_are_equal(expected_quantities, solution_quantities)

def test_symmetric_space_odd_n_with_freq_middle():
    user_constraints = dict(
        n = 3,
        pos_min = -1.5*np.pi,
        pos_max = 1.5*np.pi,
        freq_middle = 0
    )

    pos_extent = user_constraints['pos_max'] - user_constraints['pos_min']
    d_pos = pos_extent / (user_constraints['n'] - 1)
    d_freq = 1. / (d_pos * user_constraints['n'])
    freq_extent = d_freq * (user_constraints['n'] - 1)
    freq_min = user_constraints['freq_middle'] - (user_constraints['n'] - 1) / 2 * d_freq
    freq_max = user_constraints['freq_middle'] + (user_constraints['n'] - 1) / 2 * d_freq
    pos_middle = user_constraints["pos_min"] + (user_constraints["n"] - 1) / 2 * d_pos

    missing_expected_quantities = dict(
        pos_extent = pos_extent,
        d_pos = d_pos,
        d_freq = d_freq,
        pos_middle = pos_middle,
        freq_extent = freq_extent,
        freq_min = freq_min,
        freq_max = freq_max
    )

    expected_quantities = {**user_constraints, **missing_expected_quantities}
    solution_quantities = _z3_constraint_solver(
        constraints=user_constraints,
        loose_params=[],
        make_suggestions=False
    )

    print('Expected: ', expected_quantities)
    print('Solution: ', solution_quantities)

    assert_constraints_are_equal(expected_quantities, solution_quantities)

def test_matching_implicit_even_n_without_loose_params():
    user_constraints = dict(
        d_pos = 1*np.pi,
        pos_min = -1.5*np.pi,
        pos_max = 1.5*np.pi,
        freq_middle = 0,
        n="even"
    )

    pos_extent = user_constraints['pos_max'] - user_constraints['pos_min']
    n = pos_extent / user_constraints['d_pos'] + 1
    d_freq = 1. / (user_constraints['d_pos'] * n)
    pos_middle = user_constraints["pos_min"] + n/2 * user_constraints['d_pos']
    freq_extent = d_freq * (n - 1)
    freq_min = user_constraints['freq_middle'] - n / 2 * d_freq
    freq_max = user_constraints['freq_middle'] + (n / 2 - 1) * d_freq

    missing_expected_quantities = dict(
        pos_extent = pos_extent,
        n = n,
        d_freq = d_freq,
        pos_middle = pos_middle,
        freq_extent = freq_extent,
        freq_min = freq_min,
        freq_max = freq_max
    )

    expected_quantities = {**user_constraints, **missing_expected_quantities}
    solution_quantities = _z3_constraint_solver(
        constraints=user_constraints,
        loose_params=[],
        make_suggestions=False
    )

    print('Expected: ', expected_quantities)
    print('Solution: ', solution_quantities)

    assert_constraints_are_equal(expected_quantities, solution_quantities)

def test_matching_implicit_power_of_two_n_without_loose_params():
    user_constraints = dict(
        d_pos = 1,
        pos_min = 0,
        pos_max = 6,
        freq_middle = 0,
        n="power_of_two"
    )

    solution_quantities = _z3_constraint_solver(
        constraints=user_constraints,
        loose_params=["d_pos"],
        make_suggestions=False
    )

    pos_extent = user_constraints['pos_max'] - user_constraints['pos_min']
    user_constraints['d_pos'] = pos_extent / 7
    n = pos_extent / user_constraints['d_pos'] + 1
    d_freq = 1. / (user_constraints['d_pos'] * n)
    pos_middle = user_constraints["pos_min"] + n/2 * user_constraints['d_pos']
    freq_extent = d_freq * (n - 1)
    freq_min = user_constraints['freq_middle'] - n / 2 * d_freq
    freq_max = user_constraints['freq_middle'] + (n / 2 - 1) * d_freq

    missing_expected_quantities = dict(
        pos_extent = pos_extent,
        n = n,
        d_freq = d_freq,
        pos_middle = pos_middle,
        freq_extent = freq_extent,
        freq_min = freq_min,
        freq_max = freq_max
    )
    expected_quantities = {**user_constraints, **missing_expected_quantities}


    print('Expected: ', expected_quantities)
    print('Solution: ', solution_quantities)

    assert_constraints_are_equal(expected_quantities, solution_quantities)

def test_exception_implicit_odd_n_without_loose_params():
    user_constraints = dict(
        d_pos = 1*np.pi,
        pos_min = -np.pi,
        pos_max = np.pi,
        freq_middle = 0,
        n="power_of_two"
    )

    with pytest.raises(NoSolutionFoundError):
        _z3_constraint_solver(
            constraints=user_constraints,
            loose_params=[],
            make_suggestions=False
        )

def test_n_widening_odd_to_even_with_two_loose_params():
    user_constraints = dict(
        d_pos = 1,
        d_freq = 1. / 5,
        pos_min = -2*np.pi,
        freq_middle = 0,
        n="even"
    )

    solution_quantities = _z3_constraint_solver(
        constraints=user_constraints,
        loose_params=['d_pos', 'd_freq'],
        make_suggestions=False
    )

    print('Solution: ', solution_quantities)

    assert solution_quantities['n'] == 6
    assert solution_quantities['d_pos'] < user_constraints['d_pos']
    assert solution_quantities['d_freq'] < user_constraints['d_freq']

def test_n_widening_odd_to_power_of_two_with_two_loose_params():
    user_constraints = dict(
        d_pos = 1,
        d_freq = 1./5.,
        pos_min = -2*np.pi,
        freq_middle = 0,
        n="power_of_two"
    )

    solution_quantities = _z3_constraint_solver(
        constraints=user_constraints,
        loose_params=['d_pos', 'd_freq'],
        make_suggestions=False
    )

    print('Solution: ', solution_quantities)

    assert solution_quantities['n'] == 8
    assert solution_quantities['d_pos'] < user_constraints['d_pos']
    assert solution_quantities['d_freq'] < user_constraints['d_freq']

def test_n_widening_even_to_power_of_two_with_two_loose_params():
    user_constraints = dict(
        d_pos = 1,
        d_freq = 1./6.,
        pos_min = -2*np.pi,
        freq_middle = 0,
        n="power_of_two"
    )

    solution_quantities = _z3_constraint_solver(
        constraints=user_constraints,
        loose_params=['d_pos', 'd_freq'],
        make_suggestions=False
    )

    print('Solution: ', solution_quantities)

    assert solution_quantities['n'] == 8
    assert solution_quantities['d_pos'] < user_constraints['d_pos']
    assert solution_quantities['d_freq'] < user_constraints['d_freq']

def test_matching_implicit_n_with_two_loose_params():
    user_constraints = dict(
        d_pos = 1,
        d_freq = 1./8.,
        pos_min = -2*np.pi,
        freq_middle = 0,
        n="power_of_two"
    )

    solution_quantities = _z3_constraint_solver(
        constraints=user_constraints,
        loose_params=['d_pos', 'd_freq'],
        make_suggestions=False
    )

    print('Solution: ', solution_quantities)

    assert solution_quantities['n'] == 8
    assert solution_quantities['d_pos'] == user_constraints['d_pos']
    assert solution_quantities['d_freq'] == user_constraints['d_freq']

#### Random collection of different constraint combinations ####

def test_asymmetric_space_even_n_with_freq_middle():
    user_constraints = dict(
        n = 8,
        pos_min = 10.,
        pos_max = 17.,
        freq_middle = 3.
    )

    pos_extent = user_constraints['pos_max'] - user_constraints['pos_min']
    d_pos = pos_extent / (user_constraints['n'] - 1)
    d_freq = 1. / (d_pos * user_constraints['n'])
    pos_middle = user_constraints["pos_min"] + user_constraints["n"]/2 * d_pos
    freq_extent = d_freq * (user_constraints['n'] - 1)
    freq_min = user_constraints['freq_middle'] - user_constraints['n'] / 2 * d_freq
    freq_max = user_constraints['freq_middle'] + (user_constraints['n'] / 2 - 1) * d_freq

    missing_expected_quantities = dict(
        pos_extent = pos_extent,
        d_pos = d_pos,
        d_freq = d_freq,
        pos_middle = pos_middle,
        freq_extent = freq_extent,
        freq_min = freq_min,
        freq_max = freq_max
    )

    expected_quantities = {**user_constraints, **missing_expected_quantities}
    solution_quantities = _z3_constraint_solver(
        constraints=user_constraints,
        loose_params=[],
        make_suggestions=False
    )

    print('Expected: ', expected_quantities)
    print('Solution: ', solution_quantities)

    assert_constraints_are_equal(expected_quantities, solution_quantities)

def test_symmetric_freqs_even_n_with_pos_min():
    user_constraints = dict(
        n = 8,
        freq_min = -1.5*np.pi,
        freq_max = 1.5*np.pi,
        pos_min = -1.
    )

    freq_extent = user_constraints['freq_max'] - user_constraints['freq_min']
    d_freq = freq_extent / (user_constraints['n'] - 1)
    d_pos = 1. / (d_freq * user_constraints['n'])
    freq_middle = user_constraints["freq_min"] + user_constraints["n"]/2 * d_freq
    pos_extent = d_pos * (user_constraints['n'] - 1)
    pos_max = user_constraints['pos_min'] + pos_extent
    pos_middle = user_constraints["pos_min"] + user_constraints["n"]/2. * d_pos

    missing_expected_quantities = dict(
        freq_extent = freq_extent,
        d_pos = d_pos,
        d_freq = d_freq,
        pos_middle = pos_middle,
        freq_middle = freq_middle,
        pos_extent = pos_extent,
        pos_max = pos_max
    )

    expected_quantities = {**user_constraints, **missing_expected_quantities}
    solution_quantities = _z3_constraint_solver(
        constraints=user_constraints,
        loose_params=[],
        make_suggestions=False
    )

    print('Expected: ', expected_quantities)
    print('Solution: ', solution_quantities)

    assert_constraints_are_equal(expected_quantities, solution_quantities)

def test_symmetric_freqs_even_n_with_freq_middle_and_freq_max_and_pos_min():
    user_constraints = dict(
        n = 4,
        freq_middle = 0,
        freq_max = 1.5*np.pi,
        pos_min = 0.
    )

    d_freq = user_constraints['freq_max'] - user_constraints['freq_middle'] / (user_constraints["n"] / 2 - 1)
    freq_extent = (user_constraints['n'] - 1) * d_freq
    d_pos = 1. / (d_freq * user_constraints['n'])
    freq_min = user_constraints['freq_middle'] - user_constraints['n'] / 2 * d_freq
    pos_extent = d_pos * (user_constraints['n'] - 1)
    pos_max = user_constraints['pos_min'] + pos_extent
    pos_middle = user_constraints['pos_min'] + user_constraints["n"]/2 * d_pos

    missing_expected_quantities = dict(
        freq_extent = freq_extent,
        d_pos = d_pos,
        d_freq = d_freq,
        pos_middle = pos_middle,
        freq_min = freq_min,
        pos_extent = pos_extent,
        pos_max = pos_max
    )

    expected_quantities = {**user_constraints, **missing_expected_quantities}
    solution_quantities = _z3_constraint_solver(
        constraints=user_constraints,
        loose_params=[],
        make_suggestions=False
    )

    print('Expected: ', expected_quantities)
    print('Solution: ', solution_quantities)

    assert_constraints_are_equal(expected_quantities, solution_quantities)

def test_symmetric_space_even_n_with_freq_middle_and_pos_middle():
    user_constraints = dict(
        n = 4,
        pos_middle = 0.,
        pos_extent = 3*np.pi,
        freq_middle = 0.
    )

    d_pos = user_constraints['pos_extent'] / (user_constraints['n'] - 1)
    pos_min = user_constraints['pos_middle'] - user_constraints['n']/2 * d_pos
    pos_max = user_constraints['pos_middle'] + (user_constraints['n']/2 - 1) * d_pos
    d_freq = 1. / (d_pos * user_constraints['n'])
    freq_extent = d_freq * (user_constraints['n'] - 1)
    freq_min = user_constraints['freq_middle'] - user_constraints['n'] / 2 * d_freq
    freq_max = user_constraints['freq_middle'] + (user_constraints['n'] / 2 - 1) * d_freq

    missing_expected_quantities = dict(
        pos_min = pos_min,
        pos_max = pos_max,
        d_pos = d_pos,
        d_freq = d_freq,
        freq_extent = freq_extent,
        freq_min = freq_min,
        freq_max = freq_max
    )

    expected_quantities = {**user_constraints, **missing_expected_quantities}
    solution_quantities = _z3_constraint_solver(
        constraints=user_constraints,
        loose_params=[],
        make_suggestions=False
    )

    print('Expected: ', expected_quantities)
    print('Solution: ', solution_quantities)

    assert_constraints_are_equal(expected_quantities, solution_quantities)

def test_symmetric_space_even_n_with_freq_middle_and_pos_middle_and_pos_min():
    user_constraints = dict(
        n = 4,
        pos_middle = 0.,
        pos_min = -1.5*np.pi,
        freq_middle = 0.
    )

    pos_extent = (user_constraints['pos_middle'] - user_constraints['pos_min']) / (user_constraints['n']/2) * (user_constraints['n']-1)
    d_pos = pos_extent / (user_constraints['n'] - 1)
    pos_max = user_constraints['pos_middle'] + (user_constraints['n']/2 - 1) * d_pos
    d_freq = 1. / (d_pos * user_constraints['n'])
    freq_extent = d_freq * (user_constraints['n'] - 1)
    freq_min = user_constraints['freq_middle'] - user_constraints['n'] / 2 * d_freq
    freq_max = user_constraints['freq_middle'] + (user_constraints['n'] / 2 - 1) * d_freq

    missing_expected_quantities = dict(
        pos_extent = pos_extent,
        pos_max = pos_max,
        d_pos = d_pos,
        d_freq = d_freq,
        freq_extent = freq_extent,
        freq_min = freq_min,
        freq_max = freq_max
    )

    expected_quantities = {**user_constraints, **missing_expected_quantities}
    solution_quantities = _z3_constraint_solver(
        constraints=user_constraints,
        loose_params=[],
        make_suggestions=False
    )

    print('Expected: ', expected_quantities)
    print('Solution: ', solution_quantities)

    assert_constraints_are_equal(expected_quantities, solution_quantities)

def test_symmetric_space_even_n_with_freq_middle_and_pos_middle_and_pos_max():
    user_constraints = dict(
        n = 4,
        pos_middle = 0.,
        pos_max = 1.5*np.pi,
        freq_middle = 0.
    )

    pos_extent = (user_constraints['pos_max'] - user_constraints['pos_middle']) * (user_constraints["n"]-1) / (user_constraints["n"] / 2 - 1)
    d_pos = pos_extent / (user_constraints['n'] - 1)
    pos_min = user_constraints['pos_middle'] - user_constraints['n'] / 2 * d_pos
    d_freq = 1. / (d_pos * user_constraints['n'])
    freq_extent = d_freq * (user_constraints['n'] - 1)
    freq_min = user_constraints['freq_middle'] - user_constraints['n'] / 2 * d_freq
    freq_max = user_constraints['freq_middle'] + (user_constraints['n'] / 2 - 1) * d_freq

    missing_expected_quantities = dict(
        pos_extent = pos_extent,
        pos_min = pos_min,
        d_pos = d_pos,
        d_freq = d_freq,
        freq_extent = freq_extent,
        freq_min = freq_min,
        freq_max = freq_max
    )

    expected_quantities = {**user_constraints, **missing_expected_quantities}
    solution_quantities = _z3_constraint_solver(
        constraints=user_constraints,
        loose_params=[],
        make_suggestions=False
    )

    print('Expected: ', expected_quantities)
    print('Solution: ', solution_quantities)

    assert_constraints_are_equal(expected_quantities, solution_quantities)

def test_constraint_solver_without_exception():

    user_constraints = dict(
        pos_min = -50e-6,
        pos_max = 50e-6,
        freq_middle = 0.,
        freq_extent = 10*k_L,
        n = 'power_of_two'
    )

    _z3_constraint_solver(
        constraints=user_constraints,
        loose_params=["freq_extent"],
        make_suggestions=False
    )

#### Some special cases of valid/invalid constraints ####

def test_exception_n_as_float():
    user_constraints = dict(
        n = 8.,
        pos_min = 17.,
        pos_max = 10.,
        freq_middle = 3.
    )

    with pytest.raises(ConstraintValueError):
        _z3_constraint_solver(
            constraints=user_constraints,
            loose_params=[],
            make_suggestions=False
        )

def test_exception_n_rounding_mode_odd():
    user_constraints = dict(
        pos_middle = 0,
        freq_middle = 3.,
        d_pos = 1,
        d_freq = 1./9,
        n='odd'
    )

    with pytest.raises(ConstraintValueError):
        _z3_constraint_solver(
            constraints=user_constraints,
            loose_params=[],
            make_suggestions=False
        )

def test_pos_min_larger_pos_max():
    user_constraints = dict(
        n = 8,
        pos_min = 17.,
        pos_max = 10.,
        freq_middle = 3.
    )

    with pytest.raises(NoSolutionFoundError):
        _z3_constraint_solver(
            constraints=user_constraints,
            loose_params=[],
            make_suggestions=False
        )

def test_freq_min_larger_freq_max():
    user_constraints = dict(
        n = 8,
        freq_min = 17.,
        freq_max = 10.,
        pos_middle = 0.
    )

    with pytest.raises(NoSolutionFoundError):
        _z3_constraint_solver(
            constraints=user_constraints,
            loose_params=[],
            make_suggestions=False
        )

def test_equivalent_space_constraints_even_n():
    user_constraints_1 = dict(
        n = 8,
        pos_middle = 0.,
        pos_extent = 7.,
        freq_middle = 0.
    )

    user_constraints_2 = dict(
        n = 8,
        pos_min = -4,
        pos_max = 3,
        freq_middle = 0.
    )

    solution_quantities_1 = _z3_constraint_solver(
        constraints=user_constraints_1,
        loose_params=[],
        make_suggestions=False
    )
    solution_quantities_2 = _z3_constraint_solver(
        constraints=user_constraints_2,
        loose_params=[],
        make_suggestions=False
    )

    assert_constraints_are_equal(solution_quantities_1, solution_quantities_2)

def test_equivalent_freq_constraints_even_n():
    user_constraints_1 = dict(
        n = 8,
        freq_middle = 0.,
        freq_extent = 7.,
        pos_middle = 0.
    )

    user_constraints_2 = dict(
        n = 8,
        freq_min = -4,
        freq_max = 3,
        pos_middle = 0.
    )

    solution_quantities_1 = _z3_constraint_solver(
        constraints=user_constraints_1,
        loose_params=[],
        make_suggestions=False
    )
    solution_quantities_2 = _z3_constraint_solver(
        constraints=user_constraints_2,
        loose_params=[],
        make_suggestions=False
    )

    assert_constraints_are_equal(solution_quantities_1, solution_quantities_2)

def test_non_unique_constraints_1():
    user_constraints = dict(
        n = "even",
        freq_middle = 0.,
        freq_extent = 2.5,
        d_freq = 0.5
    )

    with pytest.raises(NoUniqueSolutionError):
        _z3_constraint_solver(
            constraints=user_constraints,
            loose_params=[],
            make_suggestions=False
        )

def test_non_unique_constraints_2():
    user_constraints = dict(
        pos_min = 0.,
        pos_max = 2.,
        d_freq = 0.1,
        n="power_of_two"
    )

    with pytest.raises(NoUniqueSolutionError):
        _z3_constraint_solver(
            constraints=user_constraints,
            loose_params=[],
            make_suggestions=False
        )

def test_overdetermined_constraints_no_n_widening_possible():
    user_constraints = dict(
        n = 8,
        d_pos = 1,
        d_freq = 1,
        pos_min = -1,
        pos_middle = 0,
        freq_middle = 0.,
    )

    with pytest.raises(NoSolutionFoundError):
        _z3_constraint_solver(
            constraints=user_constraints,
            loose_params=[],
            make_suggestions=False
        )

def test_overconstrained_1():
    user_constraints = dict(
        n = 8,
        d_pos = 1,
        d_freq = 1,
        freq_middle = 0.,
    )

    with pytest.raises(NoSolutionFoundError):
        _z3_constraint_solver(
            constraints=user_constraints,
            loose_params=[],
            make_suggestions=False
        )

def test_overconstrained_2():
    user_constraints = dict(
        n = "even",
        freq_middle = 0.,
        freq_extent = 2.5,
        d_pos = 0.5,
    )

    with pytest.raises(NoSolutionFoundError):
        _z3_constraint_solver(
            constraints=user_constraints,
            loose_params=[],
            make_suggestions=False
        )

def test_jax_numpy_values():
    user_constraints_1 = dict(
        pos_middle = jnp.array(0.),
        freq_middle = jnp.array(0.),
        n = jnp.array(64),
        d_pos = 1/jnp.sqrt(2)
    )
    user_constraints_2 = dict(
        pos_middle = 0.,
        freq_middle = 0.,
        n = 64,
        d_pos = 1/np.sqrt(2)
    )
    solution_quantities_1 = _z3_constraint_solver(
        constraints=user_constraints_1,
        loose_params=[],
        make_suggestions=False
    )
    solution_quantities_2 = _z3_constraint_solver(
        constraints=user_constraints_2,
        loose_params=[],
        make_suggestions=False
    )
    assert_constraints_are_equal(solution_quantities_1, solution_quantities_2)

invalid_values = [np.inf, np.nan, True, False, '0', 1+1j, [0], tuple([0])]

@pytest.mark.parametrize("test_val", invalid_values)
def test_invalid_values(test_val):
    """
    Test that an error is thrown if a constraint has an unsupported type.
    """
    user_constraints = dict(
        pos_min = -1.5*np.pi,
        pos_max = 1.5*np.pi,
        d_pos = test_val,
        freq_middle = 0.,
        n="power_of_two"
    )
    with pytest.raises(ConstraintValueError):
        _z3_constraint_solver(
            constraints=user_constraints,
            loose_params=[],
            make_suggestions=False
        )

@pytest.mark.slow
@given(
    n = st.integers(),
    # Limit to width=32 and no subnormals to keep the exponents small enough to not make z3 take too long.
    pos_min = st.floats(width=32, allow_subnormal=False),
    pos_max = st.floats(width=32, allow_subnormal=False),
    freq_middle = st.floats(width=32, allow_subnormal=False),
)
# Explicitly deactivate deadline because if hypothesis hits a very slow case it can take quite a few seconds.
# There is no upper bound to the duration so we do not test one.
@settings(max_examples=500, deadline=None)
def test_with_hypothesis_0(n: int, pos_min: float, pos_max: float, freq_middle: float):
    user_constraints = dict(
        n = n,
        pos_min = pos_min,
        pos_max = pos_max,
        freq_middle = freq_middle
    )
    note(str(user_constraints))
    note(f"int limits: [{-sys.maxsize-1}, {sys.maxsize}]")
    note(f"float limits: +-[{sys.float_info.min}, {sys.float_info.max}]")

    try:
        _z3_constraint_solver(
            constraints=user_constraints, # type: ignore
            loose_params=[],
            make_suggestions=False
        )
    except ConstraintSolverError:
        ...
    except Exception as e:
        raise e

@pytest.mark.slow
@given(
    d_pos = st.floats(width=32, allow_subnormal=False),
    d_freq = st.floats(width=32, allow_subnormal=False),
    pos_min = st.floats(width=32, allow_subnormal=False),
    freq_middle = st.floats(width=32, allow_subnormal=False),
)
@settings(max_examples=500, deadline=None)
def test_with_hypothesis_1(d_pos: float, d_freq: float, pos_min: float, freq_middle: float):
    user_constraints = dict(
        d_pos = d_pos,
        d_freq = d_freq,
        pos_min = pos_min,
        freq_middle = freq_middle,
        n="even"
    )
    note(str(user_constraints))
    note(f"float limits: +-[{sys.float_info.min}, {sys.float_info.max}]")
    try:
        _z3_constraint_solver(
            constraints=user_constraints, # type: ignore
            loose_params=['d_pos'],
            make_suggestions=False
        )
    except ConstraintSolverError:
        ...
    except Exception as e:
        raise e

accessors = ["freq_middle", "pos_middle"] # only floats
for pf in ["pos", "freq"]:
    accessors += [f"d_{pf}", f"{pf}_min", f"{pf}_max", f"{pf}_extent"]

@pytest.mark.slow
@given(
    n=st.one_of(st.sampled_from(["power_of_two", "even"]), st.integers()),
    **{a: st.one_of(st.none(), st.floats(width=32, allow_subnormal=False)) for a in accessors}
)
@settings(max_examples=500, deadline=None)
def test_with_hypothesis_2(**user_constraints):
    note(str(user_constraints))
    note(f"int limits: [{-sys.maxsize-1}, {sys.maxsize}]")
    note(f"float limits: +-[{sys.float_info.min}, {sys.float_info.max}]")
    try:
        _z3_constraint_solver(
            constraints=user_constraints,
            loose_params=[],
            make_suggestions=False
        )
    except ConstraintSolverError:
        ...
    except Exception as e:
        raise e

