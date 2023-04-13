import numpy as np

from fftarray.fft_constraint_solver import _z3_constraint_solver
from fftarray.fft_constraint_solver import NoSolutionFoundError, NoUniqueSolutionError

def test_undefined_pos_space():
    user_constraints = dict(
        n = 4,
        d_pos = 1,
        freq_middle = 0
    )

    try:
        _z3_constraint_solver(
            constraints=user_constraints,
            loose_params=[],
            make_suggestions=True
        )
    except NoUniqueSolutionError as e:
        assert e._suggested_additional_params == [["pos_min", "pos_max", "pos_middle"]]
    except Exception as e:
        raise e

def test_undefined_grid_spacing():
    user_constraints = dict(
        n=16,
        pos_middle = 0,
        freq_middle = 0
    )

    try:
        _z3_constraint_solver(
            constraints=user_constraints,
            loose_params=[],
            make_suggestions=True
        )
    except NoUniqueSolutionError as e:
        assert e._suggested_additional_params == [["d_freq", "freq_extent"], ["d_pos", "pos_extent"]]
    except Exception as e:
        raise e

def test_overconstrained_pos_space():
    user_constraints = dict(
        n=16,
        d_pos = 1,
        pos_min = -5,
        pos_middle = 0,
        freq_middle = 0
    )

    try:
        _z3_constraint_solver(
            constraints=user_constraints,
            loose_params=[],
            make_suggestions=True
        )
    except NoSolutionFoundError as e:
        assert e._suggested_removed_params == ["n", "d_pos", "pos_min", "pos_middle"]
    except Exception as e:
        raise e

def test_overconstrained_grid_spacing_1():
    user_constraints = dict(
        n=16,
        d_freq = 1,
        d_pos = 1,
        pos_middle = 0,
        freq_middle = 0
    )

    try:
        _z3_constraint_solver(
            constraints=user_constraints,
            loose_params=[],
            make_suggestions=True
        )
    except NoSolutionFoundError as e:
        assert e._suggested_removed_params == ["d_freq", "d_pos"]
    except Exception as e:
        raise e

def test_overconstrained_grid_spacing_2():
    user_constraints = dict(
        n=4,
        d_freq = 1./10.,
        d_pos = 1,
        pos_middle = 0,
        freq_middle = 0
    )

    try:
        _z3_constraint_solver(
            constraints=user_constraints,
            loose_params=[],
            make_suggestions=True
        )
    except NoSolutionFoundError as e:
        assert e._suggested_removed_params == ["n", "d_freq", "d_pos"]
    except Exception as e:
        raise e

def test_missing_loose_param_1():
    user_constraints = dict(
        n="power_of_two",
        d_freq = 1./10.,
        d_pos = 1,
        pos_middle = 0,
        freq_middle = 0
    )

    try:
        _z3_constraint_solver(
            constraints=user_constraints,
            loose_params=[],
            make_suggestions=True
        )
    except NoSolutionFoundError as e:
        assert e._suggested_loose_params == [["d_freq"], ["d_pos"]]
    except Exception as e:
        raise e

def test_missing_loose_param_2():
    user_constraints = dict(
        n="power_of_two",
        d_pos = 1,
        freq_extent = 1. * 1e-3,
        pos_min = 0,
        freq_middle = 0
    )

    try:
        _z3_constraint_solver(
            constraints=user_constraints,
            loose_params=[],
            make_suggestions=True
        )
    except NoSolutionFoundError as e:
        assert e._suggested_loose_params == [["freq_extent"]]
    except Exception as e:
        raise e
