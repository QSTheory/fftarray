"""Constraint solver based on `Z3 <https://github.com/Z3Prover/z3>`_ - a theorem
prover from Microsoft Research.

This submodule contains the functionality that initializes the coordinate grid
of the FFTWave. It contains functions that are out of scope for the average
FFTWave user. Please visit the development area to toggle their visibility.
"""

from typing import Any, Optional, Union, List, Dict
import sys
import decimal

from z3 import (
    Real, Or, Solver, CheckSatResult, Z3_L_TRUE, Z3_L_FALSE,
    AlgebraicNumRef, RatNumRef, BoolRef, ModelRef
)
import numpy as np

from .fft_constraint_solver_exceptions import *

# This dict contains all possible user constraints
# and their optimized directions for n widening
# (max <=> it may only be decreased/it is an upper bound and vice versa)
VARS_WITH_PROPS: Dict[str, Optional[str]] = {
    "n": None,
    "d_pos": "max",
    "d_freq": "max",
    "pos_min": "max",
    "pos_max": "min",
    "pos_extent": "min",
    "pos_middle": None,
    "freq_min": "max",
    "freq_max": "min",
    "freq_extent": "min",
    "freq_middle": None,
}

def _z3_constraint_solver(*, # prohibit use of positional arguments
        constraints: Dict[str, Union[int, float, str, None]],
        loose_params: List[str],
        make_suggestions: bool,
    ) -> Dict[str, Union[int, float]]:
    """Solves the constraints for an FFTDimension.

    This method solves the linear system of equations to correctly match
    the mathematics of the Fourier transform for a given set of constraints.
    For examples, please refer to the documentation of FFTDimension.

    Parameters
    ----------
    loose_params : List[str]
        If `n` is not explicitly defined it is in general not a natural number.
        Therefore it needs to be rounded up to the next natural number or
        usually for performance reasons to the next power of two. But when the
        original solution was unique this would then violate the other given
        constraints. But when we simultanously improve another parameter to
        match the increase of `n` we again get a unique solution. Here, one or
        multiple of the constraints to loosen have to be defined.
    make_suggestions : bool
        If True and no valid solution can be found, the constraint solver tries
        to make suggestions on how to modify the constraints to find a solution.

    Returns
    -------
    Dict[str, Union[int, float]]
        Numerical values for all constraints defining an FFTDimension.

    Raises
    ------
    NoSolutionFoundError
        An error occured while trying to find a solution.
    NoSolutionFoundError
        There is no unique solution to the supplied constraints.
    NoSolutionFoundError
        The supplied constraints are not satisfiable.
    NoSolutionFoundError
        An invalid value encountered for the supplied constraint values or in
        the solution.
    """

    # Prepare input arguments and check validity
    user_constraints: Dict[str, Union[float, int, str]] = {
        k: v for k,v in constraints.items() if v is not None
    }
    _validate_args(user_constraints, loose_params)
    all_constraints: List[BoolRef] = _get_constraints(user_constraints)

    # Check if system is overconstrained and suggest to remove params in that
    # case
    if _no_solution_exists(all_constraints):
        if make_suggestions:
            suggested_removed_params = _suggest_removed_params(
                user_constraints,
                loose_params
            )
            raise NoSolutionFoundError(
                all_constraints,
                suggested_removed_params=suggested_removed_params
            )
        raise NoSolutionFoundError(all_constraints)

    # Find unique model of the system or suggest additional params if not
    # existent
    model = _get_unique_model_if_exists(
        all_constraints,
        user_constraints,
        make_suggestions
    )

    # Return model if n supplied by user or the found n is a valid n (even/power
    # of two)
    current_n = _z3_to_float(model[Real("n")])
    if isinstance(user_constraints["n"], (int, np.integer)) \
        or current_n == _round_up_n(current_n, user_constraints["n"]): # type: ignore
        return _finalize_model(model)

    # If no loose params are supplied for n widening -> suggest loose params
    if len(loose_params) == 0:
        if make_suggestions:
            suggested_loose_params = _suggest_loose_params(user_constraints)
            raise NoSolutionFoundError(
                all_constraints,
                suggested_loose_params=suggested_loose_params
            )
        raise NoSolutionFoundError(all_constraints)

    # Perform distributed n widening over all loose param groups to reach next
    # valid n
    model = _perform_n_widening_all(
        user_constraints,
        loose_params,
        model=model,
        make_suggestions=make_suggestions
    )

    return _finalize_model(model)

def _get_constraints(
        user_constraints: Dict[str, Union[float, int, str]],
        loose_params: Optional[List[str]] = None,
        previous_model: Optional[ModelRef] = None,
        widening_n: Optional[Union[int, float]] = None,
        add_constraints: Optional[BoolRef] = None
    ) -> List[BoolRef]:
    """
    Merge the user supplied constraints with the general constraints of an FFT.

    If the user supplies a number for n through user_constraints, check whether
    it is odd or even and choose constraints accordingly. If no numerical value
    for n is defined, choose even constraints.

    Features for rounding to next valid n (n widening):
        1) If loose params are supplied, use the respective constraint values as
           bounds.
        2) During n-widening, a previous_model can be supplied which overwrites
           the user_constraints.
        3) Additionally, during the n widening process, a different widening_n
           as well as
        extra add_constraints can be supplied. The add_constraint are used to
        ensure symmetric widening to interdependent parameters, e.g., pos_min
        and pos_max.
    """

    z3_vars: Dict[str, Real] = {var_name: Real(var_name) for var_name in VARS_WITH_PROPS}

    # Define general FFT constraint system which applies to even and odd n
    all_constraints: List[BoolRef] = [
        z3_vars["d_freq"] * z3_vars["d_pos"] * z3_vars["n"] == 1,
        z3_vars["n"] >= 1,
        z3_vars["pos_extent"] == z3_vars["pos_max"] - z3_vars["pos_min"],
        z3_vars["pos_extent"] == z3_vars["d_pos"] * (z3_vars["n"] - 1.),
        z3_vars["freq_extent"] == z3_vars["freq_max"] - z3_vars["freq_min"],
        z3_vars["freq_extent"] == z3_vars["d_freq"] * (z3_vars["n"] - 1.),
        z3_vars["d_pos"] > 0,
        z3_vars["d_freq"] > 0,
        z3_vars["pos_extent"] >= 0,
        z3_vars["freq_extent"] >= 0,
    ]

    user_constraints = user_constraints.copy()

    # n is the only parameter that is always defined in user constraints.
    # Here we have two categories where n is either defining the
    # numerical value (if case) or the rounding mode (else case).
    # If n defines a rounding mode, we always require an even n.
    if isinstance(user_constraints["n"], (int, np.integer)):
        if widening_n is not None:
            # If explicit n is given, we are in the process of n widening where
            # we always use the constraints for even n. In this case, the n
            # supplied by widening_n is usually not an integer but a float.
            # Still, we set n_is_even to True here since we round up to an even
            # n.
            n_is_even: bool = True
        else:
            # If no widening_n is supplied, we simply check if n is even or not.
            n_is_even = user_constraints["n"] % 2 != 1
    else:
        # Remove n from user_constraints if it defines a rounding mode.
        # From here on, we treat all user constraints as numerical values.
        del user_constraints["n"]
        n_is_even = True

    # Because the defined minimum, maximum and middle/offset values should be
    # actual grid points, there are two different constraint systems for even
    # and odd n
    if n_is_even:
        all_constraints += z3_vars["pos_middle"] == z3_vars["pos_min"] \
            + z3_vars["d_pos"] * z3_vars["n"] / 2,
        all_constraints += z3_vars["freq_middle"] == z3_vars["freq_min"] \
            + z3_vars["d_freq"] * z3_vars["n"] / 2,
    else:
        all_constraints += z3_vars["pos_middle"] == z3_vars["pos_min"] \
            + z3_vars["d_pos"] * (z3_vars["n"]-1) / 2,
        all_constraints += z3_vars["freq_middle"] == z3_vars["freq_min"] \
            + z3_vars["d_freq"] * (z3_vars["n"]-1) / 2,

    if add_constraints is not None:
        all_constraints += [add_constraints]

    # If previous model and loose params are supplied, use these values instead
    # of the user_constraints
    if previous_model:
        standard_values = {
            param: previous_model[z3_vars[param]] for param in user_constraints
        }
    else:
        standard_values = user_constraints.copy()

    # During n widening it is required to reach an explicit n which is therefore
    # overwritten here
    if widening_n is not None:
        standard_values["n"] = widening_n

    if loose_params is None:
        # use user's constraints without loose params
        for var in user_constraints:
            all_constraints.append(z3_vars[var] == user_constraints[var])
    else:
        # use user's loosened constraints
        for param in standard_values:
            if param in loose_params:
                if VARS_WITH_PROPS[param] == "max":
                    # use constraint values as upper bound
                    all_constraints.append(z3_vars[param] <= standard_values[param])
                elif VARS_WITH_PROPS[param] == "min":
                    # use constraint values as lower bound
                    all_constraints.append(z3_vars[param] >= standard_values[param])
                else:
                    raise ConstraintSolverError("{var} cannot be a loose_param.")
            else:
                all_constraints.append(z3_vars[param] == standard_values[param])

    return all_constraints

def _validate_args(
        user_constraints: Dict[str, Union[float, int, str]],
        loose_params: List[str]
    ) -> None:
    """
    Check supplied values for consistency and general validity. If the values
    are provided as jax.numpy.DeviceArray these are converted to normal scalar
    values. Additionally, possible duplicates in loose_params are removed
    mutably. Apart from these mutable corrections, if check is successful, do
    nothing, otherwise raise an error.
    """

    for var, val in user_constraints.items():
        if not var in VARS_WITH_PROPS:
            raise ConstraintSolverError(f"'{var}' is not a valid argument name.")
        if hasattr(val, "shape"):
            val = np.array([val])[0]
            user_constraints[var] = val
        if not (var == 'n' and isinstance(val, str)):
            if (isinstance(val, bool) or not
                    isinstance(val, (int, float, np.integer, np.floating))):
                raise ConstraintValueError(
                    f"Supplied constraint ({val}) for {var} is not a real number. \
                    Only int and float are supported."
                )
            _check_overflow(var, val) # type: ignore

    # Check validity of user_constraints["n"]
    if "n" not in user_constraints:
        raise ConstraintSolverError('There is neither an explicit numerical ' + \
            'value for n defined, nor one of the rounding modes ("even" or ' + \
            '"power_of_two") is chosen. Please supply one of these.'
        )
    elif isinstance(user_constraints["n"], (int, np.integer)):
        if len(loose_params) != 0:
            raise ConstraintSolverError(
                "It is not supported to supply loose params with explicitly " + \
                "defined n. The loose params are only used to find a valid n " + \
                "but intentionally not for making overconstrained systems solvable."
            )
        if user_constraints["n"] < 1:
            raise ConstraintValueError(
                f"Supplied constraint ({user_constraints['n']}) for n should be at least 1."
            )
    elif isinstance(user_constraints["n"], str):
        if user_constraints["n"] not in ["even", "power_of_two"]:
            raise ConstraintValueError(
                f'The available rounding modes for n are: "even", "power_of_two". ' + \
                f'The supplied value "{user_constraints["n"]}" is not valid.'
            )
    else:
        raise ConstraintValueError(
            f"Supplied constraint ({user_constraints['n']}) for n should be " + \
            "of type int or str. The supplied value is of type " + \
            f"{type(user_constraints)}. Please set an explicit numerical value " + \
            'for n (int) or choose one of the rounding modes ("even" or ' + \
            '"power_of_two")'
        )

    loose_params = list(set(loose_params)) # remove duplicates
    for loose_param in loose_params:
        if not loose_param in user_constraints:
            raise ConstraintSolverError(
                "You can only define a used constraint as a loose_param."
            )
        if loose_param == "freq_middle":
            raise ConstraintSolverError(
                "The frequency offset cannot be a loose_param."
            )
        if loose_param == "pos_middle":
            raise ConstraintSolverError(
                "The position of the center of position space ('pos_middle') " + \
                "cannot be a loose_param."
            )

def _no_solution_exists(constraints: List[BoolRef]) -> bool:
    """
    Returns True if no solutions to the constraints exists, otherwise False.
    """
    s = Solver()
    s.add(constraints)
    return s.check() == CheckSatResult(Z3_L_FALSE)

def _get_unique_model_if_exists(
        constraints: List[BoolRef],
        user_constraints: Dict[str, Union[float, int, str]],
        make_suggestions: bool
    ) -> ModelRef:
    """
    Try to find a unique model for the supplied constraints and return it.
    If no unique model exists raise a NoUniqueSolutionError and suggest
    additional constraints if make_suggestions is True.
    If no model can be found at all raise a NoSolutionFoundError.
    """

    s = Solver()
    s.add(constraints)
    # Check if constraint system is satisfiable
    if s.check() == CheckSatResult(Z3_L_TRUE):
        # Find a model which solves the constraint system
        model = s.model()
        # Modify the constraint system to forbid the first found solution
        different_sol_constraints = [
            Real(var_name) != model[Real(var_name)]
            for var_name in VARS_WITH_PROPS
            if (not var_name in user_constraints \
            or isinstance(user_constraints[var_name], str))
        ]
        s.add(Or(different_sol_constraints))
        # Check if constraint system is still satisfiable
        # If it is, there is no unique solution to the constraints
        if s.check() == CheckSatResult(Z3_L_TRUE):
            if make_suggestions:
                suggested_additional_params = _suggest_additional_params(user_constraints)
                raise NoUniqueSolutionError(
                    constraints,
                    suggested_additional_params=suggested_additional_params
                )
            raise NoUniqueSolutionError(constraints)
        # if the constraint system is not satisfiable anymore
        # the solution is unique and the model can be returned
        elif s.check() == CheckSatResult(Z3_L_FALSE):
            return model
    raise NoSolutionFoundError(constraints)

def _perform_n_widening_all(
        user_constraints: Dict[str, Union[float, int, str]],
        loose_params: List[str],
        model: ModelRef,
        make_suggestions: bool
    ) -> ModelRef:
    """
    If the found model's n is not valid the loose params are
    modified/improved to find a solution with a valid n.

    The loose params are grouped into sets which can than be modified
    without affecting the constraints from the other groups.
    The difference between the invalid n and the valid target n is then
    equally split between all existing groups in the loose params.
    """

    grouped_loose_params = _group_loose_params(loose_params)
    n_invalid = _z3_to_float(model[Real("n")])
    n_target = _round_up_n(n_invalid, user_constraints["n"]) # type: ignore
    n_budget_per_step = (n_target - n_invalid) / len(grouped_loose_params)
    # Try to improve the loose params of one group by moving a step closer to
    # the target n
    for i, loose_param_group in enumerate(grouped_loose_params):
        group_constraint = _make_constraint_for_loose_param_group(
            loose_param_group,
            user_constraints
        )
        current_n = n_invalid + n_budget_per_step * (i+1) if i+1 < len(loose_params) else n_target
        # Define constraints for the next widening step using the values from
        # the previous model and the current_n which apprpoaches the target n
        # step by step
        all_constraints = _get_constraints(
            user_constraints,
            loose_param_group,
            previous_model=model,
            widening_n=current_n,
            add_constraints=group_constraint
        )
        try:
            model = _get_unique_model_if_exists(
                all_constraints,
                user_constraints,
                make_suggestions=False
            )
        except ConstraintSolverError:
            original_constraints = _get_constraints(
                user_constraints,
                loose_params
            )
            if make_suggestions:
                suggested_loose_params = _suggest_loose_params(user_constraints)
                raise NoSolutionFoundError(
                    original_constraints,
                    suggested_loose_params=suggested_loose_params
                )
            raise NoSolutionFoundError(original_constraints)
    return model

def _group_loose_params(
        loose_params: List[str],
        find_suggestions: bool = False
    ) -> List[List[str]]:
    """
    There are different dependencies of parameters in the FFT constraint system
    which allow sorting those into sets of independent groups such that these
    can be used for the n widening or for finding suggestions of loose params
    for the user. This method fulfills two purposes: it is mainly used to group
    user-provided loose params. By providing the user defined parameters through
    the loose_params argument and setting find_suggestions to True, this method
    can suggest loose_params that possibly match the user's constraints.
    """

    if find_suggestions:
        groups = [["d_freq"], ["d_pos"], ["freq_min", "freq_max"],
                  ["freq_extent"], ["pos_min", "pos_max"], ["pos_extent"]]
    else:
        groups = [["d_freq"], ["d_pos"],
                  ["freq_min", "freq_max", "freq_extent"],
                  ["pos_min", "pos_max", "pos_extent"]]
    grouped_loose_params = []
    for group in groups:
        # If one or multiple params of one group exist, add those to
        # grouped_loose_params
        group_params = [param for param in group if param in loose_params]
        if len(group_params) > 0:
            grouped_loose_params += [group_params]
    return grouped_loose_params

def _make_constraint_for_loose_param_group(
        loose_param_group: List[str],
        user_constraints: Dict[str, Union[float, int, str]]
    ) -> Optional[BoolRef]:
    """
    Within the loose param groups ["*_min", "*_max", "*_extent"] with *=pos/freq
    we want to widen "*_min" and "*_max" symmetrically if both are named as
    loose params. Therefore we create an additional constraint ensuring that
    here.
    """

    space = "freq" if any(["freq" in param for param in loose_param_group]) else "pos"
    if any("min" in param for param in loose_param_group) \
        and any("max" in param for param in loose_param_group):
        return user_constraints[f"{space}_min"] \
            - Real(f"{space}_min") == Real(f"{space}_max") \
            - user_constraints[f"{space}_max"]
    return None

def _round_up_n(current_n: float, rounding_mode: str) -> int:
    """
    If n is not explicitly defined the constraint solver, in general, finds a
    non-int value. Starting from that as the current_n, this method rounds up to
    the next even number or power of two.
    """

    try:
        if rounding_mode == 'power_of_two':
            valid_n = round_up_to_next_power_of_two(current_n)
        elif rounding_mode == 'even':
            valid_n = int(np.ceil(current_n))
            # if valid_n is odd, add 1 to reach the next even number
            if valid_n % 2 == 1:
                valid_n += 1
        return valid_n
    except Exception as e:
        raise ConstraintValueError(
            f"The above error occured while trying to round up n={current_n} " + \
            "to the next valid integer."
        ) from e

def _finalize_model(model: ModelRef) -> Dict[str, Union[int, float]]:
    """
    Final conversion of model to dict with values for all parameters
    and final numerical tests before returning the solution.
    """

    sol: Dict[str, Union[int, float]] = _model_as_float_dict(model)
    sol["n"] = int(sol["n"])
    for val, var in sol.items():
        _check_overflow(val, var)
    return sol

def _z3_to_float(num: Union[RatNumRef, AlgebraicNumRef]) -> float:
    """
    Convert values from special z3 types to standard float values.
    """

    if isinstance(num, AlgebraicNumRef):
        num = num.approx(precision=15)
    num = num.as_fraction()
    try:
        return num.numerator / num.denominator
    except OverflowError:
        # if normal division would result in an OverflowError
        overflow_approx = decimal.Decimal(num.numerator // num.denominator)
        raise ConstraintValueError(
            "Constraint solver results in too large value for some " + \
            f"parameter: {overflow_approx:.2e}."
        )

def _model_as_float_dict(model: ModelRef) -> Dict[str, float]:
    sol: Dict[str, float] = {}
    for param in VARS_WITH_PROPS:
        sol[param] = _z3_to_float(model[Real(param)])
    return sol

def round_up_to_next_power_of_two(value: Union[int, float]) -> int:
    """Finds the next integer power of two larger than or equal to supplied
    value.

    Parameters
    ----------
    value : Union[int, float]
        Number from which to round up to next power of two.

    Returns
    -------
    int
        Next integer power of two
    """


    return max(1, int(2**np.ceil(np.log2(value))))

def _check_overflow(var: str, val: Union[float, int]) -> None:
    """Check for int and float overflow.

    Raises
    ------
    ConstraintValueError
        If `val` is infinite or not a number or
        if `val` is an `int` and not in the range
        `[-sys.maxsize-1, sys.maxsize]` or
        if `val` is a `float` and its absolute value is not in the range
        `[0, sys.float_info.max]` and not `0.0`
    """

    if isinstance(val, float):
        if abs(val) > sys.float_info.max:
            raise ConstraintValueError(var_name=var, var_value=val)
    if isinstance(val, int):
        if val > sys.maxsize or val < -sys.maxsize-1:
            raise ConstraintValueError(var_name=var, var_value=val)
    if np.isinf(val) or np.isnan(val):
        raise ConstraintValueError(var_name=var, var_value=val)

def _suggest_loose_params(
        user_constraints: Dict[str, Union[float, int, str]]
    )-> Optional[List[List[str]]]:
    """
    Try different combinations of loose params within independent groups
    and suggest those to the user which yield a solution.
    """

    groups = _group_loose_params(
        list(user_constraints.keys()),
        find_suggestions=True
    )
    suggested_loose_params = []
    for group in groups:
        try:
            _z3_constraint_solver(
                constraints=user_constraints, # type: ignore
                loose_params=group,
                make_suggestions=False
            )
            suggested_loose_params += [group]
        except:
            pass
    return suggested_loose_params

def _suggest_additional_params(
        user_constraints: Dict[str, Union[float, int, str]]
    ) -> Optional[List[List[str]]]:
    """
    Find suggestions for additional parameters which could yield a unique
    solution. The strategy is to understand which of the required groups is
    missing and to suggest those. However, no explicit tests are performed.
    """

    smart_constraints: Dict[str, Any] = user_constraints.copy()
    groups = [
        ["n"],
        ["d_freq", "freq_extent"],
        ["d_pos", "pos_extent"],
        ["freq_min", "freq_max", "freq_middle"],
        ["pos_min", "pos_max", "pos_middle"]
    ]

    at_least_two_true = lambda a,b,c: a and (b or c) or (b and c)

    spaces = ["pos", "freq"]
    for i, space in enumerate(spaces):
        # Check if one of the groups is implicitly defined by
        # other ones and add those to the smart constraints manually
        if at_least_two_true(
            f"d_{space}" in user_constraints,
            f"{space}_extent" in user_constraints,
            isinstance(user_constraints["n"], (int, np.integer))
        ):
            smart_constraints[f"d_{spaces[i-1]}"] = None
            smart_constraints["n"] = None

    missing_groups = []
    for group in groups:
        # Add the groups to the suggestions if these are not represented
        if not any(param in group for param in list(smart_constraints.keys())):
            missing_groups += [group]
    return missing_groups

def _suggest_removed_params(
        user_constraints: Dict[str, Union[float, int, str]],
        loose_params: List[str]
    ) -> Optional[List[str]]:
    """
    Try to remove one of the constraints at a time and suggest
    those removed parameters to the user which yield a solution.
    """
    suggested_removed_params = []
    for constraint in user_constraints:
        reduced_constraints = user_constraints.copy()
        if constraint == "n":
            reduced_constraints["n"] = "even"
        else:
            del reduced_constraints[constraint]
        try:
            _z3_constraint_solver(
                constraints=reduced_constraints, # type: ignore
                loose_params=loose_params,
                make_suggestions=False
            )
            suggested_removed_params += [constraint]
        except:
            pass
    return suggested_removed_params
