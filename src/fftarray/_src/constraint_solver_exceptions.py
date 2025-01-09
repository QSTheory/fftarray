import json
import sys
from typing import Optional, Union, List
from copy import copy

from z3 import BoolRef


class ConstraintSolverError(Exception):
    """Custom exception class for the constraint solver.

    Adds special formatting to exceptions.

    Parameters
    ----------
    msg : Optional[str], optional
        The message to throw instead of the default message. Defaults to None.
    constraints : Optional[List[BoolRef]], optional
        Constraint system that a solution is searched for. Defaults to None.
    """

    def __init__(
            self,
            msg: Optional[str] = None,
            constraints: Optional[List[BoolRef]] = None
        ):
        self._msg = "Constraint solver failed." if msg is None else msg
        self._constraints = constraints
        super().__init__()

    def get_message(self) -> str:
        if self._constraints is None:
            return self._msg
        constraints_f = json.dumps(
            [str(constraint) for constraint in self._constraints],
            indent=4
        )
        return f"{self._msg} Supplied constraint system:\n {constraints_f}"

    def __str__(self):
        # color the error message:
        # \0339[91m : FAIL (terminal color for FAIL)
        # \033[0m : ENDC (resets the color)
        return "\033[91m" + self.get_message() + "\033[0m"


class NoUniqueSolutionError(ConstraintSolverError):
    """Exception raised if the supplied constraints do not yield a unique
    solution.

    Parameters
    ----------
    constraints : Optional[List[BoolRef]], optional
        Constraint system that a solution is searched for, by default None
    suggested_additional_params : Optional[List[List[str]]], optional
        List of suggested params that the user could add for finding a unique
        solution, by default None
    """

    def __init__(
            self,
            constraints: Optional[List[BoolRef]] = None,
            suggested_additional_params: Optional[List[List[str]]] = None
        ):
        self._suggested_additional_params = suggested_additional_params
        suggestion_msg = ""
        if suggested_additional_params is not None:
            if len(suggested_additional_params) > 0:
                suggestion_str = ', '.join(
                    [str(param) for param in suggested_additional_params]
                )
                multiple_groups: bool = len(suggested_additional_params) > 1
                suggestion_msg = " Choosing an additional constraint from the " + \
                    f"following undefined group{'s' if multiple_groups else ''} " + \
                    f"may yield a unique solution: {suggestion_str}."

        super().__init__(
            msg = "Could not find a unique solution." + suggestion_msg,
            constraints = constraints
        )

class NoSolutionFoundError(ConstraintSolverError):
    """Exception raised if there is no solution to the supplied constraints.

    Parameters
    ----------
    constraints : Optional[List[BoolRef]], optional
        Constraint system that a solution is searched for, by default None
    suggested_loose_params : Optional[List[List[str]]], optional
        Grouped list of loose params which yield a solution, by default None
    suggested_removed_params : Optional[List[str]], optional
        List of params to remove which then yield a solution, by default None
    """

    def __init__(
                self,
                constraints: Optional[List[BoolRef]] = None,
                suggested_loose_params: Optional[List[List[str]]] = None,
                suggested_removed_params: Optional[List[str]] = None
            ):

        self._suggested_loose_params = suggested_loose_params
        self._suggested_removed_params = copy(suggested_removed_params)

        suggestion_msg = ""
        if suggested_loose_params is not None:
            if len(suggested_loose_params) > 0:
                suggestion_str = ', '.join(
                    [str(param) for param in suggested_loose_params]
                )
                suggestion_msg = " Using one or multiple of the following " + \
                    f"list elements as loose params yields a solution: {suggestion_str}."
        if suggested_removed_params is not None:
            if "n" in suggested_removed_params:
                suggested_removed_params.remove("n")
                suggestion_msg = " Find a solution by either removing one " + \
                    f'of the following params: {", ".join(suggested_removed_params)} or choose n="even".'
            elif len(suggested_removed_params) > 0:
                suggestion_msg = " Removing one of the following params " + \
                    f"yields a solution: {', '.join(suggested_removed_params)}."

        super().__init__(
            msg = "Could not find any solution." + suggestion_msg,
            constraints = constraints
        )


class ConstraintValueError(ConstraintSolverError):
    """Exception raised if constraint values exceed specific bounds, i.e., if
    there is an int or float overflow. If `msg` is not given, `var_name` and
    `var_value` are returned.

    Parameters
    ----------
    msg : Optional[str], optional
        The message to throw (overwrites the data generated message), by default
        None
    var_name : Optional[str], optional
        The variable's name, by default None
    var_value : Optional[Union[int, float]], optional
        The variable's value, by default None
    """

    def __init__(
            self,
            msg: Optional[str] = None,
            var_name: Optional[str] = None,
            var_value: Optional[Union[int, float]] = None
        ):
        if msg is None:
            self.var_name = var_name
            self.var_value = var_value
            super().__init__(msg = self.get_msg_from_vars())
        else:
            super().__init__(msg = msg)

    def get_msg_from_vars(self) -> str:
        overflow_info = ""
        if isinstance(self.var_value, int):
            overflow_info = f"Only int in range [{-sys.maxsize-1}, {sys.maxsize}] are supported."
        elif isinstance(self.var_value, float):
            overflow_info = f"Only float in range +-[0, {sys.float_info.max}] are supported."
        var_name_str = self.var_name if self.var_name is not None else ""
        var_value_str = f"({self.var_value})" if self.var_value is not None else ""
        return f"Supplied constraint {var_value_str} for {var_name_str} is not supported. {overflow_info}"
