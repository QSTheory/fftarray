from typing import Literal, get_args

import numpy as np

import array_api_compat

_DEFAULT_XP = array_api_compat.array_namespace(np.asarray(0))

def set_default_xp(xp) -> None:
    global _DEFAULT_XP
    # We want the wrapped namespace everywhere by default.
    # If the array library fully supports the Python Array API
    # this becomes the default namespace.
    _DEFAULT_XP= array_api_compat.array_namespace(xp.asarray(0))

def get_default_xp():
    return _DEFAULT_XP


class DefaultArrayNamespaceContext:
    def __init__(self, xp):
        self.override = xp

    def __enter__(self):
        self.previous = get_default_xp()
        set_default_xp(self.override)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_default_xp(self.previous)

def default_xp(xp) -> DefaultArrayNamespaceContext:
    return DefaultArrayNamespaceContext(xp=xp)



"""
    Only floating types are allowed as default dtypes
    because only those make sense to initialize coordinate arrays.
"""
DEFAULT_PRECISION = Literal[
    "float32",
    "float64",
]

# stores the name of the attribute
_DEFAULT_PRECISION: DEFAULT_PRECISION = "float64"

def check_precision(precision: DEFAULT_PRECISION):
    """Helper function checking whether the provided precision is supported."""
    if precision not in get_args(DEFAULT_PRECISION):
        raise ValueError(
            f"Precision '{precision}' is not supported. " +
            f"Choose one from {get_args(DEFAULT_PRECISION)}."
        )

def set_default_precision(precision: DEFAULT_PRECISION) -> None:
    check_precision(precision)
    global _DEFAULT_PRECISION
    _DEFAULT_PRECISION = precision

def get_default_precision() -> DEFAULT_PRECISION:
    return _DEFAULT_PRECISION

class DefaultPrecisionContext:
    def __init__(self, precision: DEFAULT_PRECISION):
        self.override = precision

    def __enter__(self):
        self.previous = get_default_precision()
        set_default_precision(self.override)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_default_precision(self.previous)

def default_precision(precision: DEFAULT_PRECISION) -> DefaultPrecisionContext:
    check_precision(precision)
    return DefaultPrecisionContext(precision=precision)


_DEFAULT_EAGER: bool = False

def set_default_eager(eager: bool) -> None:
    global _DEFAULT_EAGER
    _DEFAULT_EAGER= eager

def get_default_eager() -> bool:
    return _DEFAULT_EAGER

class DefaultEagerContext:
    def __init__(self, eager: bool):
        self.override = eager

    def __enter__(self):
        self.previous = get_default_eager()
        set_default_eager(self.override)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_default_eager(self.previous)

def default_eager(eager: bool) -> DefaultEagerContext:
    return DefaultEagerContext(eager=eager)
