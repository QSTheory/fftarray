from typing import Literal

import numpy as np

import array_api_compat

_DEFAULT_XP = array_api_compat.array_namespace(np.array(0))

def set_default_xp(xp) -> None:
    global _DEFAULT_XP
    # We want the wrapped namespace everywhere by default.
    # If the array library fully supports the Python Array API
    # this becomes the default namespace.
    _DEFAULT_XP= array_api_compat.array_namespace(xp.asarray(0))

def get_default_xp():
    return _DEFAULT_XP

def default_xp(xp):
    return DefaultArrayNamespaceContext(xp=xp)

class DefaultArrayNamespaceContext:
    def __init__(self, xp):
        self.override = xp

    def __enter__(self):
        self.previous = get_default_xp()
        set_default_xp(self.override)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_default_xp(self.previous)


"""
    Only floating types are allowed as default dtypes
    because only those make sense to initialize coordinate arrays.
"""
DEFAULT_DTYPE = Literal[
    "float32",
    "float64",
    "complex64",
    "complex128",
]

class DefaultDTypeContext:
    def __init__(self, dtype: DEFAULT_DTYPE):
        self.override = dtype

    def __enter__(self):
        self.previous = get_default_dtype_name()
        set_default_dtype_name(self.override)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_default_dtype_name(self.previous)

# stores the name of the attribute
_DEFAULT_DTYPE_NAME: DEFAULT_DTYPE = "float64"

def set_default_dtype_name(dtype: DEFAULT_DTYPE) -> None:
    global _DEFAULT_DTYPE_NAME
    _DEFAULT_DTYPE_NAME= dtype

def get_default_dtype_name() -> DEFAULT_DTYPE:
    return _DEFAULT_DTYPE_NAME

def default_dtype_name(dtype: DEFAULT_DTYPE) -> DefaultDTypeContext:
    return DefaultDTypeContext(dtype=dtype)

_DEFAULT_EAGER: bool = False

def set_default_eager(eager: bool) -> None:
    global _DEFAULT_EAGER
    _DEFAULT_EAGER= eager

def get_default_eager() -> bool:
    return _DEFAULT_EAGER

def default_eager(eager: bool):
    return DefaultEagerContext(eager=eager)

class DefaultEagerContext:
    def __init__(self, eager: bool):
        self.override = eager

    def __enter__(self):
        self.previous = get_default_eager()
        set_default_eager(self.override)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_default_eager(self.previous)
