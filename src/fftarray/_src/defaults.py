
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
