from ..backends.backend import Backend
from ..backends.numpy import NumpyBackend


_DEFAULT_BACKEND: Backend = NumpyBackend("default")

def set_default_backend(backend: Backend) -> None:
    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND= backend

def get_default_backend() -> Backend:
    return _DEFAULT_BACKEND

def default_backend(backend: Backend):
    return DefaultBackendContext(backend=backend)

class DefaultBackendContext:
    def __init__(self, backend: Backend):
        self.override = backend

    def __enter__(self):
        self.previous = get_default_backend()
        set_default_backend(self.override)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_default_backend(self.previous)

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
