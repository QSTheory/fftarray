from typing import Union, Tuple

import numpy as np
import array_api_strict
import array_api_compat

import fftarray as fa

XPS = [array_api_strict, array_api_compat.get_namespace(np.asarray(1.))]

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    XPS.append(array_api_compat.get_namespace(jnp.asarray(1.)))
    fa.jax_register_pytree_nodes()
except ImportError:
    pass


def get_other_space(space: Union[fa.Space, Tuple[fa.Space, ...]]):
    """Returns the other space. If input space is "pos", "freq" is returned and
    vice versa. If space is a `Tuple[Space]`, a tuple is returned.
    """
    if isinstance(space, str):
        if space == "pos":
            return "freq"
        return "pos"
    return tuple(get_other_space(s) for s in space)