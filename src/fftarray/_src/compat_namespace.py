from typing import Optional, Any

import numpy as np
import array_api_compat

def get_compat_namespace(xp):
    """
        Wraps a namespace in a compatibility wrapper if necessary.
    """
    return get_array_compat_namespace(xp.asarray(1))

def get_array_compat_namespace(x):
    """
        Get the Array API compatible namespace of x.
        This is basically a single array version of `array_api_compat.array_namespace`.
        But it has a special case for torch because `array_api_compat.array_namespace`
        is currently incompatible with `torch.compile`.
    """
    # Special case torch array.
    # As of pytorch 2.6.0 and array_api_compat 1.11.0
    # torch.compile is not compatible with `array_api_compat.array_namespace`.
    if array_api_compat.is_torch_array(x):
        import array_api_compat.torch as torch_namespace
        return torch_namespace

    return array_api_compat.array_namespace(x)

def convert_xp(x, old_xp, new_xp, dtype: Optional[Any] = None, copy: Optional[bool] = None):
    """
        The Array API standard does not support conversion between array namespaces.
        Therefore this function only supports conversions between libraries where the interaction
        is specifically allowed. This implementation is currently not optimal for performance
        and always goes the safe route via NumPy.
    """
    if old_xp == new_xp:
        return new_xp.asarray(x, dtype=dtype, copy=copy)

    if array_api_compat.is_array_api_strict_namespace(old_xp):
        x_np = array_api_compat.to_device(x, old_xp.__array_namespace_info__().default_device())
    elif array_api_compat.is_numpy_array(x):
        x_np = x
    elif array_api_compat.is_jax_array(x):
        assert copy is None or copy
        x_np = np.array(x)
    elif array_api_compat.is_torch_array(x):
        assert copy is None or copy
        x_np = np.array(x.cpu())
    elif array_api_compat.is_cupy_array(x):
        assert copy is None or copy
        x_np = x.get()
    else:
        # TODO: Just raise a warning and try np.array(x)?
        raise ValueError(
            f"Tried to convert from {old_xp} " \
            "which does not have a specific implementation. " \
            "The Array API standard does not support conversion between array namespaces " \
            "and there is no specific work-around for this namespace."
        )

    if (
        not array_api_compat.is_jax_namespace(new_xp)
        and not array_api_compat.is_numpy_namespace(new_xp)
        and not array_api_compat.is_torch_namespace(new_xp)
        and not array_api_compat.is_array_api_strict_namespace(new_xp)
        and not array_api_compat.is_cupy_namespace(new_xp)
    ):
        raise ValueError(f"Tried to convert to unsupported namespace {new_xp}.")

    return new_xp.asarray(x_np, dtype=dtype, copy=copy)
