from typing import Iterable, List, Literal, Union, Tuple, Any, Optional

import array_api_strict
import array_api_compat
import array_api_compat.numpy as cnp
import numpy as np
import pytest

import fftarray as fa
from fftarray._src.array import _convert_xp

XPS = [
    array_api_strict,
    array_api_compat.get_namespace(np.asarray(1.)),
]

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    XPS.append(array_api_compat.get_namespace(jnp.asarray(1.)))
    fa.jax_register_pytree_nodes()
except ImportError:
    pass

# This is helpful for tests where we need an xp which is not the currently tested one.
XPS_ROTATED_PAIRS = [
    pytest.param(xp1, xp2) for xp1, xp2 in zip(XPS, [*XPS[1:],XPS[0]], strict=True)
]

def _get_default_device(xp) -> Any:
    """
        Returns the device on which to expect an array if device=None was passed.
        This differs from array_api_info.default_device which for jax returns None,
        see https://github.com/jax-ml/jax/issues/27606 for an extended explanation.
    """
    return array_api_compat.device(xp.empty(1))

def _generate_non_default_device_pairs(xp) -> List[Tuple[Any, Optional[Any], Any]]:
    """
        Generate for each supported library a pair of a non-default device parameter
        and expected device.
        Because some libraries provide very special devices (like the meta device of torch)
        this list is hand-coded for each array library instead of using the inspection
        apis of the standard.
    """
    res: List[Tuple[Any, Optional[Any], Any]] = []

    if array_api_compat.is_array_api_strict_namespace(xp):
        res.append(
            (xp, xp.Device("device1"), xp.Device("device1")),
        )
    elif array_api_compat.is_jax_namespace(xp):
        import jax
        if len(jax.devices()) > 1:
            res.append(
                (xp, jax.devices()[1], jax.devices()[1]),
            )

    return res

def _get_device_pair_list(xps) -> List[Tuple[Any, Any, Any]]:
    res = []
    for xp in xps:
        res.extend(_generate_non_default_device_pairs(xp))
    return res


XPS_WITH_DEFAULT_DEVICE_PAIRS = [(xp, _get_default_device(xp)) for xp in XPS]
XPS_NON_DEFAULT_DEVICE_PAIRS = _get_device_pair_list(XPS)

# Unification of default and non-default device parameters and expected devices.
XPS_DEVICE_PAIRS = [
    # When passing in None, the created array is expected to have the device _get_default_device(xp)
    *[(xp, None, _get_default_device(xp)) for xp in XPS],
    *XPS_NON_DEFAULT_DEVICE_PAIRS,
]

def get_other_space(space: Union[fa.Space, Tuple[fa.Space, ...]]):
    """Returns the other space. If input space is "pos", "freq" is returned and
    vice versa. If space is a `Tuple[Space]`, a tuple is returned.
    """
    if isinstance(space, str):
        if space == "pos":
            return "freq"
        return "pos"
    return tuple(get_other_space(s) for s in space)

DTYPE_NAME = Literal[
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
    "complex64",
    "complex128",
]
dtype_names_numeric_core = [
    "int32",
    "int64",
    "uint32",
    "uint64",
    "float32",
    "float64",
    "complex64",
    "complex128",
]
dtypes_names_pairs = [
    pytest.param("bool", "bool"),
    pytest.param("bool", None),
    *[
            pytest.param("uint8", x) for x in [
            "int32",
            "uint64",
            None,
        ]
    ],
    *[
        pytest.param("float32", x) for x in [
            "float32",
            "float64",
            "complex64",
            "complex128",
            None,
        ]
    ],
]


def get_dims(n: int) -> List[fa.Dimension]:
    return [
        fa.dim(str(i), n=4+i, d_pos=1.*(i+1.), pos_min=0., freq_min=0.)
        for i in range(n)
    ]

def get_arr_from_dims(
        xp,
        dims: Iterable[fa.Dimension],
        spaces: Union[fa.Space, Iterable[fa.Space]] = "pos",
        dtype_name: DTYPE_NAME = "float64",
    ):
    dtype=getattr(xp, dtype_name)
    dims = list(dims)
    if isinstance(spaces, str):
        spaces_norm: Iterable[fa.Space] = [spaces]*len(dims)
    else:
        spaces_norm = spaces
    arr = fa.array(
        xp.asarray(
            1.,
            dtype=dtype,
        ),
        [],
        [],
    )
    for dim, space in zip(dims, spaces_norm, strict=True):
        arr += fa.coords_from_dim(dim, space, xp=xp).into_dtype(dtype)
    return arr

def assert_fa_array_exact_equal(x1: fa.Array, x2: fa.Array) -> None:
    x1._check_consistency()
    x2._check_consistency()

    assert x1._dims == x2._dims
    assert x1._eager == x2._eager
    assert x1._factors_applied == x2._factors_applied
    assert x1._spaces == x2._spaces
    assert x1._xp == x2._xp
    assert x1.device == x2.device


    np.testing.assert_equal(
        _convert_xp(x1._values, old_xp=x1._xp, new_xp=cnp),
        _convert_xp(x2._values, old_xp=x1._xp, new_xp=cnp),
    )

def get_test_array(
        xp,
        dim: fa.Dimension,
        space: fa.Space,
        dtype,
        factors_applied: bool,
        device: Optional[Any] = None,
    ) -> fa.Array:
    """ Generates a test array for a given dimension, space and dtype.

    Parameters
    ----------
    xp : Array API namespace

    dim : fa.Dimension

    space : fa.Space

    dtype : dtype of the test data as a member of xp

    factors_applied : bool


    Returns
    -------
    fa.Array

    Raises
    ------
    ValueError
        Raises on unsupported dtype.
    """

    offset = xp.asarray(1. + getattr(dim, f"{space}_middle"), device=device)
    offset = xp.astype(offset, dtype)
    if xp.isdtype(dtype, "bool"):
        assert factors_applied
        return fa.array(xp.arange(dim.n, device=device)%2==0, dim, space, device=device)
    elif xp.isdtype(dtype, "integral"):
        assert factors_applied
        return fa.array(xp.astype(xp.arange(dim.n, device=device), dtype)+offset, dim, space, device=device)
    elif xp.isdtype(dtype, "real floating"):
        assert factors_applied
        return fa.coords_from_dim(dim, space, xp=xp, device=device).into_dtype(dtype)+offset
    elif xp.isdtype(dtype, "complex floating"):
        arr = (fa.coords_from_dim(dim, space, xp=xp, device=device).into_dtype(dtype)+offset)*(1.+1.2j)
        return arr.into_factors_applied(factors_applied)

    raise ValueError(f"Unsupported dtype {dtype}")
