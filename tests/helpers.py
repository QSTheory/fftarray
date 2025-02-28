from typing import Iterable, List, Literal, Union, Tuple

import array_api_strict
import array_api_compat
import numpy as np
import pytest

import fftarray as fa

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
    np.testing.assert_equal(
        np.array(x1._values),
        np.array(x2._values),
    )

def get_test_array(
        xp,
        dim: fa.Dimension,
        space: fa.Space,
        dtype,
        factors_applied: bool,
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

    offset = xp.asarray(1. + getattr(dim, f"{space}_middle"))
    offset = xp.astype(offset, dtype)
    if xp.isdtype(dtype, "bool"):
        assert factors_applied
        return fa.array(xp.arange(dim.n)%2==0, dim, space)
    elif xp.isdtype(dtype, "integral"):
        assert factors_applied
        return fa.array(xp.astype(xp.arange(dim.n), dtype)+offset, dim, space)
    elif xp.isdtype(dtype, "real floating"):
        assert factors_applied
        return fa.coords_from_dim(dim, space, xp=xp).into_dtype(dtype)+offset
    elif xp.isdtype(dtype, "complex floating"):
        arr = (fa.coords_from_dim(dim, space, xp=xp).into_dtype(dtype)+offset)*(1.+1.2j)
        return arr.into_factors_applied(factors_applied)

    raise ValueError(f"Unsupported dtype {dtype}")
