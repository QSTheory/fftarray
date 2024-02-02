from typing import Callable, Tuple, Any, List, Sequence
from types import ModuleType

from ..fft_array import FFTDimension, FFTArray, Space
from .tensor_lib import TensorLib, PrecisionSpec

import jax
from jax.tree_util import register_pytree_node
import jax.numpy as jnp
from jax.typing import ArrayLike


class JaxTensorLib(TensorLib):

    def __init__(self, precision: PrecisionSpec = "default"):
        TensorLib.__init__(self, precision = precision)

    def fftn(self, values: ArrayLike, axes: Sequence[int]) -> jax.Array: # type: ignore[override]
        return jnp.fft.fftn(values, axes=axes)

    def ifftn(self, values: ArrayLike, axes: Sequence[int]) -> jax.Array: # type: ignore[override]
        return jnp.fft.ifftn(values, axes=axes)

    @property
    def numpy_ufuncs(self) -> ModuleType:
        return jnp

    @property
    def array(self) -> Callable[..., jax.Array]:
        return jnp.array

    def reduce_multiply(self, values) -> Any:
        """Based on :func:`jax.lax.reduce`.

        :meta private:
        """
        values_jnp: jnp.ndarray = jnp.array(values)
        return jax.lax.reduce(
            operands=values_jnp,
            init_values=jnp.array(1, dtype = values_jnp.dtype),
            computation=jnp.multiply,
            dimensions=range(len(values_jnp.shape)),
        )


def fftarray_flatten(
    arr: FFTArray
) -> Tuple[
        Tuple[Any],
        Tuple[
            Tuple[FFTDimension, ...],
            Tuple[Space, ...],
            Tuple[bool, ...],
            Tuple[bool, ...],
            TensorLib
        ]
    ]:
    children = (arr._values,)
    aux_data = (arr._dims, arr._space, arr._eager, arr._factors_applied, arr._tlib)
    return (children, aux_data)

def fftarray_unflatten(aux_data, children) -> FFTArray:
    (values,) = children
    (dims, space, eager, factors_applied, tensor_lib) = aux_data
    # We explicitly do not want to call the constructor here.
    # The consistency check fails (needlessly) for PyTreeArrays and other special "tricks".
    self = FFTArray.__new__(FFTArray)
    self._values = values
    self._dims = dims
    self._space = space
    self._eager = eager
    self._factors_applied = factors_applied
    self._tlib = tensor_lib
    return self

register_pytree_node(
    FFTArray,
    fftarray_flatten,
    fftarray_unflatten,
)


def fft_dimension_flatten(v: FFTDimension) -> Tuple[List[Any], List[Any]]:
    """The `flatten_func` used by `jax.tree_util.register_pytree_node` to
    flatten an FFTDimension.

    :meta private:

    Parameters
    ----------
    v : FFTDimension
        The FFTDimension to flatten.

    Returns
    -------
    Tuple[List[Any], List[Any]]
        The flatted FFTDimension. Contains ``children`` and ``aux_data``.

    See Also
    --------
    jax.tree_util.register_pytree_node
    """
    aux_data: List[Any] = [
        v._n,
        v._name,
        v._default_tlib,
        v._default_eager,
        v._pos_min,
        v._freq_min,
        v._d_pos,
    ]
    children: List[Any] = []
    return (children, aux_data)

def fft_dimension_unflatten(aux_data, children) -> FFTDimension:
    """The `unflatten_func` used by `jax.tree_util.register_pytree_node` to
    unflatten an FFTDimension.

    :meta private:

    Parameters
    ----------
    aux_data : list
        Auxiliary data.
    children : list
        Flattened children.

    Returns
    -------
    FFTDimension
        The unflattened FFTDimension.

    See Also
    --------
    jax.tree_util.register_pytree_node
    """
    fftdim = FFTDimension.__new__(FFTDimension)
    fftdim._n = aux_data[0]
    fftdim._name = aux_data[1]
    fftdim._default_tlib = aux_data[2]
    fftdim._default_eager = aux_data[3]
    fftdim._pos_min = aux_data[4]
    fftdim._freq_min = aux_data[5]
    fftdim._d_pos = aux_data[6]
    return fftdim

from jax.tree_util import register_pytree_node
register_pytree_node(
    FFTDimension,
    fft_dimension_flatten,
    fft_dimension_unflatten,
)
