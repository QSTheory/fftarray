from functools import partial
from typing import Callable, Optional, TypeVar, Tuple, Any, List

import numpy as np
from numpy.typing import NDArray
import jax
from jax.tree_util import tree_unflatten, register_pytree_node
import jax.numpy as jnp

from ..fft_array import FFTDimension, FFTArray, PosArray, FreqArray, LazyState
from .tensor_lib import TensorLib, PrecisionSpec


class JaxTensorLib(TensorLib):

    def __init__(self, precision: PrecisionSpec = "default"):
        TensorLib.__init__(self, precision = precision)
        self.fftn = lambda values, precision: jax.numpy.fft.fftn(values)
        self.ifftn = lambda values, precision: jax.numpy.fft.ifftn(values)
        self.numpy_ufuncs = jax.numpy
        self.array = jnp.array

    def reduce_multiply(self, values) -> Any:
        """Based on :func:`jax.lax.reduce`.

        :meta private:
        """
        values_jnp: jnp.ndarray = jnp.array(values)
        return jax.lax.reduce(
            operands=values_jnp,
            init_values=jnp.array(1, dtype = values_jnp.dtype),
            computation=jnp.multiply,
            dimensions=range(len(values_jnp.shape))
        )


def make_matched_input(f, input, loop_arg):
    is_leaf = lambda x: isinstance(x, FFTArray)
    leaves_high_in, tree_def_high_in = jax.tree_util.tree_flatten(input, is_leaf = is_leaf)

    out_val = jax.eval_shape(f, input, loop_arg)[0]
    leaves_high_out, _ = jax.tree_util.tree_flatten(out_val, is_leaf = is_leaf)

    # This check is intentionally coarse in order to not interfere with the
    # weak-type adjustments of scan.
    assert len(leaves_high_in) == len(leaves_high_out)

    new_leaves = []
    for leaf_in, leaf_out in zip(leaves_high_in, leaves_high_out):
        if isinstance(leaf_in, FFTArray):
            lazy_state = leaf_out._lazy_state
            if isinstance(leaf_out, PosArray):
                leaf_in = leaf_in.pos_array()
            else:
                assert isinstance(leaf_out, FreqArray)
                leaf_in = leaf_in.freq_array()
            if lazy_state is not None:
                leaf_in = leaf_in._set_lazy_state(lazy_state)
            new_leaves.append(leaf_in)
        else:
            new_leaves.append(leaf_in)

    matched_input = tree_unflatten(tree_def_high_in, new_leaves)
    return matched_input


X = TypeVar('X')
Y = TypeVar('Y')


Carry = TypeVar('Carry')
def fft_array_scan(f: Callable[[Carry, X], Tuple[Carry, Y]],
        init: Carry,
        xs: X,
        length: Optional[int] = None,
        reverse: bool = False,
        unroll: int = 1,
    ) -> Tuple[Carry, Y]:

    if xs is None:
        first_x = 0
    else:
        # TODO Actually split the PyTree here
        first_x = xs[0] #type: ignore

    matched_input = make_matched_input(f, input = init, loop_arg = first_x)
    return jax.lax.scan(
        f,
        init=matched_input,
        xs=xs,
        length=length,
        reverse=reverse,
        unroll=unroll
    )


def fftarray_flatten(arr: FFTArray) -> Tuple[Tuple[Any], Tuple[Tuple[FFTDimension, ...], Optional[LazyState], TensorLib]]:
    children = (arr._values,)
    aux_data = (arr._dims, arr._lazy_state, arr._tlib)
    return (children, aux_data)

def fftarray_unflatten(aux_data, children, cls):
    (values,) = children
    (dims, lazy_phase_factors, tensor_lib) = aux_data
    # We explicitly do not want to call the constructor here.
    # The consistency check fails (needlessly) for PyTreeArrays and other special "tricks".
    self = cls.__new__(cls)
    self._values = values
    self._dims = dims
    self._lazy_state = lazy_phase_factors
    self._tlib = tensor_lib
    return self

register_pytree_node(
    PosArray,
    fftarray_flatten,
    partial(fftarray_unflatten, cls=PosArray),
)
register_pytree_node(
    FreqArray,
    fftarray_flatten,
    partial(fftarray_unflatten, cls=FreqArray),
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
        v._d_freq,
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
    fftdim._d_freq = aux_data[7]
    return fftdim

from jax.tree_util import register_pytree_node
register_pytree_node(
    FFTDimension,
    fft_dimension_flatten,
    fft_dimension_unflatten,
)


def flatten_lazy_state(state: LazyState):
    children: Tuple[Any,...] = tuple()
    aux_data = (state._phases_per_dim,state._scale)
    return (children, aux_data)

def unflatten_lazy_state(aux_data, children):
    # () = children
    (phase,scale) = aux_data
    # We explicitly do not want to call the constructor here.
    # The consistency check fails (needlessly) for PyTreeArrays and other
    # special "tricks".
    self = LazyState.__new__(LazyState)
    self._phases_per_dim = phase
    self._scale = scale
    return self

from jax.tree_util import register_pytree_node
register_pytree_node(
    LazyState,
    flatten_lazy_state,
    unflatten_lazy_state,
)