from typing import Tuple
from copy import copy
from .array import Array
from .named_array import get_axes_permute

def permute_dims(x: Array, dim_names: Tuple[str, ...], /) -> Array:
        """
            Permutes the dimensions of an Array.
        """
        new_dim_names = list(dim_names)
        old_dim_names = [dim.name for dim in x.dims]
        if len(new_dim_names) == 0:
            new_dim_names = copy(old_dim_names)
            new_dim_names.reverse()
        else:
            assert len(new_dim_names) == len(x.dims)

        axes_permute = get_axes_permute(old_dim_names, new_dim_names)
        permuted_values = x.xp.permute_dims(x._values, tuple(axes_permute))

        permuted_arr = Array(
            values=permuted_values,
            dims=tuple(x.dims[idx] for idx in axes_permute),
            spaces=tuple(x.spaces[idx] for idx in axes_permute),
            eager=tuple(x.eager[idx] for idx in axes_permute),
            factors_applied=tuple(x.factors_applied[idx] for idx in axes_permute),
            xp=x.xp,
        )
        return permuted_arr
