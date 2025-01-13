from typing import Sequence, Tuple, Any, List, Dict
from dataclasses import dataclass

def align_named_arrays(
        arrays: Sequence[Tuple[Sequence[str], Any]],
        xp,
    ) -> Tuple[Sequence[str], List[Any]]:
    """
        The arrays may have longer shapes than there are named dims.
        Those are always kept as the last dims.
        Reorders and expands dimensions so that all arrays have the same dim-names and shapes.

        Unnamed shapes may differ!
        This allows aligning all named dimensions of differently typed trees.

        Returns the new dim-names and the list of aligned arrays.
    """
    target_shape: Dict[str, int] = {}
    for dims, arr in arrays:
        for i, dim in enumerate(dims):
            if dim in target_shape:
                assert target_shape[dim] == arr.shape[i], \
                    "Cannot align arrays with different lengths "+ \
                    f"({target_shape[dim]}, {arr.shape[i]}) in the same dim {dim}"
            else:
                target_shape[dim] = arr.shape[i]

    target_indices = {name: i for i, name in enumerate(target_shape.keys())}
    aligned_arrays = []
    for dims, arr in arrays:
        dim_names = [*dims]
        for target_dim in target_shape.keys():
            if target_dim not in dims:
                arr = xp.reshape(arr, (-1, *arr.shape))
                dim_names.insert(0, target_dim)
        # TODO the list conversion of keys should not be necessary but is needed for mypy
        arr = transpose_array(
            arr,
            old_dims=dim_names,
            new_dims=list(target_shape.keys()),
            xp=xp,
        )
        aligned_arrays.append(arr)
    return list(target_indices.keys()), aligned_arrays


@dataclass
class FillDim:

    index: int

    def __hash__(self):
        return hash(self.index)

def get_axes_transpose(
            old_dims: Sequence[str],
            new_dims: Sequence[str]
        ) -> Tuple[int, ...]:
    assert len(old_dims) == len(new_dims)
    dim_index_lut = {dim: i for i, dim in enumerate(old_dims)}
    return tuple(dim_index_lut[target_dim] for target_dim in new_dims)


def transpose_array(
        array: Any,
        xp,
        old_dims: Sequence[str],
        new_dims: Sequence[str]
    ) -> Any:
    """
        `old_dims` and `new_dims` must be a transpose of one another.
        They may be shorter than array.shape. The last dims are left untouched.
    """
    axes_transpose = get_axes_transpose(old_dims, new_dims)
    array = xp.permute_dims(array, tuple(axes_transpose))
    return array
