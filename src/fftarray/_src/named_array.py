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
    for dim_names, arr in arrays:
        for i, dim_name in enumerate(dim_names):
            if dim_name in target_shape:
                assert target_shape[dim_name] == arr.shape[i], \
                    "Cannot align arrays with different lengths "+ \
                    f"({target_shape[dim_name]}, {arr.shape[i]}) in the same dim {dim_name}"
            else:
                target_shape[dim_name] = arr.shape[i]

    target_indices = {name: i for i, name in enumerate(target_shape.keys())}
    aligned_arrays = []
    for dim_names, arr in arrays:
        old_dim_names_filled_up = [*dim_names]
        for target_dim in target_shape.keys():
            if target_dim not in dim_names:
                arr = xp.reshape(arr, (-1, *arr.shape))
                old_dim_names_filled_up.insert(0, target_dim)
        # TODO the list conversion of keys should not be necessary but is needed for mypy
        arr = transpose_array(
            arr,
            old_dim_names=old_dim_names_filled_up,
            new_dim_names=list(target_shape.keys()),
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
            old_dim_names: Sequence[str],
            new_dim_names: Sequence[str]
        ) -> Tuple[int, ...]:
    assert len(old_dim_names) == len(new_dim_names)
    dim_index_lut = {dim_name: i for i, dim_name in enumerate(old_dim_names)}
    return tuple(dim_index_lut[target_dim_name] for target_dim_name in new_dim_names)


def transpose_array(
        array: Any,
        xp,
        old_dim_names: Sequence[str],
        new_dim_names: Sequence[str]
    ) -> Any:
    """
        `old_dims` and `new_dims` must be a transpose of one another.
        They may be shorter than array.shape. The last dims are left untouched.
    """
    axes_transpose = get_axes_transpose(old_dim_names, new_dim_names)
    array = xp.permute_dims(array, tuple(axes_transpose))
    return array
