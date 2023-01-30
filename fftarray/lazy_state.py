from __future__ import annotations

from typing import Dict, Union, Hashable
from dataclasses import dataclass
from copy import copy, deepcopy
from functools import reduce


# TODO Should this still be a dataclass?
@dataclass
class PhaseFactors:
    # The n-th value (counting from zero) is the scale a_n in the expression np.exp(1.j * a_n * (i**n)) for index i.
    values: Dict[int, Union[float, complex]]

    def __init__(self, values: Dict[int, Union[float, complex]]):
        # TODO: Specialize float or not?
        # Can also just check real uand imag part at kernel launch and simplify that here.
        self.values = {index: complex(value) for index, value in values.items()}
        for value in self.values.values():
            assert isinstance(value, complex)

    def __add__(self, other: PhaseFactors) -> PhaseFactors:
        assert isinstance(other, PhaseFactors)
        result = copy(self.values)
        for i, phase in other.values.items():
            if i in result:
                result[i] += phase
            else:
                result[i] = phase
            assert isinstance(result[i], complex)
        return PhaseFactors(result)

    def __sub__(self, other) -> PhaseFactors:
        assert isinstance(other, PhaseFactors)
        result = copy(self.values)
        for i, phase in other.values.items():
            if i in result:
                result[i] -= phase
            else:
                result[i] = phase
            assert isinstance(result[i], complex)
        return PhaseFactors(result)

def _get_phase_factor_change(existing: Dict[str, PhaseFactors], target: Dict[str, PhaseFactors]) -> Dict[str, PhaseFactors]:
        factor_names = set(existing.keys()).union(set(target.keys()))
        # If our existing phase factor should not exist in the target we need to apply.
        # If we want to add a new phase factor we need to apply its inverse.
        return {name: existing.get(name, PhaseFactors({})) - target.get(name, PhaseFactors({}))
            for name in factor_names
        }

def get_lazy_state_to_apply(existing: LazyState, target: LazyState) -> LazyState:
    # Iterate over dims
    phases_to_apply = {dim:
        _get_phase_factor_change(
            existing._phases_per_dim.get(dim, {}),
            target._phases_per_dim.get(dim, {})
        )
        for dim in {**existing._phases_per_dim, **target._phases_per_dim}.keys()
    }
    result = LazyState()
    result._scale = existing._scale/target._scale
    result._phases_per_dim = phases_to_apply
    return result


class LazyState:
    # There is one dict per dimension.
    _phases_per_dim: Dict[Hashable, Dict[str, PhaseFactors]]
    # TODO Currently we only have one use for that so it is less general.
    _scale: complex


    def __init__(self, scale: complex = 1.):
        self._phases_per_dim = {}
        self._scale = complex(scale)

    def __eq__(self, other) -> bool:
        if type(self) != type(other):
            return False
        return self._scale == other._scale and self._phases_per_dim == other._phases_per_dim

    @property
    def scale(self) -> complex:
        assert isinstance(self._scale, complex)
        return self._scale

    def add_phase_factor(self, dim: Hashable, factor_name: str, phase_factors: PhaseFactors) -> LazyState:
        new_lazy = deepcopy(self)
        # If you copy a concrete traced jax-array it becomes a symbolic value.
        # Reassigning fixes that.
        new_lazy._scale = self._scale
        if not dim in new_lazy._phases_per_dim:
            new_lazy._phases_per_dim[dim] = {}

        if factor_name in new_lazy._phases_per_dim[dim]:
            new_lazy._phases_per_dim[dim][factor_name] = new_lazy._phases_per_dim[dim][factor_name] + phase_factors
        else:
            new_lazy._phases_per_dim[dim][factor_name] = phase_factors
        assert isinstance(new_lazy._scale, complex)
        return new_lazy

    def add_scale(self, scale: complex) -> LazyState:
        new_lazy_state = deepcopy(self)
        new_lazy_state._scale = new_lazy_state._scale * complex(scale)
        assert isinstance(new_lazy_state._scale, complex)
        return new_lazy_state

    def phase_factors_for_dim(self, dim_name: Hashable) -> PhaseFactors:
        return reduce(lambda a,b: a+b, self._phases_per_dim.get(dim_name, {}).values(), PhaseFactors({}))

    def __add__(self, other: LazyState) -> LazyState:
        assert isinstance(other, LazyState)
        result = deepcopy(self)
        for dim_name, phase_factors in other._phases_per_dim.items():
            if not dim_name in result._phases_per_dim:
                result._phases_per_dim[dim_name] = {}
            for factor_name, factor in phase_factors.items():
                if factor_name in result._phases_per_dim[dim_name]:
                    result._phases_per_dim[dim_name][factor_name] += factor
                else:
                    result._phases_per_dim[dim_name][factor_name] = factor
        result._scale *= other._scale
        return result
