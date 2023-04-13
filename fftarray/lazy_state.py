from __future__ import annotations
from typing import Dict, Union, Hashable
from dataclasses import dataclass
from copy import copy, deepcopy
from functools import reduce


@dataclass
class PhaseFactors:
    """Dataclass containing information about the phase factors. The values are
    ofthe form `{n: a_n}` where `n` is a positive integer including zero and
    `a_n` is a float or complex number.
    Stores "phase-factors" that are evaluated by multiplying
    `np.exp(1.j * a_n * (i**n))` to the values they belong to (here `i` denotes
    the index in the array and dimension this instance belongs to).
    """

    # Dict[n: a_n]
    values: Dict[int, Union[float, complex]]

    def __init__(self, values: Dict[int, Union[float, complex]]):
        # TODO: Specialize float or not?
        # Can also just check real uand imag part at kernel launch and simplify
        # that here.
        self.values = {index: complex(value) for index, value in values.items()}
        for value in self.values.values():
            assert isinstance(value, complex)

    def __add__(self, other: PhaseFactors) -> PhaseFactors:
        """Adding the phase_factors of self and other.
        Multiplying two exponentials is equivalent to adding their arguments.
        """
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
        """Substracting the phase_factors of self and other.
        Dividing two exponentials is equivalent to substracting their arguments.
        """
        assert isinstance(other, PhaseFactors)
        result = copy(self.values)
        for i, phase in other.values.items():
            if i in result:
                result[i] -= phase
            else:
                result[i] = phase
            assert isinstance(result[i], complex)
        return PhaseFactors(result)


def _get_phase_factor_change(
        existing: Dict[str, PhaseFactors],
        target: Dict[str, PhaseFactors]
    ) -> Dict[str, PhaseFactors]:
    """Return the phase factors I need to apply in order to change from the
    `existing` to the `target` PhaseFactors state.
    Used in `get_lazy_state_to_apply`.
    """
    factor_names = set(existing.keys()).union(set(target.keys()))
    # If our existing phase factor should not exist in the target we need to apply.
    # If we want to add a new phase factor we need to apply its inverse.
    return {
        name: existing.get(name, PhaseFactors({})) - target.get(name, PhaseFactors({}))
        for name in factor_names
    }

def get_lazy_state_to_apply(existing: LazyState, target: LazyState) -> LazyState:
    """Returns the phase `LazyState` that needs to be applied to change the
    `existing` to the `target` LazyState.

    Used in `FFTArray._set_lazy_state`.

    Parameters
    ----------
    existing : LazyState
        The current LazyState.
    target : LazyState
        The target LazyState.

    Returns
    -------
    LazyState
        LazyState defining the transition from `existing` to `target`.
    """
    # Iterate over dims
    phases_to_apply = {
        dim: _get_phase_factor_change(
            existing._phases_per_dim.get(dim, {}),
            target._phases_per_dim.get(dim, {})
        )
        for dim in {**existing._phases_per_dim, **target._phases_per_dim}.keys()
    }
    result = LazyState()
    result._scale = existing._scale/target._scale
    result._phases_per_dim = phases_to_apply
    return result


@dataclass
class LazyState:
    """Represents the lazy state of a whole FFTArray. It allows the lazy
    evaluation of scale factors before broadcasted to each value of the
    FFTArray.

    Phase factors are stored per dimension under specific names in order to
    allow guarantueed cancelling in the presence of floating point rounding
    errors.

    Scale is just one complex number for all dimensions.
    """

    # There is one dict per dimension.
    _phases_per_dim: Dict[Hashable, Dict[str, PhaseFactors]]
    # TODO Currently we only have one use for that so it is less general.
    _scale: complex

    def __init__(self, scale: complex = 1.):
        self._phases_per_dim = {}
        self._scale = complex(scale)

    @property
    def scale(self) -> complex:
        """Returns the scale factor (one complex number for all dimensions)"""
        assert isinstance(self._scale, complex)
        return self._scale

    def add_phase_factor(
            self,
            dim: Hashable,
            factor_name: str,
            phase_factors: PhaseFactors
        ) -> LazyState:
        """Add the `phase_factors` to the phase factors of `self`.

        Parameters
        ----------
        dim : Hashable
            The dimension associated to the phase factor that should be added.
        factor_name : str
            Name of the phase factor.
        phase_factors : PhaseFactors
            The phase factor that should be added.

        Returns
        -------
        LazyState
            LazyState with combined phase factors.
        """
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
        """Combine the scales of `self` with `scale` (they are multiplied
        together).

        Parameters
        ----------
        scale : complex
            Scale factor that is multiplied on top of `self.scale`.

        Returns
        -------
        LazyState
            LazyState with combined scale factors.
        """
        new_lazy_state = deepcopy(self)
        new_lazy_state._scale = new_lazy_state._scale * complex(scale)
        assert isinstance(new_lazy_state._scale, complex)
        return new_lazy_state

    def phase_factors_for_dim(self, dim_name: Hashable) -> PhaseFactors:
        """Returns the sum of the phase factors associated to `dim_name`."""
        return reduce(
            lambda a,b: a+b,
            self._phases_per_dim.get(dim_name, {}).values(),
            PhaseFactors({})
        )

    def __add__(self, other: LazyState) -> LazyState:
        """Combine two lazy states into one (the new LazyState containes the sum
        of the phase factors and the product of the scales).
        The combined results in the same thing as applying both one after the
        other up to rounding errors.
        """
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
