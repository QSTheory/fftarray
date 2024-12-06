from __future__ import annotations
from typing import Literal, Iterable, List, Optional, Tuple

from .space import Space
from .fft_dimension import FFTDimension
import numpy as np

def get_transform_signs(
            input_factors_applied: Iterable[bool],
            target_factors_applied: Iterable[bool],
        ) -> Optional[List[Literal[-1, 1, 0]]]:

    do_return_list = False
    signs: List[Literal[-1, 1, 0]] = []
    for input_factor_applied, target_factor_applied in zip(input_factors_applied, target_factors_applied, strict=True):
        # If both are the same, we do not need to do anything

        if input_factor_applied == target_factor_applied:
            signs.append(0)
        else:
            do_return_list = True
            # 1: Go from applied (external) to not applied (internal)
            # -1 internal to external
            sign: Literal[1, -1] = 1 if input_factor_applied else -1
            signs.append(sign)

    if do_return_list:
        return signs
    else:
        return None

def complex_type(xp, dtype):
    match dtype:
        case xp.complex64 | xp.complex128:
            return dtype
        case xp.float32:
            return xp.complex64
        case xp.float64:
            return xp.complex128
        case _:
            raise ValueError(f"Passed in unsupported type '{dtype}'")

def real_type(xp, dtype):
    match dtype:
        case xp.float32 | xp.float64:
            return dtype
        case xp.complex64:
            return xp.float32
        case xp.complex128:
            return xp.float64
        case _:
            raise ValueError(f"Passed in unsupprted type '{dtype}'")


def apply_lazy(
        xp,
        values,
        dims: Tuple[FFTDimension, ...],
        signs: List[Literal[1,-1,0]],
        spaces: Iterable[Space],
        scale_only: bool,
    ):
    """

        values:
            if scale only:
                floating array from the namespace of xp, might be mutated
            else:
                complex array from the namespace of xp, might be mutated
        Does return the modified values and only that array is guarantueed to be changed.
                complex array from the namespace of xp, will be mutated

        sign:
            1 from external to internal (True to False)
            -1 from internal to external (False to True)
    """

    # Real-numbered scale
    scale: float = 1.
    do_apply = False
    for dim, sign, dim_space in zip(dims, signs, spaces, strict=True):
        if sign != 0 and dim_space == "freq":
            # TODO: Write as separate mul or div?
            scale = scale * (dim.d_freq*dim.n)**sign
            do_apply = True
    # We cannot test for float == 1. because that value might be dynamic under tracing.
    if do_apply:
        # as array to ensure correct precision.
        values *= scale

    if scale_only:
        return values

    # The array must have the length of the dimension in any dimension where a phase factor
    # has to be applied.
    # scale_only works for any shape.
    reps = []
    needs_expansion = False
    for length, dim, sign in zip(values.shape, dims, signs, strict=True):
        # if we do change the phase factor in a dim where the length is
        # 1 (not broadcast yet) while the dim.n is longer we need to expand
        # the array now in order to be able to apply the phase_factors.
        if sign != 0 and length != dim.n:
            assert length == 1
            reps.append(dim.n)
            needs_expansion = True
        else:
            reps.append(1)

    if needs_expansion:
        values = xp.tile(values, tuple(reps))

    per_idx_phase_factors = xp.asarray(0., dtype=real_type(xp, values.dtype))
    for dim_idx, (dim, sign, dim_space) in enumerate(zip(dims, signs, spaces, strict=True)):
        # If both are the same, we do not need to do anything

        if sign != 0:
            # Create indices with correct shape
            indices = xp.arange(0, dim.n, dtype=real_type(xp, values.dtype))
            extended_shape = np.ones(len(values.shape), dtype=int)
            extended_shape[dim_idx] = -1
            indices = xp.reshape(indices, shape=tuple(extended_shape))

            if dim_space == "pos":
                # x = indices * dim.d_pos + dim.pos_min
                per_idx_values = -sign*2*np.pi*dim.freq_min*dim.d_pos*indices
            else:
                # f = indices * dim.d_freq + dim.freq_min
                per_idx_values = sign*2*np.pi*dim.pos_min*(
                    dim.freq_min + dim.d_freq*indices
                )

            per_idx_phase_factors = per_idx_phase_factors + per_idx_values

    if len(per_idx_phase_factors.shape) > 0: # type: ignore
        # TODO: Figure out typing
        exponent = xp.asarray(1.j, dtype=values.dtype) * per_idx_phase_factors # type: ignore
        # TODO Could optimise exp into cos and sin because exponent is purely complex
        # Is that faster or more precise? Should we test that or just do it?
        values *= xp.exp(exponent)
    return values


def do_fft(
    xp,
    values,
    are_values_owned: bool,
    dims: Tuple[FFTDimension, ...],
    needs_fft: List[bool],
    space_before: Tuple[Space, ...],
    space_after: Tuple[Space, ...],
    factors_applied_before: Tuple[bool, ...],
    factors_applied_after: Tuple[bool, ...],
):
    """
        values must be of kind 'complex floating'
        If are_values_owned=True this function might mutate the input array.
        The returned values have factors_applied=False in all dimensions which were transformed.
    """
    current_factors_applied = list(factors_applied_before)

    #------------
    # Set factors_applied=False in all dimensions which need an FFT or iFFT.
    #------------
    pre_fft_applied = [
        False if fft_needed else old_factors_applied
        for fft_needed, old_factors_applied in zip(needs_fft, factors_applied_before, strict=True)
    ]
    signs = get_transform_signs(
        input_factors_applied=factors_applied_before,
        target_factors_applied=pre_fft_applied,
    )
    if signs is not None:
        # If 'not are_values_owned', we need to copy now, because apply_lazy mutates the values.
        # Also the type needs to be a complex floating point for both the phase factors and the FFT.
        if not are_values_owned:
            values = xp.asarray(values, copy=True)
            are_values_owned = True

        values = apply_lazy(
            xp=xp,
            values=values,
            dims=dims,
            signs=signs,
            spaces=space_before,
            scale_only=False,
        )

    #------------
    # Apply the actual transforms.
    # Different axes may be transformed in different directions.
    #------------
    fft_axes = []
    ifft_axes = []
    for dim_idx, (old_space, new_space) in enumerate(zip(space_before, space_after, strict=True)):
        if old_space != new_space:
            if old_space == "pos":
                fft_axes.append(dim_idx)
            else:
                ifft_axes.append(dim_idx)
            # After the transform the phase factors are missing.
            current_factors_applied[dim_idx] = False

    if len(fft_axes) > 0:
        values = xp.fft.fftn(values, axes=fft_axes)
        are_values_owned = True

    if len(ifft_axes) > 0:
        values = xp.fft.ifftn(values, axes=ifft_axes)
        are_values_owned = True

    #------------
    # Bring values into the target factors_applied state
    #------------
    signs = get_transform_signs(
        input_factors_applied=current_factors_applied,
        target_factors_applied=factors_applied_after,
    )

    if signs is not None:
        # signs can only be not None, if a FFT or iFFT was necessary.
        # And then the values have already been mutated and are therefore owned here.
        assert are_values_owned

        values = apply_lazy(
            xp=xp,
            values=values,
            dims=dims,
            signs=signs,
            spaces=space_after,
            scale_only=False,
        )

    return values, are_values_owned
