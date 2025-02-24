from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

@dataclass
class TwoOperandTransforms:
    """
        When combining two operands with addition or multiplication
        not all phase-factors need to be applied necessarily.

        The rules on how to combine the phase factors of two operands per
        dimension are stored in this class as a look-up-table.

        When defining these rules one has to take into account that forcing a
        scalar operand from ``True`` to ``False`` implies a transform to the complex
        type of the other operand.

        The inputs per dimension are three booleans:

        +----------+------------------------------------------+
        |Input     |Description                               |
        +==========+==========================================+
        | eager    | Must be the same between the two arrays, |
        |          | otherwise they cannot be combined.       |
        +----------+------------------------------------------+
        | factors1 | `factors_applied` of the first operand   |
        +----------+------------------------------------------+
        | factors2 | `factors_applied` of the first operand   |
        +----------+------------------------------------------+

        These three booleans are then converted into a binary number in the order
        ``(eager, factors1, factors2)``.
        This then yields the index between ``0`` and ``7`` for the look up.

        The actual application code assumes that `factors1 and factors2` never gets mapped
        to `factors_applied=False`.
        This property is not checked since the tables are hard-coded.
    """

    def __init__(
            self,
            factor_application_signs: npt.NDArray[np.int8],
            final_factor_state: npt.NDArray[np.bool_],
        ):
        """..

        If length 4 is passed in, this signals the result does not depend on ``eager``.
        The table then gets automatically extended to length ``8`` by duplication.

        Args:
            factor_application_signs (npt.NDArray[np.int8]): Shape 2x4 or 2x8
            final_factor_state (npt.NDArray[np.bool]): Shape 4 or 8
        """

        # If only for op-combinations are given, it is implied that eager is irrelevant
        # and the values are duplicated for both cases.
        if factor_application_signs.shape[0]==4:
            # Same for both possible values of eager.
            factor_application_signs = np.tile(factor_application_signs, (2,1))
            final_factor_state = np.tile(final_factor_state, 2)

        self.factor_application_signs = factor_application_signs
        self.final_factor_state = final_factor_state


    # Shape 2(operands)*8(input state combinations), valid values: -1, 0, 1
    factor_application_signs: npt.NDArray[np.int8]
    # Shape 8(input state combinations)
    final_factor_state: npt.NDArray[np.bool_]

"""
    Generates the required phase factor applications and factors_applied result required
    for multiplication while keeping factors_applied correct.
    The general rule is that the factors of at least one input need to be applied,
    because the square of factors cannot be represented as lazy state.

    This table shows the results of this function for different ``factors_applied``.
    A scalar input has always ``factors_applied=True``.

    +------------+--------+--------+-----------+-------------------+-------------------+-------+
    |   eager    |factors1|factors2| LUT Index |x1 sign (to target)|x2 sign (to target)| res   |
    +============+========+========+===========+===================+===================+=======+
    | False/True | False  | False  | 0/4       | 0(False)          |-1(True)           | False |
    +------------+--------+--------+-----------+-------------------+-------------------+-------+
    | False/True | False  | True   | 1/5       | 0(False)          | 0(True)           | False |
    +------------+--------+--------+-----------+-------------------+-------------------+-------+
    | False/True | True   | False  | 2/6       | 0(True)           | 0(False)          | False |
    +------------+--------+--------+-----------+-------------------+-------------------+-------+
    | False/True | True   | True   | 3/7       | 0(True)           | 0(True)           | True  |
    +------------+--------+--------+-----------+-------------------+-------------------+-------+

    The choice between the operands in the first line is arbitrary for multiplication
    but is the necessary choice for division (without reversed operand order like
    in ``__rfloordiv__`` and ``__rtruediv__``).
    In order to maximise consistency the choice for division is also used in
    multiplication.
"""
mul_transforms_lut = TwoOperandTransforms(
    factor_application_signs=np.array([
        # The choice between the operands is arbitrary for multiplication
        # but is the necessary choice for division.
        # (False, False)
        [0, -1],
        # (False, True)
        [0, 0],
        # (True,  False)
        [0,  0],
        # (True,  True)
        [0,  0],
    ]),
    final_factor_state=np.array([False, False, False, True])
)

"""
    Generates the required phase factor applications and factors_applied result required
    for division while keeping factors_applied correct.
    The general rule here is that the divisor always needs to have ``factors_applied=True``,
    because the inverse phase factors cannot be represented as lazy state.

    This table shows the results of this function for different ``factors_applied``.
    A scalar input has always ``factors_applied=True``.

    +------------+--------+--------+-----------+-------------------+-------------------+-------+
    |   eager    |factors1|factors2| LUT Index |x1 sign (to target)|x2 sign (to target)| res   |
    +============+========+========+===========+===================+===================+=======+
    | False/True | False  | False  | 0/4       | 0(False)          |-1(True)           | False |
    +------------+--------+--------+-----------+-------------------+-------------------+-------+
    | False/True | False  | True   | 1/5       | 0(False)          | 0(True)           | False |
    +------------+--------+--------+-----------+-------------------+-------------------+-------+
    | False/True | True   | False  | 2/6       | 0(True)           | -1(True)          | True  |
    +------------+--------+--------+-----------+-------------------+-------------------+-------+
    | False/True | True   | True   | 3/7       | 0(True)           | 0(True)           | True  |
    +------------+--------+--------+-----------+-------------------+-------------------+-------+

    The choice between the operands in the first table entry is arbitrary for multiplication
    but is the necessary choice for division.
    In order to maximise consistency the choice for division is also used in
    multiplication.
"""
div_transforms_lut = TwoOperandTransforms(
    factor_application_signs=np.array([
        # The choice between the operands is arbitrary for multiplication
        # but is the necessary choice for division.
        # (False, False)
        [0, -1],
        # (False, True)
        [0, 0],
        # (True,  False)
        [0,  -1],
        # (True,  True)
        [0,  0],
    ]),
    final_factor_state=np.array([False, False, True, True])
)

"""
    Generates the required phase factor applications and factors_applied result required
    for division from the right (``__rfloordiv__`` and ``rtruediv__``) while keeping
    factors_applied correct.
    The general rule here is that the divisor always needs to have ``factors_applied=True``,
    because the inverse phase factors cannot be represented as lazy state.

    This table shows the results of this function for different ``factors_applied``.
    A scalar input has always ``factors_applied=True``.

    +------------+--------+--------+-----------+-------------------+-------------------+-------+
    |   eager    |factors1|factors2| LUT Index |x1 sign (to target)|x2 sign (to target)| res   |
    +============+========+========+===========+===================+===================+=======+
    | False/True | False  | False  | 0/4       | -1(True)          | 0(False)          | False |
    +------------+--------+--------+-----------+-------------------+-------------------+-------+
    | False/True | False  | True   | 1/5       | -1(True)          | 0(True)           | True  |
    +------------+--------+--------+-----------+-------------------+-------------------+-------+
    | False/True | True   | False  | 2/6       | 0(True)           | 0(False)          | False |
    +------------+--------+--------+-----------+-------------------+-------------------+-------+
    | False/True | True   | True   | 3/7       | 0(True)           | 0(True)           | True  |
    +------------+--------+--------+-----------+-------------------+-------------------+-------+

    The choice between the operands in the first table entry is arbitrary for multiplication
    but is the necessary choice for division.
    In order to maximise consistency the choice for division is also used in
    multiplication.
"""
rdiv_transforms_lut = TwoOperandTransforms(
    factor_application_signs=np.array([
        # The choice between the operands is arbitrary for multiplication
        # but is the necessary choice for division.
        # (False, False)
        [-1, 0],
        # (False, True)
        [-1, 0],
        # (True,  False)
        [0,  0],
        # (True,  True)
        [0,  0],
    ]),
    final_factor_state=np.array([False, True, False, True])
)


"""
    Defines the required phase factor applications and factors_applied result required
    for addition and subtraction while keeping factors_applied correct.
    This requires to always have the same phase factor state for both operands.

    This tables shows the results of this function for different ``factors_applied``.
    A scalar input has always ``factors_applied=True``.

    With `eager=False` when given an arbitrary choice, `False` is preferred for the result:

    +--------+--------+--------+-----------+-------------------+-------------------+-------+
    | eager  |factors1|factors2| LUT Index |x1 sign (to target)|x2 sign (to target)| res   |
    +========+========+========+===========+===================+===================+=======+
    | False  | False  | False  | 0         | 0(False)          | 0(False)          | False |
    +--------+--------+--------+-----------+-------------------+-------------------+-------+
    | False  | False  | True   | 1         | 0(False)          | 1(False)          | False |
    +--------+--------+--------+-----------+-------------------+-------------------+-------+
    | False  | True   | False  | 2         | 1(False)          | 0(False)          | False |
    +--------+--------+--------+-----------+-------------------+-------------------+-------+
    | False  | True   | True   | 3         | 0(True)           | 0(True)           | True  |
    +--------+--------+--------+-----------+-------------------+-------------------+-------+

    With ``eager=True`` when given an arbitrary choice, ``True`` is preferred for the result:

    +--------+--------+--------+----------+-------------------+-------------------+-------+
    | eager  |factors1|factors2| LUT Index|x1 sign (to target)|x2 sign (to target)| res   |
    +========+========+========+==========+===================+===================+=======+
    | True   | False  | False  | 4        | 0(False)          | 0(False)          | False |
    +--------+--------+--------+----------+-------------------+-------------------+-------+
    | True   | False  | True   | 5        |-1(True)           | 0(True)           | True  |
    +--------+--------+--------+----------+-------------------+-------------------+-------+
    | True   | True   | False  | 6        | 0(True)           |-1(True)           | True  |
    +--------+--------+--------+----------+-------------------+-------------------+-------+
    | True   | True   | True   | 7        | 0(True)           | 0(True)           | True  |
    +--------+--------+--------+----------+-------------------+-------------------+-------+
"""
add_transforms_lut = TwoOperandTransforms(
    factor_application_signs=np.array([
        #--------
        # eager=False
        #--------
        # (False, False)
        [0,0],
        # (False, True)
        [0,1],
        # (True,  False)
        [1,0],
        # (True,  True)
        [0,0],
        #--------
        # eager=True
        #--------
        # (False, False)
        [0,0],
        # (False, True)
        [-1,0],
        # (True,  False)
        [0,-1],
        # (True,  True)
        [0,0],
    ]),
    final_factor_state=np.array([
        # eager=False
        False, False, False, True,
        # eager=True
        False, True, True, True,
    ])
)


"""
    Defines the required phase factor applications and factors_applied result
    required for any two-operands operation which requires the phase factors
    to be applied for both operands.

    This tables shows the results of this function for different ``factors_applied``.
    A scalar input has always ``factor_applied=True``.

    +------------+--------+--------+-----------+-------------------+-------------------+-------+
    |   eager    |factors1|factors2| LUT Index |x1 sign (to target)|x2 sign (to target)| res   |
    +============+========+========+===========+===================+===================+=======+
    | False/True | False  | False  | 0/4       |-1(True)           |-1(True)           | True  |
    +------------+--------+--------+-----------+-------------------+-------------------+-------+
    | False/True | False  | True   | 1/5       |-1(True)           | 0(True)           | True  |
    +------------+--------+--------+-----------+-------------------+-------------------+-------+
    | False/True | True   | False  | 2/6       | 0(True)           |-1(True)           | True  |
    +------------+--------+--------+-----------+-------------------+-------------------+-------+
    | False/True | True   | True   | 3/7       | 0(True)           | 0(True)           | True  |
    +------------+--------+--------+-----------+-------------------+-------------------+-------+
"""
default_transforms_lut = TwoOperandTransforms(
    factor_application_signs=np.array([
        # (False, False)
        [-1,-1],
        # (False, True)
        [-1,0],
        # (True,  False)
        [0,-1],
        # (True,  True)
        [0,0],
    ]),
    final_factor_state=np.array([True, True, True, True])
)
