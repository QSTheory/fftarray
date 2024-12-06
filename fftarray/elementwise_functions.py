
from .fft_array import (
    elementwise_one_operand,
    elementwise_two_operands,
    FFTArray,
)
from .op_lazy_luts import (
    add_transforms_lut,
    mul_transforms_lut,
    div_transforms_lut,
)




# This one is the only one with kwargs, so just done by hand.
def clip(x: FFTArray, /, *, min=None, max=None) -> FFTArray:
    assert isinstance(x, FFTArray)
    values = x.xp.clip(x.values(space=x.space), min=min, max=max)
    return FFTArray(
        values=values,
        space=x.space,
        dims=x.dims,
        eager=x.eager,
        factors_applied=(True,)*len(x.dims),
        xp=x.xp,
    )



# These use special shortcuts in the phase application.
add = elementwise_two_operands("add", add_transforms_lut)
subtract = elementwise_two_operands("subtract", add_transforms_lut)
multiply = elementwise_two_operands("multiply", mul_transforms_lut)
divide = elementwise_two_operands("divide", div_transforms_lut)


#------------------
# Single operand element-wise functions
#------------------
acos = elementwise_one_operand("acos")
acosh = elementwise_one_operand("acosh")
asin = elementwise_one_operand("asin")
asinh = elementwise_one_operand("asinh")
atan = elementwise_one_operand("atan")
atanh = elementwise_one_operand("atanh")
bitwise_invert = elementwise_one_operand("bitwise_invert")
ceil = elementwise_one_operand("ceil")
conj = elementwise_one_operand("conj")
cos = elementwise_one_operand("cos")
cosh = elementwise_one_operand("cosh")
exp = elementwise_one_operand("exp")
expm1 = elementwise_one_operand("expm1")
floor = elementwise_one_operand("floor")
imag = elementwise_one_operand("imag")
isfinite = elementwise_one_operand("isfinite")
isinf = elementwise_one_operand("isinf")
isnan = elementwise_one_operand("isnan")
log = elementwise_one_operand("log")
log1p = elementwise_one_operand("log1p")
log2 = elementwise_one_operand("log2")
log10 = elementwise_one_operand("log10")
logical_not = elementwise_one_operand("logical_not")
negative = elementwise_one_operand("negative")
positive = elementwise_one_operand("positive")
real = elementwise_one_operand("real")
round = elementwise_one_operand("round")
sign = elementwise_one_operand("sign")
signbit = elementwise_one_operand("signbit")
sin = elementwise_one_operand("sin")
sinh = elementwise_one_operand("sinh")
square = elementwise_one_operand("square")
sqrt = elementwise_one_operand("sqrt")
tan = elementwise_one_operand("tan")
tanh = elementwise_one_operand("tanh")
trunc = elementwise_one_operand("trunc")

#------------------
# Two operand element-wise functions
#------------------
atan2 = elementwise_two_operands("atan2")
bitwise_and = elementwise_two_operands("bitwise_and")
bitwise_left_shift = elementwise_two_operands("bitwise_left_shift")
bitwise_or = elementwise_two_operands("bitwise_or")
bitwise_right_shift = elementwise_two_operands("bitwise_right_shift")
bitwise_xor = elementwise_two_operands("bitwise_xor")
copysign = elementwise_two_operands("copysign")
equal = elementwise_two_operands("equal")
floor_divide = elementwise_two_operands("floor_divide")
greater = elementwise_two_operands("greater")
greater_equal = elementwise_two_operands("greater_equal")
hypot = elementwise_two_operands("hypot")
less = elementwise_two_operands("less")
less_equal = elementwise_two_operands("less_equal")
logaddexp = elementwise_two_operands("logaddexp")
logical_and = elementwise_two_operands("logical_and")
logical_or = elementwise_two_operands("logical_or")
logical_xor = elementwise_two_operands("logical_xor")
maximum = elementwise_two_operands("maximum")
minimum = elementwise_two_operands("minimum")
not_equal = elementwise_two_operands("not_equal")
pow = elementwise_two_operands("pow")
remainder = elementwise_two_operands("remainder")
