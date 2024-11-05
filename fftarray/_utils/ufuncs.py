from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .. import FFTArray

#-------------
# Helper functions to support type inference on binary and unary functions in FFTArray
#-------------
def binary_ufuncs(op):
    def fun(self: "FFTArray", other) -> "FFTArray":
        return op(self, other)
    def fun_ref(self: "FFTArray", other) -> "FFTArray":
        return op(other, self)
    return fun, fun_ref

def unary_ufunc(op):
    def fun(self: "FFTArray") -> "FFTArray":
        return op(self)
    return fun
