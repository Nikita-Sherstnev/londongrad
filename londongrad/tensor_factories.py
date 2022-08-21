from .tensor import Tensor
from .function import mul, pow, add

Tensor.__mul__ = mul
Tensor.__rmul__ = mul
Tensor.__pow__ = pow
Tensor.__add__ = add
Tensor.__radd__ = add


def tensor(data, name = None):
    return Tensor(data, name = None)
