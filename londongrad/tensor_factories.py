from .tensor import Tensor
from .function import mul, pow, add, neg, sub, rsub, div, rdiv


def init_tensor_operations():
    Tensor.__mul__ = mul
    Tensor.__rmul__ = mul
    Tensor.__pow__ = pow
    Tensor.__add__ = add
    Tensor.__radd__ = add
    Tensor.__neg__ = neg
    Tensor.__sub__ = sub
    Tensor.__rsub__ = rsub
    Tensor.__truediv__ = div
    Tensor.__rtruediv__ = rdiv


def tensor(data, name = None):
    return Tensor(data, name = None)
