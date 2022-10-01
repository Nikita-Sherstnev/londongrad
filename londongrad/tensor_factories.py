from .tensor import Tensor
from .function import mul, pow, add, neg, sub, rsub, div, rdiv, matmul, reshape, sum
from .function import transpose as transpose_f

def sum_f(self, axis=None, keepdims=False):
    return sum(self, axis, keepdims)


def transpose(self, *axes):
    if len(axes) == 0:
        axes = None
    elif len(axes) == 1:
        if isinstance(axes[0], (tuple, list)) or axes[0] is None:
            axes = axes[0]
    return transpose_f(self, axes)


@property
def T(self):
    return transpose_f(self)


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
    Tensor.__matmul__ = matmul
    Tensor.sum = sum_f
    Tensor.reshape = reshape
    Tensor.transpose = transpose
    Tensor.T = T


def tensor(data, name = None):
    return Tensor(data, name = None)
