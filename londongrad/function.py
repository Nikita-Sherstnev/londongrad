import weakref
import numpy as np

from .tensor import Tensor
from .config import Config


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_tensor(obj):
    if isinstance(obj, Tensor):
        return obj
    return Tensor(obj)


class Function:
    def __call__(self, *inputs):
        inputs = [as_tensor(x) for x in inputs]
        
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = [Tensor(as_array(y)) for y in ys]
        
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def __repr__(self):
        """
        For printing in graph.
        """
        raise NotImplementedError()

    def forward(self, *x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Mul(Function):
    def __repr__(self):
        return '*'

    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        return gx0, gx1

def mul(x0, x1):
    return Mul()(x0, x1)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def __repr__(self):
        return f'**{self.c}'

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx

def pow(x, c):
    return Pow(c)(x)


class Add(Function):
    def __repr__(self):
        return '+'

    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    return Add()(x0, x1)


class Neg(Function):
    def __repr__(self):
        return '-'

    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def __repr__(self):
        return '-'

    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0 = gy
        gx1 = -gy
        return gx0, gx1


def sub(x0, x1):
    return Sub()(x0, x1)

def rsub(x0, x1):
    return Sub()(x1, x0)


class Div(Function):
    def __repr__(self):
        return '/'

    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1


def div(x0, x1):
    return Div()(x0, x1)

def rdiv(x0, x1):
    return Div()(x1, x0)
