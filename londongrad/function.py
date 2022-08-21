import weakref
import numpy as np

from .tensor import Tensor


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

        self.generation = max([x.generation for x in inputs])
        outputs = [Tensor(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Mul(Function):
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
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    return Add()(x0, x1)
