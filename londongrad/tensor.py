import numpy as np


class Tensor:
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            self.data = np.array(data)
        else:
            self.data = data

        self.grad = None
        self.creator = None

    def __repr__(self):
        return str(self.data)

    def __mul__(self, x):
        return self.data * x.data

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    funcs.append(x.creator)
