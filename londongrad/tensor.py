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
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)
