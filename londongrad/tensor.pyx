import numpy as np


class Tensor:
    def __init__(self, data):
        if data is not None:
            np.array(data, dtype=np.float32)

        self.data = data

    def __repr__(self):
        return str(self.data)

    def __mul__(self, x):
        return Tensor(self.data * x.data)

