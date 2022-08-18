from londongrad.tensor import Tensor


def numerical_diff(f, x, eps=1e-6):
    x0 = Tensor(x.data - eps)
    x1 = Tensor(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)