import numpy as np

import londongrad as lg
from londongrad import function as F

from conftest import numerical_diff


class TestFunction:
    def test_square_forward(self):
        x = lg.tensor(np.array(2.0))
        y = x ** 2
        expected = np.array(4.0)

        assert y.data == expected

    def test_square_backward(self):
        x = lg.tensor(np.array(3.0))
        y = x ** 2
        y.backward()
        expected = np.array(6.0)

        assert x.grad == expected

    def test_square_gradient_check(self):
        x = lg.tensor(np.random.rand(1))
        y = x ** 2
        y.backward()
        f = lambda x: x**2
        num_grad = numerical_diff(f, x)
        
        assert np.allclose(x.grad.data, num_grad)

    def test_add_forward(self):
        x0 = lg.tensor(np.array(2.0))
        x1 = lg.tensor(np.array(2.0))
        y = x0 + x1
        expected = np.array(4.0)

        assert y.data == expected

    def test_add_backward(self):
        x0 = lg.tensor(np.array(2.0))
        x1 = lg.tensor(np.array(2.0))
        y = x0 + x1
        y.backward()
        expected = np.array(1.0) # represents upstream grad which is 1.0 in this case

        assert x0.grad == expected

    def test_add_square_backward(self):
        x = lg.tensor(np.array(2.0), name='Base')
        a = x ** 2
        y = a ** 2 + a ** 2
        y.backward()

        assert y.data == 32.0
        assert x.grad == 64.0
