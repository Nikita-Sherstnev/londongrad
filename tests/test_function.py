import numpy as np

import londongrad as lg
from londongrad import function as F

from conftest import numerical_diff


class TestFunction:
    def test_square_forward(self):
        x = lg.tensor(np.array(2.0))
        y = F.square(x)
        expected = np.array(4.0)

        assert y.data == expected

    def test_square_backward(self):
        x = lg.tensor(np.array(3.0))
        y = F.square(x)
        y.backward()
        expected = np.array(6.0)

        assert x.grad == expected

    def test_square_gradient_check(self):
        x = lg.tensor(np.random.rand(1))
        y = F.square(x)
        y.backward()
        num_grad = numerical_diff(F.square, x)

        assert np.allclose(x.grad, num_grad)

    def test_add_forward(self):
        x0 = lg.tensor(np.array(2.0))
        x1 = lg.tensor(np.array(2.0))
        y = F.add(x0, x1)
        expected = np.array(4.0)

        assert y.data == expected

    def test_add_backward(self):
        x0 = lg.tensor(np.array(2.0))
        x1 = lg.tensor(np.array(2.0))
        y = F.add(x0, x1)
        y.backward()
        expected = np.array(1.0) # represents upstream grad which is 1.0 in this case

        assert x0.grad == expected
