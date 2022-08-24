import numpy as np
import pytest

import londongrad as lg

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

        assert x.grad.data == expected

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

        assert x0.grad.data == expected

    def test_mul_forward(self):
        x0 = lg.tensor(np.array(2.0))
        x1 = lg.tensor(np.array(2.0))
        y = x0 * x1
        expected = np.array(4.0)

        assert y.data == expected

    def test_mul_backward(self):
        x0 = lg.tensor(np.array(2.0))
        x1 = lg.tensor(np.array(3.0))
        y = x0 * x1

        y.backward()
        
        assert x0.grad.data == x1.data
        assert x1.grad.data == x0.data

    def test_add_square_backward(self):
        x = lg.tensor(np.array(2.0), name='Base')
        a = x ** 2
        y = a ** 2 + a ** 2
        y.backward()

        assert y.data == 32.0
        assert x.grad.data == 64.0

    def test_negation(self):
        x = lg.tensor([2.0])
        y = -x

        y.backward()
        assert y.data == -2.0
        assert x.grad.data == -1.0

    def test_sub(self):
        x0 = lg.tensor([5.0])
        x1 = lg.tensor([2.0])
        y = x0 - x1

        y.backward()
        assert y.data == 3.0
        assert x0.grad.data == 1.0
        assert x1.grad.data == -1.0

    def test_div(self):
        x0 = lg.tensor([5.0])
        x1 = lg.tensor([2.0])
        y = x0 / x1

        y.backward()
        assert y.data == 2.5
        assert x0.grad.data == 0.5
        assert x1.grad.data == -1.25

    def test_square_without_backprop(self):
        x = lg.tensor([2.0])

        with lg.no_grad():
            a = x ** 2

            with pytest.raises(Exception) as e_info:
                a.backward()

            assert str(e_info.value) == 'Grad is not retained.'