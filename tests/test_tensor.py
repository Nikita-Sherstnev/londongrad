import time

import numpy as np

import londongrad as lg

np.random.seed(42)


class TestTensor:
    def test_tensor(self):
        lst = [0,1,2]
        lst2 = np.array([0,1,2])

        t = lg.tensor(lst)
        t2 = lg.tensor(lst2)
        assert np.all(t.data) == np.all(np.array(lst))

    def test_multiply_tensors(self):
        t1 = lg.tensor([1,2,3])
        t2 = lg.tensor([1,2,3])
        t3 = t1 * t2
        assert np.all(t3.data) == np.all(np.array([1,4,9]))
