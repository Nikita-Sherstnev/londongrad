import numpy as np

import londongrad as lg


class TestTensor:
    def test_tensor(self):
        lst = [0,1,2]

        t = lg.tensor(lst)
        assert np.all(t.data) == np.all(np.array(lst))

    def test_multiply_tensors(self):
        t1 = lg.tensor([1,2,3])
        t2 = lg.tensor([1,2,3])
        print(t1.__dir__())
        t3 = t1*t2
