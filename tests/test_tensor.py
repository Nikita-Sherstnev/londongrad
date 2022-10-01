import numpy as np

import londongrad as lg
from londongrad.utils import draw_graph

np.random.seed(42)


class TestTensor:
    def test_tensor(self):
        lst = [0,1,2]
        lst2 = np.array([0,1,2])

        t = lg.tensor(lst)
        t2 = lg.tensor(lst2)
        assert np.all(t.data) == np.all(t2.data)

    def test_multiply_tensors(self):
        t1 = lg.tensor([1,2,3])
        t2 = lg.tensor([1,2,3])
        t3 = t1 * t2
        assert np.all(t3.data) == np.all(np.array([1,4,9]))

    def test_broadcasting(self):
        t1 = lg.tensor([[1,2,3],[4,5,6]])
        t2 = lg.tensor([3])

        t3 = (t1 + t2) * t2

        t3.backward()

        print(t1.grad)
        print(t2.grad)
        graph = draw_graph(t3, format='png')
        graph.render('graph')
