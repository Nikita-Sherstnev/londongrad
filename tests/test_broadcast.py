import numpy as np
import londongrad as lg

import londongrad.function as F
from londongrad.utils import draw_graph

np.random.seed(42)


class TestBroadcast:
    def test_shape_check(self):
        x = lg.tensor(np.random.randn(1, 10))
        b = lg.tensor(np.random.randn(10))
        y = x + b
        loss = F.sum(y)
        loss.backward()

        dot = draw_graph(loss, format='png')
        dot.render('graph')
        
        assert b.grad.shape == b.shape