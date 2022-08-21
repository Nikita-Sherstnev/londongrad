import os

import londongrad as lg
from londongrad.utils import draw_graph


class TestUtils:
    def test_graph_plot(self):
        if os.path.exists('graph.png'):
            os.remove('graph.png')
        
        x = lg.tensor([2.0])
        a = x ** 2
        y = a ** 2 + a ** 2

        y.backward()

        graph = draw_graph(y, format='png')
        graph.render('graph')

        assert os.path.exists('graph.png')
        os.remove('graph')
        os.remove('graph.png')