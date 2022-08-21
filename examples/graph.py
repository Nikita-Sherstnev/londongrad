import londongrad as lg
from londongrad.utils import draw_graph


if __name__ == '__main__':
    x = lg.tensor([2.0])
    a = x ** 2
    y = ((a + x) ** 2) * 6.0

    y.backward()

    dot = draw_graph(y, format='png')
    dot.render('graph')
