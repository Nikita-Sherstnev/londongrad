import londongrad as lg
from londongrad.utils import draw_graph


if __name__ == '__main__':
    def goldstein(x, y):
        z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
            (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
        return z

    x = lg.tensor([1.0])
    y = lg.tensor([1.0])
    z = goldstein(x, y)
    z.backward()

    dot = draw_graph(z, format='png')
    dot.render('graph')
