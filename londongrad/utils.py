from graphviz import Digraph


def trace_nodes(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            if v.creator is not None:
                for child in v.creator.inputs:
                    edges.add((child, v))
                    build(child)
    build(root)
    return nodes, edges


def draw_graph(root, format='svg', rankdir='TB'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace_nodes(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
    
    for n in nodes:
        label = f"data {n.data} | grad {n.grad}"
        dot.node(name=str(id(n)), label = "{ %s }" % (label), shape='record')
        if n.creator:
            dot.node(name=str(id(n)) + str(n.creator), label=str(n.creator))
            dot.edge(str(id(n)) + str(n.creator), str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + str(n2.creator))
    
    return dot