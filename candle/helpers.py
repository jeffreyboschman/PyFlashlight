""" 
Helper functions.
"""

from graphviz import Digraph

def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, parameters, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        # for any Scalar in the graph, create a rectangular ('record') node for it
        c = 'black'
        s = ''
        f = 'black'
        if n in parameters:
            c = 'red'
        elif n.grad == 0.0:
            c = 'grey'
            f = 'grey'
        if n.data == 0.0:
            c = 'grey79'
            s = 'filled'
        dot.node(name=str(id(n)), label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record', color=c, style=s, fontcolor=f)

        if n._op:
            # if this Scalar is a result of some operation, create an op node for it
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot

def softmax(layer):
    sums = sum(neuron.exp() for neuron in layer)
    out = [neuron.exp()/sums for neuron in layer]
    return out

def is_nested_list(outer_list):
    """Checks if you have a nested list (single level). 
    Returns True if there are any instances of a list object within your original, outer list."""
    return any(isinstance(obj, list) for obj in outer_list)