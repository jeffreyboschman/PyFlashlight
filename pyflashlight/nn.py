"""
Lightweight version of torch.nn; the basic building blocks of neural network graphs. 

Heavily inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/nn.py, but with a bit more functionality (e.g., labels, multiple activation functions, etc) and a lot more documentation.
"""

import random
from pyflashlight.engine import Scalar

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin, layer_num = 0, neuron_num = 0, activ='leakyrelu'):
        self.n = neuron_num
        self.l = layer_num
        self.w = [Scalar(random.uniform(-1, 1), label=f"L{self.l}w{prev_layer_neuron_num}-\>L{self.l+1}w{self.n}") for prev_layer_neuron_num, _ in enumerate(range(nin))]
        self.b = Scalar(random.uniform(-1,1), label = f"L{self.l+1}b{self.n}")
        self.activ = activ.strip().lower()
    
    def __call__(self, x):
        # w*x + b
        z = sum((xi*wi for wi, xi in zip(self.w, x)), self.b)
        z.label = f"L{self.l+1}z{self.n}"
        if self.activ == 'tanh':
            out = z.tanh()
        elif self.activ == 'relu':
            out = z.relu()
        elif self.activ == 'leakyrelu':
            out = z.leakyrelu()
        elif self.activ == 'sigmoid':
            out = z.sigmoid()
        elif self.activ == 'none':
            out = z
        else:
            raise ValueError(f"Unknown activation functions: {self.activ}")
        out.label = f"L{self.l+1}a{self.n}"
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer(Module):
    def __init__(self, nin, nout, layer_num, activ='leakyrelu'): 
        self.neurons = [Neuron(nin, layer_num, neuron_num, activ) for neuron_num, _ in enumerate(range(nout))]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
        #return outs

    def parameters(self):
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params
        #return [p for n in self.neurons for p in n.parameters()]

class MLP(Module):
    def __init__(self, nin, nouts, activ='leakyrelu'):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], layer_num, activ) for layer_num, i in enumerate(range(len(nouts)))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            ps = layer.parameters()
            params.extend(ps)
        return params
        #return [p for n in self.neurons for p in n.parameters()]