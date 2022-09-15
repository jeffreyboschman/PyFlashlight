import random
from candle.engine import Value

class Neuron:
    def __init__(self, nin, layer_num = 0, neuron_num = 0):
        self.n = neuron_num
        self.l = layer_num
        self.w = [Value(random.uniform(-1, 1), label=f"L{self.l}w{prev_layer_neuron_num}-\>L{self.l+1}w{self.n}") for prev_layer_neuron_num, _ in enumerate(range(nin))]
        self.b = Value(random.uniform(-1,1), label = f"L{self.l+1}b{self.n}")
    
    def __call__(self, x):
        # w*x + b
        z = sum((xi*wi for wi, xi in zip(self.w, x)), self.b)
        z.label = f"L{self.l+1}z{self.n}"
        out = z.tanh()
        out.label = f"L{self.l+1}a{self.n}"
        return out

class Layer:
    def __init__(self, nin, nout, layer_num): 
        self.neurons = [Neuron(nin, layer_num, neuron_num) for neuron_num, _ in enumerate(range(nout))]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], layer_num) for layer_num, i in enumerate(range(len(nouts)))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


        