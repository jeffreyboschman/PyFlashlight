"""
Lightweight version of PyTorch's autograd engine for backpropagation (not optimized for efficiency; mainly just for educational purposes to see how gradients are calculatd and stored)

Heavily inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py, but with a bit more functionality (e.g., extension to vectors) and a lot more documentation.
"""

import math
from typing import Union, Tuple

class Scalar:
    def __init__(self, data: Union[float, int], _children: Tuple["Scalar", ...]=(), _op: str="", label: str="") -> None:
        """Initializes a Scalar with its data."""
        self.data = float(data)
        self.grad = 0.0
        self.label = label

        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        """Returns a printable representation of the given Scalar object."""
        if self.label:
            return f"Scalar(data={self.data}, grad={self.grad}, label={self.label})"
        else:
            return f"Scalar(data={self.data}, grad={self.grad})"

    def __add__(self, other: Union["Scalar", float, int]): 
        """Adds two Scalar objects (or a Scalar object plus a Python int/float). e.g., self + other"""
        _other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data + _other.data, _children = (self, _other), _op = "+")

        def _backward() -> None:
            """Going backwards through the computation graph, calculates the deriviates of the final node w.r.t. self and other (for this addition operation).
            Via the chain rule, self.grad = (the local derivative d(out)/d(self)) * (the gradient of out w.r.t. the final node (which was already calculated since we are going backward)).
            Via the chain rule, _other.grad = (the local derivative d(out)/d(_other)) * (the gradient of out w.r.t. the final node).
            The reason it is `+=` (instead of just `=`) is because each `self` and `_other` object could have been used for calculating many `out` objects, and so you need to add those gradients together.""" 
            self.grad += (1 * out.grad)     # local derivative for addition: d(out)/d(self) = 1
            _other.grad += (1 * out.grad)   # local derivative for addition: d(out)/d(_other) = 1
        out._backward = _backward

        return out

    def __mul__(self, other: Union["Scalar", float, int]):
        """Multiplies two Scalar objects (or a Scalar object with a Python int/float). e.g., self * other"""
        _other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * _other.data, _children = (self, _other), _op = "*")

        def _backward() -> None:
            """Going backwards through the computation graph, calculates the deriviates of the final node w.r.t. self and other (for this multiplication operation).""" 
            self.grad += (_other.data * out.grad)   # local derivative for multiplication: d(out)/d(self) = _other
            _other.grad += (self.data * out.grad)   # local derivative for multiplication: d(out)/d(_other) = self
        out._backward = _backward

        return out

    def __pow__(self, other: Union[float, int]):
        """Takes the power of a Scalar object to a Python float/int. e.g., self**2
        (Note, NOT for when the exponent is another Scalar object)"""
        assert isinstance(other, (float, int))
        _other = other
        out = Scalar(self.data ** _other, _children = (self,), _op = f"**{_other}")

        def _backward() -> None:
            self.grad += (_other * self.data**(_other - 1)) * out.grad   # local derivative for exponentiation: d(out)/d(self) = _other * self**(_other - 1)
        out._backward = _backward
        return out

    def exp(self):
        """Takes the power of the exponential constant e to a Scalar object. e.g., e^self"""
        x = self.data
        out = Scalar(math.exp(x), (self, ), _op = 'exp')

        def _backward() -> None:
            self.grad += out.data * out.grad    # local derivative of exp: d(out)/d(self) = out
        out._backward = _backward

        return out

    def log(self):
        """Takes the natural logarithm of of a Scalar object. e.g., ln(self)"""
        x = self.data
        out = Scalar(math.log(x), (self, ), _op = 'log')

        def _backward() -> None:
            self.grad += (1/self.data) * out.grad    # local derivative of log: d(out)/d(self) = 1 / self
        out._backward = _backward

        return out

    def sigmoid(self):
        """Takes the sigmoid of a Scalar object (i.e., squishes the data value between 0 and 1)"""
        x = self.data
        sigmoid_x = (1 / (1 + math.exp(-x)))
        out = Scalar(sigmoid_x, _children = (self, ), _op = "sigmoid")

        def _backward() -> None:
            self.grad += (sigmoid_x * (1 - sigmoid_x)) * out.grad    # local derivative of sigmoid: d(out)/d(self) = out*(1 - out)
        out._backward = _backward

        return out

    def tanh(self):
        """Takes the tanh of a Scalar object (i.e., squishes the data value between -1 and 1)"""
        x = self.data
        tanh_x = (math.exp(2*x) - 1)/(math.exp(2*x) + 1) 
        out = Scalar(tanh_x, _children = (self, ), _op = "tanh")

        def _backward() -> None:
            self.grad += (1 - tanh_x**2) * out.grad    # local derivative of tanh: d(out)/d(self) = 1 - out^2
        out._backward = _backward

        return out

    def relu(self):
        """Takes the ReLU of a Scalar object (i.e., sets negative data values to 0)"""
        x = self.data
        relu_x = max(x, 0)
        out = Scalar(relu_x, _children = (self, ), _op = "relu")

        def _backward() -> None:
            local_grad = 1 if x > 0 else 0
            self.grad += local_grad * out.grad    # local derivative of tanh: d(out)/d(self) = 1 if x > 0
        out._backward = _backward

        return out        

    def leakyrelu(self):
        """Takes the LeakyReLU of a Scalar object (i.e., multiplies negative data values by a small number)"""
        x = self.data
        relu_x = x if x > 0 else x*0.01
        out = Scalar(relu_x, _children = (self, ), _op = "leakyrelu")

        def _backward() -> None:
            local_grad = 1 if x > 0 else 0.01
            self.grad += local_grad * out.grad    # local derivative of tanh: d(out)/d(self) = 1 if x > 0 else 0.01
        out._backward = _backward

        return out  
        
    def backward(self):
        """Calculates the gradients of the final node (self) w.r.t. each node (a.k.a., backpropagation)."""
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __neg__(self):
        """Changes the sign of a Scalar object. e.g., -self"""
        return self * -1

    def __sub__(self, other: Union["Scalar", float, int]):
        """Subtracts two Scalar objects (or a Scalar object minus a Python int/float). e.g., self - other"""
        return self + (-other)
    
    def __truediv__(self, other: Union["Scalar", float, int]):
        """Divides two Scalar objects (or a Scalar object divided by a Python int/float). e.g., self / other"""
        return self * other**(-1)

    def __radd__(self, other):
        """Adds a Python int/float with a Scalar object. e.g., other + self

        Without this method, if a command were `a = 2 + Scalar(3)`, a TypeError would occur 
        because Python does not know what to do with 2.__add__(Scalar(3))."""
        return self + other

    def __rsub__(self, other):
        """Subtract a Scalar object from a Python int/float. e.g., other - self"""
        return other + (-self)

    def __rmul__(self, other):
        """Multiplies a Python int/float with a Scalar object. e.g., other * self"""
        return self * other

    def __rtruediv__(self, other):
        """Divides a Python int/float by a Scalar object. e.g., other / self"""
        return other * self**-1


class Vector:
    def __init__(self, scalars, label: str = ""):
        """Initalizes a Vector object as a list of Scalar objects."""
        self.scalars = [Scalar(item) for item in scalars]
        if label:
            for idx, scalar in enumerate(self.scalars):
                scalar.label = f"{label}{idx}"
    
    def __repr__(self):
        """Returns a printable representation of the Vector object."""
        return f"Vector({self.scalars})"
    
    def __getitem__(self, idx):
        """Returns the next Scalar object in the Vector."""
        if idx >= len(self):
            raise IndexError("Vector index out of range")
        return self.scalars[idx]

    def __len__(self):
        """Returns the number of Scalar objects in the Vector."""
        return len(self.scalars)