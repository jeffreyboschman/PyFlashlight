"""
Lightweight version of PyTorch's autograd engine for backpropagation (not optimized for efficiency; mainly just for educational purposes to see how gradients are calculatd and stored)

Heavily inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py, but with a bit more functionality (e.g., extension to vectors) and a lot more documentation.
"""

import math
from this import d
from typing import Union, Tuple, List, Set, Callable

class Scalar:
    def __init__(self, data: Union[float, int], _children: Tuple["Scalar", ...]=(), _op: str="", label: str = "") -> None:
        """Initializes a scalar with its data."""
        self.data = float(data)
        self.grad = 0.0
        self.label = label

        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        """Returns a printable representation of the given Scalar object."""
        if self.label:
            return f"Scalar(data={self.data}, label={self.label})"
        else:
            return f"Scalar(data={self.data})"


    def __add__(self, other: Union["Scalar", float, int]): 
        """Adds two Scalar objects (or a Scalar object with a Python int/float). e.g., self + other"""
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
        """The derivative expression would be different if we the power was another Scalar object. (it would be like x^y instead of x^2)"""
        assert isinstance(other, (float, int))
        _other = other
        out = Scalar(self.data ** _other, _children = (self,), _op = f"**{_other}")

        def _backward() -> None:
            """Going backwards through the computation graph, calculates the deriviates of the final node w.r.t. self and other (for this multiplication operation).""" 
            self.grad += (_other * self.data**(_other -1)) * out.grad   # local derivative for multiplication: d(out)/d(self) = _other
            #_other.grad += ???   # WHAT IS THIS
        out._backward = _backward
        return out

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)
    
    def __truediv__(self, other: Union["Scalar", float, int]):
        return self * other**(-1)

    def exp(self):
        x = self.data
        out = Scalar(math.exp(x), (self, ), _op = 'exp')

        def _backward() -> None:
            self.grad += out.data * out.grad    # local derivative of exp: d(out)/d(self) = 
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        tanh_x = (math.exp(2*x) - 1)/(math.exp(2*x) + 1) 
        out = Scalar(tanh_x, _children = (self, ), _op = "tanh")

        def _backward() -> None:
            self.grad += (1 - tanh_x**2) * out.grad    # local derivative of tanh: d(out)/d(self) = 1 - tanh(selsf)^2
        out._backward = _backward

        return out

    def backward(self):
        """Calculates the gradients of the final node (self) w.r.t. each node."""
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

    def __radd__(self, other: Union["Scalar", float, int]):
        """Adds a Python int/float with a Scalar object. e.g., other + self

        Without this method, if a command were `a = 2 + Scalar(3)`, a TypeError would occur 
        because Python does not know what to do with 2.__add__(Scalar(3))."""
        return self + other

    def __rmul__(self, other: Union["Scalar", float, int]):
        """Multiplies a Python int/float with a Scalar object. e.g., other * self

        Without this method, if a command were `a = 2 * Scalar(3)`, a TypeError would occur 
        because Python does not know what to do with 2.__mul__(Scalar(3))."""
        return self * other


class Vector:
    def __init__(self, scalars, label: str = ""):
        self.scalars = [Scalar(item) for item in scalars]
        if label:
            for idx, scalar in enumerate(self.scalars):
                scalar.label = f"{label}{idx}"
    
    def __repr__(self):
        """Returns a printable representation of the given Scalar object."""
        return f"Vector({self.scalars})"
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Vector index out of range")
        return self.scalars[idx]

    def __len__(self):
        return len(self.scalars)