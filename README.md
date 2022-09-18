# PyCandle

A lightweight library (with a PyTorch-style API) for gaining intuition on the inner workings of neural networks. Includes functions for visualizing a neural network graph (with operations, labels, values, and gradients) and building a multi-layer perceptron (and more soon!). 



## Example Usage

This example codeblock creates a simple neural network (1 input layer, 1 output layer) with 2 neurons in each layer. The tanh activation function is applied on the output neurons, and then the loss is calculated using the mean squared error loss function. 

```
x = [0.1, 0.9]
x = Vector(x, label="x")
mlp = nn.MLP(2, [2])
y = [1.0, -1.0]
y_preds = mlp(x)
loss = losses.mean_squared_error(y_preds, y)
helpers.draw_dot(loss)
```

Next, we can perform backpropagation, which calculates the gradient of each graph node, a, relative to the loss, l (dl/da). It calculates the gradient whether or not it is a parameter, as all are useful in calculating the relevant parameter gradients.

```
loss.backward()
helpers.draw_dot(loss)
```


## References

Heavily inspired by Andrej Karpathy's [micrograd engine](https://github.com/karpathy/micrograd), but with a bit more functionality (extention to vectors, creating and displaying labels, multiple loss functions) and a lot more documentation. 
