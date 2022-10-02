# PyCandle

A lightweight library for gaining intuition on the inner workings of neural networks. Includes functions for visualizing a neural network graph (with operations, labels, values, and gradients) and building a multi-layer perceptron (and more soon!). 



## Example Usage

This example demonstrates how gradients can be calculated and visualized on a neural network graph.

The following codeblocks create a simple multi-layer perceptron (MLP) neural network with 1 input layer (2 neurons) and 1 output layer (2 neurons). The tanh activation function is applied on the output neurons, and then the loss is calculated using the mean squared error loss function. All of the graph nodes are initialized with gradients of 0 because we haven't calculated them yet (when the gradient of a graph node is 0, it has a grey outline). 

```
import candle.nn as nn
import candle.losses as losses
import candle.helpers as helpers
from candle.engine import Vector
```
```
mlp = nn.MLP(2, [2])

x = [0.1, 0.9]
x = Vector(x, label="x")
y = [1.0, -1.0]
y = Vector(y, label='y')

y_preds = mlp(x)
loss = losses.mean_squared_error(y, y_preds)
helpers.draw_dot(loss, mlp.parameters())
```

![alt text](https://github.com/jeffreyboschman/PyCandle/blob/main/images/mlp_no_grads.svg?raw=true)


Next, we can perform backpropagation, which calculates the gradient of each graph node, a, relative to the loss, l (dl/da). It calculates the gradient whether or not it is a parameter, as all are useful in calculating the relevant parameter gradients (which have red outlines).

```
loss.backward()
helpers.draw_dot(loss, mlp.parameters())
```

![alt text](https://github.com/jeffreyboschman/PyCandle/blob/main/images/mlp_with_grads.svg?raw=true)


## Helpful Visualization of Cross Entropy Loss

I used to be really confused by the cross entropy loss function - do the output probabilties that are multiplied to the 0's of the one-hot encoded ground truth contribute to the loss at all? How does this affect the gradient calculations? This example helps visualize the internal workings of calculating the cross entropy loss and subsequently calculating the gradients.

This example is for a classification problem with three classes. This block of code runs one training example (with two features) through an MLP with three nodes. Then, these three nodes are converted to probabilities with softmax and compared with the one-hot encoded ground truth vector with cross entropy loss.

```
import candle.nn as nn
import candle.losses as losses
import candle.helpers as helpers
from candle.engine import Vector

mlp = nn.MLP(2, [3])

x = [0.1, -0.5]
x = Vector(x, label="x")
y = [0, 0, 1]
y = Vector(y, label='y')

y_logits = mlp(x)
y_preds = helpers.softmax(y_logits)

loss = losses.categorical_cross_entropy(y, y_preds)
helpers.draw_dot(loss, mlp.parameters())
```

![alt text](https://github.com/jeffreyboschman/PyCandle/blob/main/images/mlp_cce.svg?raw=true)

```
loss.backward()
helpers.draw_dot(loss, mlp.parameters())
```



## References

Heavily inspired by Andrej Karpathy's [micrograd engine](https://github.com/karpathy/micrograd), but with a bit more functionality (extention to vectors, creating and displaying labels, multiple loss functions) and a lot more documentation. 
