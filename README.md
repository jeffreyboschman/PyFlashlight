# PyFlashlight

A lightweight library for gaining intuition on the inner workings of neural networks. Includes functions for visualizing a neural network graph (with operations, labels, values, and gradients) and building a multi-layer perceptron (and more soon!). 




## Backpropagation Calculation and Visualization Example

With PyFlashlight, we can visualize the calculation of gradients and the updating of parameters with a neural network graph.

The following codeblocks (also found in `graph_examples.ipynb`) create a simple multi-layer perceptron (MLP) neural network with 1 input layer (2 neurons) and 1 output layer (1 neurons) for regression. The leaky ReLU activation function is applied on the output neurons, and then the loss is calculated using the mean squared error loss function (there is only one example and one output neuron, so it is actually just the squared error loss).

```
import pyflashlight.nn as nn
import pyflashlight.losses as losses
import pyflashlight.helpers as helpers
from pyflashlight.engine import Scalar, Vector
import random

random.seed(42)
```
```
mlp = nn.MLP(2, [1], activ='leakyrelu')

x = [0.1, 0.9]
x = Vector(x, label='x')
y = 0.7
y = Scalar(y, label='y')

y_pred = mlp(x)
loss = losses.mean_squared_error(y, y_pred)
```

We can visualize the set of operations as a graph. The leaf nodes are usually parameters or inputs, and the intermediate nodes are the results of (and inputs to) intermediate operations. The root node at the far right is the loss. When the gradient of a node (w.r.t. the loss) is `0`, it has a grey outline (all of the graph nodes are initialized with gradients of `0` because we haven't calculated them yet). The nodes outlined with red are the parameters (i.e., weights and biases) whose data values will get updated during gradient descent. 

```  
helpers.draw_dot(loss, mlp.parameters())
```

![alt text](https://github.com/jeffreyboschman/PyFlashlight/blob/main/images/mlp_no_grads.svg?raw=true)

Next, we can perform backpropagation, which calculates the gradient of each node relative to the loss. It calculates the gradient of a node whether or not it is a parameter (i.e., a weight or bias) as gradients of all the intermediate nodes are necessary for calculating the parameter gradients. This is because each parameter influences the loss via a set of downstream operations during the forward pass, and therefore the gradients of these operations are needed to calculate the gradient of each parameter w.r.t. the loss during the backwards pass (using the chain rule of calculus).

```
loss.backward()
helpers.draw_dot(loss, mlp.parameters())
```

![alt text](https://github.com/jeffreyboschman/PyFlashlight/blob/main/images/mlp_with_grads.svg?raw=true)

Now, we can see that all the nodes have a gradient w.r.t. the loss calculated (and therefore, they have a black outline). Great! 

The next steps are to update the values of the parameters, zero the gradients, and then do the whole forward pass again. We can see how we can use PyFlashlight to train a larger neural network for multiple epochs in the following example.

## Training a Neural Nework Example

PyFlashlight can be used for training a simple neural network. 

The notebook `training_examples.ipynb` has an example of training an MLP (2 input neurons, 16 neurons, 16 neurons, 1 output neuron) for binary classification of the `scikit-learn make_moons` dataset. It is trained using tanh activation functions, mean squared error loss, and classic gradient descent with a linearly decreasing learning rate for 50 epochs. The following gif demonstrates how the predicted background contour improves for more epochs.

![alt text](https://github.com/jeffreyboschman/PyFlashlight/blob/main/images/moons_training.gif?raw=true)

## Categorical Cross Entropy Loss Visualization Example

Creating PyFlashlight also helped me to solidify concepts related to the inner workings of machine learning with neural networks.

I used to be really confused by the cross entropy loss function - do the output probabilties that are multiplied by the `0`'s of the one-hot encoded ground truth contribute to the loss at all? How does this affect the gradient calculations? This example helps visualize the internal workings of calculating the cross entropy loss and subsequently calculating the gradients.

The following codeblocks (also found in `graph_examples.ipynb`) are for a classification problem with three classes. We run one training example (with two features) through an MLP with three output nodes. Then, these logit outputs of three nodes are converted to probabilities with softmax and compared with the one-hot encoded ground truth vector with categorical cross entropy loss.

```
import pyflashlight.nn as nn
import pyflashlight.nn as nn
import pyflashlight.losses as losses
import pyflashlight.helpers as helpers
from pyflashlight.engine import Vector
import random

random.seed(42)
```
```
mlp = nn.MLP(2, [3], activ='tanh')

x = [0.1, -0.5]
x = Vector(x, label='x')
y = [0, 0, 1]
y = Vector(y, label='y')

y_logits = mlp(x)
y_preds = helpers.softmax(y_logits)
print(y_preds)

loss = losses.categorical_cross_entropy(y, y_preds)
helpers.draw_dot(loss, mlp.parameters())
```

![alt text](https://github.com/jeffreyboschman/PyFlashlight/blob/main/images/mlp_cce_no_grads.svg?raw=true)

The nodes that are the result of multiplication operations with the 0's of the one-hot encoded ground truth `y` values (`y0` and `y1`) have a value of `0` (graph nodes with a data value of `0` are greyed out). Since these terms are added together during the categorical cross entropy loss calculation, we can see that the nodes that are multiplied by the `0`'s of the one-hot encoded vector (i.e., all the outputs of the softmax except for one) do not impact the final loss value.

```
loss.backward()
helpers.draw_dot(loss, mlp.parameters())
```

![alt text](https://github.com/jeffreyboschman/PyFlashlight/blob/main/images/mlp_cce_with_grads.svg?raw=true)

After doing backpropagation, we can also see that most of the gradients of the operations during the softmax are `0`; these `0`-gradient operations are those that lead to the softmax outputs that are multiplied by the `0`'s of the one-hot encoded ground truth vector. Since these downstream gradients are `0`, they also do not influence the gradients of the output nodes before the softmax and the parameters leading into them. In other words, when using categorical cross entropy loss, the gradients of the parameters leading to the final nodes before the softmax only depend on the softmax output of the node multiplied by the `1` of the one-hot encoded ground truth vector. In other other words, *the gradients of the parameters leading to final nodes before the softmax only depend on how close the "correct" probability is, and not at all on how wrong the other probabilities are*. 

However, all of this is *okay because the softmax operation means that making one component of the vector larger must shrink the sum of the remaining components by the same amount*. 

Hopefully that all makes sense, and that visualizing this neural network graph helps other people understand categorical cross entropy loss the way that it helped me.

## References

Heavily inspired by Andrej Karpathy's [micrograd engine](https://github.com/karpathy/micrograd), but with a bit more functionality (extension to vectors, creating and displaying labels, coloured nodes, multiple loss functions with capability for single example or batches, multiple activation functions, training animation) and a lot more documentation. 
