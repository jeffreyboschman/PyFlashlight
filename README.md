# PyFlashlight

A library for gaining intuition on the inner workings of neural networks. PyFlashlight is a bit like a lightweight implementation of PyTorch with `autograd=True`, in the sense that it can _**chain together operations**_ to yield an error between the output and some ground truth, _**keep track of what operations have been done**_, and subsequently _**traverse backwards from the error**_.

PyFlashlight is especially helpful for understanding gradient descent by _**visualizing the operations, values, and gradients in a neural network graph**_. It includes functionality to perform backpropagation with most standard arithmetic operations, multiple loss functions, multiple activation functions, and pre-defined classes for building a multi-layer perceptron (MLP). 

![cartoon image of snake holding flashlight](https://github.com/jeffreyboschman/PyFlashlight/blob/main/images/DALLE_PyFlashlight1.png?raw=true)
Image created by text-to-image generator DALL-E (2022-10-09).  

## Simple Operations Example

Below are a few examples of operations that are possible with PyFlashlight and a visualization of them as a computation graphs.

```
from pyflashlight.engine import Scalar
import pyflashlight.helpers as helpers
```
```
a = Scalar(-3.0, label='a')
b = Scalar(2, label='b')
c = 0.25*(a + b); c.label='c'
d = (b * c).leakyrelu(); d.label='d'

d.backward()
helpers.draw_dot(d)
```
![alt text](https://github.com/jeffreyboschman/PyFlashlight/blob/main/images/simple_graph_abc.svg?raw=true)

```
x = Scalar(0.2, label='x')
x /= 3.0
y = Scalar(0.35, label='y')
y -= 0.12
z = x**2 + y.log(); z.label = 'z'

helpers.draw_dot(z)
```
![alt text](https://github.com/jeffreyboschman/PyFlashlight/blob/main/images/simple_graph_xyz.svg?raw=true)
(Note that if we do not call `z.backward()` in this second example, all node outlines are grey and their `grad` values are `0.0000`. This will be explained more in the following example.) 


## Neural Network Backpropagation Calculation and Visualization Example

With PyFlashlight, we can also visualize the calculation of gradients and the updating of parameters in a neural network.

The following codeblocks (also found in `graph_examples.ipynb`) create a simple MLP neural network with 1 input layer (2 neurons) and 1 output layer (1 neuron) for regression. The leaky ReLU activation function is applied on the output neuron, and then the loss is calculated using the mean squared error loss function (there is only one example and one output neuron, so it is technically just the squared error loss).

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

Again, we can visualize the set of operations as a graph. For this simple MLP, the leaf nodes are usually parameters or inputs and the intermediate nodes are the results of (and inputs to) intermediate operations. The root node at the far right is the loss. When the gradient of the loss w.r.t. a node is `0`, the node has a grey outline (as alluded to in the second Simple Operations Example, all of the graph nodes are initialized with gradients of `0` because we haven't calculated them yet). The nodes outlined with red are the parameters (i.e., weights and biases) whose data values will get updated during gradient descent.

```  
helpers.draw_dot(loss, mlp.parameters())
```

![alt text](https://github.com/jeffreyboschman/PyFlashlight/blob/main/images/mlp_no_grads.svg?raw=true)
(You can click on an image to view it larger.)


Next, we can perform backpropagation, which calculates the gradient of the loss relative to each node (sometimes casually referred to as just the "gradient of a node"). It calculates the gradient for a node whether or not it is a parameter (i.e., a weight or bias) as gradients of all the intermediate nodes are necessary for calculating the parameter gradients. This is because each parameter influences the loss via a set of downstream operations during the forward pass, and therefore the gradients of the loss w.r.t. these operations are needed to calculate the gradients of the loss w.r.t. each parameter during the backwards pass (using the chain rule of calculus).

```
loss.backward()
helpers.draw_dot(loss, mlp.parameters())
```

![alt text](https://github.com/jeffreyboschman/PyFlashlight/blob/main/images/mlp_with_grads.svg?raw=true)

Now, we can see that the gradient of the loss is calculated for all the nodes (and therefore, the nodes have a black outline). Great! 

If we were to train this MLP, the next steps would be to update the values of the parameters, zero the gradients, and then do the whole forward pass again. We can see how we can use PyFlashlight to train a larger neural network for multiple epochs in the following example.

## Training a Neural Nework Example

PyFlashlight can be used for training a simple neural network. 

The notebook `training_examples.ipynb` has an example of training an MLP (2 input neurons, 16 neurons, 16 neurons, 1 output neuron) for binary classification of the `scikit-learn` `make_moons` dataset. It is trained using leaky ReLU activation functions for the intermediate layers, sigmoid on the last layer (which is a single neuron), binary cross entropy error loss, and classic gradient descent with a linearly decreasing learning rate for 40 epochs. The following gif demonstrates how the predicted background contour improves for more epochs.

![alt text](https://github.com/jeffreyboschman/PyFlashlight/blob/main/images/moons_training.gif?raw=true)


## Categorical Cross Entropy Loss Visualization Example

Creating PyFlashlight also helped me to solidify concepts related to the inner workings of machine learning with neural networks.

I used to be really confused by the cross entropy loss function - do the output probabilties that are multiplied by the `0`'s of the one-hot encoded ground truth contribute to the loss at all? How does this affect the gradient calculations? This example helps visualize the intermediate steps of calculating the cross entropy loss and subsequently calculating the gradients.

The following codeblocks (also found in `graph_examples.ipynb`) are for a classification problem with three classes. We run one training example (with two features) through an MLP with three output nodes. Then, the logit outputs of these three nodes are converted to probabilities with softmax and compared with the one-hot encoded ground truth vector with categorical cross entropy loss.

```
import pyflashlight.nn as nn
import pyflashlight.losses as losses
import pyflashlight.helpers as helpers
from pyflashlight.engine import Vector
import random

random.seed(42)
```
```
mlp = nn.MLP(2, [3], activ='none')

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

Some of the nodes in the image above are all greyed out. This is because the nodes that are the result of multiplication operations with the 0's of the one-hot encoded ground truth `y` values (`y0` and `y1`) have a value of `0` (graph nodes with a data value of `0` are greyed out). Since these terms are added together during the categorical cross entropy loss calculation, we can see that the nodes that are multiplied by the `0`'s of the one-hot encoded vector (i.e., all the outputs of the softmax except for one) do not impact the final loss value.

```
loss.backward()
helpers.draw_dot(loss, mlp.parameters())
```

![alt text](https://github.com/jeffreyboschman/PyFlashlight/blob/main/images/mlp_cce_with_grads.svg?raw=true)

After doing backpropagation, we can also see that most of the gradients of the operations during the softmax are `0` (still have grey outlines); these `0`-gradient operations are the ones that lead to the softmax outputs but that are multiplied by the `0`'s of the one-hot encoded ground truth vector. Since these downstream gradients are `0`, they also do not influence the gradients of the output nodes before the softmax and the parameters leading into them. In other words, when using categorical cross entropy loss, the gradients of the parameters leading to the final nodes before the softmax only depend on the softmax output of the node multiplied by the `1` of the one-hot encoded ground truth vector. In other, other words, **_the gradients of the parameters leading to final nodes before the softmax only depend on how close the "correct" probability is, and not at all on how wrong the other probabilities are_**. 

However, all of this is **_okay because the softmax operation means that making one component of the vector larger must shrink the sum of the remaining components by the same amount_**. 

Hopefully that all makes sense, and that visualizing this neural network graph helps other people understand categorical cross entropy loss the way that it helped me.

## References

Heavily inspired by Andrej Karpathy's [micrograd engine](https://github.com/karpathy/micrograd), but with a bit more functionality (extension to vectors, creating and displaying labels, coloured nodes, multiple loss functions with capability for single examples or batches, multiple activation functions, training animation) and a lot more documentation. 
