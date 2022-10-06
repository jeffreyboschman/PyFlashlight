"""
Loss functions.
"""

def sum_squared_error(y_gts, y_preds):
    loss = sum((y_gt - y_pred)**2 for y_gt,y_pred in zip(y_gts, y_preds))
    loss.label = 'SSE loss'
    return loss

def mean_squared_error(y_gts, y_preds):
    loss = (sum((y_gt - y_pred)**2 for y_gt,y_pred in zip(y_gts, y_preds))) / len(y_gts)
    loss.label = 'MSE loss'
    return loss

def binary_cross_entropy(y_gt, y_pred):
    """Calculates the binary cross entropy loss between a ground truth label and a single output neuron.
    y_gt: Scalar object or int or float. Ground truth label [0 or 1].
    y_pred: Scalar object. The probality of being class 1 for a single output neuron.
    """
    loss = y_gt*(y_pred.log()) + (1-y_gt)*((1 - y_pred).log())
    loss.label = 'BCE loss'
    return loss

def categorical_cross_entropy(y_gts, y_preds):
    """Calculates the categorical cross entropy loss between a one-hot encoded ground truth vector and a list of probabilities.
    y_gts: list of Scalar objects or ints or floats. One-hot encoded ground truth vector for a single input. 
    y_preds: list of Scalar objects. The predicted probabilities for a single input (e.g., following softmax).
    """
    loss = sum(y_gt*(y_pred.log()) for y_gt, y_pred in zip(y_gts, y_preds)) / len(y_gts)
    loss.label = 'CCE loss'
    return loss