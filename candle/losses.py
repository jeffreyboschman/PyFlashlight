"""
Loss functions.
"""
import candle.helpers as helpers

def mean_squared_error(y_gts, y_preds):
    """Calculates the mean square error between ground truth valules and preicted valules.
    
    y_gts: Ground truth values. Can be either a single value (when there is a single training example and single output neuron), 
                                            a list of values (when there is a single training example an multiple output neurons, or multiple output neurons for a single training example, or
                                            a list of a list of values (when there is multiple training examples and multiple output neurons)."""
    if not isinstance(y_preds, list):
        loss = (y_gts - y_preds)**2
    elif helpers.is_nested_list(y_preds):
        loss = (sum(mean_squared_error(y_gt, y_pred) for y_gt,y_pred in zip(y_gts, y_preds))) / len(y_gts)
    else:
        loss = (sum((y_gt - y_pred)**2 for y_gt,y_pred in zip(y_gts, y_preds))) / len(y_gts)
    loss.label = 'MSE loss'
    return loss

def binary_cross_entropy(y_gts, y_preds):
    """Calculates the binary cross entropy loss between a ground truth label and a single output neuron.
    y_gt: Scalar object or int or float. Ground truth label [0 or 1].
    y_pred: Scalar object. The probality of being class 1 for a single output neuron.
    """
    if not isinstance(y_preds, list):
        loss = -(y_gts*(y_preds.log()) + (1-y_gts)*((1 - y_preds).log()))
    elif helpers.is_nested_list(y_preds):
        loss = -(sum(binary_cross_entropy(y_gt, y_pred) for y_gt,y_pred in zip(y_gts, y_preds))) / len(y_gts)
    else: 
        loss = -(sum((y_gt*(y_pred.log()) + (1-y_gt)*((1 - y_pred).log())) for y_gt, y_pred in zip(y_gts, y_preds))) / len(y_gts)
    loss.label = 'BCE loss'
    return loss

def categorical_cross_entropy(y_gts, y_preds):
    """Calculates the categorical cross entropy loss between a one-hot encoded ground truth vector and a list of probabilities.
    y_gts: list of Scalar objects or ints or floats. One-hot encoded ground truth vector for a single input. 
    y_preds: list of Scalar objects. The predicted probabilities for a single input (e.g., following softmax).
    """
    if helpers.is_nested_list(y_preds):
        loss = (sum(categorical_cross_entropy(y_gt, y_pred) for y_gt,y_pred in zip(y_gts, y_preds))) / len(y_gts)
    else:
        loss = (sum(y_gt*(y_pred.log()) for y_gt, y_pred in zip(y_gts, y_preds))) / len(y_gts)
    loss.label = 'CCE loss'
    return loss