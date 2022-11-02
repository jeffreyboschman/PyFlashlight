"""
Loss functions.
"""
import pyflashlight.helpers as helpers

def mean_squared_error(y_gts, y_preds):
    """Calculates the mean square error between ground truth values and predicted values.
    
    y_gts:      Ground truth values. Should have same dimensions as y_preds, but can be composed of Scalar objects, or Python int/float values.
    y_preds:    Predicated output values from a neural network. Can be either a single Scalar object (when there is a single training example and single output neuron), 
                    a list of Scalar objects (when there is a single training example an multiple output neurons, or multiple training examples for a single output neuron), or
                    a list of a list of Scalar objects (when there is multiple training examples and multiple output neurons)."""
    if not isinstance(y_preds, list):
        loss = (y_gts - y_preds)**2
    elif helpers.is_nested_list(y_preds):
        loss = (sum(mean_squared_error(y_gt, y_pred) for y_gt,y_pred in zip(y_gts, y_preds))) / len(y_gts)
    else:
        loss = (sum((y_gt - y_pred)**2 for y_gt,y_pred in zip(y_gts, y_preds))) / len(y_gts)
    loss.label = 'MSE loss'
    return loss

def binary_cross_entropy(y_gts, y_preds):
    """Calculates the binary cross entropy loss between ground truth labels and a predicted values.

    y_gts:      Ground truth labels (either 0 or 1). Should have same dimensions as y_preds, but can be composed of Scalar objects, or Python int/float values.
    y_preds:    Predicted probabilities of being class 1. Can be either a single Scalar object (when there is a single training example and single output neuron), 
                    a list of Scalar objects (when there is a single training example an multiple output neurons, or multiple training examples for a single output neuron), or
                    a list of a list of Scalar objects (when there is multiple training examples and multiple output neurons).
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
    y_gts:      One-hot encoded ground truth vectors. Should have same dimensions as y_preds, but can be composed of Scalar objects, or Python int/float values.
    y_preds:    The predicted probabilities (e.g., following softmax). Can be either a list of Scalar objects (when there is a single training example), or
                    a list of a list of Scalar objects (when there are multiple training examples).
    """
    if helpers.is_nested_list(y_preds):
        loss = (sum(categorical_cross_entropy(y_gt, y_pred) for y_gt,y_pred in zip(y_gts, y_preds))) / len(y_gts)
    else:
        loss = (sum(y_gt*(y_pred.log()) for y_gt, y_pred in zip(y_gts, y_preds))) / len(y_gts)
    loss.label = 'CCE loss'
    return loss