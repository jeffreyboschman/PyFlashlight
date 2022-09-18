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