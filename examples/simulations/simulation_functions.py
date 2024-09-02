import numpy as np

def relu(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    x = scale * x
    return x * (x > 0)

def elu(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    x = scale * x
    a = np.nan_to_num(np.exp(x))
    return x * (x > 0) + (a - 1) * (x <= 0)

def sigmoid(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    x = scale * x
    a = np.nan_to_num(np.exp(-2 * x))
    return 2 / (1 + a) - 1

def elu_sigmoid(x: np.ndarray, scale: float = 1.0, weight: float = 0.0) -> np.ndarray:
    _elu = elu(x, scale=scale)
    _sigm = sigmoid(x, scale=scale)
    return (1 - weight) * _elu + weight * _sigm