import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .simulation_functions import relu, elu, sigmoid, elu_sigmoid


def simulate_exposure(
    dfE: pd.DataFrame,
    num_exp: int,
    activation: str = "relu",
    scale: float = 1.0,
    weight: float = 0.0,
    threshold_binary_sim: float = 0.25,
) -> pd.DataFrame:

    if num_exp < 1:
        return pd.DataFrame(np.zeros(len(dfE), dtype=np.float32), index=dfE.index)

    E = dfE.values
    indices = np.random.choice(E.shape[1], num_exp, replace=False)
    upper_quantiles = np.quantile(E[:, indices], 1 - threshold_binary_sim, axis=0)

    shifted_E = E[:, indices] - upper_quantiles
    if activation == "relu":
        E = relu(shifted_E, scale=scale)
    elif activation == "elu":
        E = elu(shifted_E, scale=scale)
    elif activation == "sigmoid":
        E = sigmoid(shifted_E, scale=scale)
    elif activation == "elu_sigmoid":
        print(f"weight: {weight}")
        E = elu_sigmoid(shifted_E, scale=scale, weight=weight)
    else:
        raise ValueError(f"Unknown activation function: {activation}")

    e = E.mean(axis=1).reshape(-1, 1)
    e = StandardScaler().fit_transform(e)

    return pd.DataFrame(e, index=dfE.index)
