import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def simulate_risk_factors(
    dfG: pd.DataFrame,
    var_geno: float,
    num_rfs: int,
    num_shared_qtls_rf: int,
) -> pd.DataFrame:
    
    sim_weights = lambda n: 0.3 + 0.7 * np.random.rand(n)

    # simulate betas
    beta_g = np.zeros([dfG.shape[1], num_rfs])
    idx_random = np.random.permutation(dfG.shape[1])
    idx_shared = idx_random[:num_shared_qtls_rf]
    idx_specific = idx_random[num_shared_qtls_rf:].reshape(num_rfs, -1)
    for _rf_i in range(num_rfs):
        _idx = np.concatenate([idx_shared, idx_specific[_rf_i, :]])
        beta_g[_idx, _rf_i] = np.random.choice([-1, 1], _idx.shape[0]) * sim_weights(_idx.shape[0])


    # simulate genetic component
    Y_g = StandardScaler().fit_transform(dfG) @ beta_g  # [N, P]
    Y_g = np.sqrt(var_geno) * StandardScaler().fit_transform(Y_g)

    # simulate residuals
    var_noise = 1 - var_geno
    Y_noise = np.random.randn(dfG.shape[0], beta_g.shape[1])
    Y_noise = np.sqrt(var_noise) * StandardScaler().fit_transform(Y_noise)

    # add up and standardize
    Y = Y_g + Y_noise  # [N, P]
    Y = StandardScaler().fit_transform(Y)
    assert Y.shape[0] == dfG.shape[0]
    assert Y.shape[1] == beta_g.shape[1]

    # make pandas df and return
    dfY = pd.DataFrame(
        Y, columns=[f"Risk_factor_{i}" for i in range(Y.shape[1])]
    ).astype(np.float64)
    return dfY
