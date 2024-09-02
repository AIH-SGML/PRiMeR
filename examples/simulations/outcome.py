import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


def simulate_outcome(
    dfG: pd.DataFrame,
    dfe: pd.DataFrame,
    v_caus: float,
    v_hp: float,
    frac_hp_snps: float = 0.1,
) -> pd.DataFrame:

    num_snps = dfG.shape[1]

    hpr_indices = np.random.choice(
        range(num_snps),
        int(frac_hp_snps * num_snps),
        replace=False,
    )

    G_hp = StandardScaler().fit_transform(dfG.values[:, hpr_indices.ravel()])
    beta_hp = np.random.choice([-1, 1], size=(G_hp.shape[1], 1))
    hp = G_hp @ beta_hp
    hp = StandardScaler().fit_transform(hp)

    noise = np.random.normal(size=(dfe.shape[0], 1))
    noise = StandardScaler().fit_transform(noise)

    o = np.sqrt(v_caus) * dfe.values
    o += np.sqrt(v_hp) * hp
    o += np.sqrt(1 - v_caus - v_hp) * noise

    o = StandardScaler().fit_transform(o)

    return pd.DataFrame(o, index=dfG.index)
