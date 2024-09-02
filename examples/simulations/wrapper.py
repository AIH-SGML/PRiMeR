from typing import Tuple
import pandas as pd

from .exposure import simulate_exposure
from .outcome import simulate_outcome


def simulate_agg_risk_and_dis_outc(
    dfE: pd.DataFrame,
    dfG: pd.DataFrame,
    num_exp: int,
    v_caus: float,
    v_hp: float,
    activation: str = "relu",
    scale: float = 1.0,
    weight: float = 0.0,
    nonlinearity_threshhold: float = 0.25,
    frac_hp_snps: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    e = simulate_exposure(
        dfE,
        num_exp,
        activation,
        scale,
        weight,
        nonlinearity_threshhold,
    )

    o = simulate_outcome(dfG, e, v_caus, v_hp, frac_hp_snps)

    return e, o
