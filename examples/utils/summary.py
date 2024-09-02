from typing import Union
import numpy as np
from scipy.stats import spearmanr
from torch import Tensor


def get_summary(
    model: str,
    seed: int,
    num_exp: int,
    v_caus: float,
    v_hp: float,
    scale: float,
    weight: float,
    ss_n: float,
    e_star: Union[Tensor, np.ndarray],
    e_sim: Union[Tensor, np.ndarray],
) -> dict:

    rho, _ = spearmanr(e_sim, e_star)

    res_d = {}
    res_d["model"] = model
    res_d["seed"] = seed
    res_d["num_exp"] = num_exp
    res_d["v_caus"] = v_caus
    res_d["v_hp"] = v_hp
    res_d["scale"] = scale
    res_d["weight"] = weight
    res_d["ss_n"] = ss_n
    res_d["rho"] = rho

    return res_d
