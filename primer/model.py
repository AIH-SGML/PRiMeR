import time
from tqdm import tqdm
import numpy as np

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam

from .ptgwas import PTGWAS
from .predictor import get_predictor


def lowrank_plus_diag_solve(W, d, X, return_b=False):
    # Step 1: Compute D^{-1}X efficiently
    d_inv = 1 / d
    DiX = d_inv[:, None] * X

    # Step 2: Compute the inverse of (I + W^T D^{-1} W)
    B = torch.eye(W.size(1), device=W.device) + W.t().mm(d_inv[:, None] * W)
    B_inv = torch.inverse(B)

    # Step 3: Compute the second term by multiplying matrices one by one from right to left
    low_rank_update = d_inv[:, None] * W.mm(B_inv.mm(W.t().mm(DiX)))

    # Final result combining both terms
    result = DiX - low_rank_update

    if return_b:
        return result, B

    return result


def lowrank_mvn_log_prob(y, W, d):

    Kiy, B = lowrank_plus_diag_solve(W, d, y, return_b=True)

    # logdet
    quad = -0.5 * torch.einsum("ip,ip->", y, Kiy)
    logdet = -0.5 * torch.logdet(B) - 0.5 * torch.log(d).sum()
    const = -0.5 * y.shape[0] * np.log(2.0 * np.pi)

    return quad + logdet + const


def xgower_factr(X):
    a = torch.pow(X, 2).sum()
    b = torch.mm(X, X.sum(0).unsqueeze(1)).sum()
    return torch.sqrt((a - b / X.shape[0]) / (X.shape[0] - 1))


class PRiMeR(nn.Module):

    def __init__(
        self,
        E: Tensor,
        G: Tensor,
        beta_o: Tensor,
        ste_o: Tensor,
        F: Tensor = None,
        lin: bool = False,
    ):
        super(PRiMeR, self).__init__()

        if F is None:
            F = torch.ones([E.shape[0], 1])

        # set some variables to seld
        self.E = E
        self.G = G
        self.F = F
        self.beta_o = beta_o
        self.ste_o = ste_o

        # normalize beta_o
        self.beta_o_norm = beta_o / beta_o.std(0)
        self.ste_o_norm = ste_o / torch.sqrt((ste_o**2).mean())

        # define predictor
        if lin:
            p_type = "identity"
            vi = False
        else:
            p_type = "elu"
            vi = True

        self.f = get_predictor(E.shape[1], p_type, vi=vi)

        # define modlue that performs linear regression
        self.ulr = PTGWAS(G, F)

        # define variance components
        self.log_ve = nn.Parameter(torch.Tensor([np.log(0.5)]))
        self.log_vn = nn.Parameter(torch.Tensor([np.log(0.5)]))

    @property
    def ve(self):
        return torch.exp(self.log_ve)

    @property
    def vn(self):
        return torch.exp(self.log_vn)

    @property
    def Be(self):
        fE = self.f(self.E)
        fE = (fE - fE.mean(0)) / fE.std(0)
        return self.ulr(fE)

    def _get_covar(self):
        _Be = self.Be
        cov_factor = torch.exp(0.5 * self.log_ve) * _Be / xgower_factr(_Be)
        cov_diag = self.vn * self.ste_o_norm.ravel() ** 2 + 1e-4
        return cov_factor, cov_diag

    def alphas(self):
        y = self.beta_o_norm
        cov_factor, cov_diag = self._get_covar()
        _Kiy = lowrank_plus_diag_solve(cov_factor, cov_diag, y)
        alphas = cov_factor.T.mm(_Kiy)
        return alphas

    def predict(self, Eval) -> Tensor:
        self.f.sample = False
        fEval = self.f(Eval)
        fEtrain = self.f(self.E)
        fEval = (fEval - fEtrain.mean(0)) / fEtrain.std(0)
        out = fEval.mm(self.alphas())
        self.f.sample = True
        return out

    def loss(self):
        cov_factor, cov_diag = self._get_covar()
        loss = (
            -lowrank_mvn_log_prob(self.beta_o_norm, cov_factor, cov_diag)
            + self.f.kld().sum()
        )
        return loss

    def optimize_sgd(self, epochs=1000, lr=1e-3, E=None, e_real=None, rocauc=False):

        history = {}
        history["loss"] = []
        history["ve"] = []
        history["vn"] = []
        history["rho"] = []
        history["rocauc"] = []

        opt = Adam(self.parameters(), lr=lr)
        for i in tqdm(range(epochs)):

            loss = self.loss()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1)
            opt.step()

            history["loss"].append(float(loss.data.cpu().numpy()))
            history["ve"].append(float(self.ve.data.cpu().numpy()))
            history["vn"].append(float(self.vn.data.cpu().numpy()))

            if E is not None and e_real is not None:
                with torch.no_grad():
                    e_star = self.predict(E)
                rho, _ = spearmanr(e_star.data.cpu().numpy(), e_real.data.cpu().numpy())
                history["rho"].append(abs(rho))
                if rocauc:
                    rocauc = roc_auc_score(
                        e_real.data.cpu().numpy().ravel(),
                        e_star.data.cpu().numpy().ravel(),
                    )
                    history["rocauc"].append(rocauc)

        return history

    def optimize_lbfgs(self, max_iter=20, factr=1e7, verbose=False):
        ftol = factr * np.finfo(float).eps
        optimizer = torch.optim.LBFGS(self.parameters(), line_search_fn="strong_wolfe")
        param_history = {}

        def closure():
            optimizer.zero_grad()
            loss = self.loss()
            loss.backward()
            return loss

        def grad():
            optimizer.zero_grad()
            loss = self.loss()
            loss.backward()
            return torch.cat([p.grad.flatten() for p in self.parameters()])

        t0 = time.time()
        conv = False
        previous_loss = self.loss()
        iterator = (
            tqdm.tqdm(range(max_iter), desc=f"Optimize {self.__class__.__name__}")
            if verbose
            else range(max_iter)
        )
        for _iter in iterator:
            loss = optimizer.step(closure)

            param_history[_iter] = {
                name: param.clone() for name, param in self.named_parameters()
            }
            if _iter > 2 and (loss.item() - previous_loss.item()) < ftol:
                conv = True
                if verbose:
                    print(f"Converged after {_iter} steps")
                break
            previous_loss = loss

        if verbose:
            print("Elapsed:", time.time() - t0)
        return loss, grad(), conv, param_history

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))
