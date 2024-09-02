from typing import Literal

from tqdm import tqdm
import numpy as np

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from .predictor import get_predictor


class LRM(nn.Module):

    def __init__(
        self,
        E: Tensor,
        o: Tensor,
        p_type: Literal["elu", "identity"] = "elu",
        task: Literal["classification", "regression"] = "classification",
    ):
        super(LRM, self).__init__()

        self.E = E
        self.o = o

        self.f = nn.Sequential(
            get_predictor(E.shape[1], p_type, vi=False),
            nn.BatchNorm1d(E.shape[1], affine=False),
            nn.Linear(E.shape[1], 1, bias=False),
            nn.Identity() if task == "regression" else nn.Sigmoid(),
        )

        # define criterion
        if task == "classification":
            self.criterion = nn.BCEWithLogitsLoss()
        elif task == "regression":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Task {task} not known!")

    def predict(self, Eval) -> Tensor:
        return self.f(Eval)

    def loss(self) -> Tensor:
        e = self.f(self.E)
        loss = self.criterion(e, self.o)
        return loss

    def optimize_sgd(
        self,
        epochs: int = 1000,
        lr: float = 1e-3,
        E_val: Tensor = None,
        o_val: Tensor = None,
        rocauc: bool = False,
        annealing: bool = False,
    ):

        history = {}
        history["loss"] = []
        history["rho"] = []
        history["rocauc"] = []
        history["best_val_loss"] = np.inf

        if isinstance(o_val, Tensor):
            o_val = o_val.detach().cpu().numpy().ravel()

        opt = Adam(self.parameters(), lr=lr)

        scheduler = None
        if annealing:
            scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=0)

        best_val_loss = np.inf
        best_model = None
        best_epoch = 0

        for i in tqdm(range(epochs)):

            loss = self.loss()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1)
            opt.step()

            if annealing:
                scheduler.step()

            history["loss"].append(loss.item())

            if self.E is not None and o_val is not None:
                self.eval()
                with torch.no_grad():
                    e_star = self.predict(E_val)
                    val_loss = self.criterion(e_star, o_val).item()

                if torch.isnan(e_star).any() or torch.unique(e_star).shape[0] == 1:
                    history["rho"].append(0)
                    if rocauc:
                        history["rocauc"].append(0.5)
                    self.train()
                    continue

                e_star = e_star.cpu().numpy().ravel()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = i
                    best_model = self.state_dict()
                    history["best_val_loss"] = best_val_loss

                rho, _ = spearmanr(o_val, e_star)
                history["rho"].append(abs(rho))

                if rocauc:
                    rocauc = roc_auc_score(o_val, e_star)
                    history["rocauc"].append(rocauc)
                self.train()

        if best_model is not None:
            print(f"Training finished after {epochs} epochs")
            print(
                f"Best model found at epoch {best_epoch} with val loss {best_val_loss}"
            )
            print(f"Loading best model")
            self.load_state_dict(best_model)

        return history
