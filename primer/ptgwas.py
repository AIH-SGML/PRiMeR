import torch
import torch.nn as nn


class PTGWAS(nn.Module):
    def __init__(self, G, F=None):
        super(PTGWAS, self).__init__()

        if F is None:
            F = torch.ones((G.shape[0], 1))
        self.G = G
        self.F = F

        self.FF = torch.mm(F.T, F)
        self.A0i = torch.inverse(self.FF)
        self.GG = torch.einsum("ij,ij->j", G, G)
        self.FG = torch.mm(F.T, G)

        self.A0iFG = torch.mm(self.A0i, self.FG)
        self.n = 1.0 / (self.GG - torch.einsum("ij,ij->j", self.FG, self.A0iFG))
        self.M = -self.n * self.A0iFG

    def forward(self, Y):

        FY = torch.mm(self.F.T, Y)
        GY = torch.mm(self.G.T, Y)

        beta_g = torch.einsum("ks,kp->sp", self.M, FY)
        beta_g += self.n[:, None] * GY

        return beta_g
