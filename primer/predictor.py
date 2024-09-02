import torch
import torch.nn as nn
import torch.nn.functional as F


def get_predictor(input_dim, p_type="elu", vi=True):
    if p_type == "elu":
        return EluPredictor(input_dim, vi)
    elif p_type == "identity":
        return IdentiyPredictor(input_dim)
    else:
        raise ValueError("Function type not known!")


def kld_normal(mu_q, sigma_q, mu_p, sigma_p):
    var_q = sigma_q**2
    var_p = sigma_p**2
    kld = (
        torch.log(sigma_p / sigma_q) + (var_q + (mu_q - mu_p) ** 2) / (2 * var_p) - 0.5
    )
    return kld


class EluPredictor(nn.Module):

    def __init__(self, input_dim, vi=True):
        super(EluPredictor, self).__init__()
        self.input_dim = input_dim

        self.b_mean = nn.Parameter(1e-3 * torch.ones(input_dim))
        self.c_mean = nn.Parameter(torch.zeros(input_dim))

        self.vi = vi
        self.sample = True

        if self.vi:
            self.log_b_std = nn.Parameter(torch.log(0.8 * torch.ones(input_dim)))
            self.log_c_std = nn.Parameter(torch.log(3.0 * torch.ones(input_dim)))

    @property
    def b_std(self):
        return torch.exp(self.log_b_std)

    @property
    def c_std(self):
        return torch.exp(self.log_c_std)

    def forward(self, x):
        if self.vi and self.sample:
            b_ = self.b_mean + self.b_std * torch.randn_like(self.b_std)
            c_ = self.c_mean + self.c_std * torch.randn_like(self.c_std)
        else:
            b_ = self.b_mean
            c_ = self.c_mean
        out = F.elu(x * torch.tanh(b_) + c_, inplace=True)
        return out

    def kld(self):

        if self.vi:
            # priors
            b_prior_mean = torch.zeros_like(self.b_mean)
            c_prior_mean = torch.zeros_like(self.c_mean)
            b_prior_std = 0.8 * torch.ones_like(self.b_std)
            c_prior_std = 3.0 * torch.ones_like(self.c_std)

            # klds
            kld_b = kld_normal(self.b_mean, self.b_std, b_prior_mean, b_prior_std)
            kld_c = kld_normal(self.c_mean, self.c_std, c_prior_mean, c_prior_std)

            return kld_b + kld_c
        else:
            return torch.zeros(1)


class IdentiyPredictor(nn.Module):

    def __init__(self, input_dim):
        super(IdentiyPredictor, self).__init__()
        self.input_dim = input_dim

    def forward(self, x):
        return x

    def kld(self):
        return torch.zeros(1)
