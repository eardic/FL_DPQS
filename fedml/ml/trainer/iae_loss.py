import torch
import numpy as np


class IAE:
    def __init__(self, nu, lmda, n_class, device):
        self.device = device
        self.lmda = lmda
        self.nu = nu
        self.n_class = n_class
        self.iae_R = torch.zeros(n_class, device=device).detach()
        self.iae_c = torch.randn((n_class,), device=device).detach()
        self.dists = {}

    def get_dists(self) -> dict:
        return self.dists

    def update_r(self, d_dict: dict):
        for lbl, dists in d_dict.items():
            self.iae_R[lbl] = np.quantile(dists, 1 - self.nu)

    def compute_svdd_loss(self, reps, labels, update_r=True):
        svdd_loss = 0
        c = 0
        for lbl in torch.unique(labels):
            c += 1
            dist = torch.sum((reps[labels == lbl] - self.iae_c[lbl]) ** 2, dim=1)
            scores = dist - self.iae_R[lbl] ** 2
            svdd_loss += self.lmda * (
                    self.iae_R[lbl] ** 2 + (1 / self.nu) * torch.mean(
                torch.max(torch.zeros_like(scores), scores)))
            # update radius
            dist_sqrt = np.sqrt(dist.clone().data.cpu().numpy())
            if update_r:
                self.iae_R[lbl] = np.quantile(dist_sqrt, 1 - self.nu)
            self.dists[lbl.item()] = dist_sqrt
        return svdd_loss / c if c > 0 else torch.tensor(0).detach()

    def update_center(self, reps, labels, eps=1e-10):
        u_lbls = torch.unique(labels)
        self.iae_c = torch.zeros((len(u_lbls), reps.shape[1]), device=self.device)
        for lbl in u_lbls:
            c = torch.mean(reps[labels == lbl], dim=0, keepdim=True).detach()
            # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
            c[(abs(c) < eps) & (c < 0)] = -eps
            c[(abs(c) < eps) & (c > 0)] = eps
            self.iae_c[lbl] = c
