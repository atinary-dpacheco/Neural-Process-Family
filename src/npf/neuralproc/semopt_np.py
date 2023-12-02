"""Module for vanilla  [conditional | latent] neural processes"""
import logging

import semopt.nps as semopt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal

from npf.architectures import MLP, merge_flat_input

from .attnnp import AttnLNP

logger = logging.getLogger(__name__)

__all__ = ["SemoptNP"]


class SemoptNP(AttnLNP):
    """
    (Latent) Neural process from SeMOpt.

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    encoded_path : {"latent", "both"}
        Which path(s) to use:
        - `"latent"`  the decoder gets a sample latent representation as input as in [1].
        - `"both"` concatenates both the deterministic and sampled latents as input to the decoder [2].

    kwargs :
        Additional arguments to `ConditionalNeuralProcess` and `NeuralProcessFamily`.

    References
    ----------
    [1] Atinary Technologies
    """

    def __init__(self, x_dim, y_dim, batch_size=None, **kwargs):
        super().__init__(x_dim, y_dim, **kwargs)
        hp = {
            "model": {
                "rep_size": kwargs["r_dim"],
                "batch_size": batch_size,
                "z_size": kwargs["r_dim"],
            }
        }
        self.semopt_model = semopt.NeuralProcess(
            x_dim, y_dim, hyperparams=hp, use_cross_attention=True
        )

    def forward(self, X_cntxt, Y_cntxt, X_trgt, Y_trgt=None):
        mu, sigma, dists, z_priors, z_posteriors = self.semopt_model.forward(
            X_cntxt, Y_cntxt, X_trgt, Y_trgt, training=True
        )
        # get torch distribution from mu and sigma
        zs = [d.rsample([self.nz_samples_train]) for d in z_priors]
        z = torch.stack(zs)
        z = z.view(-1, self.hp["model"]["rep_size"])
        # z: (1,batch_size,1,r_dim)
        # q_zCc: Independent(Normal(loc=mu, scale=sigma))
        return dists, z, z_priors, z_posteriors

    def encode_globally(self, X_cntxt, Y_cntxt):
        batch_size, n_cntxt, _ = X_cntxt.shape

        # encode all cntxt pair separately
        # size = [batch_size, n_cntxt, r_dim]
        dists_z, mus_z, sigmas_z = self.semopt_model.lat_encoder(X_cntxt, Y_cntxt)

        # using mean for aggregation (i.e. n_rep=1)
        # size = [batch_size, 1, r_dim]
        R = torch.mean(mus_z, dim=1, keepdim=True)

        if n_cntxt == 0:
            # arbitrarily setting the global representation to zero when no context
            R = torch.zeros(batch_size, 1, self.r_dim, device=mus_z.device)

        return R

    def trgt_dependent_representation(self, X_cntxt, _, R, X_trgt):
        batch_size, n_cntxt, _ = X_cntxt.shape

        if n_cntxt == 0:
            # arbitrarily setting each target representation to zero when no context
            R_trgt = torch.zeros(
                batch_size, X_trgt.size(1), self.r_dim, device=R.device
            )
        else:
            # size = [batch_size, n_trgt, r_dim]
            # R_trgt = self.attender(X_cntxt, X_trgt, R)  # keys, queries, values
            R_trgt = self.semopt_model.det_encoder.cross_attention(
                X_cntxt, X_trgt, R
            )  # keys, queries, values

        # n_z_samples=1. size = [1, batch_size, n_trgt, r_dim]
        return R_trgt.unsqueeze(0)
