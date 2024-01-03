"""Module for vanilla  [conditional | latent] neural processes"""
import logging
from typing import List

import semopt.nps as semopt
import torch

from npf.utils import helpers

from .attnnp import AttnLNP

logger = logging.getLogger(__name__)

__all__ = ["SemoptNP"]


class SemoptNP(AttnLNP, torch.nn.Module):
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

    def __init__(
        self,
        x_dim,
        y_dim,
        n_z_samples_train=None,
        n_z_samples_test=None,
        batch_size=None,
        **kwargs
    ):
        # super().__init__(x_dim, y_dim, **kwargs)
        torch.nn.Module.__init__(self)
        self.n_z_samples_train = n_z_samples_train
        self.n_z_samples_test = n_z_samples_test
        self.batch_size = batch_size
        hp = {
            "model": {
                "rep_size": kwargs["r_dim"],
                "batch_size": batch_size,
                "z_size": kwargs["r_dim"],
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }
        }
        self.semopt_model = semopt.NeuralProcess(
            x_dim, y_dim, hyperparams=hp, use_cross_attention=True
        )
        self.reset_parameters()

    def forward(self, X_cntxt, Y_cntxt, X_trgt, Y_trgt=None, **kwargs):
        # expected shape of X_cntxt: (batch_size, n_cntxt, x_dim)

        self.n_z_samples = (
            self.n_z_samples_train if self.training else self.n_z_samples_test
        )
        mu, sigma, *dists = self.semopt_model.forward(
            X_cntxt,
            Y_cntxt,
            X_trgt,
            Y_trgt,
            training=self.training,
            num_samples=self.n_z_samples,
        )

        if self.training:
            dists, z_priors, z_posteriors = dists
            zs = [d.rsample([self.n_z_samples]) for d in z_priors]
            z = torch.stack(zs)
            batch_size, n_z_samples, r_size = z.shape
            # z: (n_z_samples,batch_size,n_trgt,z_size)
            # q_zCc: Independent(Normal(loc=torch.Size([8,1,128]), scale=torch.size([8,1,128])))

            # for z_priors (q_zCc) and z_posteriors (p_zCt)
            # create single Independent distribution out of list of MultivariateNormal
            # add nz_dimension to mu,sigma to the left
            # mu: (batch_size,n_trgt,y_dim) -> (n_z_samples,batch_size,n_trgt,y_dim)
            z = z.view(
                self.n_z_samples,
                batch_size,
                -1,
                self.semopt_model.hp["model"]["z_size"],
            )
            q_zCc = self.multivariate_normal_diag(z_priors, batch_size)
            q_zCct = self.multivariate_normal_diag(z_posteriors, batch_size)
        else:
            # test ommits z_priors and z_posteriors
            q_zCc, q_zCct, z = None, None, None
            dists = dists[0]
        # get torch distribution from dists
        mus = torch.stack([dist.loc for dist in dists]).expand(
            self.n_z_samples, -1, -1, -1
        )
        sigmas = torch.stack([dist.variance for dist in dists]).expand(
            self.n_z_samples, -1, -1, -1
        )
        p_y_trgt = helpers.MultivariateNormalDiag(mus, sigmas)
        # dists = [normal.Normal(mu,sigma) for mu,sigma in zip(mus,sigmas)]
        # # mu = mu.unsqueeze(0).expand(self.n_z_samples,-1,-1,-1)
        # # sigma = sigma.unsqueeze(0).expand(self.n_z_samples,-1,-1,-1)
        # # use dists to get p_y_trgt
        # p_y_trgt = self.multivariate_normal_diag(dists,batch_size)
        # p_y_trgt = helpers.MultivariateNormalDiag(mu,sigma)
        return p_y_trgt, z, q_zCc, q_zCct

    # # override parameters from torch.nn.Module
    # def parameters(self, recurse: bool = True):
    #     return self.semopt_model.parameters()

    def multivariate_normal_diag(
        self, dists: List[torch.distributions.MultivariateNormal], batch_size
    ):
        exp_size = (
            batch_size or self.semopt_model.hp["model"]["batch_size"],
            1,
            self.semopt_model.hp["model"]["z_size"],
        )
        loc = torch.stack([d.loc for d in dists]).view(exp_size)
        scale = torch.stack([d.variance for d in dists]).view(exp_size)
        return helpers.MultivariateNormalDiag(loc, scale)
