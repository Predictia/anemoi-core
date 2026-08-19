# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np

import einops
import torch

from torch import Tensor
from torch.nn import Module

from anemoi.models.layers.sht import (
    CartesianRealSHT,
    CartesianInverseRealSHT,
)

# @torch.no_grad()
# def on_fit_start(self) -> None:
#
#     target_modules = [
#         module for module in self.model.modules()
#         if getattr(module, "needs_on_fit_start", False)
#     ]
#
#     train_loader = self.datamodule.train_dataloader()
#
#     if self.global_rank == 0:
#         for module in target_modules:
#             initialization_batches = []
#             for batch in train_loader:
#                 batch = self._normalize_batch(batch)
#                 batch = module.to_latent(batch)
#                 initialization_batches.append(batch)
#             module.initialize(torch.cat(initialization_batches))
#
#     for module in target_modules:
#         for buffer in module.buffers():
#             self.trainer.strategy.broadcast(buffer, src=0)
#
#     return

class HankelDMD(Module):

    needs_on_fit_start = True

    def __init__(
        self,
        nlat: int,
        nlon: int,
        grid: str = "legendre-gauss",
        hankel_weights: str = None,
        rollout_check: float = 0.1,
        **_,
    ) -> None:

        super().__init__()

        self.needs_on_fit_start = False  # True isn't implemented!
        hankel_weights = np.load(hankel_weights)
        for key, value in hankel_weights.items():
            self.register_buffer(key, torch.from_numpy(value))

        self.sht = CartesianRealSHT(
            nlat=nlat,
            nlon=nlon,
            grid=grid,
        )

        self.isht = CartesianInverseRealSHT(
            nlat=nlat,
            nlon=nlon,
            grid=grid,
            lmax=self.lmax.item(),
        )

        self.rollout_check = rollout_check
        self.register_buffer("hroll", torch.empty(0), persistent=False)

    def to_latent(self, x: Tensor) -> Tensor:

        target_idx = self.fit_start_kwargs["target_idx"]
        lmax = self.fit_start_kwargs["lmax"]

        # x (batch, nlat * nlon, var)
        x = x[..., target_idx]  # (batch, nlat * nlon)

        x = einops.rearrange(
            x,
            "... (lat lon) -> ... lat lon",
            lat=self.sht.nlat,
            lon=self.sht.nlon,
        )  # (batch, nlat, nlon)

        x = self.sht(x)  # (batch, nlat, nlon // 2 + 1)
        x = x[:, :lmax, :lmax]  # (batch, lmax, lmax)
        x = x[:, torch.arange(lmax) < torch.arange(lmax)[:, None] + 1]  # (batch, t[lmax])

        return x        
  
    def initialize(self, x: Tensor) -> None:

        xh_mean, vn, lags, phi_h, evals_h = self.create_hankel_model(x)

        self.register_buffer("xh_mean", xh_mean)
        self.register_buffer("vn", vn)
        self.register_buffer("lags", lags)
        self.register_buffer("phi_h", phi_h)
        self.register_buffer("evals_h", evals_h)
        
        return
    
    def create_hankel_model(self, xh: Tensor) -> tuple[Tensor]:

        N = self.fit_start_kwargs["N"]  # (,)
        H = self.fit_start_kwargs["H"]  # (,)
        lags = self.fit_start_kwargs["lags"]  # (len(lags),)

        # xh (T, t[lmax])
        xh_mean = xh.mean(0)  # (t[lmax],)
        xhc = xh - xh_mean  # (T, t[lmax])
        _, _, v = torch.linalg.svd(xhc, full_matrices=False)  # (t[lmax], t[lmax])

        vn = v.conj().T[:, :N]  # (t[lmax], N)
        z = xhc @ vn  # (T, N)

        lags, slag, lagged = torch.tensor(lags), max(lags), []
        for lag in lags:
            end = (-1) * lag if lag > 0 else None
            lagged.append(z[slag - lag: end, :])  # (~T, N)

        h = torch.stack(lagged, -1)  # (~T, N, len(lags))
        h = h.reshape(h.shape[0], -1)  # (~T, len(lags) * N)

        h_i = h[:-1, :]  # (~T, len(lags) * N)
        h_f = h[1:, :]  # (~T, len(lags) * N)

        u, s, v = torch.linalg.svd(h_i.T, full_matrices=False)
        # u (len(lags) * N, len(lags) * N)
        # s (len(lags) * N,)
        # v (len(lags) * N, ~T)

        uh = u[:, :H]  # (len(lags) * N, H)
        vh = v[:H, :].conj().T  # (len(lags) * N, ~T)

        sh = torch.diag(s[:H])  # (H, H)
        ish = torch.linalg.inv(sh.to(vh.dtype))  #  (H, H)

        Ah = uh.conj().T @ h_f.T @ vh @ ish  # (H, H)

        evals_h, wh = torch.linalg.eig(Ah)  # (H,), (H, H)
        phi_h = h_f.T @ vh @ ish @ wh  # (len(lags) * N, H)

        return xh_mean, vn, lags, phi_h, evals_h

    def rollout_hankel_state(self, h: Tensor) -> Tensor:

        if h.shape == self.hroll.shape:
            if (h - self.hroll).abs().mean() < self.rollout_check:
                h = self.hroll

        self.hroll = h * torch.exp(1j * torch.angle(self.evals_h))
        
        return self.hroll

    def forward(self, x: Tensor) -> Tensor:
        
        # x (..., nlon * nlat, len(lags))
        x = einops.rearrange(
            x,
            "... (lat lon) lag -> ... lag lat lon",
            lat=self.sht.nlat,
            lon=self.sht.nlon,
        )  # (..., len(lags), nlat, nlon)

        xh = self.sht(x)  # (..., len(lags), nlat, nlon // 2 + 1)
        xh = xh[..., :self.lmax, :self.lmax]  # (..., len(lags), lmax, lmax)
        xh = xh[..., torch.arange(self.lmax) < torch.arange(self.lmax)[:, None] + 1]  # (..., len(lags), t[lmax])

        # xh_mean (t[lmax],)
        # vn (t[lmax], N)
        xhc = xh - self.xh_mean  # (..., len(lags), t[lmax])
        z = xhc @ self.vn  # (..., len(lags), N)
        z = einops.rearrange(z, "... lag n -> ... (lag n)")  # (..., len(lags) * N)

        # phi_h (len(lags) * N, H)
        # evals_h (H,)
        h = torch.linalg.lstsq(self.phi_h[(None,) * (z.ndim - 2)], z.mT).solution.mT  # (..., H)
        h = self.rollout_hankel_state(h)  # (..., H)

        z = h @ self.phi_h.T  # (..., len(lags) * N)
        z = einops.rearrange(
            z,
            "... (lag n) -> ... lag n",
            lag=len(self.lags),
            n=self.vn.shape[-1],
        )  # (..., len(lags), N)

        # vn (t[lmax], N)
        # xh_mean (t[lmax],)
        xhc = z @ self.vn.conj().T  # (..., len(lags), t[lmax])
        xht = xhc + self.xh_mean  # (..., len(lags), t[lmax])

        xh = torch.zeros(
            (*xht.shape[:-1], self.lmax, self.lmax),
            dtype=xht.dtype,
            device=xht.device,
        )  # (..., len(lags), lmax, lmax)
        i, j = torch.tril_indices(self.lmax, self.lmax)  # (t[lmax],), (t[lmax],)
        xh[..., i, j] = xht  # (..., len(lags), lmax, lmax)
        xh[..., 0, 0] = xh[..., 0, 0].abs()  # (..., len(lags), lmax, lmax)

        x = self.isht(xh)  # (..., len(lags), nlat, nlon)
        x = einops.rearrange(x, "... lag lat lon -> ... (lat lon) lag")  # (..., nlon * nlat, len(lags))

        return x
