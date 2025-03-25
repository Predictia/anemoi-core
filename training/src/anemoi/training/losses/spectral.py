# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


# (C) Copyright 2022 The torch-harmonics Authors. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from __future__ import annotations

import logging

import torch
import numpy as np

from anemoi.training.losses.weightedloss import BaseWeightedLoss

LOGGER = logging.getLogger(__name__)


def legendre_gauss_weights(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    r"""
    Helper routine which returns the Legendre-Gauss nodes and weights
    on the interval [a, b].
    """

    xlg, wlg = np.polynomial.legendre.leggauss(n)
    xlg = (b - a) * 0.5 * xlg + (b + a) * 0.5
    wlg = wlg * (b - a) * 0.5

    return xlg, wlg


def clenshaw_curtiss_weights(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    r"""
    Computation of the Clenshaw-Curtis quadrature nodes and weights.

    This implementation follows
    [1] Joerg Waldvogel, Fast Construction of the Fejer and Clenshaw-Curtis Quadrature Rules; BIT Numerical Mathematics, Vol. 43, No. 1, pp. 001-018.
    """

    assert n > 1

    tcc = np.cos(np.linspace(np.pi, 0, n))

    if n == 2:
        wcc = np.array([1.0, 1.0])
    else:

        n1 = n - 1
        N = np.arange(1, n1, 2)
        l = len(N)
        m = n1 - l

        v = np.concatenate([2 / N / (N - 2), 1 / N[-1:], np.zeros(m)])
        v = 0 - v[:-1] - v[-1: 0: -1]

        g0 = (-1) * np.ones(n1)
        g0[l] = g0[l] + n1
        g0[m] = g0[m] + n1
        g = g0 / (n1 ** 2 - 1 + (n1 % 2))
        wcc = np.fft.ifft(v + g).real
        wcc = np.concatenate((wcc, wcc[:1]))

    # Rescale
    tcc = (b - a) * 0.5 * tcc + (b + a) * 0.5
    wcc = wcc * (b - a) * 0.5

    return tcc, wcc


def legpoly(
    mmax: int,
    lmax: int,
    x: np.ndarray,
    norm: str = "ortho",
    inverse: bool = False,
    csphase: bool = True,
) -> np.ndarray:
    r"""
    Computes the values of (-1)^m c^l_m P^l_m(x) at the positions specified by x.
    The resulting tensor has shape (mmax, lmax, len(x)). The Condon-Shortley Phase (-1)^m
    can be turned off optionally.

    Method of computation follows
    [1] Schaeffer, N.; Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Rapp, R.H.; A Fortran Program for the Computation of Gravimetric Quantities from High Degree Spherical Harmonic Expansions, Ohio State University Columbus; report; 1982; https://apps.dtic.mil/sti/citations/ADA123406.
    [3] Schrama, E.; Orbit integration based upon interpolated gravitational gradients.
    """

    # Compute the tensor P^m_n:
    nmax = max(mmax, lmax)
    vdm = np.zeros((nmax, nmax, len(x)), dtype=np.float64)
        
    norm_factor = 1. if norm == "ortho" else np.sqrt(4 * np.pi)
    norm_factor = 1. / norm_factor if inverse else norm_factor

    # Initial values to start the recursion
    vdm[0, 0, :] = norm_factor / np.sqrt(4 * np.pi)

    # Fill the diagonal and the lower diagonal
    for l in range(1, nmax):
        vdm[l - 1, l, :] = np.sqrt(2 * l + 1) * x * vdm[l - 1, l - 1, :]
        vdm[l, l, :] = np.sqrt((2 * l + 1) * (1 + x) * (1 - x) / 2 / l) * vdm[l - 1, l - 1, :]

    # Fill the remaining values on the upper triangle and multiply b
    for l in range(2, nmax):
        for m in range(0, l - 1):
            vdm[m, l, :] = (
                + x * np.sqrt((2 * l - 1) / (l - m) * (2 * l + 1) / (l + m)) * vdm[m, l - 1, :]
                - np.sqrt((l + m - 1) / (l - m) * (2 * l + 1) / (2 * l - 3) * (l - m - 1) / (l + m)) * vdm[m, l - 2, :]
            )

    if norm == "schmidt":
        for l in range(0, nmax):
            if inverse:
                vdm[:, l, :] = vdm[:, l, :] * np.sqrt(2 * l + 1)
            else:
                vdm[:, l, :] = vdm[:, l, :] / np.sqrt(2 * l + 1)

    vdm = vdm[:mmax, :lmax]

    if csphase:
        for m in range(1, mmax, 2):
            vdm[m] *= -1

    return vdm


def precompute_legpoly(
    mmax: int,
    lmax: int,
    t: np.ndarray,
    norm: str = "ortho",
    inverse: bool = False,
    csphase: bool = True,
) -> np.ndarray:
    r"""
    Computes the values of (-1)^m c^l_m P^l_m(\cos \theta) at the positions specified by t (theta).
    The resulting tensor has shape (mmax, lmax, len(x)). The Condon-Shortley Phase (-1)^m
    can be turned off optionally.

    Method of computation follows
    [1] Schaeffer, N.; Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Rapp, R.H.; A Fortran Program for the Computation of Gravimetric Quantities from High Degree Spherical Harmonic Expansions, Ohio State University Columbus; report; 1982; https://apps.dtic.mil/sti/citations/ADA123406.
    [3] Schrama, E.; Orbit integration based upon interpolated gravitational gradients.
    """

    return legpoly(mmax, lmax, np.cos(t), norm=norm, inverse=inverse, csphase=csphase)


class RealSHT(torch.nn.Module):

    def __init__(self, nlat: int, nlon: int, grid: str) -> None:

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon

        self.lmax = self.nlat
        self.mmax = min(self.lmax, self.nlon // 2 + 1)

        if grid == "legendre-gauss":
            cost, w = legendre_gauss_weights(nlat, -1, 1)
        elif grid == "equiangular":
            cost, w = clenshaw_curtiss_weights(nlat, -1, 1)
        else:
            raise NotImplementedError(f"Unknown grid {grid}.")

        tq = np.flip(np.arccos(cost))

        pct = precompute_legpoly(self.mmax, self.lmax, tq)
        pct = torch.from_numpy(pct)

        weights = torch.from_numpy(w)
        weights = torch.einsum('mlk, k -> mlk', pct, weights)

        self.register_buffer('weights', weights, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = 2.0 * torch.pi * torch.fft.rfft(x, dim=-1, norm="forward")

        x = torch.view_as_real(x)

        out_shape = list(x.size())
        out_shape[-3] = self.lmax
        out_shape[-2] = self.mmax
        xout = torch.zeros(out_shape, dtype=x.dtype, device=x.device)

        xout[..., 0] = torch.einsum('...km, mlk -> ...lm', x[..., :self.mmax, 0], self.weights.to(x.dtype))
        xout[..., 1] = torch.einsum('...km, mlk -> ...lm', x[..., :self.mmax, 1], self.weights.to(x.dtype))
        x = torch.view_as_complex(xout)

        return x


class SpectralLoss(BaseWeightedLoss):
    """Spectral Loss (Spherical Harmonics)"""

    name = "spectral"

    def __init__(
        self,
        node_weights: torch.Tensor,
        ignore_nans: bool = False,
        grid: str = "legendre-gauss",
        node_order: str = "lat-lon",
        nlat: int = 0,
        nlon: int = 0,
        bias: float = 0.0,
        loss: str = "amse",
        root_fagg: bool = False,
        log_scale: bool = False,
    ) -> None:
        
        super().__init__(
            node_weights=node_weights,
            ignore_nans=ignore_nans,
        )

        assert nlat * nlon == len(node_weights), "Only tensor-product grids are supported."

        self.sht = RealSHT(nlat, nlon, grid)

        if node_order == "lat-lon":
            self.reshaper = (self.sht.nlat, self.sht.nlon)
            self.permuter = (0, 0)
        elif node_order == "lon-lat":
            self.reshaper = (self.sht.nlon, self.sht.nlat)
            self.permuter = (-1, -2)
        else:
            raise NotImplementedError("Only lat-lon or lon-lat orderings are supported.")
        
        if loss == "amse":
            self.loss_function = self.spectral_loss_amse
        elif loss == "fmse":
            self.loss_function = self.spectral_loss_mse
        elif loss == "power":
            self.loss_function = self.spectral_loss_power
        else:
            raise NotImplementedError("Only AMSE, FMSE or POWER loss functions are supported.")
        
        l_weights = torch.arange(self.sht.lmax) ** bias
        l_weights = l_weights * self.sht.lmax / torch.sum(l_weights)

        m_weights = 2 * torch.ones(self.sht.mmax - 1)
        m_weights = torch.cat((torch.tensor([1.]), m_weights))

        self.register_buffer("l_weights", l_weights)
        self.register_buffer("m_weights", m_weights)

        self.amplitude_m = torch.sqrt if loss == "power" else lambda x: x
        self.amplitude_l = torch.sqrt if root_fagg else lambda x: x

        self.logscaler_m = torch.log1p if log_scale else lambda x: x

    def integrate_m_modes(
        self,
        modes: torch.Tensor,
    ) -> torch.Tensor:
        
        modes = modes * self.m_weights
        modes = self.sum_function(modes, -1)
        modes = self.amplitude_m(modes)
        modes = self.logscaler_m(modes)

        return modes
    
    def integrate_l_modes(
        self,
        modes: torch.Tensor,            
    ) -> torch.Tensor:
        
        modes = modes * self.l_weights
        modes = self.sum_function(modes, -1)
        modes = modes / (4 * torch.pi)
        modes = self.amplitude_l(modes)

        return modes

    def spectrum(
        self,
        field: torch.Tensor,
    ) -> torch.Tensor:
        
        field = field.transpose(-1, -2)

        field = field.reshape(*field.shape[:-1], *self.reshaper)
        field = field.transpose(*self.permuter)

        return self.sht(field)

    def power(
        self,
        field: torch.Tensor,
    ) -> torch.Tensor:
        
        coeff = self.spectrum(field)
        coeff = torch.view_as_real(coeff)

        power = coeff[..., 0] ** 2 + coeff[..., 1] ** 2
        power = self.integrate_m_modes(power)

        return power
    
    def spectrum_and_power(
        self,
        field: torch.Tensor,
    ) -> torch.Tensor:
        
        coeff = self.spectrum(field)
        coeff = torch.view_as_real(coeff)

        power = coeff[..., 0] ** 2 + coeff[..., 1] ** 2
        power = self.integrate_m_modes(power)

        return coeff, power

    def spectral_loss_power(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        power_p = self.power(pred)
        power_t = self.power(target)

        sp_loss = torch.square(power_p - power_t)
        sp_loss = self.integrate_l_modes(sp_loss)
        
        return sp_loss

    def spectral_loss_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        sp_loss = self.power(pred - target)
        sp_loss = self.integrate_l_modes(sp_loss)
        
        return sp_loss    
    
    def safe_coherence(
        self,
        coherence: torch.Tensor,
        amplitude: torch.Tensor,
    ) -> torch.Tensor:
        
        idx = amplitude > 0

        amplitude[idx] = torch.sqrt(amplitude[idx])
        coherence[idx] = coherence[idx] / amplitude[idx]

        return coherence, amplitude
    
    def spectral_loss_amse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        
        coeff_p, power_p = self.spectrum_and_power(pred)
        coeff_t, power_t = self.spectrum_and_power(target)

        coheren = coeff_p[..., 0] * coeff_t[..., 0] + coeff_p[..., 1] * coeff_t[..., 1]
        coheren = coheren * self.m_weights
        coheren = self.sum_function(coheren, -1)

        coheren, crossam = self.safe_coherence(coheren, power_p * power_t)

        sp_loss = (
            + (power_p + power_t - 2.0 * crossam)
            + (2.0 * torch.max(power_p, power_t))
            * (1.0 - coheren)
        )

        sp_loss = self.integrate_l_modes(sp_loss)

        return sp_loss

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        scalar_indices: tuple[int, ...] | None = None,
        without_scalars: list[str] | list[int] | None = None,
    ) -> torch.Tensor:
        """Calculates the spectral (spherical harmonics) loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)
        squash : bool, optional
            Average last dimension, by default True
        scalar_indices: tuple[int,...], optional
            Indices to subset the calculated scalar with, by default None
        without_scalars: list[str] | list[int] | None, optional
            list of scalars to exclude from scaling. Can be list of names or dimensions to exclude.
            By default None

        Returns
        -------
        torch.Tensor
            Spectral loss
        """
        loss = self.loss_function(pred, target)
        loss = self.scale(loss, scalar_indices, without_scalars=without_scalars)
        loss = self.avg_function(loss) if squash else self.avg_function(loss, (0, 1))

        return loss
