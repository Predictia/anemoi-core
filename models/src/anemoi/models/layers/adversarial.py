# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import torch
import numpy as np

from torch.nn import Module


def training(func):
    
    def wrapper(self, x): return func(self, x) if self.training else x

    return wrapper


class ZeroNoise(Module):

    def __init__(self, **_): super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor: return x


class RandomNoise(Module):

    def __init__(
        self,
        vars_idx: list[int] = [],
        mean_noise: float = 0.05,
        true_noise: bool = False,
        **_,
    ):
        
        super().__init__()

        self.vars_idx = vars_idx
        self.mean_noise = np.clip(0, 0.66, mean_noise)

        self.noise = self.true_noise if true_noise else self.easy_noise

    def easy_noise(self, shape: tuple[int]) -> torch.Tensor:

        return torch.randn(shape)

    def true_noise(self, shape: tuple[int]) -> torch.Tensor:

        raise NotImplementedError

    def varsbatch_noise(self, nbatch: int) -> torch.Tensor:

        vars_noise = np.random.uniform(0, 1, len(self.vars_idx))
        btch_noise = np.random.beta(3 * self.mean_noise / (4 - 6 * self.mean_noise), 1, nbatch)
        
        afin_noise = np.sqrt(np.outer(btch_noise, vars_noise))[:, None, None, None, :]

        return torch.from_numpy(afin_noise)

    @training
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        afin = self.varsbatch_noise(x.shape[0])
        afin = afin.to(x.device, x.dtype)
        
        noise = self.noise((*x.shape[:-1], len(self.vars_idx)))
        noise = noise.to(x.device, x.dtype)

        x[..., self.vars_idx] = (
            + (1 - afin) * x[..., self.vars_idx]
            + afin * noise
        )

        return x


class SpottyRandomNoise(RandomNoise):

    def __init__(
        self,
        lats: torch.Tensor,
        lons: torch.Tensor,
        vars_idx: list[int] = [],
        mean_noise: float = 0.50,
        true_noise: bool = False,
        mean_spot_size: float = 0.3,
        oval_spot_axis: float = 2.0,
    ):
        
        super().__init__(
            vars_idx=vars_idx,
            mean_noise=mean_noise,
            true_noise=true_noise,
        )

        nodes_xyz = torch.stack([
            torch.cos(lats) * torch.cos(lons),
            torch.cos(lats) * torch.sin(lons),
            torch.sin(lats),
        ], -1)

        self.register_buffer("nodes_xyz", nodes_xyz)

        self.mean_spot_size = mean_spot_size
        self.oval_spot_axis = oval_spot_axis

    def spot_location_and_shape(self) -> tuple[torch.Tensor, torch.Tensor, float]:

        loc = torch.randn(3)
        loc = loc / (torch.sqrt(loc.square().sum()) + 1e-3)

        shape = torch.randn(3, 3)
        shape = shape @ shape.T / (3 * self.oval_spot_axis) + torch.eye(3)
        shape = shape / (1 + 1 / self.oval_spot_axis)

        size = self.mean_spot_size * torch.abs(torch.randn(1)) / 0.7979
        size = size.item()

        return loc, shape, size
    
    def spotty_noise_mask(self) -> torch.Tensor:

        loc, shape, size = self.spot_location_and_shape()
        loc, shape = loc.to(self.nodes_xyz.device), shape.to(self.nodes_xyz.device)

        mask = ((self.nodes_xyz - loc) @ shape * (self.nodes_xyz - loc)).sum(-1)
        mask = torch.exp((-1) * mask / (2 * size ** 2 + 1e-3))

        return mask.unsqueeze(-1)

    @training
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        vars = self.varsbatch_noise(x.shape[0])
        mask = self.spotty_noise_mask()

        vars = vars.to(x.device, x.dtype)
        mask = mask.to(x.device, x.dtype)

        afin = vars * mask
        
        noise = self.noise((*x.shape[:-1], len(self.vars_idx)))
        noise = noise.to(x.device, x.dtype)

        x[..., self.vars_idx] = (
            + (1 - afin) * x[..., self.vars_idx]
            + afin * noise
        )

        return x
