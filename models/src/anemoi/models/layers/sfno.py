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


import logging
import torch
import numpy as np

from torch.utils.checkpoint import checkpoint
from torch import Tensor
from torch.nn import (
    Parameter,
    Module,
    Conv1d,
    Conv2d,
    Sequential,
    Dropout1d,
    Identity,
    GELU,
    init,
)

from typing import Tuple
from anemoi.models.layers.block import BaseBlock

LOGGER = logging.getLogger(__name__)


class ChannelSimpleMixer(Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gain: float = 1.0,
        flat: bool = True,
        bias: bool = True,
    ) -> None:
        
        super().__init__()

        mixer_builder = Conv1d if flat else Conv2d

        self.mixer = mixer_builder(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )

        init.normal_(
            tensor=self.mixer.weight,
            std=np.sqrt(gain / in_channels),
        )

        if self.mixer.bias is not None:

            init.constant_(
                tensor=self.mixer.bias,
                val=0.0,
            )
    
    def forward(self, x: Tensor) -> Tensor:

        return self.mixer(x)


class ChannelDeepMixer(Module):

    def __init__(
        self,
        in_features: int,
        out_features: int | None = None,
        hidden_features: int | None = None,
        activation: Module = GELU,
        drop_rate: float = 0.0,
        gain: float = 1.0,
        output_bias: bool = False,
        checkpointing: bool = False,
    ) -> None:
        
        super().__init__()

        self.checkpointing = checkpointing
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Input dense layer
        mix_in = ChannelSimpleMixer(
            in_channels=in_features,
            out_channels=hidden_features,
            gain=2.0,
        )

        # Activation layer
        activation = activation()

        # Output dense layer
        mix_out = ChannelSimpleMixer(
            in_channels=hidden_features,
            out_channels=out_features,
            gain=gain,
            bias=output_bias,
        )

        # Dropout layer
        dropout = Dropout1d(drop_rate) if drop_rate > 0.0 else Identity()

        # Full network
        self.mlp = Sequential(mix_in, activation, dropout, mix_out, dropout)

    @torch.jit.ignore
    def checkpoint_forward(self, x: Tensor) -> Tensor:
        return checkpoint(self.mlp, x)

    def forward(self, x: Tensor) -> Tensor:
        return self.checkpoint_forward(x) if self.checkpointing else self.mlp(x)


class SpectralConvS2(Module):
    """
    Spectral Convolution according to Driscoll & Healy. Designed for convolutions on the two-sphere S2
    using the Spherical Harmonic Transforms in torch-harmonics, but supports convolutions on the periodic
    domain via the RealFFT2 and InverseRealFFT2 wrappers.
    """

    def __init__(
        self,
        forward_transform: Module,
        inverse_transform: Module,
        in_channels: int,
        out_channels: int,
        operator: str = "driscoll-healy",
        gain: float = 2.0,
        bias: bool = False,
        trainable: int = 0,
    ) -> None:

        super().__init__()

        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.modes_lat = getattr(self.inverse_transform, "lmax")
        self.modes_lon = getattr(self.inverse_transform, "mmax")

        # Inverse transform for residual (skip) connection
        forlat = getattr(self.forward_transform, "nlat")
        invlat = getattr(self.inverse_transform, "nlat")
        forlon = getattr(self.forward_transform, "nlon")
        invlon = getattr(self.inverse_transform, "nlon")

        if (forlat != invlat) or (forlon != invlon):
            self.scale_residual = self.inverse_residual
        else:
            self.scale_residual = self.no_inverse_residual

        # Mimics Anemoi's trainable parameters
        self.trainable = Parameter(torch.zeros(trainable, self.modes_lat, self.modes_lon, 2))
        in_channels = in_channels + trainable

        # Define spectral filter operation
        self.einsum, shape = self.build_filter_operator(operator)

        shape = (out_channels, in_channels) + shape
        scale = np.sqrt(gain / in_channels)

        weight = torch.randn(*shape, 2) / np.sqrt(2)
        weight = scale * weight

        self.weight = Parameter(weight)

        # Bias, if any
        if bias: self.bias = Parameter(torch.zeros(1, out_channels, 1))
    
    def build_filter_operator(self, operator: str) -> Tuple[str, Tuple[int]]:

        EINSUM = {
            "diagonal": "...ilm, oilm -> ...olm",
            "block-diagonal": "...ilm, oilnm -> ...oln",
            "driscoll-healy": "...ilm, oil -> ...olm",
        }

        SHAPES = {
            "diagonal": (self.modes_lat, self.modes_lon),
            "block-diagonal": (self.modes_lat, self.modes_lon, self.modes_lon),
            "driscoll-healy": (self.modes_lat,),            
        }

        assert operator in set(EINSUM) & set(SHAPES), f"Unknown operator type {operator}."

        return EINSUM[operator], SHAPES[operator]

    def inverse_residual(self, x: Tensor, residual: Tensor) -> Tensor:

        return self.inverse_transform(x)
    
    def no_inverse_residual(self, x: Tensor, residual: Tensor) -> Tensor:

        return residual
    
    def concat_trainable(self, x: Tensor) -> Tensor:

        trainable = torch.view_as_complex(self.trainable)
        trainable = trainable.expand(x.shape[0], -1, -1, -1)

        return torch.cat([x, trainable], dim=1)

    def forward(self, x: Tensor) -> Tensor:

        dtype = x.dtype
        device = x.device.type

        x = x.float()
        residual = x

        with torch.autocast(device, enabled=False):
            x = self.forward_transform(x)
            residual = self.scale_residual(x, residual)

        x = self.concat_trainable(x)
        w = torch.view_as_complex(self.weight)
        x = torch.einsum(self.einsum, x, w)

        with torch.autocast(device, enabled=False):
            x = self.inverse_transform(x)

        if hasattr(self, "bias"):
            x = x + self.bias

        return x.type(dtype), residual


class SFNO2HEALPixBlock(BaseBlock):

    def __init__(
        self,
        forward_transform: Module,
        inverse_transform: Module,
        input_dim: int,
        output_dim: int,
        operator: str = "driscoll-healy",
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.0,
        drop_path: float = 0.0,
        activation: Module = GELU,
        norm_layer: Module = Identity,
        inner_skip: str | None = None,
        outer_skip: str | None = "linear",
        mix_first: bool = False,
        mix_after: bool = True,
        filter_bias: bool = False,
        trainable: int = 0,
    ) -> None:
        
        super().__init__()

        # Check dimensions
        if "identity" in (inner_skip, outer_skip):
            assert input_dim == output_dim, "Identity skip requires matching sizes."
        
        # Mix channels before spectral filter
        if mix_first:
            self.mix_first = ChannelSimpleMixer(input_dim, input_dim, 0.0, False)

        # Define "gain" between layers
        gain = 1.0 if activation == Identity else 2.0

        if inner_skip:
            gain = gain / 2.0

        # Spectral filter (Driscoll-Healy)
        self.global_conv = SpectralConvS2(
            forward_transform=forward_transform,
            inverse_transform=inverse_transform,
            in_channels=input_dim,
            out_channels=output_dim,
            operator=operator,
            gain=gain,
            bias=filter_bias,
            trainable=trainable,
        )

        # Inner skip: x -> f(x) + x
        if inner_skip == "linear":
            self.inner_skip = ChannelSimpleMixer(input_dim, output_dim, gain)
        elif inner_skip == "identity":
            self.inner_skip = Identity()
        elif inner_skip is None:
            pass
        else:
            raise NotImplementedError(f"Unknown skip {inner_skip}.")

        # Normalisation layer
        self.norm_before_mix = norm_layer()

        # Dropout layer (drop path)
        assert drop_path == 0.0, "DropPath not implemented."
        self.drop_path = Identity()

        # Mix channels after spectral filter (deep mixer)
        gain = 0.5 if outer_skip else 1.0

        if mix_after:
            self.mlp_mixer = ChannelDeepMixer(
                in_features=output_dim,
                out_features=output_dim,
                hidden_features=int(output_dim * mlp_ratio),
                activation=activation,
                drop_rate=drop_rate,
                gain=gain,
            )

        # Outer skip: x -> F(x) + x
        if outer_skip == "linear":
            self.outer_skip = ChannelSimpleMixer(input_dim, output_dim, gain)
        elif outer_skip == "identity":
            self.outer_skip = Identity()
        elif outer_skip is None:
            pass
        else:
            raise NotImplementedError(f"Unknown skip {outer_skip}.")

        # Normalisation layer
        self.norm_after_mix = norm_layer()

    def forward(self, x: Tensor) -> Tensor:

        if hasattr(self, "mix_first"):
            x = x + self.mix_first(x)

        x, residual = self.global_conv(x)
        x = self.norm_before_mix(x)

        if hasattr(self, "inner_skip"):
            x = x + self.inner_skip(residual)

        if hasattr(self, "mlp_mixer"):
            x = self.mlp_mixer(x)

        x = self.norm_after_mix(x)
        x = self.drop_path(x)

        if hasattr(self, "outer_skip"):
            x = x + self.outer_skip(residual)

        return x
