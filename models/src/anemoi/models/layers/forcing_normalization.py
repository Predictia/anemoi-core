# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Forcing-conditioned normalization layers for climate modeling.

This module provides layers that condition the model on global atmospheric
forcings (e.g., CO2, CH4, N2O) via conditional layer normalization.

Typical usage:
    1. ForcingEmbedder at model level extracts and embeds forcings from input
    2. ForcingConditionedLayerNorm at layer level applies conditional normalization
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from torch import Tensor
from torch import nn
import torch.distributed as dist


class ForcingEmbedder(nn.Module):
    """Extracts and embeds forcing variables from input tensor.

    It extracts specified forcing variables by name, computes spatial
    mean (since forcings are typically replicated across grid points),
    and embeds them into a conditioning vector.

    The conditioning vector can then be passed to ForcingConditionedLayerNorm
    layers throughout the model architecture.

    Attributes
    ----------
    forcing_indices : list[int]
        Indices of forcing variables in the input tensor.
    embed_mlp : nn.Sequential
        MLP that embeds forcing values into conditioning space.
    """

    def __init__(
        self,
        forcing_variables: list[str],
        embed_dim: int = 32,
        condition_dim: int = 16,
        variables: Optional[dict[str, int]] = None,
    ) -> None:
        """Initialize ForcingEmbedder.

        Parameters
        ----------
        forcing_variables : list[str]
            Names of forcing variables to extract (e.g., ["ch4global", "co2mass"]).
            Variable names must exist in the `variables` dict.
        embed_dim : int, optional
            Hidden dimension for the embedding MLP, by default 32.
        condition_dim : int, optional
            Output dimension of the conditioning vector, by default 16.
        variables : dict[str, int], optional
            Mapping from variable names to indices in the input tensor.
            This is typically injected by the Anemoi framework at model
            instantiation time, by default None (empty dict).

        Raises
        ------
        ValueError
            If any forcing variable name is not found in the variables dict.
        """
        super().__init__()

        if variables is None:
            variables = {}

        # Convert variable names to indices (following CompleteOrnsteinResidual pattern)
        self.forcing_indices = []
        for var_name in forcing_variables:
            if var_name not in variables:
                raise ValueError(
                    f"Forcing variable '{var_name}' not found in variables dict. "
                    f"Available variables: {list(variables.keys())}"
                )
            self.forcing_indices.append(variables[var_name])

        self.num_forcings = len(self.forcing_indices)

        # Embedding MLP: raw forcings -> conditioning vector
        # Two-layer MLP with GELU activation
        self.embed_mlp = nn.Sequential(
            nn.Linear(self.num_forcings, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, condition_dim),
        )

        # Store variable names for debugging/inspection
        self.forcing_variables = forcing_variables

    def forward(self, x: Tensor) -> Tensor:
        """Extract and embed forcing variables from input.

        Parameters
        ----------
        x : Tensor
            Input tensor containing all variables.
            Expected shapes:
            - [batch, vars]: Already batch-aggregated forcings
            - [batch, grid, vars]: Standard format
            - [batch, time, ensemble, grid, vars]: Full rollout format

        Returns
        -------
        Tensor
            Conditioning vector of shape [batch, condition_dim].
            This vector is spatially-averaged (global) and can be broadcast
            to all grid points in subsequent layers.

        Notes
        -----
        The method automatically handles spatial averaging since forcings
        are typically replicated across all grid points (no spatial variation).
        """
        if x.ndim < 2:
            raise ValueError(
                f"Expected forcing input with at least 2 dimensions [batch, ..., vars], got shape {tuple(x.shape)}."
            )

        # Extract forcing variables
        forcings = x[..., self.forcing_indices]

        # Compute spatial mean to get batch-level forcings
        # We reduce all dimensions except batch and forcings
        while forcings.ndim > 2:
            # Keep reducing dimensions until we have [batch, num_forcings]
            # Mean over dimension 1 (grid, time, or ensemble depending on input shape)
            forcings = forcings.mean(dim=1)

        # Embed forcings into conditioning space
        cond = self.embed_mlp(forcings)  # [batch, condition_dim]

        return cond


class ForcingConditionedLayerNorm(nn.Module):
    """Conditional Layer Normalization conditioned on forcing variables.

    This layer applies layer normalization with scale and bias modulated by
    a conditioning vector (typically from ForcingEmbedder). It follows the
    same pattern as ConditionalLayerNorm used for diffusion models, but is
    designed specifically for forcing-based conditioning.

    The normalization formula is:
        x_norm = scale(cond) * LayerNorm(x) + bias(cond)

    where scale and bias are learned linear transformations of the conditioning
    vector.

    Attributes
    ----------
    norm : nn.LayerNorm
        Standard layer normalization without learnable affine parameters.
    scale : nn.Linear
        Linear layer that transforms conditioning to scale parameters.
    bias : nn.Linear
        Linear layer that transforms conditioning to bias parameters.
    """

    def __init__(
        self,
        normalized_shape: int,
        condition_dim: int = 16,
        zero_init: bool = True,
        autocast: bool = True,
    ) -> None:
        """Initialize ForcingConditionedLayerNorm.

        Parameters
        ----------
        normalized_shape : int
            Dimension over which to normalize (typically the hidden dimension).
        condition_dim : int, optional
            Dimension of the conditioning vector from ForcingEmbedder,
            by default 16.
        zero_init : bool, optional
            If True, initializes scale and bias transformation weights to zeros.
            This makes the layer behave like standard layer normalization
            initially, allowing the model to gradually learn forcing influence,
            by default True.
        autocast : bool, optional
            If True, automatically cast output to match input dtype for
            mixed precision training, by default True.
        """
        super().__init__()

        # Layer normalization without learnable affine parameters
        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)

        # Linear transformations for scale and bias modulation
        self.scale = nn.Linear(condition_dim, normalized_shape)
        self.bias = nn.Linear(condition_dim, normalized_shape)

        self.autocast = autocast

        if zero_init:
            # Initialize to zeros so layer starts as standard LayerNorm
            nn.init.zeros_(self.scale.weight)
            nn.init.zeros_(self.scale.bias)
            nn.init.zeros_(self.bias.weight)
            nn.init.zeros_(self.bias.bias)

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        """Apply forcing-conditioned layer normalization.

        Parameters
        ----------
        x : Tensor
            Input tensor to normalize. Shape: [..., normalized_shape]
        cond : Tensor, optional
            Conditioning vector from ForcingEmbedder.
            Shape: [batch, condition_dim]
            If None, applies standard layer normalization, by default None.

        Returns
        -------
        Tensor
            Normalized and conditioned output with same shape as input.

        Notes
        -----
        The conditioning vector is broadcast across spatial/sequence dimensions
        automatically. This allows a single global forcing value to modulate
        all grid points uniformly.
        """
        # Apply standard layer normalization
        out = self.norm(x)

        if cond is not None:
            # Compute scale and bias from conditioning
            scale = self.scale(cond)  # [batch, normalized_shape]
            bias = self.bias(cond)  # [batch, normalized_shape]

            # Scale and bias are uniformly applied to all grid points.
            # Reshape for broadcasting if needed
            # x shape: [batch, grid, normalized_shape]
            # scale/bias shape: [batch, normalized_shape]
            # Need to add grid dimension: [batch, 1, normalized_shape]
            if x.ndim > scale.ndim:
                for _ in range(x.ndim - scale.ndim):
                    scale = scale.unsqueeze(1)
                    bias = bias.unsqueeze(1)

            # Apply modulation: scale * norm(x) + bias
            out = out * (1 + scale) + bias

            ######## DEBUG #########
            # if not dist.is_initialized() or dist.get_rank() == 0:
            #     print("**************************************************")
            #     print(f"scale min: {scale.min().item():.4e}, bias min: {bias.min().item():.4e}")
            #     print(f"scale mean: {scale.mean().item():.4e}, bias mean: {bias.mean().item():.4e}")
            #     print(f"scale max: {scale.max().item():.4e}, bias max: {bias.max().item():.4e}")
            #     print(f"scale std: {scale.std().item():.4e}, bias std: {bias.std().item():.4e}")
            #     print("**************************************************")
            ######## DEBUG #########

        # Cast back to input dtype if needed (for mixed precision)
        return out.type_as(x) if self.autocast else out