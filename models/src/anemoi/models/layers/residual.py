# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from abc import ABC
from abc import abstractmethod
from typing import Optional

import numpy as np
import einops
import torch
from torch import nn
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import gather_channels
from anemoi.models.distributed.graph import shard_channels
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.layers.sparse_projector import build_sparse_projector

from torch.nn import Parameter, Module
from anemoi.models.layers.sht import (
    CartesianRealSHT,
    CartesianInverseRealSHT,
    OctahedralRealSHT,
    OctahedralInverseRealSHT,
)


class BaseResidualConnection(nn.Module, ABC):
    """Base class for residual connection modules."""

    def __init__(self, graph: HeteroData | None = None) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, grid_shard_shapes=None, model_comm_group=None) -> torch.Tensor:
        """Define the residual connection operation.

        Should be overridden by subclasses.
        """
        pass


class SkipConnection(BaseResidualConnection):
    """Skip connection module

    This layer returns the most recent timestep from the input sequence.

    This module is used to bypass processing layers and directly pass the latest input forward.
    """

    def __init__(self, step: int = -1, **_) -> None:
        super().__init__()
        self.step = step

    def forward(self, x: torch.Tensor, grid_shard_shapes=None, model_comm_group=None) -> torch.Tensor:
        """Return the last timestep of the input sequence."""
        return x[:, self.step, ...]  # x shape: (batch, time, ens, nodes, features)


class TruncatedConnection(BaseResidualConnection):
    """Truncated skip connection

    This connection applies a coarse-graining and reconstruction of input features using sparse
    projections to truncate high frequency features.

    This module uses two projection operators: one to map features from the full-resolution
    grid to a truncated (coarse) grid, and another to project back to the original resolution.

    Parameters
    ----------
    graph : HeteroData, optional
        The graph containing the subgraphs for down and up projections.
    data_nodes : str, optional
        Name of the nodes representing the data nodes.
    truncation_nodes : str, optional
        Name of the nodes representing the truncated (coarse) nodes.
    edge_weight_attribute : str, optional
        Name of the edge attribute to use as weights for the projections.
    src_node_weight_attribute : str, optional
        Name of the source node attribute to use as weights for the projections.
    autocast : bool, default False
        Whether to use automatic mixed precision for the projections.
    truncation_up_file_path : str, optional
        File path (.npz) to load the up-projection matrix from.
    truncation_down_file_path : str, optional
        File path (.npz) to load the down-projection matrix from.

    Example
    -------
    >>> from torch_geometric.data import HeteroData
    >>> import torch
    >>> # Assume graph is a HeteroData object with the required edges and node types
    >>> graph = HeteroData()
    >>> # ...populate graph with nodes and edges for 'data' and 'int'...
    >>> # Example creating the projection matrices from the graph
    >>> conn = TruncatedConnection(
    ...     graph=graph,
    ...     data_nodes="data",
    ...     truncation_nodes="int",
    ...     edge_weight_attribute="gauss_weight",
    ... )
    >>> x = torch.randn(2, 4, 1, 40192, 44)  # (batch, time, ens, nodes, features)
    >>> out = conn(x)
    >>> print(out.shape)
    torch.Size([2, 4, 1, 40192, 44])

    >>> # Example specifying .npz files for projection matrices
    >>> conn = TruncatedConnection(
    ...     truncation_down_file_path="n320_to_o96.npz",
    ...     truncation_up_file_path="o96_to_n320.npz",
    ... )
    >>> x = torch.randn(2, 4, 1, 40192, 44)
    >>> out = conn(x)
    >>> print(out.shape)
    torch.Size([2, 4, 1, 40192, 44])
    """

    def __init__(
        self,
        graph: Optional[HeteroData] = None,
        data_nodes: Optional[str] = None,
        truncation_nodes: Optional[str] = None,
        edge_weight_attribute: Optional[str] = None,
        src_node_weight_attribute: Optional[str] = None,
        truncation_up_file_path: Optional[str] = None,
        truncation_down_file_path: Optional[str] = None,
        autocast: bool = False,
    ) -> None:
        super().__init__()
        up_edges, down_edges = self._get_edges_name(
            graph,
            data_nodes,
            truncation_nodes,
            truncation_up_file_path,
            truncation_down_file_path,
            edge_weight_attribute,
        )

        self.project_down = build_sparse_projector(
            graph=graph,
            edges_name=down_edges,
            edge_weight_attribute=edge_weight_attribute,
            src_node_weight_attribute=src_node_weight_attribute,
            file_path=truncation_down_file_path,
            autocast=autocast,
        )

        self.project_up = build_sparse_projector(
            graph=graph,
            edges_name=up_edges,
            edge_weight_attribute=edge_weight_attribute,
            src_node_weight_attribute=src_node_weight_attribute,
            file_path=truncation_up_file_path,
            autocast=autocast,
        )

    def _get_edges_name(
        self,
        graph,
        data_nodes,
        truncation_nodes,
        truncation_up_file_path,
        truncation_down_file_path,
        edge_weight_attribute,
    ):
        are_files_specified = truncation_up_file_path is not None and truncation_down_file_path is not None
        if not are_files_specified:
            assert graph is not None, "graph must be provided if file paths are not specified."
            assert data_nodes is not None, "data nodes name must be provided if file paths are not specified."
            assert (
                truncation_nodes is not None
            ), "truncation nodes name must be provided if file paths are not specified."
            up_edges = (truncation_nodes, "to", data_nodes)
            down_edges = (data_nodes, "to", truncation_nodes)
            assert up_edges in graph.edge_types, f"Graph must contain edges {up_edges} for up-projection."
            assert down_edges in graph.edge_types, f"Graph must contain edges {down_edges} for down-projection."
        else:
            assert (
                data_nodes is None or truncation_nodes is None or edge_weight_attribute is None
            ), "If file paths are specified, node and attribute names should not be provided."
            up_edges = down_edges = None  # Not used when loading from files
        return up_edges, down_edges

    def forward(self, x: torch.Tensor, grid_shard_shapes=None, model_comm_group=None) -> torch.Tensor:
        """Apply truncated skip connection."""
        batch_size = x.shape[0]
        x = x[:, -1, ...]  # pick latest step
        shard_shapes = apply_shard_shapes(x, 0, grid_shard_shapes) if grid_shard_shapes is not None else None

        x = einops.rearrange(x, "batch ensemble grid features -> (batch ensemble) grid features")
        x = self._to_channel_shards(x, shard_shapes, model_comm_group)
        x = self.project_down(x)
        x = self.project_up(x)
        x = self._to_grid_shards(x, shard_shapes, model_comm_group)
        x = einops.rearrange(x, "(batch ensemble) grid features -> batch ensemble grid features", batch=batch_size)

        return x

    def _to_channel_shards(self, x, shard_shapes=None, model_comm_group=None):
        return self._reshard(x, shard_channels, shard_shapes, model_comm_group)

    def _to_grid_shards(self, x, shard_shapes=None, model_comm_group=None):
        return self._reshard(x, gather_channels, shard_shapes, model_comm_group)

    def _reshard(self, x, fn, shard_shapes=None, model_comm_group=None):
        if shard_shapes is not None:
            x = fn(x, shard_shapes, model_comm_group)
        return x


class IdentityResidual(Module):

    def __init__(self, input_idx: list[int] = [], **_) -> None:

        super().__init__()
        self._internal_input_idx = input_idx
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x[..., self._internal_input_idx]


class NoResidual(Module):

    def __init__(self, input_idx: list[int] = [], **_) -> None:

        super().__init__()
        self._internal_input_idx = input_idx
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.zeros_like(x[..., self._internal_input_idx])


class SimpleOrnsteinResidual(Module):

    def __init__(
        self,
        theta_init: float = 0.00,
        theta_buff: float = 0.00,
        theta_train: bool = True,
        input_idx: list[int] = [],
        statistics: dict[str, np.ndarray] = {},
        **_
    ) -> None:

        super().__init__()

        theta_init = self.init_theta(theta_init, theta_buff, statistics)
        theta_init = np.log(theta_init / (1 - theta_init))
        theta_init = np.array(theta_init)

        weight = torch.zeros(len(input_idx))
        weight[:] = torch.from_numpy(theta_init)

        self.weight = Parameter(weight, theta_train)
        self.theta_buff = theta_buff
        self._internal_input_idx = input_idx

    def init_theta(
        self,
        theta_init: float,
        theta_buff: float,
        statistics: dict[str, np.ndarray],
    ) -> np.ndarray:
        
        theta_init = (
            theta_init
            if (theta_init != 0) or any(s not in statistics for s in {"stdev", "stdev_tend"})
            else 0.5 * (statistics["stdev_tend"] / statistics["stdev"]) ** 2
        )
        
        theta_init = (theta_init - theta_buff) / (1 - theta_buff)
        theta_init = np.where(theta_init < 1, theta_init, 0.99)
        theta_init = np.where(theta_init > 0, theta_init, 0.01)

        return theta_init

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return (
            + (1 - torch.sigmoid(self.weight) * (1 - self.theta_buff) - self.theta_buff)
            * x[..., self._internal_input_idx]
        )


class BasicOrnsteinResidual(Module):

    def __init__(
        self,
        nlat: int,
        nlon: int,
        lmax: int = 2,
        grid: str = "legendre-gauss",
        node_order: str = "lat-lon",
        theta_init: float = 0.00,
        theta_buff: float = 0.00,
        zmean_term: bool = True,
        regressors: list[str] = [],
        input_idx: list[int] = [],
        variables: dict[str, int] = {},
        statistics: dict[str, np.ndarray] = {},
    ) -> None:
        
        super().__init__()

        theta_init = self.init_theta(theta_init, theta_buff, statistics)
        theta_init = np.sqrt(4 * np.pi) * np.log(theta_init / (1 - theta_init))
        theta_init = np.array(theta_init)
        
        weight = torch.zeros(len(regressors) + 2, len(input_idx), lmax, lmax, 2)
        weight[0, :, 0, 0, 0] = torch.from_numpy(theta_init)

        self.weight = Parameter(weight)

        if grid == "octahedral":
            self.isht = OctahedralInverseRealSHT(nlat, lmax, lmax)
            self.values_reshape_for = f"... values var -> ... var values"
            self.values_reshape_inv = f"... var values -> ... values var"
            self.kwargs_reshape_for = {}
        else:
            self.isht = CartesianInverseRealSHT(nlat, nlon, lmax, grid)
            self.values_reshape_for = f"... ({node_order.replace("-", " ")}) var -> ... var lat lon"
            self.values_reshape_inv = f"... var lat lon -> ... ({node_order.replace("-", " ")}) var"
            self.kwargs_reshape_for = {"lat": nlat, "lon": nlon}

        self._regressors_input_idx = [variables[f] for f in regressors]
        self._internal_input_idx = input_idx

        muzero = torch.ones_like(weight)
        muzero[1, :, :, :, :] = 1.0 if zmean_term else 0.0

        self.register_buffer("muzero", muzero)
        self.theta_buff = theta_buff

    def init_theta(
        self,
        theta_init: float,
        theta_buff: float,
        statistics: dict[str, np.ndarray],
    ) -> np.ndarray:
        
        theta_init = (
            theta_init
            if (theta_init != 0) or any(s not in statistics for s in {"stdev", "stdev_tend"})
            else 0.5 * (statistics["stdev_tend"] / statistics["stdev"]) ** 2
        )
        
        theta_init = (theta_init - theta_buff) / (1 - theta_buff)
        theta_init = np.where(theta_init < 1, theta_init, 0.99)
        theta_init = np.where(theta_init > 0, theta_init, 0.01)

        return theta_init

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        weight = self.isht(torch.view_as_complex(self.weight * self.muzero))
        weight = einops.rearrange(weight, self.values_reshape_inv)

        return (
            + (1 - torch.sigmoid(weight[0, ...]) * (1 - self.theta_buff) - self.theta_buff)
            * x[..., self._internal_input_idx]
            + weight[1, ...]
            + sum(
                weight[i + 2, ...] * x[..., k].unsqueeze(-1)
                for i, k in enumerate(self._regressors_input_idx)
            )
        )


class CompleteOrnsteinResidual(BasicOrnsteinResidual):

    def __init__(
        self,
        nlat: int,
        nlon: int,
        lmax: int = 2,
        grid: str = "legendre-gauss",
        node_order: str = "lat-lon",
        theta_init: float = 0.05,
        theta_buff: float = 0.00,
        zmean_term: bool = True,
        regressors: list[str] = [],
        anti_aliasing: bool = True,
        skip_blur: list[str] = [],
        input_idx: list[int] = [],
        variables: dict[str, int] = {},
        statistics: dict[str, np.ndarray] = {},
    ) -> None:
        
        super().__init__(
            nlat=nlat,
            nlon=nlon,
            grid=grid,
            lmax=lmax,
            node_order=node_order,
            theta_init=theta_init,
            theta_buff=theta_buff,
            zmean_term=zmean_term,
            regressors=regressors,
            input_idx=input_idx,
            variables=variables,
            statistics=statistics,
        )

        if grid == "octahedral":
            self.x_fsht = OctahedralRealSHT(nlat)
            self.x_isht = OctahedralInverseRealSHT(nlat, self.x_fsht.lmax, self.x_fsht.mmax)
        else:
            self.x_fsht = CartesianRealSHT(nlat, nlon, grid)
            self.x_isht = CartesianInverseRealSHT(nlat, nlon, self.x_fsht.lmax, grid)

        self._blurring_input_idx = [
            int(idx)
            for idx in input_idx
            if idx not in [variables.get(v) for v in skip_blur]
        ]

        self._var_axis_tail = (
            (slice(None),) * len(self.kwargs_reshape_for)
            if len(self.kwargs_reshape_for) > 0
            else (slice(None),)
        )

        filter = torch.ones(len(self._blurring_input_idx), self.x_fsht.lmax)
        filter = filter * max(theta_init, 0.01) / (0.5 - max(theta_init, 0.01))
        filter = torch.sqrt(filter / self.x_fsht.lmax)

        walias = torch.zeros(len(self._blurring_input_idx), lmax, lmax, 2)

        self.filter = Parameter(filter)
        self.walias = Parameter(walias)

        self.lpass_filter = (
            self.blur_with_anti_aliasing
            if anti_aliasing
            else self.blur_without_anti_aliasing
        )

    def x_filter(self) -> torch.Tensor:

        filter = torch.square(self.filter)
        filter = torch.cumsum(filter, -1)
        filter = filter / (1 + filter)

        return filter

    def w_filter(self) -> torch.Tensor:

        walias = self.isht(torch.view_as_complex(self.walias))

        return torch.sigmoid(walias)

    def blur_without_anti_aliasing(self, x_blur: torch.Tensor) -> torch.Tensor:

        x_blur = self.x_fsht(x_blur)
        filter = self.x_filter()

        x_blur = x_blur * (1 - filter.unsqueeze(-1))

        return self.x_isht(x_blur)

    def blur_with_anti_aliasing(self, x_blur: torch.Tensor) -> torch.Tensor:
        
        x_skip = self.x_fsht(x_blur)
        filter = self.x_filter()
        walias = self.w_filter()

        x_skip = x_skip * (1 - filter.unsqueeze(-1))

        return (
            + walias * x_blur
            + (1 - walias) * self.x_isht(x_skip)
        )

    def blurring(self, x: torch.Tensor) -> torch.Tensor:

        x = einops.rearrange(x, self.values_reshape_for, **self.kwargs_reshape_for)

        x[..., self._blurring_input_idx, *self._var_axis_tail] = self.lpass_filter(
            x[..., self._blurring_input_idx, *self._var_axis_tail]
        )

        x = einops.rearrange(x, self.values_reshape_inv)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return super().forward(self.blurring(x))
