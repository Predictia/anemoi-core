# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Optional
from functools import reduce

import numpy as np
import einops
import numpy as np
import torch
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import get_shape_shards
from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiModelEncProcDec(nn.Module):
    """Message passing graph neural network."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
        truncation_data: dict,
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        model_config : DotDict
            Model configuration
        data_indices : dict
            Data indices
        graph_data : HeteroData
            Graph definition
        """
        super().__init__()
        model_config = DotDict(model_config)
        self._graph_data = graph_data
        self._graph_name_data = model_config.graph.data
        self._graph_name_hidden = model_config.graph.hidden

        self.multi_step = model_config.training.multistep_input
        self.num_channels = model_config.model.num_channels

        self.node_attributes = NamedNodesAttributes(model_config.model.trainable_parameters.hidden, self._graph_data)

        self._calculate_shapes_and_indices(data_indices)
        self._assert_matching_indices(data_indices)
        self.data_indices = data_indices
        self.statistics = statistics

        # read config.model.layer_kernels to get the implementation for certain layers
        self.layer_kernels_encoder = load_layer_kernels(model_config.model.layer_kernels.get("encoder", {}))
        self.layer_kernels_decoder = load_layer_kernels(model_config.model.layer_kernels.get("decoder", {}))
        self.layer_kernels_processor = load_layer_kernels(model_config.model.layer_kernels.get("processor", {}))

        self.multi_step = model_config.training.multistep_input
        self.num_channels = model_config.model.num_channels

        self.node_attributes = NamedNodesAttributes(model_config.model.trainable_parameters.hidden, self._graph_data)

        self._truncation_data = truncation_data

        self.input_dim = self._calculate_input_dim(model_config)

        # we can't register these as buffers because DDP does not support sparse tensors
        # these will be moved to the GPU when first used via sefl.interpolate_down/interpolate_up
        self.A_down, self.A_up = None, None
        if "down" in self._truncation_data:
            self.A_down = self._make_truncation_matrix(self._truncation_data["down"])
            LOGGER.info("Truncation: A_down %s", self.A_down.shape)
        if "up" in self._truncation_data:
            self.A_up = self._make_truncation_matrix(self._truncation_data["up"])
            LOGGER.info("Truncation: A_up %s", self.A_up.shape)

        # Guess lat-lon sizes (data graph) & node order (hidden graph)
        x_data = self._graph_data[self._graph_name_data].x
        x_hidden = self._graph_data[self._graph_name_hidden].x
        sfno_kwargs = {
            "nlat": torch.unique(x_data[:, 0]).shape[0],
            "nlon": torch.unique(x_data[:, 1]).shape[0],
            "output_order": np.lexsort((x_hidden[:, 1], (-1) * x_hidden[:, 0])).tolist(),
        }
        sfno_kwargs = (
            sfno_kwargs if "SFNO" in model_config.model.encoder._target_
            else {}
        )

        # Encoder data -> hidden
        self.encoder = instantiate(
            model_config.model.encoder,
            in_channels_src=self.input_dim,
            in_channels_dst=self.node_attributes.attr_ndims[self._graph_name_hidden],
            hidden_dim=self.num_channels,
            sub_graph=self._graph_data[(self._graph_name_data, "to", self._graph_name_hidden)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_data],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            layer_kernels=self.layer_kernels_encoder,
            **sfno_kwargs,
        )

        # Processor hidden -> hidden
        self.processor = instantiate(
            model_config.model.processor,
            num_channels=self.num_channels,
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_hidden)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            layer_kernels=self.layer_kernels_processor,
        )

        # Decoder hidden -> data
        self.decoder = instantiate(
            model_config.model.decoder,
            in_channels_src=self.num_channels,
            in_channels_dst=self.input_dim,
            hidden_dim=self.num_channels,
            out_channels_dst=self.num_output_channels,
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_data)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_data],
            layer_kernels=self.layer_kernels_decoder,
        )

        # Instantiation of model output bounding functions (e.g., to ensure outputs like TP are positive definite)
        self.boundings = nn.ModuleList(
            [
                instantiate(
                    cfg,
                    name_to_index=self.data_indices.internal_model.output.name_to_index,
                    statistics=self.statistics,
                    name_to_index_stats=self.data_indices.data.input.name_to_index,
                )
                for cfg in getattr(model_config.model, "bounding", [])
            ]
        )

        # Regularization (via weight normalization, w -> w / ||w||)
        self._set_normalization(
            normalization=getattr(model_config.model, "normalization", {})
        )

    def _set_normalization(self, normalization: dict) -> None:
        """
        Spectral normalization, following J. Schreck et al., Community Research Earth
        Digital Intelligence Twin (2024). Bounds the `spectral norm` of the model, in
        order to avoid explosive behaviour in autoregressive rollout (possibly caused
        by `resonant` directions in the input space of one or more operators, which
        are amplified in each forward pass). Impose [max{||Wx||/||x||} < 1] to avoid
        any `resonant` directions. Notice that, (i) chaotic behaviour is usually
        characterized by greater-than-one spectral norms (see Lyapunov exponents or
        Lipschitz continuity for a better characterization), (ii) individually
        bounding the spectral norm of the components of a model doesn't necessarily
        bound the spectral norm of the full model. Moreover, this implementation
        only works when using the (soon) deprecated `torch.nn.utils.spectral_norm`
        (instead of the recommended `torch.nn.utils.parametrizations.spectral_norm`,
        which uses torch's `parametrizations`, but only supports serialization with
        the `state_dict`, and thus is incompatible with Anemoi's implementation of
        the checkpointing callback) (early tests show that both functions produce
        exactly the same results).

        > model:
        >   normalization:
        >     modules:
        >       Linear:
        >       - _target_: torch.nn.utils.spectral_norm
        """
        # Freeze the model's state, because it may change
        named_parameters = [n for n, p in self.named_parameters() if p.requires_grad]
        modules = [m for m in self.modules()]

        # Normalizations by parameter or by module
        norm_by_params: dict = normalization.get("parameters", {})
        norm_by_module: dict = normalization.get("modules", {})
        
        # Loop through all (parameter, normalizations) pairs
        for name, norms in norm_by_params.items():
            # Loop through all normalizations for the target parameter
            for norm in norms:
                # Loop through all parameters in the model
                for n in named_parameters:
                    # If the parameter's name matches the target's name
                    if n.split(".")[-1] == name:
                        # Get the parameter's parent module
                        module = reduce(getattr, n.split(".")[:-1], self)
                        # We assume the normalization is a callable (not a class)
                        instantiate(norm, module=module, name=name)

        # Loop through all (module, normalizations) pairs
        for name, norms in norm_by_module.items():
            # Loop through all normalizations for the target module
            for norm in norms:
                # Name of the target parameter (defaults to `weight`)
                weight_name = norm.pop("weight_name", "weight")
                # Loop through all modules in the model
                for module in modules:
                    # If the module's name matches the target's name
                    if (
                        any(c.__name__ == name for c in module.__class__.__mro__) and
                        hasattr(module, weight_name) and
                        getattr(module, weight_name).requires_grad
                    ):
                        # We assume the normalization is a callable (not a class)
                        instantiate(norm, module=module, name=weight_name)
        return
    def _make_truncation_matrix(self, A, data_type=torch.float32):
        A_ = torch.sparse_coo_tensor(
            torch.tensor(np.vstack(A.nonzero()), dtype=torch.long),
            torch.tensor(A.data, dtype=data_type),
            size=A.shape,
        ).coalesce()
        return A_

    def _multiply_sparse(self, x, A):
        return torch.sparse.mm(A, x)

    def _truncate_fields(self, x, A, batch_size=None, grad_checkpoint=False):
        if not batch_size:
            batch_size = x.shape[0]
        out = []
        with torch.amp.autocast(device_type="cuda", enabled=False):
            for i in range(batch_size):
                if grad_checkpoint:
                    out.append(torch.utils.checkpoint(self.multiply_sparse, x[i, ...], A, use_reentrant=False))
                else:
                    out.append(self._multiply_sparse(x[i, ...], A))
        return torch.stack(out)

    def _assemble_input(self, x, batch_size):
        # normalize and add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                self.node_attributes(self._graph_name_data, batch_size=batch_size),
            ),
            dim=-1,  # feature dimension
        )

        x_skip = x[:, -1, ...]
        if self.A_down is not None or self.A_up is not None:
            x_skip = einops.rearrange(x_skip, "batch ensemble grid vars -> (batch ensemble) grid vars")
            # these can't be registered as buffers because ddp does not like to broadcast sparse tensors
            # hence we check that they are on the correct device ; copy should only happen in the first forward run
            if self.A_down is not None:
                self.A_down = self.A_down.to(x_skip.device)
                x_skip = self._truncate_fields(x_skip, self.A_down)  # to coarse resolution
            if self.A_up is not None:
                self.A_up = self.A_up.to(x_skip.device)
                x_skip = self._truncate_fields(x_skip, self.A_up)  # back to high resolution
            x_skip = einops.rearrange(
                x_skip, "(batch ensemble) grid vars -> batch ensemble grid vars", batch=batch_size
            )

        return x_data_latent, x_skip

    def _assemble_output(self, x_out, x_skip, batch_size, ensemble_size, dtype):
        x_out = (
            einops.rearrange(
                x_out,
                "(batch ensemble grid) vars -> batch ensemble grid vars",
                batch=batch_size,
                ensemble=ensemble_size,
            )
            .to(dtype=dtype)
            .clone()
        )

        # residual connection (just for the prognostic variables)
        x_out[..., self._internal_output_idx] += x_skip[..., self._internal_input_idx]

        for bounding in self.boundings:
            # bounding performed in the order specified in the config file
            x_out = bounding(x_out)
        return x_out

    def _calculate_input_dim(self, model_config):
        return self.multi_step * self.num_input_channels + self.node_attributes.attr_ndims[self._graph_name_data]

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        self.num_input_channels = len(data_indices.internal_model.input)
        self.num_output_channels = len(data_indices.internal_model.output)
        self.num_input_channels_prognostic = len(data_indices.model.input.prognostic)
        self._internal_input_idx = data_indices.internal_model.input.prognostic
        self._internal_output_idx = data_indices.internal_model.output.prognostic
        self.input_dim = (
            self.multi_step * self.num_input_channels + self.node_attributes.attr_ndims[self._graph_name_data]
        )

    def _assert_matching_indices(self, data_indices: dict) -> None:

        assert len(self._internal_output_idx) == len(data_indices.internal_model.output.full) - len(
            data_indices.internal_model.output.diagnostic
        ), (
            f"Mismatch between the internal data indices ({len(self._internal_output_idx)}) and "
            f"the internal output indices excluding diagnostic variables "
            f"({len(data_indices.internal_model.output.full) - len(data_indices.internal_model.output.diagnostic)})",
        )
        assert len(self._internal_input_idx) == len(
            self._internal_output_idx,
        ), f"Internal model indices must match {self._internal_input_idx} != {self._internal_output_idx}"

    def _run_mapper(
        self,
        mapper: nn.Module,
        data: tuple[Tensor],
        batch_size: int,
        shard_shapes: tuple[tuple[int, int], tuple[int, int]],
        model_comm_group: Optional[ProcessGroup] = None,
        use_reentrant: bool = False,
    ) -> Tensor:
        """Run mapper with activation checkpoint.

        Parameters
        ----------
        mapper : nn.Module
            Which processor to use
        data : tuple[Tensor]
            tuple of data to pass in
        batch_size: int,
            Batch size
        shard_shapes : tuple[tuple[int, int], tuple[int, int]]
            Shard shapes for the data
        model_comm_group : ProcessGroup
            model communication group, specifies which GPUs work together
            in one model instance
        use_reentrant : bool, optional
            Use reentrant, by default False

        Returns
        -------
        Tensor
            Mapped data
        """
        return checkpoint(
            mapper,
            data,
            batch_size=batch_size,
            shard_shapes=shard_shapes,
            model_comm_group=model_comm_group,
            use_reentrant=use_reentrant,
        )

    def forward(self, x: Tensor, *, model_comm_group: Optional[ProcessGroup] = None, **kwargs) -> Tensor:
        batch_size = x.shape[0]
        ensemble_size = x.shape[2]

        x_data_latent, x_skip = self._assemble_input(x, batch_size)
        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)

        shard_shapes_data = get_shape_shards(x_data_latent, 0, model_comm_group)
        shard_shapes_hidden = get_shape_shards(x_hidden_latent, 0, model_comm_group)

        x_data_latent, x_latent = self._run_mapper(
            self.encoder,
            (x_data_latent, x_hidden_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_data, shard_shapes_hidden),
            model_comm_group=model_comm_group,
        )

        x_latent_proc = self.processor(
            x_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        x_latent_proc = x_latent_proc + x_latent

        x_out = self._run_mapper(
            self.decoder,
            (x_latent_proc, x_data_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hidden, shard_shapes_data),
            model_comm_group=model_comm_group,
        )

        x_out = self._assemble_output(x_out, x_skip, batch_size, ensemble_size, x.dtype)

        return x_out
