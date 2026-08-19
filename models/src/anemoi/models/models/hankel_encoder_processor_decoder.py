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

import einops
import torch

from hydra.utils import instantiate
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.models import AnemoiModelEncProcDec
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiHankelModelEncProcDec(AnemoiModelEncProcDec):

    def _build_networks(self, model_config: DotDict) -> None:

        super()._build_networks(model_config)

        self.hankel = instantiate(
            model_config.model.hankel,
            nlat=len(torch.unique(self._graph_data[self._graph_name_data].x[:, 0])),
            nlon=len(torch.unique(self._graph_data[self._graph_name_data].x[:, 1])),
        )

        latent_variables = model_config.model.hankel.latent_variables
        target_variables = model_config.model.hankel.target_variables
        in_idx = self.data_indices.model.input.name_to_index
        out_idx = self.data_indices.model.output.name_to_index

        self.hankel_latent_input_idx = [in_idx[v] for v in latent_variables]
        self.hankel_latent_output_idx = [out_idx[v] for v in latent_variables]
        self.hankel_target_output_idx = [out_idx[v] for v in target_variables]
        self.hankel_target_latent_idx = [
            self.hankel_latent_input_idx.index(in_idx[v])
            for v in target_variables.values()
        ]
        self.hankel_target_internal_idx = [
            list(self._internal_input_idx).index(in_idx[v])
            for v in target_variables
        ]

    def _assemble_output(self, x_out, x_skip, x_hankel, batch_size, ensemble_size, dtype):

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

        x_hankel = x_hankel[:, -1, ...]
        x_out_bk = x_out[..., self.hankel_target_output_idx].clone()

        x_out[..., self._internal_output_idx] += x_skip
        x_out[..., self.hankel_latent_output_idx] = x_hankel
        x_out[..., self.hankel_target_output_idx] = (
            + 0.5 * x_skip[..., self.hankel_target_internal_idx] 
            + 0.5 * x_hankel[..., self.hankel_target_latent_idx]
            + x_out_bk
        )

        for bounding in self.boundings:
            x_out = bounding(x_out)
        
        return x_out

    def forward(
        self,
        x: Tensor,
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        **kwargs,
    ) -> Tensor:

        batch_size = x.shape[0]
        ensemble_size = x.shape[2]
        in_out_sharded = grid_shard_shapes is not None
        self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded, model_comm_group)

        x_hankel = self.hankel(x[..., self.hankel_latent_input_idx]).to(x.dtype)
        x[..., self.hankel_latent_input_idx] = x_hankel

        x_data_latent, shard_shapes_data = self._assemble_input(x, batch_size, grid_shard_shapes, model_comm_group)
        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)
        shard_shapes_hidden = get_shard_shapes(x_hidden_latent, 0, model_comm_group)

        if hasattr(self, "residual_connection"):
            x_skip = self.residual_connection(
                x,
                grid_shard_shapes=grid_shard_shapes,
                model_comm_group=model_comm_group,
            )[..., self._internal_input_idx]
        elif hasattr(self, "learnable_residual"):
            x_skip = self.learnable_residual(x[:, -1, ...])

        if self.residual_only_mode:
            return self._residual_only_mode(x_skip, batch_size, ensemble_size)

        x_data_latent, x_latent = self.encoder(
            (x_data_latent, x_hidden_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_data, shard_shapes_hidden),
            model_comm_group=model_comm_group,
            x_src_is_sharded=in_out_sharded,
            x_dst_is_sharded=False,
            keep_x_dst_sharded=True,
        )

        x_latent_proc = self.processor(
            x=x_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        x_latent_proc = (
            x_latent_proc + x_latent
            if self.latent_residual
            else x_latent_proc
        )

        x_out = self.decoder(
            (x_latent_proc, x_data_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hidden, shard_shapes_data),
            model_comm_group=model_comm_group,
            x_src_is_sharded=True,
            x_dst_is_sharded=in_out_sharded,
            keep_x_dst_sharded=in_out_sharded,
        )

        x_out = self._assemble_output(x_out, x_skip, x_hankel, batch_size, ensemble_size, x.dtype)

        return x_out
