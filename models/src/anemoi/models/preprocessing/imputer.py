# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import ABC
from typing import Optional

import torch
import numpy as np

from omegaconf import DictConfig
from omegaconf import OmegaConf

from torch_geometric.data import HeteroData
from scipy.spatial import KDTree

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import BasePreprocessor

LOGGER = logging.getLogger(__name__)


class BaseImputer(BasePreprocessor, ABC):
    """Base class for Imputers."""

    supports_skip_imputation = True

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        """Initialize the imputer.

        Parameters
        ----------
        config : DotDict
            configuration object of the processor
        data_indices : IndexCollection
            Data indices for input and output variables
        statistics : dict
            Data statistics dictionary
        """
        super().__init__(config, data_indices, statistics)

        # These masks describe the current batch or local batch shard of the rank.
        self.nan_locations: torch.Tensor | None = None
        self.loss_mask_training: torch.Tensor | None = None

    def _validate_indices(self):
        assert len(self.index_training_input) == len(self.index_inference_input) <= len(self.replacement), (
            f"Error creating imputation indices {len(self.index_training_input)}, "
            f"{len(self.index_inference_input)}, {len(self.replacement)}"
        )
        assert len(self.index_training_output) == len(self.index_inference_output) <= len(self.replacement), (
            f"Error creating imputation indices {len(self.index_training_output)}, "
            f"{len(self.index_inference_output)}, {len(self.replacement)}"
        )

    def _create_imputation_indices(
        self,
        statistics=None,
    ):
        """Create the indices for imputation."""
        name_to_index_training_input = self.data_indices.data.input.name_to_index
        name_to_index_inference_input = self.data_indices.model.input.name_to_index
        name_to_index_training_output = self.data_indices.data.output.name_to_index
        name_to_index_inference_output = self.data_indices.model.output.name_to_index

        self.num_training_input_vars = len(name_to_index_training_input)
        self.num_inference_input_vars = len(name_to_index_inference_input)
        self.num_training_output_vars = len(name_to_index_training_output)
        self.num_inference_output_vars = len(name_to_index_inference_output)

        (
            self.index_training_input,
            self.index_inference_input,
            self.index_training_output,
            self.index_inference_output,
            self.replacement,
        ) = ([], [], [], [], [])

        # Create indices for imputation
        for name in name_to_index_training_input:

            method = self.methods.get(name, self.default)
            if method == "none":
                LOGGER.debug(f"Imputer: skipping {name} as no imputation method is specified")
                continue

            if name_to_index_inference_input.get(name, None) is None:
                # if the variable is not in inference input (diagnostic variable), we cannot place NaNs in its inference output
                if method != self.default:
                    LOGGER.warning(
                        f"If placement of NaNs for diagnostic variables in inference output is desired, this needs to be handled by postprocessors: {name}"
                    )

            self.index_training_input.append(name_to_index_training_input[name])
            self.index_training_output.append(name_to_index_training_output.get(name, None))
            self.index_inference_input.append(name_to_index_inference_input.get(name, None))
            self.index_inference_output.append(name_to_index_inference_output.get(name, None))

            if statistics is None:
                self.replacement.append(method)
            elif isinstance(statistics, dict):
                assert method in statistics, f"{method} is not a method in the statistics metadata"
                self.replacement.append(statistics[method][name_to_index_training_input[name]])
            else:
                raise TypeError(f"Statistics {type(statistics)} is optional and not a dictionary")

            LOGGER.info(f"Imputer: replacing NaNs in {name} with value {self.replacement[-1]}")

    def get_nans(self, x: torch.Tensor) -> torch.Tensor:
        """Get NaN mask from data

        The mask is only saved for the first two dimensions (batch, timestep) and the last two dimensions (grid, variable)
        For the rest of the dimensions we select the first element since we assume the nan locations do not change along these dimensions.
        This means for the ensemble dimension: we assume that the NaN locations are the same for all ensemble members.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch, time, ..., grid, variable)

        Returns
        -------
        torch.Tensor
            Tensor with NaN locations of shape (batch, time, ..., grid)
        """
        idx = [slice(None), slice(None)] + [0] * (x.ndim - 4) + [slice(None), slice(None)]
        return torch.isnan(x[tuple(idx)])

    def _expand_subset_mask(self, x: torch.Tensor, idx_src: int, nan_locations: torch.Tensor) -> torch.Tensor:
        """Expand the subset of the nan location mask to the correct shape.

        The mask is only saved for the first dimension (batch) and the last two dimensions (grid, variable).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch, time, ..., grid, variable)
        idx_src : int
            Index of the source variable in the nan locations mask
        nan_locations : torch.Tensor
            Tensor with NaN locations of shape (batch, grid, variable)

        Returns
        -------
        torch.Tensor
            Expanded tensor with NaN locations of shape (batch, time, ..., grid)
        """
        for i in x.shape[1:-2]:
            nan_locations = nan_locations.unsqueeze(1)

        return nan_locations[..., idx_src].expand(-1, *x.shape[1:-2], -1)

    def fill_with_value(
        self, x: torch.Tensor, index_x: list[int], nan_locations: torch.Tensor, index_nl: list[int]
    ) -> torch.Tensor:
        """Fill NaN locations in the input tensor with the specified values.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        index_x : list[int]
            List of indices for the variables to be imputed
        nan_locations : torch.Tensor
            Tensor with NaN locations
        index_nl : list[int]
            List of indices in the NaN mask.

        Returns
        -------
        torch.Tensor
            Tensor where NaN locations are filled with the specified values
        """
        # Expand the nan locations to match the shape of the input tensor
        for i in x.shape[2:-2]:
            nan_locations = nan_locations.unsqueeze(2)
        for idx_src, (idx_dst, value) in zip(index_nl, zip(index_x, self.replacement)):
            if idx_src is not None and idx_dst is not None:
                x[..., idx_dst][nan_locations[..., idx_src]] = value
        return x

    def transform(
        self,
        x: torch.Tensor,
        in_place: bool = True,
        skip_imputation: bool = False,
        **_kwargs,
    ) -> torch.Tensor:
        """Impute missing values in the input tensor."""
        if not in_place:
            x = x.clone()
        if skip_imputation:
            return x

        # recalculate NaN locations every forward pass and save for backward pass
        nan_locations = self.get_nans(x)

        # choose correct index based on number of variables which are different for training and inference
        if x.shape[-1] == self.num_training_input_vars:
            # training input

            # Select the first timestep used by the loss mask and postprocessing.
            self.nan_locations = nan_locations[:, 0, ..., self.data_indices.data.input.full]

            # data indices for training input
            index = self.index_training_input

            # set training loss mask to match shape of training input
            self.loss_mask_training = torch.ones(
                (x.shape[0], x.shape[-2], len(self.data_indices.model.output.name_to_index)), device=x.device
            )  # shape (batchsize, grid, n_outputs)

            # for all variables that are imputed and part of the model output, set the loss weight to zero at NaN location
            for idx_src, idx_dst in zip(self.index_training_input, self.index_inference_output):
                if idx_src is not None and idx_dst is not None:
                    self.loss_mask_training[..., idx_dst] = (~nan_locations[:, 0, ..., idx_src]).int()

        elif x.shape[-1] == self.num_inference_input_vars:
            # inference input

            # save nan masks of inference input for inverse transform
            self.nan_locations = nan_locations[:, 0]
            self.loss_mask_training = None

            # data indices for training input
            index = self.index_inference_input
        else:
            raise ValueError(
                f"Input tensor ({x.shape[-1]}) does not match the training "
                f"({self.num_training_input_vars}) or inference shape ({self.num_inference_input_vars})",
            )

        # Replace values
        return self.fill_with_value(x, index, nan_locations, index)

    def inverse_transform(
        self,
        x: torch.Tensor,
        in_place: bool = True,
        skip_imputation: bool = False,
        **_kwargs,
    ) -> torch.Tensor:
        """Impute missing values in the input tensor."""
        if not in_place:
            x = x.clone()
        if skip_imputation:
            return x

        # Replace original nans with nan again
        if x.shape[-1] == self.num_training_output_vars:
            index = self.index_training_output
        elif x.shape[-1] == self.num_inference_output_vars:
            index = self.index_inference_output
        else:
            raise ValueError(
                f"Input tensor ({x.shape[-1]}) does not match the training "
                f"({self.num_training_output_vars}) or inference shape ({self.num_inference_output_vars})",
            )

        if self.nan_locations is None:
            msg = "Cannot inverse-transform before the corresponding input has been transformed."
            raise RuntimeError(msg)

        assert (
            x.shape[0] == self.nan_locations.shape[0]
        ), f"Batch dimension of input tensor ({x.shape[0]}) does not match the batch dimension of nan locations ({self.nan_locations.shape[0]}). Are you using the postprocessors without running the preprocessor first?"

        # Replace values
        for idx_src, idx_dst in zip(self.index_inference_input, index):
            if idx_src is not None and idx_dst is not None:
                x[..., idx_dst][self._expand_subset_mask(x, idx_src, self.nan_locations)] = torch.nan
        return x


class InputImputer(BaseImputer):
    """Imputes missing values using the statistics supplied.

    Expects the config to have keys corresponding to available statistics
    and values as lists of variables to impute.:
    ```
    default: "none"
    mean:
        - y
    maximum:
        - x
    minimum:
        - q
    ```
    """

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        super().__init__(config, data_indices, statistics)

        if isinstance(statistics, DictConfig):
            statistics = OmegaConf.to_container(statistics, resolve=True)
        self._create_imputation_indices(statistics)

        self._validate_indices()


class ConstantImputer(BaseImputer):
    """Imputes missing values using the constant value.

    Expects the config to have keys corresponding to available statistics
    and values as lists of variables to impute.:
    ```
    default: "none"
    1:
        - y
    5.0:
        - x
    3.14:
        - q
    ```
    """

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        super().__init__(config, data_indices, statistics)

        self._create_imputation_indices()

        self._validate_indices()


class NearestNeighbourImputer(BaseImputer):

    needs_graph_data = True

    def __init__(
        self,
        config: Optional[DictConfig] = None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
        graph_data: Optional[HeteroData] = None,
    ) -> None:

        super().__init__(
            config=(config | {"default": "none"}),
            data_indices=data_indices,
            statistics=statistics,
        )

        self._create_imputation_indices()
        self._validate_indices()

        assert graph_data is not None, "Missing nodes information!"
        assert (
            len({k for k in config if k != "default"}) == 1,
            "A single number of neighbours must be specified!",
        )

        x = graph_data[config.get("graph_name_data", "data")].x
        x = np.stack(
            (
                np.cos(x[:, 0]) * np.cos(x[:, 1]),
                np.cos(x[:, 0]) * np.sin(x[:, 1]),
                np.sin(x[:, 0]),
            ),
            axis=-1
        )

        k = next(k for k in config if k != "default")

        if isinstance(k, int):
            k, step = k, 1
        elif k == "guess":
            k, step = self.guess_best_strategy(x)
        elif (
            isinstance(k, str)
            and all(ss.isnumeric() for ss in k.split("x"))
        ):
            k_total = k.split("x")
            k, step = int(k_total[0]), int(k_total[1])
        else:
            raise ValueError("This imputation strategy doesn't exist!")

        k, step = max(1, k), max(1, step)
        neighbours = list(range(1, k * step + 1, step))
        nn_all2all = KDTree(x).query(x, neighbours)[-1]
        
        self.register_buffer("_nn_all2all", torch.from_numpy(nn_all2all), persistent=False)
        self.register_buffer("_saved_nanloc", torch.empty(0, dtype=torch.bool), persistent=False)

        self.x = x
        self._saved_nn_all2notnan = None

    def guess_best_strategy(
        self,
        x: np.ndarray,
    ) -> tuple[int, int]:

        raise NotImplementedError

    def get_new_nn_all2notnan(
        self,
        nan_locations: torch.Tensor,
    ) -> np.ndarray:
        
        nn_all2notnan = np.stack(
            [
                KDTree(np.where(nanloc_v[:, None].cpu(), [0, 0, 3], self.x)).query(self.x)[-1]
                for nanloc_v in nan_locations.unbind(-1)
            ],
            axis=-1
        )

        self._saved_nn_all2notnan = nn_all2notnan
        self._saved_nanloc = nan_locations

        return self._saved_nn_all2notnan

    def get_nn_all2notnan(
        self,
        nan_locations: torch.Tensor,
    ) -> np.ndarray:

        return (
            self._saved_nn_all2notnan
            if torch.equal(nan_locations, self._saved_nanloc)
            else self.get_new_nn_all2notnan(nan_locations)
        )

    def fill_with_value(
        self,
        x: torch.Tensor,
        index_x: list[int],
        nan_locations: torch.Tensor,
        index_nl: list[int],
    ) -> torch.Tensor:
        
        idx = [
            (idx_src, idx_dst)
            for idx_src, idx_dst in zip(index_nl, index_x)
            if idx_src is not None and idx_dst is not None
        ]

        nan_locations = nan_locations[*((0,) * (nan_locations.ndim - 2)), :, :]
        nn_all2notnan = self.get_nn_all2notnan(nan_locations[:, [i for i, _ in idx]])

        for k, (idx_src, idx_dst) in enumerate(idx):

            x[..., idx_dst] = x[..., nn_all2notnan[:, k], idx_dst]

            x[..., nan_locations[:, idx_src], idx_dst] = (
                x[..., self._nn_all2all[nan_locations[:, idx_src]], idx_dst]
            ).mean(-1)

        return x


class CopyImputer(BaseImputer):
    """Imputes missing values copying them from another variable.
    ```
    default: "none"
    variable_to_copy:
        - variable_missing_1
        - variable_missing_2
    ```
    """

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        super().__init__(config, data_indices, statistics)

        self._create_imputation_indices()

        self._validate_indices()

    def _create_imputation_indices(
        self,
    ):
        """Create the indices for imputation."""
        name_to_index_training_input = self.data_indices.data.input.name_to_index
        name_to_index_inference_input = self.data_indices.model.input.name_to_index
        name_to_index_training_output = self.data_indices.data.output.name_to_index
        name_to_index_inference_output = self.data_indices.model.output.name_to_index

        self.num_training_input_vars = len(name_to_index_training_input)
        self.num_inference_input_vars = len(name_to_index_inference_input)
        self.num_training_output_vars = len(name_to_index_training_output)
        self.num_inference_output_vars = len(name_to_index_inference_output)

        (
            self.index_training_input,
            self.index_inference_input,
            self.index_training_output,
            self.index_inference_output,
            self.replacement,
        ) = ([], [], [], [], [])

        # Create indices for imputation
        for name in name_to_index_training_input:
            key_to_copy = self.methods.get(name, self.default)

            if key_to_copy == "none":
                LOGGER.debug(f"Imputer: skipping {name} as no imputation method is specified")
                continue

            self.index_training_input.append(name_to_index_training_input[name])
            self.index_training_output.append(name_to_index_training_output.get(name, None))
            self.index_inference_input.append(name_to_index_inference_input.get(name, None))
            self.index_inference_output.append(name_to_index_inference_output.get(name, None))

            self.replacement.append(key_to_copy)

            LOGGER.debug(f"Imputer: replacing NaNs in {name} with value coming from variable :{self.replacement[-1]}")

    def fill_with_value(
        self, x: torch.Tensor, index_x: list[int], nan_locations: torch.Tensor, index_nl: list[int]
    ) -> torch.Tensor:
        for i in x.shape[2:-2]:
            nan_locations = nan_locations.unsqueeze(2)
        # Replace values
        for idx_src, (idx_dst, value) in zip(index_nl, zip(index_x, self.replacement)):
            if idx_dst is not None:
                assert not torch.isnan(
                    x[..., self.data_indices.data.input.name_to_index[value]][nan_locations[..., idx_src]]
                ).any(), f"NaNs found in variable {value} to be copied."
                x[..., idx_dst][nan_locations[..., idx_src]] = x[
                    ..., self.data_indices.data.input.name_to_index[value]
                ][nan_locations[..., idx_src]]
        return x
