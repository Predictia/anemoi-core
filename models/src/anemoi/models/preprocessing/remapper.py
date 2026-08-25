# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Optional

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import BasePreprocessor
from anemoi.models.preprocessing.mappings import affine_transform
from anemoi.models.preprocessing.mappings import asinh_converter
from anemoi.models.preprocessing.mappings import atanh_converter
from anemoi.models.preprocessing.mappings import boxcox_converter
from anemoi.models.preprocessing.mappings import displace_boundary_atoms
from anemoi.models.preprocessing.mappings import expm1_converter
from anemoi.models.preprocessing.mappings import inverse_affine_transform
from anemoi.models.preprocessing.mappings import inverse_asinh_converter
from anemoi.models.preprocessing.mappings import inverse_atanh_converter
from anemoi.models.preprocessing.mappings import inverse_boxcox_converter
from anemoi.models.preprocessing.mappings import inverse_displace_boundary_atoms
from anemoi.models.preprocessing.mappings import inverse_power_transform
from anemoi.models.preprocessing.mappings import inverse_sqrt_converter
from anemoi.models.preprocessing.mappings import log1p_converter
from anemoi.models.preprocessing.mappings import noop
from anemoi.models.preprocessing.mappings import power_transform
from anemoi.models.preprocessing.mappings import sqrt_converter

LOGGER = logging.getLogger(__name__)


class Remapper(BasePreprocessor):
    """Remap and convert variables for single variables."""

    supported_methods = {
        method: [f, inv]
        for method, f, inv in zip(
            ["log1p", "sqrt", "boxcox", "atanh", "asinh", "power", "displace_boundary_atoms", "affine", "none"],
            [
                log1p_converter,
                sqrt_converter,
                boxcox_converter,
                atanh_converter,
                asinh_converter,
                power_transform,
                displace_boundary_atoms,
                affine_transform,
                noop,
            ],
            [
                expm1_converter,
                inverse_sqrt_converter,
                inverse_boxcox_converter,
                inverse_atanh_converter,
                inverse_asinh_converter,
                inverse_power_transform,
                inverse_displace_boundary_atoms,
                inverse_affine_transform,
                noop,
            ],
        )
    }

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        super().__init__(config, data_indices, statistics)
        self._create_remapping_indices(statistics)
        self._validate_indices()

    def _validate_indices(self):
        assert (
            len(self.index_training_input)
            == len(self.index_inference_input)
            == len(self.index_inference_output)
            == len(self.index_training_out)
            == len(self.remappers)
        ), (
            f"Error creating conversion indices {len(self.index_training_input)}, "
            f"{len(self.index_inference_input)}, {len(self.index_training_input)}, {len(self.index_training_out)}, {len(self.remappers)}"
        )

    def _create_remapping_indices(
        self,
        statistics=None,
    ):
        """Create the parameter indices for remapping."""
        # list for training and inference mode as position of parameters can change
        name_to_index_training_input = self.data_indices.data.input.name_to_index
        name_to_index_inference_input = self.data_indices.model.input.name_to_index
        name_to_index_training_output = self.data_indices.data.output.name_to_index
        name_to_index_inference_output = self.data_indices.model.output.name_to_index
        self.num_training_input_vars = len(name_to_index_training_input)
        self.num_inference_input_vars = len(name_to_index_inference_input)
        self.num_training_output_vars = len(name_to_index_training_output)
        self.num_inference_output_vars = len(name_to_index_inference_output)

        (
            self.remappers,
            self.backmappers,
            self.index_training_input,
            self.index_training_out,
            self.index_inference_input,
            self.index_inference_output,
            self.remapper_kwargs,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        # Create parameter indices for remapping variables
        for name in name_to_index_training_input:
            method = self.methods.get(name, self.default)
            if method in self.supported_methods:
                self.remappers.append(self.supported_methods[method][0])
                self.backmappers.append(self.supported_methods[method][1])
                self.index_training_input.append(name_to_index_training_input[name])
                if name in name_to_index_training_output:
                    self.index_training_out.append(name_to_index_training_output[name])
                else:
                    self.index_training_out.append(None)
                if name in name_to_index_inference_input:
                    self.index_inference_input.append(name_to_index_inference_input[name])
                else:
                    self.index_inference_input.append(None)
                if name in name_to_index_inference_output:
                    self.index_inference_output.append(name_to_index_inference_output[name])
                else:
                    # this is a forcing variable. It is not in the inference output.
                    self.index_inference_output.append(None)
                if method in self.method_kwargs:
                    self.remapper_kwargs.append(self.method_kwargs[method])
                else:
                    self.remapper_kwargs.append({})
            else:
                raise KeyError(f"Unknown remapping method for {name}: {method}")

    def transform(self, x, in_place: bool = True) -> torch.Tensor:
        if not in_place:
            x = x.clone()
        if x.shape[-1] == self.num_training_input_vars:
            idx = self.index_training_input
        elif x.shape[-1] == self.num_inference_input_vars:
            idx = self.index_inference_input
        else:
            raise ValueError(
                f"Input tensor ({x.shape[-1]}) does not match the training "
                f"({self.num_training_input_vars}) or inference shape ({self.num_inference_input_vars})",
            )
        for i, remapper, kwargs in zip(idx, self.remappers, self.remapper_kwargs):
            if i is not None:
                x[..., i] = remapper(x[..., i], **kwargs)
        return x

    def inverse_transform(self, x, in_place: bool = True) -> torch.Tensor:
        if not in_place:
            x = x.clone()
        if x.shape[-1] == self.num_training_output_vars:
            idx = self.index_training_out
        elif x.shape[-1] == self.num_inference_output_vars:
            idx = self.index_inference_output
        else:
            raise ValueError(
                f"Input tensor ({x.shape[-1]}) does not match the training "
                f"({self.num_training_output_vars}) or inference shape ({self.num_inference_output_vars})",
            )
        for i, backmapper, kwargs in zip(idx, self.backmappers, self.remapper_kwargs):
            if i is not None:
                x[..., i] = backmapper(x[..., i], **kwargs)
        return x


class GHGtoERF(BasePreprocessor):

    # SARF_CO2  = alpha(C, N) * ln(C / C0)
    # alpha     = a1*(C - C0)**2 + b1*|C - C0| + c1*sqrt(Nbar) + d1
    # Nbar      = 0.5*(N + N0)   [N2O in ppb]
    _CO2_COEF = {
        "a1": -2.4785e-7,   # W m-2 ppm-2
        "b1":  7.5906e-4,   # W m-2 ppm-1
        "c1": -2.1492e-3,   # W m-2 ppb-1/2
        "d1":  5.2488,      # W m-2
        "C0":  277.15,      # ppm  [1750 CE preindustrial baseline]
    }

    # SARF_N2O  = alpha(C, N, M) * (sqrt(N) - sqrt(N0))
    # alpha     = a2*sqrt(Cbar) + b2*sqrt(Nbar) + c2*sqrt(Mbar) + d2
    # Cbar      = 0.5*(C + C0) [ppm], Nbar = 0.5*(N + N0) [ppb], Mbar = 0.5*(M + M0) [ppb]
    _N2O_COEF = {
        "a2": -3.4197e-4,   # W m-2 ppm-1/2 ppb-1/2
        "b2":  2.5455e-4,   # W m-2 ppb-1
        "c2": -2.4357e-4,   # W m-2 ppb-1
        "d2":  0.12173,     # W m-2 ppb-1/2
        "N0":  273.87,      # ppb  [1750 CE preindustrial baseline]
    }

    # SARF_CH4 = alpha(M, N) * (sqrt(M) - sqrt(M0))
    # alpha    = a3*sqrt(Mbar) + b3*sqrt(Nbar) + d3
    # Mbar     = 0.5*(M + M0) [ppb], Nbar = 0.5*(N + N0) [ppb]
    _CH4_COEF = {
        "a3": -8.9603e-5,   # W m-2 ppb-1
        "b3": -1.2462e-4,   # W m-2 ppb-1
        "d3":  0.045194,    # W m-2 ppb-1/2
        "M0":  731.41,      # ppb  [1750 CE preindustrial baseline]
    }

    # SARF_CFC = RE * (X - X0)   [W m-2]
    # Radiative efficiencies (RE) from AR6 Table 7.SM.7 (Hodnebrog et al., 2020)
    _CFC_COEF = {
        "cfc11_re": 0.259,   # W m-2 ppb-1
        "cfc12_re": 0.320,   # W m-2 ppb-1
        "cfc11_0":  0.0,     # ppb  [1750 CE preindustrial baseline]
        "cfc12_0":  0.0,     # ppb  [1750 CE preindustrial baseline]
    }

    # ---- ERF/SARF tropospheric-adjustment scale factors ----
    # Derived from RFMIP model-ensemble means in AR6 WGI Table 7.5.
    # Consistent with the NOAA AGGI 2022 transition from direct RF to ERF.
    # CH4 < 1.0 because rapid adjustments (stratospheric cooling, H2O changes)
    # partially offset the positive direct forcing.
    # CFC-11 and CFC-12 > 1.0 because tropospheric adjustments amplify their
    # long-wave direct forcing (note: ozone-depletion indirect effect is excluded).
    ERF_SARF_RATIO = {
        "co2":   1.05,
        "ch4":   0.95,
        "n2o":   1.07,
        "cfc11": 1.13,
        "cfc12": 1.12,
    }

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        
        super().__init__(config, data_indices, statistics)

        self.name_to_index_training_input = self.data_indices.data.input.name_to_index
        self.name_to_index_inference_input = self.data_indices.model.input.name_to_index

        self.variable_names = config.get("variable_names", {})
        self.unit_conversion = config.get("unit_conversion", {})
        self.unit_conversion = {
            var: self.unit_conversion.get(var, 1.0)
            for var in ["co2", "n2o", "ch4", "cfc11", "cfc12"]
        }

        assert {"co2", "n2o", "ch4"}.issubset(self.variable_names)

        assert set(self.variable_names.values()).issubset(self.name_to_index_training_input)
        assert set(self.variable_names.values()).issubset(self.name_to_index_inference_input)
    
    def transform(self, x: torch.Tensor, in_place: bool = True) -> torch.Tensor:

        if not in_place:
            x = x.clone()
        
        if x.shape[-1] == len(self.name_to_index_training_input):
            name_to_index = self.name_to_index_training_input
        elif x.shape[-1] == len(self.name_to_index_inference_input):
            name_to_index = self.name_to_index_inference_input
        else:
            raise ValueError(
                f"Input tensor ({x.shape[-1]}) does not match the training "
                f"({len(self.name_to_index_training_input)}) or inference shape "
                f"({len(self.name_to_index_inference_input)})",
            )

        co2_idx = name_to_index[self.variable_names["co2"]]
        n2o_idx = name_to_index[self.variable_names["n2o"]]
        ch4_idx = name_to_index[self.variable_names["ch4"]]

        x[..., co2_idx] = x[..., co2_idx] * self.unit_conversion["co2"]
        x[..., n2o_idx] = x[..., n2o_idx] * self.unit_conversion["n2o"]
        x[..., ch4_idx] = x[..., ch4_idx] * self.unit_conversion["ch4"]

        co2_sarf = (
            (
                self._CO2_COEF["a1"] * (x[..., co2_idx] - self._CO2_COEF["C0"]) ** 2
                + self._CO2_COEF["b1"] * (x[..., co2_idx] - self._CO2_COEF["C0"]).abs()
                + self._CO2_COEF["c1"] * (0.5 * (x[..., n2o_idx] + self._N2O_COEF["N0"])) ** 0.5
                + self._CO2_COEF["d1"]
            )
            * torch.log(x[..., co2_idx] / self._CO2_COEF["C0"])
        )

        n2o_sarf = (
            (
                self._N2O_COEF["a2"] * (0.5 * (x[..., co2_idx] + self._CO2_COEF["C0"])) ** 0.5
                + self._N2O_COEF["b2"] * (0.5 * (x[..., n2o_idx] + self._N2O_COEF["N0"])) ** 0.5
                + self._N2O_COEF["c2"] * (0.5 * (x[..., ch4_idx] + self._CH4_COEF["M0"])) ** 0.5
                + self._N2O_COEF["d2"]
            )
            * (x[..., n2o_idx] ** 0.5 - self._N2O_COEF["N0"] ** 0.5)
        )

        ch4_sarf = (
            (
                self._CH4_COEF["a3"] * (0.5 * (x[..., ch4_idx] + self._CH4_COEF["M0"])) ** 0.5
                + self._CH4_COEF["b3"] * (0.5 * (x[..., n2o_idx] + self._N2O_COEF["N0"])) ** 0.5
                + self._CH4_COEF["d3"]
            )
            * (x[..., ch4_idx] ** 0.5 - self._CH4_COEF["M0"] ** 0.5)
        )

        x[..., co2_idx] = co2_sarf * self.ERF_SARF_RATIO["co2"]
        x[..., n2o_idx] = n2o_sarf * self.ERF_SARF_RATIO["n2o"]
        x[..., ch4_idx] = ch4_sarf * self.ERF_SARF_RATIO["ch4"]

        if {"cfc11", "cfc12"}.issubset(self.variable_names):

            cfc11_idx = name_to_index[self.variable_names["cfc11"]]
            cfc12_idx = name_to_index[self.variable_names["cfc12"]]

            x[..., cfc11_idx] = (
                (
                    x[..., cfc11_idx] * self.unit_conversion["cfc11"]
                    - self._CFC_COEF["cfc11_0"]
                )
                * self.ERF_SARF_RATIO["cfc11"]
                * self._CFC_COEF["cfc11_re"]
            )

            x[..., cfc12_idx] = (
                (
                    x[..., cfc12_idx] * self.unit_conversion["cfc12"]
                    - self._CFC_COEF["cfc12_0"]
                )
                * self.ERF_SARF_RATIO["cfc12"]
                * self._CFC_COEF["cfc12_re"]
            )

        return x
