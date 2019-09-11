# Copyright (c) 2017 The Khronos Group Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division, print_function, absolute_import

import math
import os
import sys
import typing

import numpy as np
import six
import torch

from nnef_tools.backend.pytorch.nnef_module import NNEFModule, to_torch_tensor, to_numpy_array
from nnef_tools.core import json_utils
from nnef_tools.core import utils
from nnef_tools.io.nnef import nnef_io
from nnef_tools.io.nnef.nnef_graph import *


def run(nnef_graph,  # type: NNEFGraph
        inputs,  # type: typing.Tuple[np.ndarray, ...]
        device=None,  # type: typing.Optional[str]
        custom_operations=None,  # type: typing.Optional[typing.Dict[str, typing.Callable]]
        tensor_hooks=None,  # type: typing.Optional[typing.List[typing.Callable[[NNEFTensor, torch.Tensor], None]]]
        ):
    # type: (...) -> typing.Tuple[np.ndarray, ...]

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Info: Using device {}'.format(device))

    assert len(inputs) == len(nnef_graph.inputs)
    torch_inputs = (to_torch_tensor(input, tensor.dtype).to(device)
                    for input, tensor in zip(inputs, nnef_graph.inputs))

    with torch.no_grad():  # Without this, gradients are calculated even in eval mode
        nnef_module = NNEFModule(nnef_graph=nnef_graph,
                                 custom_operations=custom_operations,
                                 tensor_hooks=tensor_hooks)
        nnef_module.to(device)
        nnef_module.eval()
        torch_outputs = nnef_module.forward(*torch_inputs)

    assert len(torch_outputs) == len(nnef_graph.outputs)
    return tuple(to_numpy_array(output, nnef_dtype=tensor.dtype)
                 for output, tensor in zip(torch_outputs, nnef_graph.outputs))


def try_to_fix_unsupported_attributes(nnef_graph):
    # type: (NNEFGraph)->None
    printed_warnings = set()

    def warn_once(message):
        if message not in printed_warnings:
            print(message, file=sys.stderr)
            sys.stderr.flush()
            printed_warnings.add(message)

    for op in nnef_graph.operations:
        if op.name in ("pad",) and op.attribs['border'] not in ("constant", "reflect", "replicate"):
            warn_once("{}: only constant, reflect and replicate border is supported, "
                      "given: {}. Using reflect border.".format(op.name, op.attribs['border']))
            op.attribs['border'] = 'reflect'
        elif op.name in ("deconv", "debox") and op.attribs['border'] not in ("constant",):
            warn_once("{}: Only constant border is supported, "
                      "given: '{}'. Using constant border.".format(op.name, op.attribs['border']))
            op.attribs['border'] = 'constant'
        elif op.name in ("multilinear_upsample",):
            method, border = op.attribs['method'], op.attribs['border']
            if (method, border) in (('symmetric', 'constant'),
                                    ('asymmetric', 'constant'),
                                    ('asymmetric', 'replicate')):
                if op.attribs['factor'] not in ([2], [2, 2]):
                    warn_once(
                        "{}: (symmetric, constant), (asymmetric, constant/replicate) "
                        "is only implemented for 3D, 4D tensors, and factor=2 respectively."
                        "Setting method, border to symmetric, replicate".format(op.name))
                    op.attribs['method'], op.attribs['border'] = 'symmetric', 'replicate'
            elif (method, border) != ('symmetric', 'replicate') and method != 'aligned':
                warn_once("Multilinear upsample is only implemented if (method, border) are "
                          "(symmetric, constant) "
                          "or (asymmetric, constant), "
                          "or (symmetric, replicate), "
                          "or (asymmetric, replicate), "
                          "or (aligned, [anything]), "
                          "given: ({}, {})."
                          "Setting method, border to symmetric, replicate".format(method, border))
                op.attribs['method'], op.attribs['border'] = 'symmetric', 'replicate'


class ActivationExportHook(object):
    def __init__(self, tensor_names, output_directory):
        self.output_directory = output_directory
        self.tensor_names_to_export = set(tensor_names)

    def __call__(self, nnef_tensor, torch_tensor):
        # type: (NNEFTensor, torch.Tensor)->None

        if nnef_tensor.name in self.tensor_names_to_export:
            nnef_io.write_nnef_tensor(filename=os.path.join(self.output_directory, nnef_tensor.name + ".dat"),
                                      array=to_numpy_array(torch_tensor, nnef_dtype=nnef_tensor.dtype))


class Statistics(object):
    def __init__(self, min, max, mean, std, name=""):
        if not all(math.isfinite(x) for x in (min, max, mean, std)):
            raise utils.NNEFToolsException(
                "{}: Statistics has infinite/NaN values: min={}, max={}, mean={}, std={}".format(
                    name, min, max, mean, std))
        self.min = min
        self.max = max
        self.mean = mean
        self.std = std


class StatisticsHook(object):
    def __init__(self):
        self._stat_tensor = {}

    def __call__(self, nnef_tensor, torch_tensor):
        # type: (NNEFTensor, torch.Tensor)->None
        if nnef_tensor.dtype == "scalar" and not (nnef_tensor.is_constant and nnef_tensor.rank == 0):
            self._stat_tensor[nnef_tensor.name] = self._calculate_stat_tensor(torch_tensor)

    def get_statistics(self):
        tensor_names = sorted(six.iterkeys(self._stat_tensor))
        stat_tensor_tuple = tuple(self._stat_tensor[name] for name in tensor_names)
        if not stat_tensor_tuple:  # seems like there were only bool or int tensors in the graph
            return {}
        stat_tensor = torch.stack(stat_tensor_tuple)
        stat_numpy = to_numpy_array(stat_tensor, nnef_dtype="scalar")
        stats = {}
        for i, name in enumerate(tensor_names):
            stats[name] = Statistics(min=float(stat_numpy[i][0]),
                                     max=float(stat_numpy[i][1]),
                                     mean=float(stat_numpy[i][2]),
                                     std=float(stat_numpy[i][3]),
                                     name=name)
        return stats

    def save_statistics(self, path):
        stats_dir = os.path.dirname(path)
        utils.makedirs(stats_dir, exist_ok=True)
        json_utils.dump(self.get_statistics(), path, add_class_name=False)

    @staticmethod
    def _calculate_stat_tensor(torch_tensor):
        # type: (torch.Tensor)->torch.Tensor
        count = utils.product(torch_tensor.shape)
        if count == 0:  # Avoid nans
            return torch.zeros([4], dtype=torch_tensor.dtype, device=torch_tensor.device)
        elif count == 1:  # Avoid nans
            elem = torch.min(torch_tensor)
            return torch.stack((elem, elem, elem, 0.0 * elem))
        else:
            return torch.stack((torch.min(torch_tensor),
                                torch.max(torch_tensor),
                                torch.mean(torch_tensor),
                                torch.std(torch_tensor, unbiased=True)))


__all__ = [
    "run",
    "try_to_fix_unsupported_attributes",
    "ActivationExportHook",
    "Statistics",
    "StatisticsHook",
    "NNEFModule",
    "to_torch_tensor",
    "to_numpy_array",
]
