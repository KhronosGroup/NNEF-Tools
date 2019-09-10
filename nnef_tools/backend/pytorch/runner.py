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
import typing

import numpy as np
import torch
import six

from nnef_tools.backend.pytorch.nnef_module import NNEFModule
from nnef_tools.core import utils
from nnef_tools.io.nnef import nnef_io
from nnef_tools.io.nnef.nnef_graph import *
from nnef_tools.core import json_utils


def run(nnef_graph,  # type: NNEFGraph
        inputs,  # type: typing.Tuple[np.ndarray, ...]
        device=None,  # type: typing.Optional[str]
        custom_operations=None,  # type: typing.Optional[typing.Dict[str, typing.Callable]]
        fix_batch_size=False,  # type: bool # TODO
        permissive=False,  # type: bool # TODO
        tensor_hooks=None,  # type: typing.Optional[typing.List[typing.Callable[[NNEFTensor, torch.Tensor], None]]]
        ):
    # type: (...) -> typing.Tuple[np.ndarray, ...]

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Info: Using device {}'.format(device))

    assert len(inputs) == len(nnef_graph.inputs)
    torch_inputs = (NNEFModule.to_torch_tensor(input, tensor.dtype).to(device) for input, tensor in
                    zip(inputs, nnef_graph.inputs))

    torch_outputs = NNEFModule(nnef_graph=nnef_graph,
                               custom_operations=custom_operations,
                               fix_batch_size=fix_batch_size,
                               permissive=permissive,
                               tensor_hooks=tensor_hooks).to(device).eval().forward(*torch_inputs)

    assert len(torch_outputs) == len(nnef_graph.outputs)
    return tuple(NNEFModule.to_numpy_array(output, nnef_dtype=tensor.dtype)
                 for output, tensor in zip(torch_outputs, nnef_graph.outputs))


class ActivationExportHook(object):
    def __init__(self, tensor_names, output_directory):
        self.output_directory = output_directory
        self.tensor_names_to_export = set(tensor_names)

    def __call__(self, nnef_tensor, torch_tensor):
        # type: (NNEFTensor, torch.Tensor)->None

        if nnef_tensor.name in self.tensor_names_to_export:
            nnef_io.write_nnef_tensor(filename=os.path.join(self.output_directory, nnef_tensor.name + ".dat"),
                                      array=NNEFModule.to_numpy_array(torch_tensor, nnef_dtype=nnef_tensor.dtype))


class Statistics(object):
    def __init__(self, min, max, mean, std, name=""):
        if not all(math.isfinite(x) for x in (min, max, mean, std)):
            raise utils.NNEFToolsException(
                "{}: Statistics has infinite/NaN values: min={}, max={}, mean={}, std={}".format(name, min, max, mean, std))
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
        stat_numpy = NNEFModule.to_numpy_array(stat_tensor, nnef_dtype="scalar")
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


class TopKHook(object): # TODO remove
    def __init__(self, tensor_names, k=5):
        self.k = k
        self.tensor_names = tensor_names

    def __call__(self, nnef_tensor, torch_tensor):
        # type: (NNEFTensor, torch.Tensor)->None

        if nnef_tensor.name in self.tensor_names:
            if len(nnef_tensor.shape) >= 2:
                topk_result = torch_tensor.topk(dim=1, k=self.k)
                topk_result = (NNEFModule.to_numpy_array(topk_result[0], nnef_tensor.dtype),
                               NNEFModule.to_numpy_array(topk_result[1], nnef_tensor.dtype))
                if len(topk_result[0].shape) == 3 and topk_result[0].shape[1] == topk_result[0].shape[2] == 1:
                    topk_result = (np.squeeze(topk_result[0], axis=(1, 2)), np.squeeze(topk_result[1], axis=(1, 2)))
                po = np.get_printoptions()
                np.set_printoptions(precision=3, suppress=True, floatmode='fixed')
                print("TopK({}, k={}, axis=1):\n{}\n{}".format(
                    nnef_tensor.name, self.k, topk_result[0], topk_result[1]))
                np.set_printoptions(**po)
            else:
                print("Tensor({}):\n{}".format(nnef_tensor.name, NNEFModule.to_numpy_array(torch_tensor,
                                                                                           nnef_tensor.dtype)))
