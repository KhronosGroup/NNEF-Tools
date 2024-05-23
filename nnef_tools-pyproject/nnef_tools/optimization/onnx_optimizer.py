# Copyright (c) 2020 The Khronos Group Inc.
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

from ..model.utils import replace_chain
import six


class Optimizer:

    def __init__(self, custom_optimizers=None):
        self._custom_optimizers = custom_optimizers or {}

    def __call__(self, graph, only_required=False):
        self._fix_batchnorm_spatial(graph)
        for chain, replacer in six.iteritems(self._custom_optimizers):
            replace_chain(graph, chain, replacer)
        return graph

    @staticmethod
    def _fix_batchnorm_spatial(graph):
        for op in graph.operations:
            if op.type == 'BatchNormalization':
                spatial = op.attribs.get('spatial')
                if spatial == 0 and op.inputs[1].rank == 1:
                    del op.attribs['spatial']
