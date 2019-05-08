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

import importlib
import typing

import nnef

from nnef_tools.core import utils


class NNEFParserConfig(object):
    STANDARD_CONFIG = None  # It is loaded after the class definition

    def __init__(self, fragments=None, shapes=None, lowered=None):
        if fragments is None:
            fragments = ""
        if shapes is None:
            shapes = {}
        if lowered is None:
            lowered = list()
        if not isinstance(lowered, list):
            lowered = list(lowered)

        self.fragments = fragments
        self.shapes = shapes
        self.lowered = lowered

    def parse_string(self, graph_str, quant_str=None):
        return nnef.parse_string(graph_str=graph_str,
                                 quant_str=quant_str,
                                 stdlib=self.fragments,
                                 lowered=self.lowered)

    def load_graph(self, path):
        return nnef.load_graph(path=path,
                               stdlib=self.fragments,
                               lowered=self.lowered)

    def infer_shapes(self, graph):
        nnef.infer_shapes(graph=graph, custom_shapes=self.shapes)
        return graph

    @property
    def empty(self):
        return not self.fragments and not self.shapes and not self.lowered

    @staticmethod
    def load_config(module_name):
        # type: (str)->NNEFParserConfig

        """

        :param module_name: "package.module"
        :return: NNEFParserConfig
        """

        module = importlib.import_module(module_name)

        custom_fragments = ""
        if hasattr(module, "NNEF_FRAGMENTS"):
            custom_fragments = module.NNEF_FRAGMENTS
        custom_lowered = []
        if hasattr(module, "NNEF_LOWERED"):
            custom_lowered = module.NNEF_LOWERED
        custom_shapes = {}
        if hasattr(module, "NNEF_SHAPES"):
            custom_shapes = module.NNEF_SHAPES

        return NNEFParserConfig(fragments=custom_fragments, shapes=custom_shapes, lowered=custom_lowered)

    @staticmethod
    def load_configs(module_names="", load_standard=True):
        # type: (typing.Union[str, typing.List[str], None], bool)->typing.List[NNEFParserConfig]

        """

        :param module_names: "package.module" or "p1.m1,p2.m2", or ["p1.m1", "p2.m2"]
        :param load_standard: Load the standard NNEF op definitions as well
        :return: parser configs
        """

        if module_names is None:
            module_names = []
        if utils.is_anystr(module_names):
            module_names = [name.strip() for name in module_names.split(',')] if module_names.strip() else []

        configs = [NNEFParserConfig.STANDARD_CONFIG] if load_standard else []
        for module_name in module_names:
            config = NNEFParserConfig.load_config(module_name)

            if not config.empty:
                configs.append(config)
        return configs

    @staticmethod
    def combine_configs(configs):
        assert all(isinstance(config, NNEFParserConfig) for config in configs)

        shapes = {}
        lowered = []
        for config in configs:
            shapes.update(config.shapes)
            lowered += config.lowered

        return NNEFParserConfig(fragments='\n\n'.join(config.fragments for config in configs),
                                shapes=shapes,
                                lowered=utils.unique(lowered))


NNEFParserConfig.STANDARD_CONFIG = NNEFParserConfig.load_config("nnef_tools.io.nnef.parser_config_std")
