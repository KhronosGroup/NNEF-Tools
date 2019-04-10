from __future__ import division, print_function, absolute_import

import importlib

import nnef

from nnef_tools.core import utils


class NNEFParserConfig(object):

    def __init__(self, source=None, shapes=None, expand=None):
        if source is None:
            source = ""
        if shapes is None:
            shapes = {}
        if expand is None:
            expand = list()
        if not isinstance(expand, list):
            expand = list(expand)

        self._source = source
        self._shapes = shapes
        self._expand = expand

    def parse_string(self, graph_str, quant_str=None):
        return nnef.parse_string(graph_str=graph_str,
                                 quant_str=quant_str,
                                 stdlib=self._source,
                                 lowered=self._expand)

    def load_graph(self, path):
        return nnef.load_graph(path=path,
                               stdlib=self._source,
                               lowered=self._expand)

    def infer_shapes(self, graph):
        nnef.infer_shapes(graph=graph, custom_shapes=self._shapes)
        return graph

    @staticmethod
    def load_config(name):
        # type: (str)->NNEFParserConfig

        try:
            return importlib.import_module("nnef_tools.io.nnef.parser_config_" + name).CONFIG
        except ImportError:
            pass

        try:
            return importlib.import_module("nnef_parser_config_" + name).CONFIG
        except ImportError:
            pass

        return importlib.import_module(name).CONFIG

    @staticmethod
    def combine_configs(configs):
        assert all(isinstance(config, NNEFParserConfig) for config in configs)

        shapes = {}
        for config in configs:
            shapes.update(config._shapes)

        expand = []
        for config in configs:
            expand += config._expand

        expand = utils.unique(expand)

        return NNEFParserConfig(source='\n\n'.join(config._source for config in configs),
                                shapes=shapes,
                                expand=expand)
