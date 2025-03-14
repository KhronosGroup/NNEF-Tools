from ..model.utils import replace_chain
import six


class Optimizer:

    def __init__(self, custom_optimizers=None):
        self._custom_optimizers = custom_optimizers or {}

    def __call__(self, model, only_required=False):
        for graph in model.graphs:
            self._fix_batchnorm_spatial(graph)
            for chain, replacer in six.iteritems(self._custom_optimizers):
                replace_chain(graph, chain, replacer)

    @staticmethod
    def _fix_batchnorm_spatial(graph):
        for op in graph.operations:
            if op.type == 'BatchNormalization':
                spatial = op.attribs.get('spatial')
                if spatial == 0 and op.inputs[1].rank == 1:
                    del op.attribs['spatial']
