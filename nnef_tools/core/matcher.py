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

import itertools
import typing

import six

from nnef_tools.core import graph_utils
from nnef_tools.core import utils
from nnef_tools.core.base_graph import BaseGraph, BaseTensor, BaseOperation

_TensorOrListOrTupleOrSetOrDict = typing.Union['Tensor',
                                               typing.List['Tensor'],
                                               typing.Tuple['Tensor', ...],
                                               typing.Set['Tensor'],
                                               typing.Dict[int, typing.Optional["Tensor"]]]


class Pattern(object):

    def _match(self, value, settings):
        # type: (typing.Any, _MatchSettings)->Match
        return Match()


class _Ignore(Pattern):
    def _match(self, value, settings):
        # type: (typing.Any, _MatchSettings)->Match
        return Match(did_match=True, root=value, dict_={})


class Tensor(Pattern):

    def __init__(self):
        self._producer_pattern = None  # type: typing.Optional[Operation]

    def _match(self, value, settings):
        assert isinstance(value, BaseTensor)

        if settings.dict_so_far.get(self, value) != value:
            return Match

        if self._producer_pattern and settings.follow_producer:
            if value.producer is None:
                return Match()

            match = self._producer_pattern._match(
                value.producer,
                settings.copy(dict_so_far=utils.dict_union(settings.dict_so_far, {self: value})))

            if not match:
                return Match()

            return Match(did_match=True, root=value, dict_=utils.dict_union(match.dict, {self: value}))
        else:
            return Match(did_match=True, root=value, dict_={self: value})


def tensors(n):
    assert n >= 1
    return tuple(Tensor() for _ in range(n)) if n > 1 else Tensor()


class _Const(Pattern):
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value

    def _match(self, value, settings):
        assert self not in settings.dict_so_far, "Const cannot be matched multiple times"
        return Match(did_match=True, root=value, dict_={self: value}) if value == self.value else Match()


class Operation(Pattern):
    def __init__(self,
                 name=None,  # type: typing.Union[None, str, typing.List[str]]
                 inputs=None,  # type: typing.Optional[_TensorOrListOrTupleOrSetOrDict]
                 outputs=None,  # type: typing.Optional[_TensorOrListOrTupleOrSetOrDict]
                 attribs=None  # type: typing.Optional[typing.Dict[str, typing.Any]]
                 ):
        # type: (...) -> None
        if isinstance(inputs, Tensor):
            inputs = (inputs,)
        if isinstance(outputs, Tensor):
            outputs = (outputs,)
        assert name is None or isinstance(name, (str, list, tuple))
        assert inputs is None or isinstance(inputs, (list, tuple, set, dict))
        assert outputs is None or isinstance(outputs, (list, tuple, set, dict))
        assert attribs is None or isinstance(attribs, dict)

        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.attribs = attribs

        if self.outputs:
            def visit(tensor):
                assert tensor._producer_pattern is None
                tensor._producer_pattern = self

            utils.recursive_visit(outputs, visit)

    def _match(self, value, settings):
        assert self not in settings.dict_so_far, "Operation cannot be matched multiple times"

        assert isinstance(value, BaseOperation)
        op = value

        if not settings.allow_multi_consumer and any(len(r.consumers) > 1 for r in op.outputs):
            return Match()

        if self.name is not None and op.name not in utils.listify(self.name):
            return Match()

        match_ = Match(True, root=op, dict_={self: op})

        if self.inputs is not None:
            match2 = Match()
            for input_patterns in self._pattern_list_list(self.inputs, op.inputs):
                match2 = self._match_inputs(op, settings, input_patterns)
                if match2:
                    break
            if not match2:
                return Match()
            match_ = Match(True, root=op, dict_=utils.dict_union(match_.dict, match2.dict))

        if self.attribs is not None:
            assert isinstance(self.attribs, dict)
            match2 = self._match_attribs(op, settings, self.attribs)
            if not match2:
                return Match()
            match_ = Match(True, root=op, dict_=utils.dict_union(match_.dict, match2.dict))

        if self.outputs is not None:
            match2 = Match()
            for output_patterns in self._pattern_list_list(self.outputs, op.outputs):
                match2 = self._match_outputs(op, settings, output_patterns)
                if match2:
                    break
            if not match2:
                return Match()
            match_ = Match(True, root=op, dict_=utils.dict_union(match_.dict, match2.dict))

        return match_

    @staticmethod
    def _pattern_list_list(patterns, values):
        if isinstance(patterns, dict):
            max_index = max(max(six.iterkeys(patterns)), len(values))
            pattern_list = []
            for i in range(0, max_index):
                if i in patterns:
                    ith_pattern = patterns[i]
                    if ith_pattern is None:
                        break
                    pattern_list.append(ith_pattern)
                else:
                    pattern_list.append(_Ignore())
            return [pattern_list]
        elif isinstance(patterns, (list, tuple)):
            return [list(patterns)]
        elif isinstance(patterns, set):
            return list(itertools.permutations(list(patterns)))
        else:
            assert False

    def _match_inputs(self, op, settings, input_patterns):
        if len(op.inputs) != len(input_patterns):
            return Match()

        dict_ = {self: op}
        for input, input_pattern in zip(op.inputs, input_patterns):
            # noinspection PyProtectedMember
            match_ = input_pattern._match(input,
                                          settings.copy(allow_multi_consumer=settings.allow_multi_consumer_inside,
                                                        dict_so_far=utils.dict_union(settings.dict_so_far, dict_)))
            if not match_:
                return Match()
            dict_.update(match_.dict)

        return Match(did_match=True, root=op, dict_=dict_)

    def _match_outputs(self, op, settings, output_patterns):
        if len(op.outputs) != len(output_patterns):
            return Match()

        dict_ = {self: op}
        for output, output_pattern in zip(op.outputs, output_patterns):
            # noinspection PyProtectedMember
            match_ = output_pattern._match(output,
                                           settings.copy(allow_multi_consumer=settings.allow_multi_consumer_inside,
                                                         dict_so_far=utils.dict_union(settings.dict_so_far, dict_),
                                                         follow_producer=False))
            if not match_:
                return Match()
            dict_.update(match_.dict)

        return Match(did_match=True, root=op, dict_=dict_)

    def _match_attribs(self, op, settings, attrib_patterns):
        def trafo(arg):
            return arg if isinstance(arg, Pattern) else _Const(arg)

        attrib_patterns = utils.recursive_transform(attrib_patterns, trafo)  # type: typing.Dict[str, Pattern]

        dict_ = {self: op}
        for attrib_name, attrib_pattern in six.iteritems(attrib_patterns):
            attrib_value = op.attribs[attrib_name]
            match_ = attrib_pattern._match(attrib_value,
                                           settings.copy(allow_multi_consumer=settings.allow_multi_consumer_inside,
                                                         dict_so_far=utils.dict_union(settings.dict_so_far, dict_)))
            if not match_:
                return Match()
            dict_.update(match_.dict)

        return Match(did_match=True, root=op, dict_=dict_)


class OrPattern(Pattern):
    def __init__(self, *patterns):
        # type: (typing.Tuple[Operation, ...]) -> None
        self._patterns = patterns

    @property
    def patterns(self):
        return self._patterns

    def _match(self, value, settings):
        assert self not in settings.dict_so_far, "OrPattern cannot be matched multiple times"

        match_ = Match()

        for pattern in self.patterns:
            match_ = pattern._match(value, settings)
            if match_:
                break

        if not match_:
            return match_

        return Match(did_match=True, root=match_.root, dict_=utils.dict_union(match_.dict, {self: match_.root}))


class SetParams(Pattern):
    def __init__(self, pattern, allow_multi_consumer_inside=None):
        self._pattern = pattern  # type: Pattern
        self._allow_multi_consumer_inside = allow_multi_consumer_inside

    @property
    def allow_multi_consumer_inside(self):
        return self._allow_multi_consumer_inside

    @property
    def pattern(self):
        return self._pattern

    def _match(self, value, settings):
        new_settings = settings.copy(allow_multi_consumer_inside=self._allow_multi_consumer_inside)
        match_ = self._pattern._match(value, new_settings)
        if not match_:
            return match_
        return Match(did_match=True, root=match_.root, dict_=utils.dict_union(match_.dict, {self: match_.root}))


class _MatchSettings(object):
    def __init__(self,
                 dict_so_far,
                 allow_multi_consumer,
                 allow_multi_consumer_inside,
                 follow_producer):
        self.dict_so_far = dict_so_far
        self.allow_multi_consumer = allow_multi_consumer
        self.allow_multi_consumer_inside = allow_multi_consumer_inside
        self.follow_producer = follow_producer

    def copy(self,
             dict_so_far=None,
             allow_multi_consumer=None,
             allow_multi_consumer_inside=None,
             follow_producer=None):
        return _MatchSettings(dict_so_far=utils.first_set(dict_so_far, self.dict_so_far),
                              allow_multi_consumer=utils.first_set(allow_multi_consumer, self.allow_multi_consumer),
                              allow_multi_consumer_inside=utils.first_set(allow_multi_consumer_inside,
                                                                          self.allow_multi_consumer_inside),
                              follow_producer=utils.first_set(follow_producer, self.follow_producer))


class Match(object):
    def __init__(self, did_match=False, root=None, dict_=None):
        if dict_ is None:
            dict_ = {}
        self._root = root
        self._did_match = did_match
        self._dict = dict_

    @property
    def operations(self):
        assert self._did_match
        return list(set(v for v in six.itervalues(self._dict) if isinstance(v, BaseOperation)))

    def __nonzero__(self):  # for python 2
        return self._did_match

    def __bool__(self):  # for python 3
        return self._did_match

    def __getitem__(self, item):
        # type: (Pattern) -> typing.Any
        assert self._did_match
        return self._dict[item]

    def __contains__(self, item):
        return self._did_match and item in self._dict

    @property
    def root(self):
        assert self._did_match
        return self._root

    @property
    def dict(self):
        assert self._did_match
        return self._dict


def match(g, op, pattern):
    # type: (BaseGraph, BaseOperation, Pattern) -> Match
    # assert g.is_unique
    assert op.graph

    return pattern._match(op, _MatchSettings(
        dict_so_far={},
        allow_multi_consumer=True,
        allow_multi_consumer_inside=False,
        follow_producer=True
    ))


def for_each(graph,  # type: BaseGraph
             pattern,  # type: Pattern
             action,  # type: typing.Callable[[Match], typing.Any] # result is not used
             condition=None  # type: typing.Optional[typing.Callable[[Match], bool]]
             ):
    # type: (...)->None
    for op in list(graph.operations):
        if op.graph is not None:  # op can be removed if the graph is not topological-sorted
            m = match(graph, op, pattern)
            if m and (condition is None or condition(m)):
                action(m)


def replace(graph,  # type: BaseGraph
            pattern,  # type: Pattern
            replacement,  # type: typing.Callable[[Match], typing.Any] # result is not used
            condition=None  # type: typing.Optional[typing.Callable[[Match], bool]]
            ):
    # type: (...)->None
    for op in list(graph.operations):
        if op.graph is not None:  # op can be removed if the graph is not topological-sorted
            m = match(graph, op, pattern)
            if m and (condition is None or condition(m)):
                replacement(m)
                graph_utils.remove_subgraph(graph, m.operations)
