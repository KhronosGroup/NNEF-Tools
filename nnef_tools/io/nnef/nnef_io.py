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

import os
import re
import shutil
import sys
import tempfile
import typing
from collections import OrderedDict

import nnef
import numpy as np
import six

from nnef_tools.core import utils
from nnef_tools.io.nnef import nnef_unifier
from nnef_tools.io.nnef.nnef_graph import *
from nnef_tools.io.nnef.parser_config import NNEFParserConfig

NNEFDTypeByNumpyDType = {
    'float16': 'scalar',
    'float32': 'scalar',
    'float64': 'scalar',
    'int8': 'integer',
    'uint8': 'integer',
    'int16': 'integer',
    'uint16': 'integer',
    'int32': 'integer',
    'uint32': 'integer',
    'int64': 'integer',
    'uint64': 'integer',
    'bool': 'logical',
}

NumpyDTypeByNNEFDType = {
    'scalar': 'float32',
    'integer': 'int32',
    'logical': 'bool',
}

_InputShapeType = typing.Union[None,
                               typing.List[int],
                               typing.Tuple[typing.List[int]],
                               typing.Dict[str, typing.List[int]]]


def read(path,  # type: str
         parser_configs=None,  # type: typing.Optional[typing.List[NNEFParserConfig]]
         input_shape=None,  # type: _InputShapeType
         ):
    # type: (...)->NNEFGraph

    if not (path.endswith('.tgz') or path.endswith('.nnef') or path.endswith('.txt') or os.path.isdir(path)):
        raise utils.NNEFToolsException("Only .tgz or .nnef or .txt files or directories are supported")

    parser_config = NNEFParserConfig.combine_configs(parser_configs if parser_configs else [])

    path_to_load = None
    compressed = False

    try:
        if os.path.isdir(path):
            compressed = False
            with_weights = True
            path_to_load = path
        elif path.endswith('.tgz'):
            compressed = True
            with_weights = True
            path_to_load = tempfile.mkdtemp(prefix="nnef_")
            utils.tgz_extract(path, path_to_load)
        elif path.endswith('.nnef') or path.endswith('.txt'):
            compressed = False
            with_weights = False
            path_to_load = path
        else:
            assert False

        # If there are fragments in the graph and also in parser_config
        # we remove the non-standard fragments from parser_config to avoid duplicate fragment definition
        if parser_config.fragments:
            re_graph = re.compile(r"^graph\s|\sgraph\s")
            re_fragment = re.compile(r"^fragment\s|\sfragment\s")
            graph_nnef_path = os.path.join(path_to_load, 'graph.nnef') if os.path.isdir(path_to_load) else path_to_load
            with open(graph_nnef_path, 'r') as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if re_fragment.search(line):
                        parser_config.fragments = NNEFParserConfig.STANDARD_CONFIG.fragments
                        break
                    if re_graph.search(line):
                        break

        parser_graph = parser_config.load_graph(path_to_load)

        if input_shape is not None:
            if not isinstance(input_shape, (tuple, list, dict)):
                raise utils.NNEFToolsException("input_shape must be list or dict")

            for op in parser_graph.operations:
                if op.name == 'external':
                    name = op.outputs['output']
                    if isinstance(input_shape, dict):
                        if name in input_shape:
                            op.attribs['shape'] = input_shape[name]
                    elif isinstance(input_shape, tuple):
                        op.attribs['shape'] = input_shape[parser_graph.inputs.index(name)]
                    else:
                        op.attribs['shape'] = input_shape

        parser_config.infer_shapes(parser_graph)
        return _read(parser_graph=parser_graph, with_weights=with_weights)

    finally:
        if compressed and path_to_load:
            shutil.rmtree(path_to_load)


def write(nnef_graph,  # type: NNEFGraph
          tgz_or_dir_path,  # type: str
          write_weights=True,  # type: bool
          raise_on_missing_weight=True,  # type: bool
          extensions=None,  # type: typing.Optional[typing.List[str]]
          fragments=None,  # type: typing.Optional[str]
          only_print_used_fragments=False,  # type: bool
          compression_level=0,  # type: int
          ):
    # type: (...) -> None

    compressed = tgz_or_dir_path.endswith('.tgz')
    dir_path = None

    try:
        if compressed:
            dir_path = tempfile.mkdtemp(prefix="nnef_")
        else:
            dir_path = tgz_or_dir_path
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        with open(os.path.join(dir_path, "graph.nnef"), "w") as f:
            _print(nnef_graph,
                   file_handle=f,
                   extensions=extensions,
                   fragments=fragments,
                   only_print_used_fragments=only_print_used_fragments)

        if any(t.quantization is not None for t in nnef_graph.tensors):
            with open(os.path.join(dir_path, "graph.quant"), "w") as f:
                _print_quantization(nnef_graph, file_handle=f)

        if write_weights:
            _write_weights(nnef_graph, dir_path=dir_path, raise_on_missing_weight=raise_on_missing_weight)

        if compressed:
            utils.tgz_compress(dir_path, tgz_or_dir_path, compression_level=compression_level)
    finally:
        if compressed and dir_path:
            shutil.rmtree(dir_path)


def _debug_print(nnef_graph, file_handle=None):
    # type: (NNEFGraph, typing.Optional[typing.TextIO]) -> None

    if file_handle is None:
        file_handle = sys.stderr

    _print(nnef_graph, file_handle=file_handle)


def _read(parser_graph, with_weights=True):
    # type: (typing.Any, bool)->NNEFGraph

    tensor_by_name = {}
    g = NNEFGraph(name=parser_graph.name)

    def add_to_tensor_by_name(tensor):
        assert tensor.name not in tensor_by_name, "Tensor {} defined multiple times".format(tensor.name)
        tensor_by_name[tensor.name] = tensor

    def transform_input(input_):
        if isinstance(input_, nnef.Identifier):
            assert str(input_) in tensor_by_name, "Tensor {} not defined before use".format(str(input_))
            return tensor_by_name[str(input_)]
        else:
            return NNEFTensor(graph=g,
                              name=None,
                              shape=[],
                              dtype=NNEFDTypeByNumpyDType[np.array(input_).dtype.name],
                              data=[input_])

    def transform_result(result_):
        if isinstance(result_, nnef.Identifier):
            quantization = parser_graph.tensors[str(result_)].quantization
            if quantization:
                quantization = NNEFQuantization(name=quantization['op-name'], attribs=quantization)
                del quantization.attribs['op-name']
            else:
                quantization = None

            tensor = NNEFTensor(graph=g,
                                name=str(result_),
                                shape=list(parser_graph.tensors[str(result_)].shape),
                                dtype=parser_graph.tensors[str(result_)].dtype,
                                quantization=quantization)

            add_to_tensor_by_name(tensor)
            return tensor
        else:
            return result_

    for parser_op in parser_graph.operations:

        inputs = utils.recursive_transform(parser_op.inputs, transform_input)
        if any(isinstance(i, list) for i in six.itervalues(inputs)):
            inputs = utils.recursive_collect(inputs)
        else:
            inputs = tuple(utils.recursive_collect(inputs))

        outputs = utils.recursive_transform(parser_op.outputs, transform_result)
        if any(isinstance(o, list) for o in six.itervalues(outputs)):
            outputs = utils.recursive_collect(outputs)
        else:
            outputs = tuple(utils.recursive_collect(outputs))

        if parser_op.name == "variable":
            outputs[0].label = parser_op.attribs["label"]
            if with_weights:
                outputs[0].data = parser_graph.tensors[parser_op.outputs["output"]].data
                assert outputs[0].data is not None
            else:
                outputs[0].data = np.array([], dtype=NumpyDTypeByNNEFDType[parser_op.dtype])
        if parser_op.name == "constant":
            outputs[0].data = parser_op.attribs["value"]

        if parser_op.name not in ["external", "constant", "variable"]:
            NNEFOperation(graph=g, name=parser_op.name, attribs=dict(parser_op.attribs), inputs=inputs, outputs=outputs,
                          dtype=parser_op.dtype)

    input_tensors = []

    for input_ in parser_graph.inputs:
        assert str(input_) in tensor_by_name, "Input tensor {} was not declared".format(str(input_))
        input_tensors.append(tensor_by_name[str(input_)])

    output_tensors = []

    for output_ in parser_graph.outputs:
        assert str(output_) in tensor_by_name, "Output tensor {} was not declared".format(str(output_))
        output_tensors.append(tensor_by_name[str(output_)])

    g.inputs = OrderedDict((t.name, t) for t in input_tensors)
    g.outputs = OrderedDict((t.name, t) for t in output_tensors)

    g.generate_missing_names()
    return g


def get_used_fragments(nnef_graph, fragments):
    # type: (NNEFGraph, str)->str
    ops = {op.name for op in nnef_graph.operations}
    ops.update(tensor.quantization.name for tensor in nnef_graph.tensors if tensor.quantization)

    fragment_list = [f.strip() for f in re.split(r"^fragment\s|\sfragment\s", fragments) if f.strip()]
    used_fragment_list = []
    for fragment in fragment_list:
        fragment_name = fragment.split('(')[0].split('<')[0].strip()
        if fragment_name in ops:
            used_fragment_list.append(fragment)
    if used_fragment_list:
        return 'fragment ' + '\nfragment '.join(used_fragment_list)
    return ""


def _print(nnef_graph,  # type: NNEFGraph
           file_handle,  # type: typing.TextIO
           extensions=None,  # type: typing.Optional[typing.List[str]]
           fragments=None,  # type: typing.Optional[str]
           only_print_used_fragments=False,  # type: bool
           ):
    # type: (...)->None

    generate_source_operations(nnef_graph)
    nnef_graph.sort()
    try:
        if extensions is None:
            extensions = []

        if fragments is None:
            fragments = ""

        fragments = add_tflite_quantization_fragment_if_needed(nnef_graph, fragments)

        if only_print_used_fragments:
            fragments = get_used_fragments(nnef_graph, fragments)

        if fragments:
            if "KHR_enable_fragment_definitions" not in extensions:
                extensions.append("KHR_enable_fragment_definitions")
            if "KHR_enable_operator_expressions" not in extensions:
                extensions.append("KHR_enable_operator_expressions")

        f = file_handle
        indent = 4 * " "

        print(nnef.format_version((1, 0)), file=f)
        if extensions:
            print(nnef.format_extensions(extensions), file=f)
        if fragments:
            print(file=f)
            print(fragments, file=f)
        print(file=f)

        graph_name = _recursive_check_str(nnef_graph.name) if nnef_graph.name is not None else "network"
        graph_inputs = _recursive_check_str([input_.name for input_ in nnef_graph.inputs])
        graph_outputs = _recursive_check_str([output_.name for output_ in nnef_graph.outputs])

        print("graph {}({}) -> ({})".format(graph_name, ', '.join(graph_inputs), ', '.join(graph_outputs)), file=f)
        print("{", file=f)

        for op in nnef_graph.operations:
            inputs = _transform_inputs_before_print(list(op.inputs) if not isinstance(op.inputs, tuple) else op.inputs)
            invocation = nnef.format_invocation(
                name=_recursive_check_str(op.name),
                attribs=_recursive_check_str(_sorted_ordered_dict(op.attribs)),
                inputs=_recursive_check_str([inputs] if isinstance(inputs, list) else list(inputs)),
                outputs=_recursive_check_str(
                    _result_to_identifiers(list(op.outputs) if not isinstance(op.outputs, tuple) else op.outputs)),
                dtype=_recursive_check_str(op.dtype))

            comment = "  # {}".format(_recursive_check_str(op.comment)) if op.comment else ""
            print("{}{};{}".format(indent, invocation, comment), file=f)

        print("}", file=f)
    finally:
        remove_source_operations(nnef_graph)


def _print_quantization(nnef_graph, file_handle):
    # type: (NNEFGraph, typing.TextIO)->None
    for tensor in nnef_graph.tensors:
        if tensor.quantization is None:
            print('# "{}": not quantized'.format(tensor.name), file=file_handle)
        else:
            print('"{}": {}({});'.format(tensor.name,
                                         tensor.quantization.name,
                                         ', '.join("{} = {}".format(k, v)
                                                   for k, v in sorted(six.iteritems(tensor.quantization.attribs)))),
                  file=file_handle)


def _write_weights(nnef_graph, dir_path, raise_on_missing_weight=True):
    # type: (NNEFGraph, str, bool) -> None
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for tensor in nnef_graph.tensors:
        if tensor.is_variable:
            if tensor.data.size == tensor.count:
                write_nnef_tensor(filename=os.path.join(dir_path, tensor.label + ".dat"),
                                  array=np.asarray(tensor.data, order='C'))
            elif tensor.data.size == 0:
                if raise_on_missing_weight:
                    utils.NNEFToolsException("Missing value for variable: {}".format(tensor.name))
            else:
                utils.NNEFToolsException(
                    "Invalid data size for variable: {}, expected: {}, got: {}".format(tensor.name,
                                                                                       tensor.count,
                                                                                       tensor.data.size))


def read_nnef_tensor(filename):
    with open(filename, "rb") as file:
        return nnef.read_tensor(file)[0]


def write_nnef_tensor(filename, array):
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, "wb") as file:
        nnef.write_tensor(file=file, tensor=array)


def _transform_inputs_before_print(inputs):
    def transform(input_):
        if isinstance(input_, NNEFTensor):
            if input_.is_constant and input_.rank == 0:
                return input_.data[0]
            else:
                return nnef.Identifier(input_.name)
        return input_

    return utils.recursive_transform(inputs, transform)


def _sorted_ordered_dict(d):
    return OrderedDict(sorted((k, v) for k, v in six.iteritems(d)))


def _result_to_identifiers(result):
    def transform(result_):
        assert isinstance(result_, NNEFTensor), "Results must be NNEF tensors, or lists/tuples of that."
        return nnef.Identifier(result_.name)

    result = utils.recursive_transform(result, transform)
    if isinstance(result, nnef.Identifier) or isinstance(result, list):
        return [result]
    elif isinstance(result, tuple):
        return list(result)
    else:
        assert False, "Unexpected result type: {}".format(type(result))


def _recursive_check_str(data):
    if sys.version_info[0] < 3:
        def check(arg):
            # noinspection PyUnresolvedReferences
            assert not isinstance(arg, unicode), \
                "NNEF module does not accept unicode strings in python2. Use NNEFGraph with str only."

        utils.recursive_visit(data, check)
    return data


def generate_source_operations(nnef_graph, gen_for_rank0_also=False):
    # type: (NNEFGraph, bool) -> None
    # assert nnef_graph.is_unique

    for t in list(nnef_graph.tensors):
        if t.producer is None:
            if t.is_constant:
                if t.rank > 0 or t in nnef_graph.outputs or gen_for_rank0_also:
                    NNEFOperation(graph=nnef_graph,
                                  name="constant",
                                  attribs=dict(shape=t.shape, value=t.data),
                                  inputs=tuple(),
                                  outputs=t,
                                  dtype=t.dtype)
            elif t.is_variable:
                NNEFOperation(graph=nnef_graph,
                              name="variable",
                              attribs=dict(shape=t.shape, label=t.label),
                              inputs=tuple(),
                              outputs=t,
                              dtype=t.dtype)
            elif t.producer is None:
                NNEFOperation(graph=nnef_graph,
                              name="external",
                              attribs=dict(shape=t.shape),
                              inputs=tuple(),
                              outputs=t,
                              dtype=t.dtype)
            else:
                assert False, "All non-source tensors must have a producer in an NNEF graph"


def remove_source_operations(nnef_graph):
    # type: (NNEFGraph) -> None
    # assert nnef_graph.is_unique

    nnef_graph.remove_operations([op for op in list(nnef_graph.operations)
                                  if op.name in {"constant", "variable", "external"}],
                                 unlink=True)


TFLITE_QUANTIZATION_FRAGMENT = """\
fragment tflite_quantize(x: tensor<scalar>, min: scalar, max: scalar, scale: scalar, zero_point: integer, bits: integer)
-> ( y: tensor<scalar> )
{
    rounded = round(x / scale + scalar(zero_point));
    q = clamp(rounded, 0.0, 255.0) if bits == 8 else clamp(rounded, -2147483648.0, 2147483647.0);
    y = (q - scalar(zero_point)) * scale;
}\
"""


def add_tflite_quantization_fragment_if_needed(nnef_graph, fragments):
    # type:(NNEFGraph, str)->str
    if (any(tensor.quantization is not None and tensor.quantization.name == "tflite_quantize"
            and any(item != 0 for item in six.itervalues(tensor.quantization.attribs))
            for tensor in nnef_graph.tensors)):
        if fragments:
            return fragments + '\n' + TFLITE_QUANTIZATION_FRAGMENT
        else:
            return TFLITE_QUANTIZATION_FRAGMENT
    return fragments


class Reader(object):

    def __init__(self, parser_configs=None, unify=False, input_shape=None):
        self._parser_configs = parser_configs
        self._unify = unify
        self._input_shape = input_shape

    def __call__(self, filename):
        g = read(filename, parser_configs=self._parser_configs, input_shape=self._input_shape)
        if self._unify:
            nnef_unifier.unify(g)
        return g


class Writer(object):

    def __init__(self,
                 write_weights=True,
                 extensions=None,
                 fragments=None,
                 only_print_used_fragments=False,
                 compression_level=0):
        self._write_weights = write_weights
        self._extensions = extensions
        self._fragments = fragments
        self._only_print_used_fragments = only_print_used_fragments
        self._compression_level = compression_level

    def __call__(self, graph, filename):
        write(graph, filename,
              write_weights=self._write_weights,
              extensions=self._extensions,
              fragments=self._fragments,
              only_print_used_fragments=self._only_print_used_fragments,
              compression_level=self._compression_level)
        return None
