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

from __future__ import division, print_function

from collections import OrderedDict

import nnef

from . import dog
from . import utils
from .types import *
from .utils import StringIO
from .nnef_dog_types import NnefGraph


def nnefdog_to_source(nnefdog, file_handle=None, custom_fragments="", enable_shape_of=False):
    if file_handle is None:
        string_io = StringIO()
        try:
            # noinspection PyUnresolvedReferences, PyTypeChecker
            return _nnefdog_to_source_impl(nnefdog, string_io,
                                           custom_fragments=custom_fragments,
                                           enable_shape_of=enable_shape_of).getvalue()
        finally:
            string_io.close()
    else:
        return _nnefdog_to_source_impl(nnefdog, file_handle,
                                       custom_fragments=custom_fragments,
                                       enable_shape_of=enable_shape_of)


def _nnefdog_to_source_impl(nnefdog, file_handle, custom_fragments="", enable_shape_of=False):
    # type: (NnefGraph, TextIO, str, bool)->TextIO

    f = file_handle
    fix_str = utils.ensure_not_unicode_in_python2
    indent = 4 * " "

    print(nnef.format_version((1, 0)), file=f)

    extensions = []
    if enable_shape_of:
        extensions.append("KHR_enable_operator_expressions")
    print(nnef.format_extensions(extensions), file=f)

    print(file=f)

    if custom_fragments:
        print(custom_fragments, file=f)
        print(file=f)

    graph_params = nnef.format_graph(name=fix_str(nnefdog.name),
                                     inputs=[fix_str(name) for name in nnefdog.input_dn_names],
                                     outputs=[fix_str(name) for name in nnefdog.output_dn_names])

    print("graph {}".format(graph_params), file=f)
    print("{", file=f)

    for op in nnefdog.ops:
        dtype = op.result.dtype if op.name in ["external", "constant", "variable"] else None
        invocation = nnef.format_invocation(name=fix_str(op.name),
                                            args=[],
                                            kwargs=_preprocess_args(op.args),
                                            results=_results_to_result_names(op.results.values()),
                                            dtype=dtype)

        comments = utils.without_nones([dn.extra.get(dog.EXTRA_COMMENT) for dn in op.get_result_nodes()])
        comment = "  # {}".format(", ".join(comments)) if comments else ""
        print("{}{};{}".format(indent, invocation, comment), file=f)

    print("}", file=f)

    return file_handle


def _preprocess_args(args):
    # type: (OrderedDict[str, Any])->OrderedDict[str, Any]
    def transform(arg):
        if isinstance(arg, dog.DataNode):
            return nnef.Identifier(utils.ensure_not_unicode_in_python2(arg.name))
        else:
            return utils.ensure_not_unicode_in_python2(arg)

    return utils.recursive_transform(args, transform)


def _results_to_result_names(results):
    results2 = []
    for r in results:
        if isinstance(r, (list, tuple)):
            results2.append(_results_to_result_names(r))
        else:
            results2.append(nnef.Identifier(utils.ensure_not_unicode_in_python2(r.name)))
    if isinstance(results, tuple):
        return tuple(results2)
    else:
        return results2
