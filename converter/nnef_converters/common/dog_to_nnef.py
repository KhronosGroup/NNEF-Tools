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


def nnefdog_to_source(nnefdog, file_handle=None, custom_fragments=""):
    if file_handle is None:
        string_io = StringIO()
        try:
            # noinspection PyUnresolvedReferences, PyTypeChecker
            return _nnefdog_to_source_impl(nnefdog, string_io,
                                           custom_fragments=custom_fragments).getvalue()
        finally:
            string_io.close()
    else:
        return _nnefdog_to_source_impl(nnefdog, file_handle,
                                       custom_fragments=custom_fragments)


def _nnefdog_to_source_impl(nnefdog, file_handle, custom_fragments=""):
    # type: (NnefGraph, TextIO, str)->TextIO

    f = file_handle
    fix_str = utils.ensure_not_unicode_in_python2
    indent = 4 * " "

    print(nnef.format_version((1, 0)), file=f)

    extensions = []
    print(nnef.format_extensions(extensions), file=f)

    print(file=f)

    if custom_fragments:
        print(custom_fragments, file=f)
        print(file=f)

    graph_name = fix_str(nnefdog.name)
    graph_inputs = [fix_str(name) for name in nnefdog.input_dn_names]
    graph_outputs = [fix_str(name) for name in nnefdog.output_dn_names]

    print("graph {}({}) -> ({})".format(graph_name, ', '.join(graph_inputs), ', '.join(graph_outputs)), file=f)
    print("{", file=f)

    for op in nnefdog.ops:
        dtype = op.result.dtype if op.name in ["external", "constant", "variable"] else None
        invocation = nnef.format_invocation(name=fix_str(op.name),
                                            attribs=_preprocess_args(op.args),
                                            inputs=tuple(),
                                            outputs=_results_to_result_names(op.results.values()),
                                            dtype=dtype)

        comments = utils.without_nones([op.extra.get(dog.EXTRA_COMMENT)]
                                       + [dn.extra.get(dog.EXTRA_COMMENT) for dn in op.get_result_nodes()])
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
