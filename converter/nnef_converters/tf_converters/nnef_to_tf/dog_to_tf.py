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

import nnef
import numpy as np
import tensorflow as tf

from ...common import dog
from ...common import utils


def _transform_arg(arg):
    if isinstance(arg, dog.DataNode):
        return arg.name
    elif isinstance(arg, str):
        return "'{}'".format(arg)
    elif isinstance(arg, tf.DType):
        return "tf." + arg.name
    elif isinstance(arg, np.ndarray):
        return arg.tolist()
    elif isinstance(arg, float):
        s = "{:.12f}".format(arg)
        if "." in s:
            s = s.rstrip('0')
            if s[-1] == '.':
                return s + '0'
            else:
                return s
        return s
    else:
        return arg


def _format_rec(arg):
    if isinstance(arg, (list, tuple)):
        parts = []
        for a in arg:
            parts.append(_format_rec(a))
        s = ", ".join(parts)
        if isinstance(arg, list):
            s = "[" + s + "]"
        if isinstance(arg, tuple):
            s = "(" + s + ")"
    else:
        s = str(arg)
    return s


def _format_args(args):
    parts = []
    can_remove_name = True
    for k, v in args.items():
        if (can_remove_name and (isinstance(v, dog.DataNode)
                                 or (isinstance(v, list) and all([isinstance(w, dog.DataNode) for w in v])))):
            arg = utils.recursive_transform(v, _transform_arg)
            arg_str = _format_rec(arg)
            parts.append("{}".format(arg_str))
        else:
            can_remove_name = False
            arg = utils.recursive_transform(v, _transform_arg)
            arg_str = _format_rec(arg)
            parts.append("{}={}".format(k, arg_str))
    return ", ".join(parts)


def tfdog_to_source(tfdog, left_pad=0, indent=4, generate_name_map=False):
    indent_str = " " * indent
    all_result_names = []
    lines = []
    header = "import tensorflow as tf\n"
    if any(op.name.startswith("tf_gen_nn_ops") for op in tfdog.ops):
        header += "from tensorflow.python.ops import gen_nn_ops as tf_gen_nn_ops\n"
    header += "\n\n"
    for tfop in tfdog.ops:
        result_names = [tfdn.name for tfdn in tfop.results.values()]
        all_result_names += result_names
        results_str = ", ".join(result_names)
        if len(results_str) < left_pad:
            results_str += ' ' * (left_pad - len(results_str))

        comments = []
        for tfdn in tfop.get_result_nodes():
            if dog.EXTRA_COMMENT in tfdn.extra and tfdn.extra[dog.EXTRA_COMMENT]:
                comments.append(str(tfdn.extra[dog.EXTRA_COMMENT]))
        comment = "  # " + " ".join(comments) if len(comments) > 0 else ""

        lines.append("{} = {}({}){}".format(results_str, tfop.name, _format_args(tfop.args), comment))

    lines.append("")
    if not generate_name_map:
        lines.append("return {")
        for output_name in tfdog.output_dn_names:
            lines.append(indent_str + '"{}": {},'.format(output_name, output_name))
        lines.append("}")
    else:
        lines.append("inputs__  = [{}]".format(", ".join(tfdog.input_dn_names)))
        lines.append("outputs__ = [{}]".format(", ".join(tfdog.output_dn_names)))
        lines.append("names__ = {{\n{}{}\n{}}}".format(2 * indent_str,
                                                       (",\n" + 2 * indent_str).join([
                                                           "'{}':{}.name".format(name, name)
                                                           for name in all_result_names
                                                           if not name.endswith("__")]),  # todo is the if needed
                                                       indent_str))
        lines.append("")
        lines.append("return inputs__, outputs__, names__")

    src = indent_str + ('\n' + indent_str).join(lines) + '\n'
    return "{}def {}():\n{}".format(header, tfdog.name, src)
