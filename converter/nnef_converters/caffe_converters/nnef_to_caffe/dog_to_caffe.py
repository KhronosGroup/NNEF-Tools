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

from ...caffe_converters import common
from ...common import dog
from ...common.types import *
from ...common.utils import StringIO

EXTRA_CAFFE_PARAM_NAME = "caffe_param_name"


def _print_name_value(name, value, file, indent_level):
    # type: (str, Any, TextIO, int)->None

    spaces = indent_level * "  "
    if value is not None:
        if isinstance(value, str):
            print('{}{}: "{}"'.format(spaces, name, value), file=file)
        else:
            print('{}{}: {}'.format(spaces, name, value), file=file)


def _print_name_dict(name, value, file, indent_level):
    # type: (str, Dict[str, Any], TextIO, int)->None

    spaces = indent_level * "  "
    print('{}{} {{'.format(spaces, name), file=file)
    for k, v in value.items():
        if k == "shape":
            _print_name_dict(k, dict(dim=v), file, indent_level + 1)
        elif isinstance(v, (dict, OrderedDict)):
            _print_name_dict(k, v, file, indent_level + 1)
        elif isinstance(v, (list, tuple)):
            for w in v:
                _print_name_value(k, w, file, indent_level + 1)
        else:
            _print_name_value(k, v, file, indent_level + 1)

    print('{}}}'.format(spaces), file=file)


# noinspection PyTypeChecker
def caffedog_to_prototxt(caffedog):
    # type: (CaffeGraph)->str

    sio = StringIO()
    try:
        for op in caffedog.ops:
            dn_args = op.get_arg_nodes()
            dn_results = op.get_result_nodes()

            print('layer {', file=sio)
            print('  name: "{}"'.format(common.get_layer_name(op)), file=sio)
            print('  type: "{}"'.format(op.name), file=sio)
            for dn in dn_args:
                print('  bottom: "{}"'.format(dn.name), file=sio)
            for dn in dn_results:
                print('  top: "{}"'.format(dn.name), file=sio)

            param_name = op.extra.get(EXTRA_CAFFE_PARAM_NAME, op.name.lower() + "_param")
            param_dict = OrderedDict([(k, v) for k, v in op.args.items() if not k.startswith("arg") and v is not None])

            if param_dict:
                _print_name_dict(param_name, param_dict, file=sio, indent_level=1)

            print('}', file=sio)

        return sio.getvalue()
    finally:
        sio.close()
