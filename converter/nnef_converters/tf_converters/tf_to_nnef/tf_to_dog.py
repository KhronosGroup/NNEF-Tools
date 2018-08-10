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

import sys
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from .invocation_tracer import Invocation, InvocationTracer
from ...common import dog
from ...common import utils
from ...common.utils import Queue
from ...common.utils import get_qualified_name
from ...tf_converters import tf_compat

EXTRA_TFTENSOR_BY_NAME = "tftensor_by_name"
EXTRA_OUTPUT_TFDN_BY_NAME = "output_tfdn_by_name"

EXTRA_NESTING_LEVEL = "nesting_level"


def tfnetfun_to_tfdog(tfnetfun, functions_to_export):
    result, invocations, call_count = InvocationTracer.get_result_and_invocations_and_call_count(
        func=tfnetfun,
        functions=functions_to_export,
        functions_for_statistics_only=[tf.gradients, tf_compat.gen_array_ops_mirror_pad_grad],
        invocation_cls=_TFInvocation)

    _check_unsupported_operations(call_count)

    for invocation in invocations:
        if 'self' in invocation.args.keys():
            del invocation.args['self']

    tftensor_by_name, tfdn_by_name, tfops = _build_graph(invocations)

    for tfop in tfops:
        tfop.args = utils.recursive_transform(tfop.args, _normalize_types)

    output_tfdn_by_name = _get_result_data_nodes(result, tfdn_by_name)
    tfdns, tfops = _get_reachable_subgraph(
        tfdn_by_name, tfops, output_tfdn_by_name)

    tfdn_by_name = {tfdn.name: tfdn for tfdn in tfdns}
    input_tfdns = [tfdn for tfdn in tfdns if tfdn.producer is not None and tfdn.producer.name ==
                   get_qualified_name(tf.placeholder)]
    output_tfdns = list(output_tfdn_by_name.values())

    tfdog = dog.Graph(tfnetfun.__name__, tfops, tfdn_by_name,
                      [dn.name for dn in input_tfdns], [dn.name for dn in output_tfdns])
    tfdog.extra[EXTRA_TFTENSOR_BY_NAME] = tftensor_by_name
    tfdog.extra[EXTRA_OUTPUT_TFDN_BY_NAME] = output_tfdn_by_name

    return tfdog


def _check_unsupported_operations(call_count):
    if call_count[tf_compat.gen_array_ops_mirror_pad_grad] > 0:
        utils.print_error("Exporting the gradient of REFLECT or SYMMETRIC pad is unsupported.")
    if call_count[tf.gradients] > 0:
        if call_count[tf.nn.dropout] > 0:
            utils.print_warning("Exporting dropout is not supported with gradients "
                                "(tf.nn.dropout and tf.gradients were both called).\n"
                                "If you have used tf.gradients on a subgraph not containing tf.nn.dropout, "
                                "please ignore this message.")
        if call_count[tf.nn.relu6] > 0:
            utils.print_warning("Exporting relu6 is not supported with gradients "
                                "(tf.nn.relu6 and tf.gradients were both called).\n"
                                "If you have used tf.gradients on a subgraph not containing tf.nn.relu6, "
                                "please ignore this message.")
        for k, v in call_count.items():
            name = k.__name__
            if "resize_" in name and v > 0:
                utils.print_warning("Exporting {} is not supported with gradients "
                                    "({} and tf.gradients were both called).\n"
                                    "If you have used tf.gradients on a subgraph not containing {}, "
                                    "please ignore this message."
                                    .format(name, name, name))


def _normalize_types(arg):
    if isinstance(arg, bytes):
        try:
            return arg.decode('utf-8')
        except UnicodeError:
            return arg
    elif isinstance(arg, np.ndarray):
        return arg.tolist()
    elif isinstance(arg, tf.TensorShape):
        return [0 if dim is None else int(dim) for dim in arg.as_list()]
    elif isinstance(arg, tf.Dimension):
        return arg.value
    elif isinstance(arg, tf.DType):
        return arg.name
    elif isinstance(arg, (np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64,
                          np.float16, np.float32, np.float64,
                          np.bool_)):
        return arg.item()
    else:
        return arg


def _tuple_to_ordered_dict(tuple_):
    od = OrderedDict()
    for i, item in enumerate(tuple_):
        od["result" + str(i)] = item
    return od


def _tensor_to_data_node(tensor, tensor_name=None, data_node=None):
    if data_node is None:
        data_node = dog.DataNode()
        data_node.name = tensor_name
        if tensor.shape.dims is not None:
            data_node.shape = tensor.shape.as_list()
            if len(data_node.shape) == 0:
                data_node.shape = [1]
        else:
            # TODO find a better way to handle the case (this only happens with tf.gradients sometimes)
            # No need to print warning, it will fail if used
            # utils.print_warning("shape of {} is unknown".format(data_node.name))
            data_node.shape = None
        data_node.dtype = tensor.dtype.name
    return data_node


def _build_graph(invocations):
    tftensor_by_name = {}
    tfdn_by_name = {}
    operation_nodes = []

    for invocation in invocations:
        # invocation.debug_print()

        op_node = dog.OperationNode()
        op_node.name = invocation.get_op_name()
        op_node.extra[EXTRA_NESTING_LEVEL] = invocation.nesting_level

        tensors, name_by_tensor = invocation.get_arg_tensors()
        for tensor in tensors:
            tensor_name = name_by_tensor.get(tensor)
            tftensor_by_name[tensor_name] = tensor
            tfdn = _tensor_to_data_node(tensor,
                                        tensor_name,
                                        tfdn_by_name.get(tensor_name))
            tfdn.consumers.append(op_node)
            tfdn_by_name[tensor_name] = tfdn

        tensors2, name_by_tensor2 = invocation.get_result_tensors()
        for tensor in tensors2:
            tensor_name = name_by_tensor2.get(tensor)
            tftensor_by_name[tensor_name] = tensor
            tfdn = _tensor_to_data_node(tensor,
                                        tensor_name,
                                        tfdn_by_name.get(tensor_name))

            if tfdn.producer is None or op_node.extra[EXTRA_NESTING_LEVEL] < tfdn.producer.extra[EXTRA_NESTING_LEVEL]:
                tfdn.producer = op_node
            tfdn_by_name[tensor_name] = tfdn

        op_node.args = _change_tensors_to_data_nodes_in_ordered_dict(
            OrderedDict(invocation.args), name_by_tensor, tfdn_by_name)

        op_node.results = _change_tensors_to_data_nodes_in_ordered_dict(
            _tuple_to_ordered_dict(invocation.results), name_by_tensor2, tfdn_by_name)

        operation_nodes.append(op_node)

    return tftensor_by_name, tfdn_by_name, operation_nodes


def _change_tensors_to_data_nodes_in_ordered_dict(ordered_dict, name_by_tensor, data_node_by_name):
    for key, value in ordered_dict.items():
        if isinstance(value, tf.Tensor):
            ordered_dict[key] = data_node_by_name[name_by_tensor[value]]
        elif isinstance(value, tf.Variable):
            ordered_dict[key] = data_node_by_name[name_by_tensor[value.value()]]
        elif isinstance(value, list):
            ordered_dict[key] = _change_tensors_to_data_nodes_in_list(value, name_by_tensor, data_node_by_name)
        elif isinstance(value, tuple):
            ordered_dict[key] = tuple(_change_tensors_to_data_nodes_in_list(list(value),
                                                                            name_by_tensor,
                                                                            data_node_by_name))
    return ordered_dict


def _change_tensors_to_data_nodes_in_list(list_, name_by_tensor, data_node_by_name):
    new_list = []
    for value in list_:
        if isinstance(value, tf.Tensor):
            new_list.append(data_node_by_name[name_by_tensor[value]])
        elif isinstance(value, tf.Variable):
            new_list.append(data_node_by_name[name_by_tensor[value.value()]])
        else:
            new_list.append(value)
    return new_list


def _get_result_data_nodes(result, data_node_by_name):
    outputs = OrderedDict()
    if isinstance(result, tf.Tensor):
        outputs['output'] = result
    elif isinstance(result, tf.Variable):
        outputs['output'] = result.value()
    elif isinstance(result, (list, tuple)):
        for i, r in enumerate(result):
            if isinstance(r, (tf.Tensor, tf.Variable)):
                outputs['output' +
                        str(i + 1)] = r.value() if isinstance(r, tf.Variable) else r
    elif isinstance(result, dict):
        names = result.keys() if isinstance(result, OrderedDict) else sorted(result.keys())
        for name in names:
            r = result[name]
            if isinstance(r, (tf.Tensor, tf.Variable)):
                outputs[name] = r.value() if isinstance(r, tf.Variable) else r

    for tensor in outputs.values():
        if tensor.name not in data_node_by_name.keys():
            utils.print_error("Error no node for output tensor: {}".format(tensor.name))
            data_node_by_name[tensor.name] = dog.get_dummy_dn()

    return OrderedDict([(k, data_node_by_name[v.name]) for k, v in outputs.items()])


def _get_reachable_subgraph(data_node_by_name, operation_nodes, result_data_node_by_name):
    q = Queue()
    visited_data_nodes = set()
    visited_operation_nodes = set()

    def insert_node(data_node, a_consumer):
        if data_node not in visited_data_nodes:
            q.put((data_node, a_consumer))
            visited_data_nodes.add(data_node)

    for data_node in result_data_node_by_name.values():
        insert_node(data_node, None)

    while not q.empty():
        data_node, a_consumer = q.get()

        producer = data_node.producer

        if producer is None:
            if a_consumer is not None:
                utils.print_error("tensor '{}' used by operation {} is not the result of any exported operation"
                                  .format(data_node.name, a_consumer.name))
            else:
                utils.print_error("output tensor '{}' is not the result of any exported operation"
                                  .format(data_node.name))
            continue

        exclusions = {
            get_qualified_name(tf.nn.conv2d_transpose): ['output_shape'],
            get_qualified_name(tf.nn.conv3d_transpose): ['output_shape']
        }

        for data_arg in producer.get_arg_nodes(except_args=exclusions.get(producer.name, [])):
            insert_node(data_arg, producer)

        visited_operation_nodes.add(producer)

    visited_data_node_by_name = {}
    visited_operation_nodes_list = []

    for name, data_node in data_node_by_name.items():
        if data_node in visited_data_nodes:
            visited_data_node_by_name[name] = data_node

    for operation_node in operation_nodes:
        if operation_node in visited_operation_nodes:
            visited_operation_nodes_list.append(operation_node)

    return visited_data_nodes, visited_operation_nodes_list


class _TFInvocation(Invocation):
    def __init__(self, func, args, results, stack, nesting_level):
        super(_TFInvocation, self).__init__(func, args, results, stack, nesting_level)
        self._add_scope_if_var()
        self._fix_special_grad_invocations()

    def _add_scope_if_var(self):
        if self.func == tf.Variable or self.func == tf.get_variable:
            name = self.args['name']
            if name is not None:
                scope = tf.get_variable_scope().name
                self.args['name'] = scope + '/' + \
                                    name if len(scope) != 0 else name

    def _fix_special_grad_invocations(self):
        output_grad = self.args.get("grad", self.args.get("grad_softmax"))
        if isinstance(self.args.get("op"), tf.Operation) and isinstance(output_grad, tf.Tensor):
            self.args = OrderedDict([
                ("orig_inputs", [t for t in self.args["op"].inputs]),
                ("orig_outputs", [t for t in self.args["op"].outputs]),
                ("output_grad", output_grad),
                ("op_name", self.args.get("op").name)
            ])

    def get_arg_tensors(self):
        tensors = set()
        name_by_tensor = {}

        for arg in self.args.values():
            arg_elems = arg if isinstance(arg, (list, tuple)) else (arg,)
            for arg_elem in arg_elems:
                if isinstance(arg_elem, tf.Variable):
                    tensors.add(arg_elem.value())
                    name_by_tensor[arg_elem.value()] = arg_elem.name
                if isinstance(arg_elem, tf.Tensor):
                    tensors.add(arg_elem)
                    name_by_tensor[arg_elem] = arg_elem.name

        return tensors, name_by_tensor

    def get_result_tensors(self):
        tensors = set()
        name_by_tensor = {}

        for result in self.results:
            result_elems = result if isinstance(
                result, (list, tuple)) else [result]
            for result_elem in result_elems:
                if isinstance(result_elem, tf.Variable):
                    tensors.add(result_elem.value())
                    name_by_tensor[result_elem.value()] = result_elem.name
                if isinstance(result_elem, tf.Tensor):
                    tensors.add(result_elem)
                    name_by_tensor[result_elem] = result_elem.name

        return tensors, name_by_tensor

    def get_op_name(self):
        return get_qualified_name(self.func)

    def debug_print(self):
        result_names = ", ".join([t.name for t in self.get_result_tensors()[0]])
        arg_names = ", ".join([t.name for t in self.get_arg_tensors()[0]])
        sys.stderr.write(
            "{}{}={}({}) nl={}\n".format(self.nesting_level * "  ", result_names, self.get_op_name(), arg_names,
                                         self.nesting_level))
        sys.stderr.flush()
