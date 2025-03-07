# Copyright (c) 2020 The Khronos Group Inc.
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

from .reader import Reader
from .writer import Writer
from .composite import replace_composites_with_py_functions, reset_composites
from .utils import set_input_shapes, fold_constant_tensors, retain_reachables_from_outputs, insert_rename_identities
from .utils import import_graph_def, export_graph_def, check_finite, check_variables
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf


composite_function = composite.function


def save_default_graph(filename, session, outputs, input_shapes=None, fold_constants=True, collapse_composites=True):
    check_variables(session)

    if not isinstance(outputs, dict):
        outputs = {tensor: tensor.name[:-2] if tensor.name.endswith(':0') else tensor.name.replace(':', '_')
                   for tensor in outputs}

    output_names = list(outputs.values())

    graph_def = export_graph_def(tf.get_default_graph())
    graph_def = insert_rename_identities(graph_def, outputs)
    graph_def = tf.graph_util.convert_variables_to_constants(session, graph_def, output_names)
    graph_def = retain_reachables_from_outputs(graph_def, output_names)

    check_finite(graph_def)

    if input_shapes:
        graph_def = set_input_shapes(graph_def, input_shapes)
    if fold_constants:
        graph_def = fold_constant_tensors(graph_def)
    if collapse_composites:
        graph_def = replace_composites_with_py_functions(graph_def)

    check_finite(graph_def)

    with open(filename, 'wb') as file:
        file.write(graph_def.SerializeToString())


def load_default_graph(filename):
    from .protobuf import GraphDef
    with tf.io.gfile.GFile(filename, "rb") as file:
        graph_def = GraphDef()
        graph_def.ParseFromString(file.read())
        tf.import_graph_def(graph_def, name='')
