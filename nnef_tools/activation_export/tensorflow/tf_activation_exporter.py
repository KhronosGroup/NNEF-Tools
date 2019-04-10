from __future__ import division, print_function, absolute_import

import os
import typing

import nnef
import numpy as np
import tensorflow as tf

from nnef_tools.conversion.conversion_info import ConversionInfo


class ActivationExporterException(Exception):
    pass


class _TensorInfo(object):
    def __init__(self, internal_name, external_name, transforms, tensor, target_shape):
        self.internal_name = internal_name
        self.external_name = external_name
        self.transforms = transforms
        self.tensor = tensor
        self.target_shape = target_shape


def _write_nnef_tensor(filename, tensor):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, "wb") as file:
        nnef.write_tensor(file, tensor, version=(1, 0))


def export(output_path,  # type: str
           feed_dict,  # type: typing.Dict[typing.Any, np.ndarray]
           conversion_info,  # type: ConversionInfo
           graph=None,  # type: typing.Optional[tf.Graph]
           checkpoint_path=None,  # type: str
           tensors_per_iter=25,  # type: int
           verbose=False,  # type: bool
           init_variables=False,  # type: bool
           input_output_only=False,  # type: bool
           ):
    # type: (...)->None

    if graph is None:
        graph = tf.get_default_graph()

    has_error = False
    tensor_infos = []

    for tensor_info in conversion_info.tensors:
        if tensor_info.is_variable or (input_output_only and not tensor_info.is_input and not tensor_info.is_output):
            continue
        try:
            tensor = graph.get_tensor_by_name(tensor_info.source_name)

            if not graph.is_fetchable(tensor):
                print("Warning: Tensor is not fetchable: {}".format(tensor_info.source_name))
            elif not isinstance(tensor, tf.Tensor):
                print("Warning: Not a tensor: {}".format(tensor_info.source_name))
            else:
                tensor_infos.append(_TensorInfo(internal_name=tensor_info.source_name,
                                                external_name=tensor_info.target_name,
                                                transforms=tensor_info.transforms,
                                                tensor=tensor,
                                                target_shape=tensor_info.target_shape))
        except KeyError:
            print("Warning: Tensor not found: {}".format(tensor_info.source_name))

    with tf.Session() as sess:
        saver = tf.train.Saver() if checkpoint_path else None
        start = 0
        while start < len(tensor_infos):
            tensors = [info.tensor for info in tensor_infos[start:start + tensors_per_iter]]

            if init_variables:
                sess.run(tf.global_variables_initializer())
            if checkpoint_path is not None:
                saver.restore(sess, checkpoint_path)
            values = sess.run(tensors, feed_dict)

            for i, arr in enumerate(values):
                info = tensor_infos[start + i]
                filename = os.path.join(output_path, info.external_name + ".dat")

                if np.isnan(arr).any():
                    print("Error: '{}' has nan's".format(info.external_name))
                    has_error = True
                elif not np.isfinite(arr).all():
                    print("Error: '{}' has inf's".format(info.external_name))
                    has_error = True
                try:
                    for transform in info.transforms:
                        arr = transform.apply_np(arr)
                    _write_nnef_tensor(filename, np.asarray(arr, order='C'))
                except ValueError as e:
                    print("Error: Can not export '{}': {}".format(info.external_name, e))

            start += len(tensors)

            if verbose:
                print("Info: Exported {}/{}".format(start, len(tensor_infos)))
    if has_error:
        raise ActivationExporterException("There were errors!")
