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

import os
import shutil
import sys
import tarfile
import tempfile
from collections import OrderedDict

import nnef
import numpy as np
import tensorflow as tf

from . import converters
from . import hooks as hooks_module
from . import tf_to_dog
from . import transformations as trafos
from ...common import dog
from ...common import dog_to_nnef
from ...common import nnef_shape_optimizer
from ...common import utils
from ...common.types import *
from ...common.utils import get_qualified_name


class TF2NNEFConverter(object):
    def __init__(self):
        self.vars_names_labels_to_export = []
        self.const_by_tfdn = {}
        self.tensor_counts = {}
        self.nnefdn_by_tfdn = None
        self.converter_by_name = {}
        self.tftensor_by_name = {}
        self.output_name_by_tfname = OrderedDict()
        self.tfop_by_nnefop = OrderedDict()
        self.checkpoint_reader = None
        self.output_path = ""
        self.hooks = {}
        self.add_comments = False
        self.trafo_dict = OrderedDict()
        self.nnefdog = None

    def reset(self):
        self.vars_names_labels_to_export = []
        self.const_by_tfdn = {}
        self.tensor_counts = {}
        self.nnefdn_by_tfdn = None
        self.converter_by_name = {}
        self.tftensor_by_name = {}
        self.output_name_by_tfname = OrderedDict()
        self.tfop_by_nnefop = OrderedDict()
        self.checkpoint_reader = None
        self.output_path = ""
        self.hooks = {}
        self.add_comments = False
        self.trafo_dict = OrderedDict()
        self.nnefdog = None

    def export_network(
            self,
            net_func,
            checkpoint=None,
            custom_converters=None,
            custom_fragments='',
            output_path=None,
            compress=False,
            verbose=True,
            hooks=None,
            add_comments=False,
    ):
        # if utils.tf_version_greater_equal(1, 10):
        #     override_dynamic_shapes_with_static()

        self.reset()
        if custom_converters is None:
            custom_converters = {}

        if hooks is not None:
            self.hooks = hooks

        self.add_comments = add_comments

        self.checkpoint_reader = None
        if checkpoint is not None:
            self.checkpoint_reader = tf.contrib.framework.load_checkpoint(checkpoint)

        if output_path is None and checkpoint is not None:
            output_path = os.path.splitext(checkpoint)[0] + '-nnef'

        if output_path is not None and not os.path.exists(output_path):
            os.makedirs(output_path)

        self.output_path = output_path

        converter_dict = converters.DefaultConverters
        if custom_converters:
            converter_dict.update(custom_converters)

        self.converter_by_name = {get_qualified_name(k): converters.ConversionRule.from_old_style(v) for k, v in
                                  converter_dict.items()}

        tfdog = tf_to_dog.tfnetfun_to_tfdog(net_func, converter_dict.keys())
        tfops = tfdog.ops
        tfdns = list(tfdog.dn_by_name.values())

        self.tftensor_by_name = tfdog.extra[tf_to_dog.EXTRA_TFTENSOR_BY_NAME]

        self.output_name_by_tfname = OrderedDict([
            (tfdn.name, output_name)
            for output_name, tfdn
            in tfdog.extra[tf_to_dog.EXTRA_OUTPUT_TFDN_BY_NAME].items()
        ])

        output_index_by_tfname = OrderedDict([
            (tfdn.name, i)
            for i, tfdn
            in enumerate(tfdog.extra[tf_to_dog.EXTRA_OUTPUT_TFDN_BY_NAME].values())
        ])

        def default_add_extras_to_tfdn_hook(tfdn, tf_tensor=None):
            pass

        for tfdn in tfdns:
            self.call_hook_list(hooks_module.HOOK_ADD_EXTRAS_TO_TFDN, default_add_extras_to_tfdn_hook,
                                tfdn, self.tftensor_by_name.get(tfdn.name, None))

        tfops = trafos.transform_tile_if_const(tfops, self)

        # TODO consider collapsing these into a single operation
        tfops = trafos.transform_sqrt_grad(tfops, self)
        tfops = trafos.transform_elu_grad(tfops, self)
        tfops = trafos.transform_relu_grad(tfops, self)
        tfops = trafos.transform_softplus_grad(tfops, self)
        tfops = trafos.transform_rsqrt_grad(tfops, self)
        tfops = trafos.transform_sigmoid_grad(tfops, self)
        tfops = trafos.transform_tanh_grad(tfops, self)
        tfops = trafos.transform_reciprocal_grad(tfops, self)
        tfops = trafos.transform_bias_add_grad(tfops, self)
        tfops = trafos.transform_lrn_grad(tfops, self)

        tfops = trafos.transform_transpose_grad(tfops, self)
        tfops = trafos.transform_min_or_max_grad(tfops, self)

        tfops = trafos.transform_evaluate_shapes(tfops, self)
        tfops = trafos.transform_evaluate_multiples(tfops, self)
        tfops = trafos.transform_fill_to_constant(tfops, self)
        tfops = trafos.transform_evaluate_other_constants(tfops, self)

        tfops = trafos.transform_remove_unreachables(tfops, tfdog.output_dn_names, self)

        tfops = trafos.transform_casts(tfops, self)
        tfops = trafos.transform_eliminate_tf_passthroughs(tfops, self)
        tfops = trafos.transform_bts_conv_stb(tfops, self)
        tfops = trafos.transform_pad(tfops, self)
        tfops = trafos.transform_fused_batch_norm(tfops, self)

        tfops = trafos.transform_zeros_ones_like(tfops, self)
        tfops = trafos.transform_strided_slice(tfops, self)
        tfops = trafos.transform_strided_slice_grad(tfops, self)
        tfops = trafos.transform_add_conv(tfops, self)

        tfops = trafos.transform_range(tfops, self)

        tfops = trafos.transform_remove_unreachables(tfops, tfdog.output_dn_names, self)

        # BEGIN TRANSLATION PHASE
        self.nnefdn_by_tfdn = {}

        for tfop in tfops:
            converter = self.converter_by_name.get(tfop.name)
            if converter is not None:
                converter.converter_fun(tfop, self)
            else:
                utils.print_error("No converter for {}".format(tfop.name))

        self.nnefdn_by_tfdn = None
        # END TRANSLATIONS PHASE

        nnefops = self.tfop_by_nnefop.keys()
        nnefops = trafos.transform_eliminate_nnef_passthroughs(nnefops, self)
        nnefops = nnef_shape_optimizer.transform_shape_optimizations(
            nnefops=nnefops,
            is_source_lang=False,
            dg_is_output_nnefdn=lambda nnefdn: nnefdn.source_name and nnefdn.source_name in self.output_name_by_tfname)

        input_nnefdns = []
        output_nnefdns = [None] * len(self.output_name_by_tfname)

        for nnefop in nnefops:
            if nnefop.name == "external":
                input_nnefdns.append(nnefop.result)

        nnefdns = utils.flatten([nnefop.get_result_nodes() for nnefop in nnefops])

        if self.add_comments:
            nnef_shape_optimizer.add_comments(nnefdns)

        if self.checkpoint_reader:
            for nnefop, var_name, var_label in self.vars_names_labels_to_export:
                if self.checkpoint_reader.has_tensor(var_name):
                    tensor = self.checkpoint_reader.get_tensor(var_name)
                    filename = self.output_path + '/' + var_label + '.dat'
                    applied_trafos = nnefop.result.extra.get(nnef_shape_optimizer.EXTRA_APPLIED_TRANSFORMATIONS)
                    utils.write_nnef_tensor(filename, utils.np_apply_transforms(tensor, applied_trafos))
                    # utils.print_warning("Exported variable {} {} {}".format(var_name, var_label, applied_trafos))
                else:
                    utils.print_error("variable '{}' not found in checkpoint".format(var_name))

        for nnefdn in nnefdns:
            if nnefdn.source_name and nnefdn.source_name in self.output_name_by_tfname:
                output_name = self.output_name_by_tfname[nnefdn.source_name]
                output_index = output_index_by_tfname[nnefdn.source_name]

                if nnefdn not in input_nnefdns:
                    nnefdn.name = output_name
                    if output_nnefdns[output_index]:
                        utils.print_error("Duplicate output: {}".format(output_name))
                    output_nnefdns[output_index] = nnefdn
                else:  # input cannot be output, so we add an identity op
                    nnefop = dog.OperationNode("squeeze")
                    nnefop.add_arg("input", nnefdn)
                    nnefop.add_arg("axes", [])
                    nnefop.add_result("output", self.make_nnefdn(None, output_name, False))
                    nnefops.append(nnefop)
                    nnefdns.append(nnefop.result)
                    if output_nnefdns[output_index]:
                        utils.print_error("Duplicate output: {}".format(output_name))
                    output_nnefdns[output_index] = nnefop.result

        for i, output_name in enumerate(self.output_name_by_tfname.values()):
            if len(output_nnefdns) <= i or output_nnefdns[i] is None or output_nnefdns[i].name != output_name:
                utils.print_error("Missing output: {}".format(output_name))

        nnefdn_by_name = {nnefdn.name: nnefdn for nnefdn in nnefdns}

        self.trafo_dict = nnef_shape_optimizer.get_trafo_dict(nnefdns, output_nnefdns)

        self.nnefdog = dog.Graph(tfdog.name,
                                 nnefops,
                                 nnefdn_by_name,
                                 [dn.name for dn in input_nnefdns],
                                 [dn.name for dn in output_nnefdns])

        file_ = open(output_path + '/graph.nnef', 'w') if output_path is not None else sys.stdout

        dog_to_nnef.nnefdog_to_source(self.nnefdog,
                                      file_handle=file_,
                                      custom_fragments=custom_fragments)

        if file_ != sys.stdout:
            file_.close()

        if compress and file_ != sys.stdout:
            if verbose:
                utils.print_info("Compressing files...")

            filename = output_path + '.tgz'
            tar = tarfile.open(filename, 'w:gz')
            for file_ in os.listdir(output_path):
                tar.add(output_path + '/' + file_, file_)
            tar.close()
            shutil.rmtree(output_path)

    def get_tftensor_by_tfdn(self, tfdn):
        return self.tftensor_by_name.get(tfdn.name)

    def is_binary_op(self, op_name):
        converter = self.converter_by_name.get(op_name)
        return converter is not None and converter.converter_fun == converters.convert_binary

    @staticmethod
    def _get_first_existing_key(dict_, keys):
        for key in keys:
            if key in dict_.keys():
                return key
        return None

    @staticmethod
    def nnef_array(value, rank):
        if isinstance(value, list):
            return value
        elif isinstance(value, tuple):
            return list(value)
        else:
            return [value] * rank

    @staticmethod
    def nnef_bool(value):
        if value is None:
            value = False
        return True if value else False

    @staticmethod
    def nnef_axis(axis, rank):
        if axis < 0:
            return axis + rank
        else:
            return axis

    @staticmethod
    def nnef_axes(axes, rank):
        if isinstance(axes, (list, tuple)):
            return [TF2NNEFConverter.nnef_axis(axis, rank) for axis in axes]
        else:
            return [TF2NNEFConverter.nnef_axis(axes, rank)]

    @staticmethod
    def spatial_size(shape, is_nhwc):
        return list(shape[1:-1] if is_nhwc else shape[2:])

    @staticmethod
    def spatial_size_of_tf_filter(shape):
        return list(shape[:-2])

    @staticmethod
    def full_shape(shape, is_nhwc, unit_elem=1):
        return [unit_elem] + shape + [unit_elem] if is_nhwc else [unit_elem, unit_elem] + shape

    def nnef_op(self, func_name):
        exporter = self.converter_by_name.get(func_name)
        if exporter is None:
            return None
        return exporter.op_name

    # this is not spatial, it has all dimensions
    @staticmethod
    def nnef_padding(padding, rank):
        if isinstance(padding, (list, tuple)):
            if len(padding) != rank:
                utils.print_error("Padding length: {} != rank: {}".format(len(padding), rank))
            return [(a, b) for a, b in padding]
        else:
            return [] if padding.upper() == 'SAME' else [(0, 0)] * rank

    @staticmethod
    def nnef_border(border):
        border = border.lower()
        if border == 'symmetric':
            border = 'reflect-even'
        return border

    # spatial
    @staticmethod
    def nnef_padding_ex(padding, input_sizes, filter_sizes, strides):
        def same_padding(input_size, filter_size, stride):
            output_size = int(np.ceil(float(input_size) / float(stride)))
            pad_total = (output_size - 1) * stride + filter_size - input_size
            if pad_total >= 0:
                pad_front = pad_total // 2
                pad_back = pad_total - pad_front
                return pad_front, pad_back
            else:
                return 0, pad_total

        def valid_padding(input_size, filter_size, stride):
            output_size = int(np.ceil(float(input_size - filter_size + 1) / float(stride)))
            pad_total = (output_size - 1) * stride + filter_size - input_size
            return 0, pad_total

        return [
            same_padding(input_size, filter_size, stride)
            if padding.upper() == 'SAME'
            else valid_padding(input_size, filter_size, stride)

            for (input_size, filter_size, stride)
            in zip(input_sizes, filter_sizes, strides)
        ]

    @staticmethod
    def dilated_size(size, dilation):
        return [(s - 1) * d + 1 for (s, d) in zip(size, dilation)]

    @staticmethod
    def numpy_dtype(tfdtype):
        if not tfdtype:
            utils.print_error("Unknown DType, reverting to float32")
            tfdtype = "float32"

        npdtype = utils.try_tf_dtype_to_np(tfdtype)

        if not npdtype:
            utils.print_error("DType {} not convertible to np dtype, reverting to float32".format(tfdtype))
            npdtype = np.dtype(np.float32)

        return npdtype

    @staticmethod
    def nnef_dtype(tfdtype):
        npdtype = TF2NNEFConverter.numpy_dtype(tfdtype)

        nnefdtype = utils.try_np_dtype_to_nnef(npdtype)

        if not nnefdtype:
            utils.print_error("Unsupported dtype: {}, reverting to scalar".format(npdtype.name))
            nnefdtype = "scalar"
        return nnefdtype

    def make_nnefdn(self, tfdn, nnef_name, indexed=True, source_name=""):
        if source_name == "" and tfdn and tfdn.name:
            source_name = tfdn.name

        if tfdn is not None:
            nnefdn = self.nnefdn_by_tfdn.get(tfdn)
            if nnefdn is not None:
                return nnefdn

        if indexed:
            count = self.tensor_counts.get(nnef_name, 0)
            self.tensor_counts[nnef_name] = count + 1
            indexed_name = nnef_name + str(count + 1)
        else:
            indexed_name = nnef_name

        nnefdn = dog.DataNode()
        nnefdn.name = indexed_name

        if source_name:
            nnefdn.source_name = source_name
        if tfdn is not None:
            self.nnefdn_by_tfdn[tfdn] = nnefdn

        return nnefdn

    def get_nnefdn(self, tfdn_or_other):
        if isinstance(tfdn_or_other, bool):
            return tfdn_or_other
        elif isinstance(tfdn_or_other, (float, int)):
            return float(tfdn_or_other)
        elif isinstance(tfdn_or_other, (list, tuple)) and len(tfdn_or_other) == 1:
            return float(tfdn_or_other[0])
        elif tfdn_or_other in self.const_by_tfdn:
            return self.const_by_tfdn[tfdn_or_other]
        elif tfdn_or_other in self.nnefdn_by_tfdn:
            return self.nnefdn_by_tfdn[tfdn_or_other]
        elif isinstance(tfdn_or_other, dog.DataNode):
            utils.print_error("Undefined TF DataNode: {}".format(tfdn_or_other.name))
            return dog.get_dummy_dn()
        else:
            utils.print_error("Undefined TF DataNode: {}".format(tfdn_or_other))
            return dog.get_dummy_dn()

    @staticmethod
    def get_shape_safe(tfdn_or_other):
        return dog.get_shape_safe(tfdn_or_other)

    @staticmethod
    def get_rank_safe(tfdn_or_other):
        return len(dog.get_shape_safe(tfdn_or_other))

    def make_constant(self, tensor, nnef_value):
        self.const_by_tfdn[tensor] = nnef_value

    def call_hook_list(self, hook_name, default_hook, *args):
        hooks = self.hooks.get(hook_name, [])

        if isinstance(hooks, (list, tuple)):
            hooks = list(reversed(hooks))
        else:
            hooks = [hooks]

        hooks.append(default_hook)

        for hook in hooks:
            res = hook(*args)
            if res is not hooks_module.PROPAGATE:
                return res

        utils.print_error("All hooks should not propagate.")
        return None

    def add_nnefop(self, nnefop, tfop=None):
        assert nnefop not in self.tfop_by_nnefop
        self.tfop_by_nnefop[nnefop] = tfop

    def add_transpose_to_input_if_nhwc(self, tfop, tfdn_input, is_nhwc=None):
        if is_nhwc is None:
            is_nhwc = utils.is_nhwc(tfop)

        input_rank = self.get_rank_safe(tfdn_input)
        nnefdn_input = self.get_nnefdn(tfdn_input)

        if is_nhwc:
            nnefop_transpose_input = dog.OperationNode("transpose")
            nnefop_transpose_input.add_arg("input", nnefdn_input)
            nnefop_transpose_input.add_arg("axes", utils.transpose_axes_nhwc_to_nchw(input_rank))
            nnefop_transpose_input.add_result("output", self.make_nnefdn(dog.DataNode(), "_nchw"))
            nnefop_transpose_input.extra[nnef_shape_optimizer.EXTRA_GENERATED_TRANSPOSE] = True
            self.add_nnefop(nnefop_transpose_input, tfop)
            return nnefop_transpose_input.result
        return nnefdn_input

    def add_transpose_to_filter_hwcn(self, tfop, tfdn_filter):
        filter_rank = self.get_rank_safe(tfdn_filter)
        nnefdn_filter = self.get_nnefdn(tfdn_filter)

        nnefop_transpose_filter = dog.OperationNode("transpose")
        nnefop_transpose_filter.add_arg("input", nnefdn_filter)
        nnefop_transpose_filter.add_arg("axes", utils.transpose_axes_hwcn_to_nchw(filter_rank))
        nnefop_transpose_filter.add_result("output", self.make_nnefdn(dog.DataNode(), "_nchw"))
        nnefop_transpose_filter.extra[nnef_shape_optimizer.EXTRA_GENERATED_TRANSPOSE] = True
        self.add_nnefop(nnefop_transpose_filter, tfop)
        return nnefop_transpose_filter.result

    def add_trafos_to_group_conv_filter(self, tfop, tfdn_filter, group_count):
        filter_rank = self.get_rank_safe(tfdn_filter)
        filter_shape = self.get_shape_safe(tfdn_filter)
        reshape_shape = filter_shape[:-2] + [group_count, filter_shape[-2] // group_count, filter_shape[-1]]
        transpose_axes = list(range(filter_rank - 2)) + [filter_rank - 1, filter_rank - 2, filter_rank]
        reshape_shape2 = filter_shape[:-2] + [filter_shape[-2] // group_count, filter_shape[-1] * group_count]
        transpose_axes2 = utils.transpose_axes_hwcn_to_nchw(filter_rank)
        nnefdn_filter = self.get_nnefdn(tfdn_filter)

        nnefop_reshape = dog.OperationNode("reshape")
        nnefop_reshape.add_arg("input", nnefdn_filter)
        nnefop_reshape.add_arg("shape", reshape_shape)
        nnefop_reshape.add_result("output", self.make_nnefdn(dog.DataNode(), "_reshape"))
        nnefop_reshape.extra[nnef_shape_optimizer.EXTRA_GENERATED_RESHAPE] = True
        self.add_nnefop(nnefop_reshape, tfop)

        nnefop_transpose = dog.OperationNode("transpose")
        nnefop_transpose.add_arg("input", nnefop_reshape.result)
        nnefop_transpose.add_arg("axes", transpose_axes)
        nnefop_transpose.add_result("output", self.make_nnefdn(dog.DataNode(), "_transpose"))
        nnefop_transpose.extra[nnef_shape_optimizer.EXTRA_GENERATED_TRANSPOSE] = True
        self.add_nnefop(nnefop_transpose, tfop)

        nnefop_reshape2 = dog.OperationNode("reshape")
        nnefop_reshape2.add_arg("input", nnefop_transpose.result)
        nnefop_reshape2.add_arg("shape", reshape_shape2)
        nnefop_reshape2.add_result("output", self.make_nnefdn(dog.DataNode(), "_reshape"))
        nnefop_reshape2.extra[nnef_shape_optimizer.EXTRA_GENERATED_RESHAPE] = True
        self.add_nnefop(nnefop_reshape2, tfop)

        nnefop_transpose2 = dog.OperationNode("transpose")
        nnefop_transpose2.add_arg("input", nnefop_reshape2.result)
        nnefop_transpose2.add_arg("axes", transpose_axes2)
        nnefop_transpose2.add_result("output", self.make_nnefdn(dog.DataNode(), "_nchw"))
        nnefop_transpose2.extra[nnef_shape_optimizer.EXTRA_GENERATED_TRANSPOSE] = True
        self.add_nnefop(nnefop_transpose2, tfop)

        return nnefop_transpose2.result

    def add_reshape_transpose_to_filter_hwcm(self, tfop, tfdn_filter):
        filter_rank = self.get_rank_safe(tfdn_filter)
        filter_shape = self.get_shape_safe(tfdn_filter)
        reshape_shape = filter_shape[:-2] + [1, filter_shape[-1] * filter_shape[-2]]
        nnefdn_filter = self.get_nnefdn(tfdn_filter)

        nnefop_reshape = dog.OperationNode("reshape")
        nnefop_reshape.add_arg("input", nnefdn_filter)
        nnefop_reshape.add_arg("shape", reshape_shape)
        nnefop_reshape.add_result("output", self.make_nnefdn(dog.DataNode(), "_reshape"))
        nnefop_reshape.extra[nnef_shape_optimizer.EXTRA_GENERATED_RESHAPE] = True
        self.add_nnefop(nnefop_reshape, tfop)

        nnefop_transpose = dog.OperationNode("transpose")
        nnefop_transpose.add_arg("input", nnefop_reshape.result)
        nnefop_transpose.add_arg("axes", utils.transpose_axes_hwcn_to_nchw(filter_rank))
        nnefop_transpose.add_result("output", self.make_nnefdn(dog.DataNode(), "_nchw"))
        nnefop_transpose.extra[nnef_shape_optimizer.EXTRA_GENERATED_TRANSPOSE] = True
        self.add_nnefop(nnefop_transpose, tfop)

        return nnefop_transpose.result

    def add_unsqueeze_to_arg_if_rank_1(self, tfop, tfdn_arg, enabled=True):
        nnefdn_arg = self.get_nnefdn(tfdn_arg)

        if not enabled:
            return nnefdn_arg

        if self.get_rank_safe(tfdn_arg) == 1:
            # TODO add nnef types
            nnefop_unsqueeze_bias = dog.OperationNode("unsqueeze")
            nnefop_unsqueeze_bias.add_arg("input", nnefdn_arg)
            nnefop_unsqueeze_bias.add_arg("axes", [0])
            nnefop_unsqueeze_bias.add_result("output", self.make_nnefdn(dog.DataNode(), "_unsqueeze"))
            nnefop_unsqueeze_bias.extra[nnef_shape_optimizer.EXTRA_GENERATED_UNSQUEEZE] = True
            self.add_nnefop(nnefop_unsqueeze_bias, tfop)
            return nnefop_unsqueeze_bias.result
        return nnefdn_arg

    def add_unsqueeze_to_arg_if_broadcast(self, tfop, tfdn_arg, tfdn_other):
        nnefdn_arg = self.get_nnefdn(tfdn_arg)

        if self.is_broadcast(tfdn_arg, tfdn_other):
            rank_diff = len(tfdn_other.shape) - len(tfdn_arg.shape)
            nnefop_unsqueeze = dog.OperationNode("unsqueeze")
            nnefop_unsqueeze.add_arg("input", nnefdn_arg)
            nnefop_unsqueeze.add_arg("axes", list(range(rank_diff)))
            nnefop_unsqueeze.add_result("output", self.make_nnefdn(dog.DataNode(), "_unsqueeze"))
            nnefop_unsqueeze.extra[nnef_shape_optimizer.EXTRA_GENERATED_UNSQUEEZE] = True
            self.add_nnefop(nnefop_unsqueeze, tfop)
            return nnefop_unsqueeze.result
        return nnefdn_arg

    def add_unsqueeze_or_transpose_to_arg_if_needed(self, tfop, tfdn_arg, is_nhwc=None):
        if tfdn_arg is None:
            return None
        if self.get_rank_safe(tfdn_arg) == 1:
            return self.add_unsqueeze_to_arg_if_rank_1(tfop, tfdn_arg)
        else:
            return self.add_transpose_to_input_if_nhwc(tfop, tfdn_arg, is_nhwc=is_nhwc)

    def add_nnefop_with_result_transposed_if_nhwc(self, tfop, nnefop, result_name, result_var_name, is_nhwc=None):
        if is_nhwc is None:
            is_nhwc = utils.is_nhwc(tfop)

        if not isinstance(result_name, (list, tuple)):
            result_name = (result_name,)

        if not isinstance(result_var_name, (list, tuple)):
            result_var_name = (result_var_name,)

        if is_nhwc:
            for i, tfdn_result in enumerate(tfop.results.values()):
                output_rank = self.get_rank_safe(tfdn_result)
                perm = utils.transpose_axes_nchw_to_nhwc(output_rank)
                perm_inv = utils.get_inverse_permutation(perm)

                nnefop.add_result(result_name[i],
                                  self.make_nnefdn(None, result_var_name[i], source_name=tfdn_result.name))
                nnefop.results[result_name[i]].extra[nnef_shape_optimizer.EXTRA_APPLIED_TRANSFORMATIONS] = [
                    ("transpose", perm_inv)
                ]
            self.add_nnefop(nnefop, tfop)

            for tfdn_result, nnefdn_result in zip(tfop.results.values(), nnefop.results.values()):
                output_rank = self.get_rank_safe(tfdn_result)
                perm = utils.transpose_axes_nchw_to_nhwc(output_rank)

                nnefop_transpose_output = dog.OperationNode("transpose")
                nnefop_transpose_output.add_arg("input", nnefdn_result)
                nnefop_transpose_output.add_arg("axes", perm)
                nnefop_transpose_output.add_result('output', self.make_nnefdn(tfdn_result, "_nhwc", source_name=None))
                nnefop_transpose_output.extra[nnef_shape_optimizer.EXTRA_GENERATED_TRANSPOSE] = True
                self.add_nnefop(nnefop_transpose_output, tfop)
        else:
            for i, tfdn_result in enumerate(tfop.results.values()):
                nnefop.add_result(result_name[i], self.make_nnefdn(tfdn_result, result_var_name[i]))
            self.add_nnefop(nnefop, tfop)

    def add_nnefop_with_result_transposed_hwcn(self, tfop, nnefop, result_name, result_var_name):
        if not isinstance(result_name, (list, tuple)):
            result_name = (result_name,)

        if not isinstance(result_var_name, (list, tuple)):
            result_var_name = (result_var_name,)

        for i, tfdn_result in enumerate(tfop.results.values()):
            output_rank = self.get_rank_safe(tfdn_result)
            perm = utils.transpose_axes_nchw_to_hwcn(output_rank)
            perm_inv = utils.get_inverse_permutation(perm)

            nnefop.add_result(result_name[i], self.make_nnefdn(None, result_var_name[i], source_name=tfdn_result.name))
            nnefop.results[result_name[i]].extra[nnef_shape_optimizer.EXTRA_APPLIED_TRANSFORMATIONS] = [
                ("transpose", perm_inv)
            ]
        self.add_nnefop(nnefop, tfop)

        for tfdn_result, nnefdn_result in zip(tfop.results.values(), nnefop.results.values()):
            output_rank = self.get_rank_safe(tfdn_result)
            perm = utils.transpose_axes_nchw_to_hwcn(output_rank)

            nnefop_transpose_output = dog.OperationNode("transpose")
            nnefop_transpose_output.add_arg("input", nnefdn_result)
            nnefop_transpose_output.add_arg("axes", perm)
            nnefop_transpose_output.add_result('output', self.make_nnefdn(tfdn_result, "_hwcn", source_name=None))
            nnefop_transpose_output.extra[nnef_shape_optimizer.EXTRA_GENERATED_TRANSPOSE] = True
            self.add_nnefop(nnefop_transpose_output, tfop)

    def add_nnefop_with_result_transposed_reshaped_hwcm(self, tfop, nnefop, result_name, result_var_name,
                                                        tf_output_shape):
        if not isinstance(result_name, (list, tuple)):
            result_name = (result_name,)

        if not isinstance(result_var_name, (list, tuple)):
            result_var_name = (result_var_name,)

        for i, tfdn_result in enumerate(tfop.results.values()):
            output_rank = self.get_rank_safe(tfdn_result)
            perm = utils.transpose_axes_nchw_to_hwcn(output_rank)
            perm_inv = utils.get_inverse_permutation(perm)

            nnefop.add_result(result_name[i], self.make_nnefdn(None, result_var_name[i], source_name=tfdn_result.name))
            nnefop.results[result_name[i]].extra[nnef_shape_optimizer.EXTRA_APPLIED_TRANSFORMATIONS] = [
                ("reshape", tf_output_shape[:-2] + [1, tf_output_shape[-2] * tf_output_shape[-1]]),
                ("transpose", perm_inv)
            ]

        self.add_nnefop(nnefop, tfop)

        for tfdn_result, nnefdn_result in zip(tfop.results.values(), nnefop.results.values()):
            output_rank = self.get_rank_safe(tfdn_result)

            nnefop_transpose = dog.OperationNode("transpose")
            nnefop_transpose.add_arg("input", nnefdn_result)
            nnefop_transpose.add_arg("axes", utils.transpose_axes_nchw_to_hwcn(output_rank))
            nnefop_transpose.add_result('output', self.make_nnefdn(None, "_hwcn", source_name=None))
            nnefop_transpose.extra[nnef_shape_optimizer.EXTRA_GENERATED_TRANSPOSE] = True
            self.add_nnefop(nnefop_transpose, tfop)

            nnefop_reshape = dog.OperationNode("reshape")
            nnefop_reshape.add_arg("input", nnefop_transpose.result)
            nnefop_reshape.add_arg("shape", tf_output_shape)
            nnefop_reshape.add_result("output", self.make_nnefdn(tfdn_result, "_reshape", source_name=None))
            nnefop_reshape.extra[nnef_shape_optimizer.EXTRA_GENERATED_RESHAPE] = True
            self.add_nnefop(nnefop_reshape, tfop)

    # TODO rename
    @staticmethod
    def is_broadcast(tfdn_broadcasted, tfdn_other, accept_rank_0_or_1=False):
        return (isinstance(tfdn_broadcasted, dog.DataNode)
                and isinstance(tfdn_other, dog.DataNode)
                and tfdn_broadcasted.shape is not None
                and (tfdn_broadcasted.shape != [] or accept_rank_0_or_1)
                and (tfdn_broadcasted.shape != [1] or accept_rank_0_or_1)
                and tfdn_other.shape is not None
                and len(tfdn_other.shape) > len(tfdn_broadcasted.shape)
                and utils.can_broadcast_from_right(tfdn_other.shape, tfdn_broadcasted.shape))

    def print_trafos(self, keys=None, file=None):
        nnef_shape_optimizer.print_trafos(self.trafo_dict, keys, file)


# TODO remove (use convert)
def export_network(
        net_func,
        checkpoint=None,
        custom_converters=None,
        custom_fragments='',
        output_path=None,
        compress=False,
        verbose=True,
        hooks=None,
        add_comments=False,
):
    """Exports a TensorFlow network to NNEF

    Args:
        net_func: A function which defines a TensorFlow network and returns its output nodes
    """

    converter = TF2NNEFConverter()

    converter.export_network(
        net_func,
        checkpoint,
        custom_converters,
        custom_fragments,
        output_path,
        compress,
        verbose,
        hooks,
        add_comments,
    )

    utils.raise_if_had_error()
    return converter


def export_activations(converter_or_net_func, checkpoint, feed_dict, custom_converters=None, custom_fragments=None,
                       output_path=None, compress=None, verbose=True,
                       hooks=None, add_comments=None,
                       evaluate_count_per_iter=25, nnefname_by_tf=None, input_output_only=False):
    if isinstance(converter_or_net_func, TF2NNEFConverter):
        converter = converter_or_net_func
        message = "If converter is given, its parameters should not be give in export_activations"
        assert custom_converters is None, message
        assert custom_fragments is None, message
        assert compress is None, message
        assert hooks is None, message
        assert add_comments is None, message
    else:
        if custom_fragments is None:
            custom_fragments = ""
        if compress is None:
            compress = False
        if add_comments is None:
            add_comments = False

        converter = TF2NNEFConverter()
        converter.export_network(
            converter_or_net_func,
            checkpoint=checkpoint,
            custom_converters=custom_converters,
            custom_fragments=custom_fragments,
            output_path=output_path,
            compress=compress,
            verbose=verbose,
            hooks=hooks,
            add_comments=add_comments,
        )

    path = output_path if output_path is not None else os.path.splitext(checkpoint)[0] + '-activations'
    if not os.path.exists(path):
        os.makedirs(path)

    if verbose:
        utils.print_info('Evaluating activations..')

    def get_nnef_name(nnefname, tfname):
        if nnefname_by_tf is None:
            return nnefname
        else:
            return nnefname_by_tf.get(tfname)

    tensors_names_nnefdns = [
        (
            converter.tftensor_by_name[nnefdn.source_name],
            get_nnef_name(nnefdn.name, nnefdn.source_name),
            nnefdn
        )

        for nnefdn in converter.nnefdog.dn_by_name.values()
        if nnefdn.source_name and get_nnef_name(nnefdn.name, nnefdn.source_name) and nnefdn.producer.name != "variable"
    ]

    if input_output_only:
        input_and_output_names = set(converter.nnefdog.input_dn_names + converter.nnefdog.output_dn_names)
        tensors_names_nnefdns = [
            (t, n, dn)
            for t, n, dn in tensors_names_nnefdns
            if dn.name in input_and_output_names
        ]

    nnefdn_by_name = {name: nnefdn for _tensor, name, nnefdn in tensors_names_nnefdns}

    with tf.Session() as sess:
        saver = None
        if checkpoint is not None:
            saver = tf.train.Saver()

        graph = tf.get_default_graph()

        total = 0
        for tensor, name, _nnefdn in tensors_names_nnefdns:
            if graph.is_fetchable(tensor) and isinstance(tensor, tf.Tensor):
                total += 1

        next_ = 0
        evaluated = 0
        while next_ < len(tensors_names_nnefdns):
            tensor_by_name = {}
            while next_ < len(tensors_names_nnefdns) and len(tensor_by_name) < evaluate_count_per_iter:
                tensor, name, _nnefdn = tensors_names_nnefdns[next_]
                if graph.is_fetchable(tensor) and isinstance(tensor, tf.Tensor):
                    tensor_by_name[name] = tensor
                    evaluated += 1
                next_ += 1

            if checkpoint is not None:
                saver.restore(sess, checkpoint)

            values = sess.run(tensor_by_name, feed_dict)

            if verbose:
                utils.print_info("Evaluated {}/{}".format(evaluated, total))

            for name, arr in values.items():
                nnefdn = nnefdn_by_name[name]
                trafo = nnefdn.extra.get(nnef_shape_optimizer.EXTRA_APPLIED_TRANSFORMATIONS)
                filename = path + '/' + name + '.dat'
                if utils.is_np_dtype_exportable_to_nnef(arr.dtype):
                    if np.isnan(arr).any():
                        utils.print_error("Activation export error: {} has nan's".format(name))
                    elif not np.isfinite(arr).all():
                        utils.print_error("Activation export error: {} has inf's".format(name))
                    utils.write_nnef_tensor(filename, utils.np_apply_transforms(arr, trafo))
                else:
                    utils.print_error(
                        "Could not export '{}', unsupported dtype '{}'".format(filename, arr.dtype.name))

    utils.raise_if_had_error()


def convert(network_function,  # type: Callable[(), Dict[str, tf.Tensor]]
            checkpoint_path=None,  # type: Optional[str]
            output_path=".",  # type: str
            verbose=False,  # type: bool
            _compress=True,  # type: bool
            _print_trafos=True,  # type: bool
            ):
    # type: (...)->TF2NNEFConverter

    tmp_dir_name = None
    try:
        if _compress:
            tmp_dir_name = tempfile.mkdtemp(prefix="tf_to_nnef_")

        if verbose:
            print("Converting...")

        converter = TF2NNEFConverter()
        converter.export_network(net_func=network_function,
                                 checkpoint=checkpoint_path,
                                 output_path=tmp_dir_name if _compress else output_path,
                                 verbose=verbose,
                                 add_comments=verbose)

        if _print_trafos:
            converter.print_trafos(["input", "output"])

        if _compress:
            if verbose:
                print("Compressing...")
            utils.ensure_dir(output_path)
            tgz_file_name = os.path.join(output_path, network_function.__name__ + ".nnef.tgz")
            utils.tgz_compress(tmp_dir_name, tgz_file_name)
            if verbose:
                print("Wrote {}".format(tgz_file_name))
        else:
            if verbose:
                print("Wrote {}/*".format(output_path))

        utils.raise_if_had_error()
        return converter

    finally:
        if tmp_dir_name:
            shutil.rmtree(tmp_dir_name)


def override_dynamic_shapes_with_static():
    import tensorflow as tf
    from tensorflow.python.ops import gen_array_ops

    def rank(input, name=None):
        return len(input.shape)

    def shape(input, out_type=tf.int32, name=None):
        return input.shape

    def shape_n(input, out_type=tf.int32, name=None):
        return [t.shape for t in input]

    gen_array_ops.rank = rank
    gen_array_ops.shape = shape
    gen_array_ops.shape_n = shape_n


def _nnef_get_shape(args, shapes, arg_name):
    return shapes[args[arg_name]] if isinstance(args[arg_name], nnef.Identifier) else []


def _nnef_broadcasted_shape(shape1, shape2):
    rank_diff = len(shape2) - len(shape1)

    if rank_diff > 0:
        shape1 += [1] * rank_diff
    else:
        shape2 += [1] * -rank_diff

    shape = [1] * len(shape1)

    assert len(shape) == len(shape1) == len(shape2)

    for i, (s, t) in enumerate(zip(shape1, shape2)):
        assert s == t or s == 1 or t == 1, \
            "Broadcast can only happen when the corresponding dimesions are either equal or one of them is 1"
        if s != 1:
            shape[i] = s
        elif t != 1:
            shape[i] = t

    return shape


def propagate_shape_unary(proto, args, shapes):
    shapes[args["y"]] = _nnef_get_shape(args, shapes, "x")


def propagate_shape_binary(proto, args, shapes):
    args["z"] = _nnef_broadcasted_shape(_nnef_get_shape(args, shapes, "x"), _nnef_get_shape(args, shapes, "y"))
