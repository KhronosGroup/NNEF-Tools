#!/usr/bin/env python

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

import sys

# Python2: Ensure that load from current directory is enabled, but load from the directory of the script is disabled
if len(sys.path) == 0:
    sys.path.append('')
if sys.path[0] != '':
    sys.path[0] = ''

import argparse
import os
import importlib
import tempfile
import shutil
import nnef

import numpy as np

from nnef_tools.io.input_source import InputSources
from nnef_tools.core import utils
from nnef_tools.io.nnef import nnef_io
from nnef_tools.io.nnef.parser_config import NNEFParserConfig
from nnef_tools.io.nnef.nnef_graph import *
from nnef_tools.backend.pytorch import runner


def ensure_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_args(argv):
    parser = argparse.ArgumentParser(description="NNEFTools/run: NNEF runner tool",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog="""Tips:
- If you refer to a Python package or module that is not in the current directory,
please add its location to PYTHONPATH.
- Quote parameters if they contain spaces or special characters.
""")

    parser.add_argument("--input-model",
                        required=True,
                        help="""Path of NNEF file, directory or archive""")

    parser.add_argument('--input',
                        default=None,
                        help="""An input_source or a tensor_name->input_source dictionary in Python syntax.
The dictionary can also have "*" or a tuple of names as key, "*" meaning "for all other". 
The following input sources are supported:
- Random(algo, *args) for all types:
    - Random('uniform', min, max) for int and float, range: [min, max]
    - Random('normal', mean, std) for float
    - Random('binomial', num, true_prob) for int, range: [0, num]
    - Random('bernoulli', true_prob) for bool
    Keyword arguments can not be used with Random.
- Image(filename, color_format='RGB', data_format='NCHW', range=None, norm=None) for int and float:
  Arguments:
    - filename: string or list of strings, path(s) of jpg/png images, can have * in them
    - color_format: RGB or BGR
    - data_format: NCHW or NHWC
    - range: [start, end] closed range
    - norm: [mean, std] or [[mean0, mean1, mean2], [std0, std1, std2]]
  The image is processed as follows:
    - The image is loaded to float32[height, width, channels], values ranging from 0 to 255.
    - The image is reordered to RGB or BGR, as requested.
    - If range is not None, the image is transformed to the specified range. 
      (The transform does not depend on the content of the image.)
    - If norm is not None: image = (image - mean) / std
    - The image is transformed to NCHW or NHWC as requested. 
    - The image is casted to the target data type.
- Tensor(filename) for all types:
  Filename must be the path of an NNEF tensor file (.dat).
The tokens [RGB, BGR, NCHW, NHWC, uniform, normal, binomial, bernoulli] can be written without quotes.
Default: 
  - float: Random('normal', 0.0, 1.0) 
  - int: Random('binomial', 255, 0.5) 
  - bool: Random('bernoulli', 0.5)""")

    parser.add_argument('--input-shape',
                        help="""Use this to override input shapes.

    Set all input shapes to [10, 224, 224, 3]:
    --input-shape="[10, 224, 224, 3]"

    Set different input shapes for each input:
    --input-shape="{'input_name_1': [10, 224, 224, 3], 'input_name_2': [10, 299, 299, 3]}" """)

    parser.add_argument('--stats', action="store_true", help="Save statistics used for quantization")

    parser.add_argument("--stats-path",
                        required=False,
                        default='graph.stats',
                        help="""Path of stats file, relative to the directory of the NNEF model. 
Default: graph.stats (inside model directory/archive)""")

    parser.add_argument("--activations",
                        nargs='*',
                        help="""Set this to dump activations.
--activations: Dump all activations. 
--activations tensor_1 tensor2: Dump activations for the listed tensors.""")

    parser.add_argument("--activations-path",
                        required=False,
                        default='.',
                        help="""Directory of activations, relative to the directory of the NNEF model. 
Default: . (inside model directory/archive)""")

    parser.add_argument("--permissive",
                        action="store_true",
                        help="""Allow some imprecise evaluations""")

    parser.add_argument("--resize",
                        action="store_true",
                        help="""Try to make the network batch-size agnostic.""")

    parser.add_argument("--device",
                        required=False,
                        help="""Set device: cpu, cuda, cuda:0, cuda:1, etc.
Default: cuda if available, cpu otherwise.""")

    parser.add_argument('--custom-operations',
                        nargs='*',
                        help="""Custom modules: e.g. package.module1 package.module2""")

    parser.add_argument("--generate-weights",
                        nargs='?',
                        default=False,  # don't generate
                        const=None,  # generate with default options
                        help="""Generate and write random weight tensors. 
Existing weights are not overwritten.
Given my_network.nnef (or my_network.txt), it generates my_network.nnef.tgz.
Optionally an input_source or a tensor_name->input_source dictionary can be provided.
  - Please note that tensor *names* must be used, not labels.
  - For the syntax and the defaults, see: --input.
""")

    parser.add_argument("--random-seed",
                        required=False,
                        default=-1,
                        type=int,
                        help="""Random seed to use in random generation. 
Default: -1 (Get the seed from /dev/urandom or the clock)""")

    args = parser.parse_args(args=argv[1:])
    args.input = InputSources(args.input)
    if args.generate_weights is not False:
        args.generate_weights = InputSources(args.generate_weights)

    has_weights = not (os.path.isfile(args.input_model) and not args.input_model.endswith('.tgz'))
    if not has_weights and args.generate_weights is False:
        print("Error: Seems like you have specified an NNEF file without weights. "
              "Specify the whole directory/tgz or use --generate-weights.")
        exit(1)

    return args


def get_custom_runners(module_names):
    if not module_names:
        return {}
    runners = {}
    for module_name in module_names:
        module = importlib.import_module(module_name)
        if hasattr(module, "TORCH_BACKEND_OPERATIONS"):
            runners.update(module.TORCH_BACKEND_OPERATIONS)
    return runners


def generate_weights(g, nnef_path, output_dir, input_sources):
    # type: (NNEFGraph, str, str, InputSources)->bool
    did_write = False
    warned = False
    for tensor in g.tensors:
        if tensor.is_variable:
            dat_path = os.path.join(output_dir, tensor.label + '.dat')
            if os.path.exists(dat_path):
                if not warned:
                    print("Warning: leaving existing weights unchanged".format(tensor.name, tensor.label))
                    warned = True
            else:
                array = input_sources.create_input(tensor.name, np_dtype=tensor.get_numpy_dtype(), shape=tensor.shape)
                nnef_io.write_nnef_tensor(dat_path, array)
                did_write = True
    if not os.path.isdir(nnef_path):
        shutil.copy(nnef_path, os.path.join(output_dir, 'graph.nnef'))
        did_write = True
    return did_write


def _is_inside(dir, rel_path):
    prefix = utils.path_without_trailing_separator(os.path.abspath(dir)) + os.sep
    str = utils.path_without_trailing_separator(os.path.abspath(os.path.join(dir, rel_path))) + os.sep
    return str.startswith(prefix)


def _evaluate_input_shape(s):
    try:
        if not s:
            return None
        else:
            return eval(s)
    except Exception:
        print("Error: Can not evaluate --input-shape {}".format(s), file=sys.stderr)
        exit(1)


def run_using_argv(argv):
    try:
        args = get_args(argv)
        if args.random_seed != -1:
            np.random.seed(args.random_seed)
        parent_dir_of_input_model = os.path.dirname(utils.path_without_trailing_separator(args.input_model))
        tmp_dir = None
        if args.input_model.endswith('.tgz'):
            nnef_path = tmp_dir = tempfile.mkdtemp(prefix="nnef_", dir=parent_dir_of_input_model)
            utils.tgz_extract(args.input_model, nnef_path)
        else:
            nnef_path = args.input_model
        try:
            parser_configs = NNEFParserConfig.load_configs(args.custom_operations, load_standard=True)
            reader = nnef_io.Reader(parser_configs=parser_configs, input_shape=_evaluate_input_shape(args.input_shape))

            did_generate_weights = False
            if args.generate_weights is not False:
                # read without weights
                graph = reader(os.path.join(nnef_path, 'graph.nnef') if os.path.isdir(nnef_path) else nnef_path)
                if os.path.isdir(nnef_path):
                    output_path = nnef_path
                elif nnef_path.endswith('.nnef') or nnef_path.endswith('.txt'):
                    output_path = tmp_dir = tempfile.mkdtemp(prefix="nnef_", dir=parent_dir_of_input_model)
                else:
                    assert False
                did_generate_weights = generate_weights(graph,
                                                        nnef_path,
                                                        output_path,
                                                        input_sources=args.generate_weights)
                nnef_path = output_path

            graph = reader(nnef_path)

            inputs = tuple(args.input.create_input(name=input.name,
                                                   np_dtype=input.get_numpy_dtype(),
                                                   shape=input.shape,
                                                   allow_bigger_batch=True) for input in graph.inputs)

            tensor_hooks = []

            if args.activations is None:
                pass
            elif not args.activations:
                tensor_hooks.append(
                    runner.ActivationExportHook(tensor_names=[t.name
                                                              for t in graph.tensors
                                                              if not t.is_constant and not t.is_variable],
                                                output_directory=os.path.join(nnef_path, args.activations_path)))
            else:
                tensor_hooks.append(
                    runner.ActivationExportHook(tensor_names=args.activations,
                                                output_directory=os.path.join(nnef_path, args.activations_path)))

            stats_hook = None
            if args.stats:
                stats_hook = runner.StatisticsHook()
                tensor_hooks.append(stats_hook)

            if args.permissive:
                runner.try_to_fix_unsupported_attributes(graph)
            
            runner.run(nnef_graph=graph,
                       inputs=inputs,
                       device=args.device,
                       custom_operations=get_custom_runners(args.custom_operations),
                       fix_batch_size=args.resize,
                       tensor_hooks=tensor_hooks)

            if stats_hook:
                if args.stats_path.endswith('/') or args.stats_path.endswith('\\'):
                    stats_path = os.path.join(nnef_path, args.stats_path, 'graph.stats')
                else:
                    stats_path = os.path.join(nnef_path, args.stats_path)
                stats_hook.save_statistics(stats_path)

            if tmp_dir and (did_generate_weights
                            or (args.stats and _is_inside(nnef_path, args.stats_path))
                            or (args.activations is not None and _is_inside(nnef_path, args.activations_path))):
                if args.input_model.endswith('.tgz'):
                    print("Info: Changing input archive")
                    shutil.move(args.input_model, args.input_model + '.nnef-tools-backup')
                    utils.tgz_compress(dir_path=nnef_path, file_path=args.input_model)
                    os.remove(args.input_model + '.nnef-tools-backup')
                else:
                    output_path = args.input_model.rsplit('.', 1)[0] + '.nnef.tgz'
                    backup_path = output_path + '.nnef-tools-backup'
                    if os.path.exists(output_path):
                        shutil.move(output_path, backup_path)
                    utils.tgz_compress(dir_path=nnef_path, file_path=output_path)
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
        finally:
            if tmp_dir:
                shutil.rmtree(tmp_dir)
    except utils.NNEFToolsException as e:
        print("Error: " + str(e), file=sys.stderr)
        exit(1)
    except nnef.Error as e:
        print("Error: " + str(e), file=sys.stderr)
        exit(1)


# Call this if you don't want to reload the whole program for each run
def run_using_command(command):
    return run_using_argv(utils.command_to_argv(command))


if __name__ == '__main__':
    run_using_argv(sys.argv)
