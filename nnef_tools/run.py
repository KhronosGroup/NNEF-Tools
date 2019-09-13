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

from nnef_tools.core import utils
from nnef_tools.io.nnef import nnef_io
from nnef_tools.io.nnef.parser_config import NNEFParserConfig
import nnef_tools.backend.pytorch as backend


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

    parser.add_argument("network",
                        help="Path of NNEF file, directory or archive")

    parser.add_argument('--input',
                        required=False,
                        nargs='+',
                        help="Path of input tensors.\n"
                             "By default they are read from the standard input through a pipe or redirect.")

    parser.add_argument('--output',
                        required=False,
                        help="The path of the output directory.\n"
                             "By default the standard output is used, but only if the command is piped or redirected.")

    parser.add_argument("--output-names",
                        nargs='*',
                        help="""List tensor names to ensure that those tensors are exported.
If this option is not specified the graph's output tensors are exported. 
--output-names: Export nothing
--output-names a b c: Export the tensors a, b and c
--output-names '*': Export all activation tensors
 """)

    parser.add_argument("--stats",
                        nargs='?',
                        default=None,
                        const='graph.stats',
                        help="""Set this to export statistics. 
If a path is given it must be relative to the directory of the NNEF model.
--stats: Write stats to graph.stats  
--stats stats.txt: Write stats to stats.txt""")

    parser.add_argument("--permissive",
                        action="store_true",
                        help="""Allow some imprecise evaluations""")

    parser.add_argument("--device",
                        required=False,
                        help="""Set device: cpu, cuda, cuda:0, cuda:1, etc.
Default: cuda if available, cpu otherwise.""")

    parser.add_argument('--custom-operations',
                        nargs='*',
                        help="""Custom modules: e.g. package.module1 package.module2""")

    args = parser.parse_args(args=argv[1:])

    has_weights = not (os.path.isfile(args.network) and not args.network.endswith('.tgz'))
    if not has_weights:
        raise utils.NNEFToolsException("Error: Seems like you have specified an NNEF file without weights. "
                                       "Please use generate_weights.py")

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


def _is_inside(dir, rel_path):
    prefix = utils.path_without_trailing_separator(os.path.abspath(dir)) + os.sep
    str = utils.path_without_trailing_separator(os.path.abspath(os.path.join(dir, rel_path))) + os.sep
    return str.startswith(prefix)


def run_using_argv(argv):
    try:
        args = get_args(argv)
        write_outputs = args.output_names is None or args.output_names

        if args.input is None:
            if sys.stdin.isatty():
                raise utils.NNEFToolsException("No input provided!")
            utils.set_stdin_to_binary()

        if write_outputs:
            if args.output is None:
                if sys.stdout.isatty():
                    raise utils.NNEFToolsException("No output provided!")
                utils.set_stdout_to_binary()

        parent_dir_of_input_model = os.path.dirname(utils.path_without_trailing_separator(args.network))
        tmp_dir = None

        if args.network.endswith('.tgz'):
            nnef_path = tmp_dir = tempfile.mkdtemp(prefix="nnef_", dir=parent_dir_of_input_model)
            utils.tgz_extract(args.network, nnef_path)
        else:
            nnef_path = args.network

        try:
            parser_configs = NNEFParserConfig.load_configs(args.custom_operations, load_standard=True)

            if args.input is None:
                reader = nnef_io.Reader(parser_configs=parser_configs)
                # read without weights
                graph = reader(os.path.join(nnef_path, 'graph.nnef') if os.path.isdir(nnef_path) else nnef_path)
                input_count = len(graph.inputs)

                inputs = tuple(nnef.read_tensor(sys.stdin)[0] for _ in range(input_count))
            else:
                inputs = tuple(nnef_io.read_nnef_tensor(path) for path in args.input)

            reader = nnef_io.Reader(parser_configs=parser_configs,
                                    input_shape=tuple(list(input.shape) for input in inputs))

            graph = reader(nnef_path)

            tensor_hooks = []

            stats_hook = None
            if args.stats:
                stats_hook = backend.StatisticsHook()
                tensor_hooks.append(stats_hook)

            if write_outputs and args.output_names is not None:
                if '*' in args.output_names:
                    tensor_hooks.append(backend.ActivationExportHook(
                        tensor_names=[t.name
                                      for t in graph.tensors
                                      if not t.is_constant and not t.is_variable],
                        output_directory=args.output))
                else:
                    tensor_hooks.append(backend.ActivationExportHook(
                        tensor_names=args.output_names,
                        output_directory=args.output))

            if args.permissive:
                backend.try_to_fix_unsupported_attributes(graph)

            outputs = backend.run(nnef_graph=graph,
                                  inputs=inputs,
                                  device=args.device,
                                  custom_operations=get_custom_runners(args.custom_operations),
                                  tensor_hooks=tensor_hooks)

            if write_outputs and args.output_names is None:
                if args.output is None:
                    for array in outputs:
                        nnef.write_tensor(sys.stdout, array)
                else:
                    for tensor, array in zip(graph.outputs, outputs):
                        nnef_io.write_nnef_tensor(os.path.join(nnef_path, args.output, tensor.name + '.dat'), array)

            if stats_hook:
                if args.stats.endswith('/') or args.stats.endswith('\\'):
                    stats_path = os.path.join(nnef_path, args.stats, 'graph.stats')
                else:
                    stats_path = os.path.join(nnef_path, args.stats)
                stats_hook.save_statistics(stats_path)

            if tmp_dir and (args.stats and _is_inside(nnef_path, args.stats)):
                if args.network.endswith('.tgz'):
                    print("Info: Changing input archive", file=sys.stderr)
                    shutil.move(args.network, args.network + '.nnef-tools-backup')
                    utils.tgz_compress(dir_path=nnef_path, file_path=args.network)
                    os.remove(args.network + '.nnef-tools-backup')
                else:
                    output_path = args.network.rsplit('.', 1)[0] + '.nnef.tgz'
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


if __name__ == '__main__':
    run_using_argv(sys.argv)
