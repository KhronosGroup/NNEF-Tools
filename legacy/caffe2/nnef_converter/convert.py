# Copyright (c) 2018 The Khronos Group Inc.
# Copyright (c) 2018 Au-Zone Technologies Inc.
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

import argparse
import logging
import logging.config
import shutil
import sys

from os import path, remove, makedirs

from .common.converter import Converter

def set_log_output(log_file, level):

    log_dir = path.dirname(log_file)
    if not path.exists(log_dir) and log_dir != '':
        makedirs(log_dir)

    if path.isfile(log_file):
        remove(log_file)

    logger = logging.getLogger('nnef_convert')
    hdlr = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    if level.lower() == "debug":
        logger.setLevel(logging.DEBUG)
    elif level.lower() == "info":
        logger.setLevel(logging.INFO)
    elif level.lower() == "warning":
        logger.setLevel(logging.WARNING)
    elif level.lower() == "error":
        logger.setLevel(logging.ERROR)
    elif level.lower() == "critical":
        logger.setLevel(logging.CRITICAL)
    logger.info('NNEF CONVERTER')
    return logger

def main_loop():
    options = argparse.ArgumentParser(description='NNEF Model Converter Tool.')
    options.add_argument('--input-framework',
                         choices=['tensorflow', 'caffe2', 'NNEF'],
                         help='Framework of input model.')
    options.add_argument('--output-framework',
                         choices=['tensorflow', 'caffe2', 'NNEF'],
                         help='Framework of output model.')
    options.add_argument('--input-model',
                         help='Network model input file')
    options.add_argument('--data-model',
                         help='Network data input file (Caffe2)')
    options.add_argument('--value-info',
                         help='Information regarding input sizes (Caffe2)')
    options.add_argument('--output-model',
                         default='',
                         help='Network model output file')
    options.add_argument('--input-nodes', help='Network model input nodes')
    options.add_argument('--output-nodes', help='Network model output nodes')
    options.add_argument('--compress',
                         default='false',
                         choices=['true', 'false'],
                         help='If the model should be compressed')
    options.add_argument('--log-level',
                         default='info',
                         choices=['off', 'debug', 'info', 'warning', 'error', 'critical'],
                         help='Logging output level.')
    try:
        args = options.parse_args()
    except Exception as err:
        print('Argument Error: {}'.format(err))
        return

    if args.output_model == '':
        if args.output_framework == 'tensorflow':
            output_dir = 'tf_output'
        elif args.output_framework == 'caffe2':
            output_dir = 'caffe2_output'
        elif args.output_framework == 'NNEF':
            output_dir = 'nnef_output'
    else:
        output_dir = path.dirname(args.output_model)

    invalid_dirs = ['caffe2', 'common', 'documentation', 'google', 'nnef_tools',
                    'tensorflow', 'visualization_tools']
    for dir_name in invalid_dirs:
        if dir_name == output_dir:
            raise Exception("Invalid output directory, please use one that does not contain: " + str(invalid_dirs))

    if path.exists(output_dir) and output_dir not in args.input_model:
        shutil.rmtree(output_dir)

    if args.log_level != 'off' and args.input_framework != 'NNEF':
        log_file = path.normpath(path.join(output_dir, 'nnef_convert.log'))
        print("Set output log: ", log_file)
        logger = set_log_output(log_file, args.log_level)

    if args.input_framework == 'tensorflow':
        if args.input_model is None:
            raise ValueError("Need to provide 'input-model' for tensorflow importer.")
        from .tensorflow.tf_converter import TensorflowImporter
        importer = TensorflowImporter(args.input_model, args.input_nodes, args.output_nodes, args.log_level)

    elif args.input_framework == 'caffe2':
        if args.input_model is None:
            raise ValueError("Need to provide 'input-model' for caffe2 importer.")
        if args.data_model is None:
            raise ValueError("Need to provide 'data-model' for caffe2 importer.")
        if args.value_info is None:
            raise ValueError("Need to provide 'value-info' for caffe2 importer.")
        from .caffe2.caffe2_converter import Caffe2Importer
        importer = Caffe2Importer(args.input_model, args.data_model, args.value_info,
                                  args.output_nodes, args.log_level)
    elif args.input_framework == 'NNEF':
        from .common.nnef_converter import NNEFImporter
        importer = NNEFImporter(args.input_model)
    else:
        raise ValueError("Unknown input framework [{}].".format(args.input_framework))

    if args.output_framework == 'tensorflow':
        from .tensorflow.tf_converter import TensorflowExporter
        output_model = args.output_model if args.output_model != '' else path.join('tf_output', 'tf.pb')
        exporter = TensorflowExporter(output_model)
    elif args.output_framework == 'caffe2':
        from .caffe2.caffe2_converter import Caffe2Exporter
        output_model = args.output_model if args.output_model != '' else path.join('caffe2_output', 'net.pb')
        exporter = Caffe2Exporter(output_model)
    elif args.output_framework == 'NNEF':
        from .common.nnef_converter import NNEFExporter
        output_model = args.output_model if args.output_model != '' else path.join('nnef_output', 'graph.nnef')
        exporter = NNEFExporter(output_model)
    else:
        raise ValueError("Unknown output framework [{}].".format(args.output_framework))

    if args.log_level != 'off' and args.input_framework != 'NNEF':
        logger.info("Converting the %s model %s --> %s model %s" % (args.input_framework, args.input_model,
                                                                    args.output_framework, output_model))
        logger.info("\tInput(s) used: %s" % args.input_nodes)
        logger.info("\tOutput(s) used: %s" % args.output_nodes)

    converter = Converter(importer, exporter)
    converter.run()

    if args.compress == 'true' and args.output_framework == 'NNEF':
        print("Compressing output folder...")
        shutil.make_archive('nnef_output', 'zip', output_dir)
        shutil.move('nnef_output.zip', output_dir + '/nnef_output.zip')

    print("Done converting model.")
    return

if __name__ == '__main__':
    try:
        main_loop()
    except Exception as err:
        print('Error: {}'.format(err))
        sys.stdout.flush()
