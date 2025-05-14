# Copyright (c) 2017-2025 The Khronos Group Inc.
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

import skriptnd as sknd
import numpy as np
import sys
import os


BASE_FOLDER = os.path.normpath(os.path.join(os.path.dirname(__file__), 'skriptnd/test/'))


def roundtrip_test():
    print('Reading model...')
    model = sknd.read_model(BASE_FOLDER + '/test.sknd')
    if not model:
        exit(-1)

    print('Writing model back...')
    sknd.print_model(model, file=sys.stdout)
    sknd.write_model(model, BASE_FOLDER + '/test_round_trip.sknd')
    print('Reading model again...')
    sknd.read_model(BASE_FOLDER + '/test_round_trip.sknd')


def compile_test():
    print('Reading model...')
    model = sknd.read_model(BASE_FOLDER + '/test.sknd')
    if not model:
        exit(-1)

    print("Compiling model '{}'...".format(model.name))
    compiled = sknd.compile_model(model, keep_generated_code=True)
    print("Compiled model '{}'".format(model.name))
    print("Inputs:", compiled.input_info())
    print("Outputs:", compiled.output_info())


def execution_test():
    print('Reading model...')
    model = sknd.read_model(BASE_FOLDER + '/../../alexnet.sknd')
    if not model:
        exit(-1)

    print("Compiling model '{}'...".format(model.name))
    compiled = sknd.compile_model(model)

    print("Executing model '{}'...".format(model.name))
    input = np.random.random((1,3,224,224)).astype(np.float32)
    output, = compiled(input)
    print("Done", output.dtype, output.shape)


roundtrip_test()
