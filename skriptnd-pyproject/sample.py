import skriptnd as sknd
import numpy as np
import sys
import os


BASE_FOLDER = os.path.normpath(os.path.join(os.path.dirname(__file__), 'skriptnd/test/'))


def roundtrip_test():
    print('Reading model...')
    model = sknd.read_model(BASE_FOLDER + '/test.nds')
    if not model:
        exit(-1)

    print('Writing model back...')
    sknd.print_model(model, file=sys.stdout)
    sknd.write_model(model, BASE_FOLDER + '/test_round_trip.nds')
    print('Reading model again...')
    sknd.read_model(BASE_FOLDER + '/test_round_trip.nds')


def compile_test():
    print('Reading model...')
    model = sknd.read_model(BASE_FOLDER + '/test.nds')
    if not model:
        exit(-1)

    print("Compiling model '{}'...".format(model.name))
    compiled = sknd.compile_model(model, keep_generated_code=True)
    print("Compiled model '{}'".format(model.name))
    print("Inputs:", compiled.input_info())
    print("Outputs:", compiled.output_info())


def execution_test():
    print('Reading model...')
    model = sknd.read_model(BASE_FOLDER + '/../../alexnet.nds')
    if not model:
        exit(-1)

    print("Compiling model '{}'...".format(model.name))
    compiled = sknd.compile_model(model)

    print("Executing model '{}'...".format(model.name))
    input = np.random.random((1,3,224,224)).astype(np.float32)
    output, = compiled(input)
    print("Done", output.dtype, output.shape)


roundtrip_test()
