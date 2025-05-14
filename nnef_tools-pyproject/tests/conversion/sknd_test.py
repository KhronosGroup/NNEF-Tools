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

import unittest
import numpy as np
import skriptnd as sknd


class TestEnv(unittest.TestCase):

    def _convert_to_sknd(self, filename, input_shape=None):
        raise NotImplementedError()

    @staticmethod
    def _exec_orig_model(filename, input_shape=None, input_range=None):
        raise NotImplementedError()

    @staticmethod
    def _exec_sknd_model(path, input_shape=None, input_range=None):
        np.random.seed(0)

        model = sknd.read_model(path)
        if not model:
            return None

        compiled_model = sknd.compile_model(model, keep_generated_code=False)

        if not isinstance(input_shape, list):
            input_shape = [input_shape] * len(model.graphs[0].inputs)
        if not isinstance(input_range, list):
            input_range = [input_range] * len(model.graphs[0].inputs)

        inputs = [TestEnv._random_data(sknd.DtypeToNumpy[input.dtype],
                                       input_shape[idx] or input.shape,
                                       input_range[idx])
                  for idx, input in enumerate(model.graphs[0].inputs)]

        return compiled_model(*inputs)

    @staticmethod
    def _compile_sknd_model(path):
        model = sknd.read_model(path)
        if not model:
            return None

        return sknd.compile_model(model)

    @staticmethod
    def _random_data(dtype, shape, range=None):
        if dtype == np.bool or dtype == np.bool_:
            return np.array(np.random.random(shape) > 0.5)
        elif dtype == np.float32 or dtype == np.float64:
            data = np.random.random(shape).astype(dtype)
            if range:
                lo, hi = range
                data *= hi - lo
                data += lo
            return data
        else:
            lo, hi = range if range else (0, 100)
            return np.random.randint(low=lo, high=hi, size=shape, dtype=dtype)

    def _test_conversion_from_file(self, filename, epsilon=1e-5, input_shape=None, input_range=None):
        self._convert_to_sknd(filename, input_shape=input_shape)

        if not self._execute:
            assert self._compile_sknd_model(filename + '.nnef2') is not None
            return

        original_outputs = self._exec_orig_model(filename, input_shape=input_shape, input_range=input_range)
        converted_outputs = self._exec_sknd_model(filename + '.nnef2', input_shape=input_shape, input_range=input_range)

        self.assertTrue(original_outputs is not None)
        self.assertTrue(converted_outputs is not None)

        self.assertEqual(len(original_outputs), len(converted_outputs))
        for idx, (original, converted) in enumerate(zip(original_outputs, converted_outputs)):
            self.assertEqual(original.shape, converted.shape)
            if all(s != 0 for s in original.shape):
                if original.dtype != np.float32 and original.dtype != np.float64:
                    self.assertTrue(np.all(original == converted))
                else:
                    max = np.max(np.abs(original))
                    diff = np.max(np.abs(original - converted))
                    print("Max absolute difference for output #{}: {}".format(idx, diff))
                    if max != 0:
                        print("Max relative difference for output #{}: {}".format(idx, diff / max))
                    self.assertLess(diff / max if max != 0 else diff, epsilon)
