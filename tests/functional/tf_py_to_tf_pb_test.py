from __future__ import division, print_function, absolute_import

import unittest

from nnef_tools.conversion.tensorflow import tf_py_to_tf_pb
from nnef_tools.io.tensorflow.tf_graph import *


class TestTFPyToTFPb(unittest.TestCase):
    def assertChain(self, *ops):
        for i in range(0, len(ops) - 1):
            a = ops[i]
            b = ops[i + 1]
            self.assertIs(a.output, b.input)

    def test_expand_softmax1(self):
        g = TFGraph(name="g")
        input = TFTensor(graph=g, name="input", shape=[10, 1000], dtype="float32")
        softmax = TFOperation(graph=g,
                              name="tf.nn.softmax",
                              inputs=input,
                              outputs=TFTensor(graph=g, name="softmax1", shape=[10, 1000], dtype="float32"))
        g.inputs = (input,)
        g.outputs = (softmax.output,)

        tf_py_to_tf_pb.expand_softmax(tf_graph=g, tf_op=softmax)

        self.assertEqual(1, len(g.operations))
        softmax = g.operations[0]
        self.assertEqual("tf.nn.softmax", softmax.name)
        self.assertTrue('axis' in softmax.attribs)
        self.assertEqual(-1, softmax.attribs['axis'])
        self.assertIs(g.inputs[0], softmax.input)
        self.assertIs(g.outputs[0], softmax.output)

    def test_expand_softmax2(self):
        g = TFGraph(name="g")
        input = TFTensor(graph=g, name="input", shape=[10, 1000], dtype="float32")
        softmax = TFOperation(graph=g,
                              name="tf.nn.softmax",
                              inputs=input,
                              attribs=dict(axis=0),
                              outputs=TFTensor(graph=g, name="softmax1", shape=[10, 1000], dtype="float32"))
        g.inputs = (input,)
        g.outputs = (softmax.output,)

        tf_py_to_tf_pb.expand_softmax(tf_graph=g, tf_op=softmax)
        g.sort()

        self.assertEqual(3, len(g.operations))
        transpose = g.operations[0]
        softmax = g.operations[1]
        transpose_inv = g.operations[2]

        self.assertEqual("tf.transpose", transpose.name)
        self.assertEqual("tf.nn.softmax", softmax.name)
        self.assertEqual("tf.transpose", transpose_inv.name)

        self.assertTrue('axis' in softmax.attribs)
        self.assertEqual(-1, softmax.attribs['axis'])

        self.assertIs(g.inputs[0], transpose.input)
        self.assertIs(g.outputs[0], transpose_inv.output)
        self.assertChain(transpose, softmax, transpose_inv)

        self.assertEqual([10, 1000], transpose.input.shape)
        self.assertEqual([1000, 10], softmax.input.shape)
        self.assertEqual([1000, 10], transpose_inv.input.shape)
        self.assertEqual([10, 1000], transpose_inv.output.shape)

        self.assertEqual([1, 0], transpose.attribs['perm'])
        self.assertEqual([1, 0], transpose_inv.attribs['perm'])

    def test_expand_softmax3(self):
        g = TFGraph(name="g")
        input = TFTensor(graph=g, name="input", shape=[10, 1000, 20, 30], dtype="float32")
        softmax = TFOperation(graph=g,
                              name="tf.nn.softmax",
                              inputs=input,
                              attribs=dict(axis=-3),
                              outputs=TFTensor(graph=g, name="softmax1", shape=[10, 1000, 20, 30], dtype="float32"))
        g.inputs = (input,)
        g.outputs = (softmax.output,)

        tf_py_to_tf_pb.expand_softmax(tf_graph=g, tf_op=softmax)
        g.sort()

        self.assertEqual(5, len(g.operations))
        transpose = g.operations[0]
        reshape = g.operations[1]
        softmax = g.operations[2]
        reshape_inv = g.operations[3]
        transpose_inv = g.operations[4]

        self.assertEqual("tf.transpose", transpose.name)
        self.assertEqual("tf.reshape", reshape.name)
        self.assertEqual("tf.nn.softmax", softmax.name)
        self.assertEqual("tf.reshape", reshape_inv.name)
        self.assertEqual("tf.transpose", transpose_inv.name)

        self.assertTrue('axis' in softmax.attribs)
        self.assertEqual(-1, softmax.attribs['axis'])

        self.assertIs(g.inputs[0], transpose.input)
        self.assertChain(transpose, reshape, softmax, reshape_inv, transpose_inv)
        self.assertIs(g.outputs[0], transpose_inv.output)

        self.assertEqual([10, 1000, 20, 30], transpose.input.shape)
        self.assertEqual([10, 20, 30, 1000], reshape.input.shape)
        self.assertEqual([6000, 1000], softmax.input.shape)
        self.assertEqual([6000, 1000], reshape_inv.input.shape)
        self.assertEqual([10, 20, 30, 1000], transpose_inv.input.shape)
        self.assertEqual([10, 1000, 20, 30], transpose_inv.output.shape)

        self.assertEqual([0, 2, 3, 1], transpose.attribs['perm'])
        self.assertEqual([0, 3, 1, 2], transpose_inv.attribs['perm'])
        self.assertEqual([-1, 1000], reshape.attribs['shape'])
        self.assertEqual([10, 20, 30, 1000], reshape_inv.attribs['shape'])


if __name__ == '__main__':
    unittest.main()
