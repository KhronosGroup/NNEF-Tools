#!/usr/bin/env bash

echo "Creating dummy model"
create_dummy_caffe_model small.prototxt
echo

echo "Small test (with model)"
caffe_to_nnef small.prototxt -m small.caffemodel -o converted_to_nnef
echo

echo "Resnet 50"
caffe_to_nnef resnet50.prototxt -o converted_to_nnef
echo

echo "Squeezenet 1.1"
caffe_to_nnef squeezenet1_1.prototxt -o converted_to_nnef
echo

echo "Vgg"
caffe_to_nnef vgg.prototxt -o converted_to_nnef
echo
