#!/usr/bin/env bash

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
