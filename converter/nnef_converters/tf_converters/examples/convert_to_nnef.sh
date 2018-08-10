#!/usr/bin/env bash

echo "Creating dummy checkpoints"
create_dummy_tf_checkpoint small_networks.small_net1
create_dummy_tf_checkpoint small_networks.small_net2
create_dummy_tf_checkpoint standard_networks.alexnet_v2
echo

echo "Small_net1 (with model)"
tf_to_nnef small_networks.small_net1 -m small_net1_ckpt -o converted_to_nnef
echo

echo "Small_net2 (with model)"
tf_to_nnef small_networks.small_net2 -m small_net2_ckpt -o converted_to_nnef
echo

echo "Alexnet_v2 (with model)"
tf_to_nnef standard_networks.alexnet_v2 -m alexnet_v2_ckpt -o converted_to_nnef
echo

echo "Resnet_v2_50"
tf_to_nnef standard_networks.resnet_v2_50 -o converted_to_nnef
echo

echo "Inception_v3"
tf_to_nnef standard_networks.inception_v3 -o converted_to_nnef
echo

echo "Vgg16"
tf_to_nnef standard_networks.vgg16 -o converted_to_nnef
echo
