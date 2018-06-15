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


from nnef_format import *
import argparse


def export_nnef_heatmaps(net):
    dirname = net.name
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    net.save_nnef_bins_heatmaps()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Provide a graph and a weight (optional) file.')
    parser.add_argument('--graph', type=str, help='the name of the prototxt file')
    parser.add_argument('--weights', type=str, default="",
                        help='the name of the caffemodel file')

    args = parser.parse_args()
    p = args.graph.split(".prototxt")[0]
    w = args.weights
    if w == "":
        w = p + ".caffemodel"
    p = p + ".prototxt"
    s = sys.argv[1].split(".prototxt")[0]
    net = buildNet(p,w,deconv_as_resamp=True,forward=True)
    net.replace_forbidden_characters()
    net.merge_batchnorm_operations()
    net.convert_scalebias_to_mul_add()
    net.convert_global_pooling_to_reduce()
    net.resolve_inplace_operations()
    export_nnef_heatmaps(net)
    log("Success")
