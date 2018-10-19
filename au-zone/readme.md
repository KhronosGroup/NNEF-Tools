# NNEF Converters

## Overview

The NNEF converter provides conversion between deep learning framework models and NNEF's representation. Trained models 
from popular frameworks can then be converted to run on NNEF compliant inference engines. 

The NNEF converter uses the trained files and doesn't require deep learning frameworks to be installed 
(uses protobufs for tensorflow and caffe2).

## Requirements

- The TensorFlow importer requires a frozen protobuf file. Resource for freezing [here](https://www.tensorflow.org/extend/tool_developers/#freezing).

## Dependencies

Requires Networkx, Numpy, Six, Protobuf and NNEF to be installed:

Follow documentation from [here](https://github.com/KhronosGroup/NNEF-Tools/tree/master/parser/cpp) for installing NNEF.

	$ python setup.py bdist_wheel
	$ pip install dist/wheel_package_generated    
	
Follow documentation from [here](https://github.com/KhronosGroup/NNEF-Tools/tree/master/parser/cpp) for installing NNEF.

## Frameworks Support

NNEF converter currently supports the following operations: 

   - Tensorflow -> NNEF 
   - NNEF       -> Tensorflow
   - Caffe2     -> NNEF
   - NNEF       -> Caffe2
   

## Usage
Example for converting Tensorflow to NNEF:  

    python -m nnef_converter.convert --input-framework tensorflow --output-framework NNEF 
                      --input-model tf_models/mobilenet_v1_1_0_224.pb  --output-model mobilenet_v1_1_0_224/graph.nnef 
                      --input-nodes Placeholder --output-nodes final_result 
                      --log-level info 

Example for converting NNEF to Tensorflow:  

    python -m nnef_converter.convert --input-framework NNEF --output-framework tensorflow 
                      --input-model  mobilenet_v1_1_0_224/graph.nnef --output-model tf_models/mobilenet_v1_1_0_224_copy.pb 
                      --input-nodes input --output-nodes output
                      --log-level info 

Example for converting Caffe2 to NNEF:

	python -m nnef_converter.convert --input-framework caffe2 --output-framework NNEF
					  --input-model predict_net.pb --data-model init_net.pb
					  --value-info value_info.json --output-model nnef_output/graph.nnef

Example for converting NNEF to Caffe2:

	python -m nnef_converter.convert --input-framework NNEF --output-framework caffe2
					  --input-model nnef_output/graph.nnef --output-model caffe2_output/net.pb

For complete list of options:

    python -m nnef_converter.convert --help 

Mandatory Arguments:
	
	--input-framework
	--output-framework
	--input-model
	--data-model (Caffe2)
	--value-info (Caffe2)
