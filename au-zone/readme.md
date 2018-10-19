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
    

## Project Architecture

For a given conversion task, convert.py loads an importer and exporter and runs the conversion by importing into a NNEF 
graph representation (based on NetworkX) and exports to the proper format. 

### Framework support
Each supported framework implements the import and export functions. Framework implementations are self contained into 
their directory ('tensorflow', caffe2', 'common' for NNEF, etc). Existing framework implementations should be used as 
references for supporting additional frameworks (once import-export functions are implemented, the implementer just needs 
to support it in convert.py).

### NNEF representation
Importers/exporters are based on NNEF Fragment and NNEF Graph objects (from "common" directory). 

The importers are parsing data from the input format and are reinterpreting it as NNEF definitions to create a NNEF 
Fragment for each operation (layer).The importers then use the Fragments created to create a NNEF Graph representation.
 
The exporters are using a NNEF Graph object to export either to NNEF format or another framework. 

Here is a simplified example showing the NNEF Fragments and NNEF Graph APIs: 

    # Creating NNEF 'External' fragment 
    input = op.External(shape=in_shape, _uid='input')

    # Creating NNEF 'Variable' fragment
    var1 = op.Variable(label='vars/var1', shape=in_shape, _np_tensor = np_array, _uid='var1')

    # Creating NNEF 'Add' fragment
    output = op.Add(x=input, y=var1, _uid='output')

    # Creating NNEF Graph 
    model = NNEFGraph('test_model', 
                      input, 
                      output)
                      
    # Running inference (assuming np_array has previously been set...)
    input_data = {'input':input_tensor}
    output_tensors = ['output']
    model_output = nnef_model.run(output_tensors, input_data)

    # 'ground_truth' holds the inference result as a numpy array
    ground_truth = model_output['output']


## Unit Tests

The project contains 2 levels of unit tests. 

- Syntax testing: Verifies that a conversion from a framework to NNEF and back to the original framework 
still provides a valid model (in the original framework). Those tests are implemented for each of the NNEF primitives, for each supported framework.

- Model testing: Verifies that the graph result of networks are equivalent before and after converting them to NNEF 
and back to their original format. Those tests are implemented for a few reference networks (those provided by the TensorFlow and Caffe2 Model Zoos).
