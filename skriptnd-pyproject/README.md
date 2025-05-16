SkriptND project
===================

Introduction
------------

SkriptND is a domain specific language to decribe operations on N-dimensional arrays (tensors) and graphs composed from such operations. 
It allows complete definition of operations including types and shapes of inputs and outputs (shape propagation) 
and the mathematical formulae to compute the output of primitive operations (lowering to scalar computation) using compact notations.

The code consists of a C++ library for parsing SkriptND syntax. This library can be used to build tools
that require parsing SkriptND files. It requires a C++17 compatible compiler. 

Furthermore, the repository contains Python wrapper code around the C++ parser and also adds some further utilities to load and save 
computational parameters easily and to generate sample execution code (C++) from the sources, and ultimately an executable python object.

Documentation of the C++ library: [cpp_api.md](cpp_api.md)

Documentation of the Python package: [package_info.md](package_info.md)

Coding examples
---------------

SkriptND models are defined in .skdn text files. There are two main entities, `operator` for defining operations on n-dimensional data, and `graph` for defining complete models or sub-graphs of them.

Here's a simple example of defining a `linear` operator (matrix multiplication with bias) that demonstrates how basic input-output shape mapping and lowering the computation to the scalar level works:

```
operator linear {
    @input {
        input: real[b,c];
        filter: real[n,c];
        bias: optional real[n];
    }
    @output {
        output: real[b,n];
    }
    @lower {
        output[bi,ni] = bias[ni,] ?? 0.0,
            bi < b, ni < n;
        output[bi,ni] += input[bi,ci] * filter[ni,ci],
            bi < b, ci < c, ni < n;
    }
}
```

And here's a more complex example of defining a `convolution` operator, that demonstrates the use of attributes, variable ranks for input-output shapes and attributes, and attribute and shape validity asserts with error messages:

```
operator conv {
    @attrib {
        stride: int..(d) = 1;
        dilation: int..(d) = 1;
        padding: optional int..(d);
    }
    @input {
        input: real[b,c,is..(d)];
        filter: real[n,c,fs..(d)];
        bias: optional real[n];
    }
    @using {
        fd = (fs - 1) * dilation + 1;
        paddings = 2 * padding ?? (is \ stride - 1) * stride + fd - is;
        os = (is + paddings - fd) / stride + 1;
    }
    @output {
        output: real[b,n,os..];
    }
    @assert {
        stride > 0: "'stride' must be positive; got {stride}";
        dilation > 0: "'dilation' must be positive; got {dilation}";
        is + paddings >= fd: "padded input-size must be greater than (dilated) filter-size;
                              got input-size={is}, filter-size={fs}, dilation={dilation},
                              total-padding={paddings}";
    }
    @lower {
        output[bi,ni,i..] = bias[ni] ?? 0.0,
            bi < b, ni < n, i < os;
        output[bi,ni,i..] += input[bi,ci,|stride * i + dilation * j - paddings / 2|..]
                           * filter[ni,ci,j..],
            bi < b, ni < n, ci < c, i < os, j < fs;
    }
}
```

However, many such operators are already pre-defined in a so called standard library of operators, grouped into various modules that can be imported and used to build complete models easily.

Here's an example for defining the good old AlexNet (parameterized by batch size and number of output classes):

```
import nn;
import layout;

graph AlexNet {
    @attrib {
        batch: int = 1;
        classes: int = 1000;
    }
    @input {
        input: real[batch,3,224,224];
    }
    @output {
        output: real[batch,classes];
    }
    @variable {
        kernel1: real[64, 3, 11, 11];
        bias1: real[64];
        kernel2: real[192, 64, 5, 5];
        bias2: real[192];
        kernel3: real[384, 192, 3, 3];
        bias3: real[384];
        kernel4: real[384, 384, 3, 3];
        bias4: real[384];
        kernel5: real[256, 384, 3, 3];
        bias5: real[256];
        kernel6: real[4096, 256, 5, 5];
        bias6: real[4096];
        kernel7: real[4096, 4096];
        bias7: real[4096];
        kernel8: real[classes, 4096];
        bias8: real[classes];
    }
    @compose {
        conv1 = nn.conv{padding=0, stride=4}(input, kernel1, bias1);
        relu1 = nn.relu(conv1);
        pool1 = nn.max_pool{padding=0, size=3, stride=2}(relu1);
        conv2 = nn.conv{padding=2}(pool1, kernel2, bias2);
        relu2 = nn.relu(conv2);
        pool2 = nn.max_pool{padding=0, size=3, stride=2}(relu2);
        conv3 = nn.conv{padding=1}(pool2, kernel3, bias3);
        relu3 = nn.relu(conv3);
        conv4 = nn.conv{padding=1}(relu3, kernel4, bias4);
        relu4 = nn.relu(conv4);
        conv5 = nn.conv{padding=1}(relu4, kernel5, bias5);
        relu5 = nn.relu(conv5);
        pool3 = nn.max_pool{padding=0, size=3, stride=2}(relu5);
        conv6 = nn.conv{padding=0}(pool3, kernel6, bias6);
        relu6 = nn.relu(conv6);
        flat1 = layout.flatten{axis=1}(relu6);
        conv7 = nn.linear(flat1, kernel7, bias7);
        relu7 = nn.relu(conv7);
        conv8 = nn.linear(relu7, kernel8, bias8);
        output = nn.softmax(conv8);
    }
}
```
