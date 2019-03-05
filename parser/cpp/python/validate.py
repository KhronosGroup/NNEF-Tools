import nnef
import sys
import os


if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print('input path must be supplied')
        exit(-1)
    
    path = sys.argv[1]
    
    try:
        graph = nnef.load_model(path)
        print(nnef.format_graph(graph.name, graph.inputs, graph.outputs, graph.operations))
        print('Validation succeeded')
    except nnef.Error as err:
        print(err)
