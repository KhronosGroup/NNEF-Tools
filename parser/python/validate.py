import nnef
import argparse


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('path', type=str, help='path to the model to validate')
    ap.add_argument('--shapes', action="store_true", help='perform shape validation as well')
    args = ap.parse_args()
    
    try:
        graph = nnef.load_graph(args.path)
    except nnef.Error as err:
        print('Parse error: ' + str(err))
        exit(-1)

    if args.shapes:
        try:
            nnef.infer_shapes(graph)
        except nnef.Error as err:
            print('Shape error: ' + str(err))
            exit(-1)

    print(nnef.format_graph(graph.name, graph.inputs, graph.outputs, graph.operations))
    print('Validation succeeded')
