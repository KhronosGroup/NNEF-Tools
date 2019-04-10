import nnef
import argparse


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('path', type=str, help='path to the model to validate')
    ap.add_argument('--stdlib', type=str, help='file name of alternate standard operation definitions '
                                               '(defaults to all-primitive definitions)', default='')
    ap.add_argument('--lower', type=str, help='comma separated list of operations to lower (if defined as compound)',
                    default='')
    ap.add_argument('--shapes', action="store_true", help='perform shape validation as well')
    args = ap.parse_args()

    stdlib = ''
    if args.stdlib:
        try:
            with open(args.stdlib) as file:
                stdlib = file.read()
        except FileNotFoundError as e:
            print('Could not open file: ' + args.stdlib)
            exit(-1)
    
    try:
        graph = nnef.load_graph(args.path, stdlib=stdlib, lowered=args.lower.split(','))
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
