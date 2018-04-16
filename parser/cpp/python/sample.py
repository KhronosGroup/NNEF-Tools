import nnef

attrs, ops, shapes = nnef.parse_string(
    "version 1.0"
    "graph Net( input ) -> ( output )"
    "{"
    "   input = external(shape = [1,3,224,224])"
    "   filter = variable(shape = [32,3,5,5], label = 'conv/filter')"
    "   output = conv(input, filter)"
    "}"
)

print('graph ' + nnef.format_declaration(name=attrs["name"], inputs=attrs["inputs"], outputs=attrs["outputs"]))
print('{')
for (name, args, res) in ops:
    args, kwargs = nnef.extract_positional_args(args)
    print('\t' + nnef.format_invocation(name, args=args.values(), kwargs=kwargs, results=res.values()))
print('}')
