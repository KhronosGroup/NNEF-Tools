import _nnef


def format_version(version):
    major, minor = version
    return 'version {}.{};'.format(major, minor)


def format_extensions(extensions):
    string = str()
    for i, ext in enumerate(extensions):
        if i != 0:
            string += '\n'
        string += 'extension {};'.format(ext)
    return string


def format_argument(value):
    if isinstance(value, _nnef.Identifier):
        return value
    elif isinstance(value, str):
        return "'" + value + "'"
    elif isinstance(value, bool):
        return 'true' if value else 'false'
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, (list, tuple)):
        string = '[' if isinstance(value, list) else '('
        for idx, item in enumerate(value):
            if idx != 0:
                string += ', '
            string += format_argument(item)
        string += ']' if isinstance(value, list) else ')'
        return string
    else:
        raise TypeError('arguments must be of type int, float, str, nnef.Identifier or list/tuple of such, found: ' + str(type(value)))


def format_result(value):
    if isinstance(value, (list, tuple)):
        string = '[' if isinstance(value, list) else '('
        for idx, item in enumerate(value):
            if idx != 0:
                string += ', '
            string += format_result(item)
        string += ']' if isinstance(value, list) else ')'
        return string
    elif isinstance(value, _nnef.Identifier):
        return value
    else:
        raise TypeError('results must be of type nnef.Identifier or list/tuple of such, found: ' + str(type(value)))


def format_invocation(name, attribs, inputs, outputs=None, dtype=None):
    string = str()

    if outputs is not None:
        string += ', '.join([format_result(output) for output in outputs])
        string += ' = '

    string += name

    if dtype is not None:
        string += '<' + dtype + '>'

    string += '('
    string += ', '.join([format_argument(input) for input in inputs])
    if len(inputs) and len(attribs):
        string += ', '
    string += ', '.join(key + ' = ' + format_argument(value) for (key, value) in attribs.items())
    string += ')'

    return string


def format_graph(name, inputs, outputs, operations):
    string = 'graph ' + name + '( ' + ', '.join(inputs) + ' ) -> ( ' + ', '.join(outputs) + ' )\n'
    string += '{\n'
    for operation in operations:
        invocation = format_invocation(operation.name, operation.attribs, operation.inputs.values(), operation.outputs.values(), operation.dtype)
        string += '\t' + invocation + ';\n'
    string += '}\n'
    return string
