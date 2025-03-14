import _skriptnd as _nd
import numpy as np
import re


def valid_id(name):
    id = re.sub('[^~_0-9a-zA-Z]+', '_', name)
    if len(id) > 0 and id[0].isdigit():
        id = "__" + id
    return id


class Printer:

    def __init__(self, inline_subgraphs=False):
        self._inline_subgraphs = inline_subgraphs

    def __call__(self, model, file=None):
        self._block_scope = ''
        self._used_ids = {self._make_id(tensor.name) for tensor in model.tensors}
        self._next_shape_id = 0

        self._print_graph(model, model.graphs[0], idx=0, file=file)

        if not self._inline_subgraphs:
            for i, graph in enumerate(model.graphs[1:]):
                self._print_graph(model, graph, idx=i + 1, file=file)

    def _make_id(self, name):
        if self._block_scope and name.startswith(self._block_scope) and len(name) > len(self._block_scope):
            name = name[len(self._block_scope):]
        return valid_id(name)

    def _can_inline(self, tensor):
        return tensor.shape is not None and len(tensor.shape) == 0 and \
               tensor.value is not None and not isinstance(tensor.value, (np.ndarray, list))

    def _format_value(self, value):
        if isinstance(value, _nd.Tensor):
            if self._can_inline(value):
                if isinstance(value.value, bool):
                    return "true" if value.value else "false"
                else:
                    return str(value.value)
            elif value.dtype == "type":
                return "~"  # null tensor
            else:
                return self._make_id(value.name)
        elif isinstance(value, _nd.TensorPack):
            if not value.name.startsWith('.'):
                return value.name
            else:
                return "[" + ", ".join(self._format_value(v) for v in value) + "]"
        elif isinstance(value, _nd.Graph):
            return value.name
        elif isinstance(value, list):
            return "[" + ", ".join(self._format_value(v) for v in value) + "]"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, str):
            return '"' + value + '"'
        else:
            return str(value)

    def _format_param(self, name, dtype, shape, default=None, optional=False, packed=False, repeats=None):
        text = self._make_id(name) + ": "
        if optional:
            text += "optional "
        text += dtype.name.lower()
        if shape is not None:
            text += "[" + ",".join(str(s) if s is not None else "~" for s in shape) + "]"
        if packed:
            text += ".."
        if repeats is not None:
            text += "(" + str(repeats) + ")"
        if default is not None:
            text += " = " + self._format_value(default)
        return text

    def _format_result(self, result):
        if result is None:
            return "~"
        elif isinstance(result, _nd.Tensor):
            return self._make_id(result.name)
        elif isinstance(result, _nd.TensorPack):
            if not result.name.startsWith('.'):
                return result.name + "..(" + str(len(result)) + ")"
            else:
                return "[" + ", ".join(self._format_result(r) for r in result) + "]"
        elif isinstance(result, tuple):
            return ", ".join(self._format_result(r) for r in result)
        elif isinstance(result, list):
            return "[" + ", ".join(self._format_result(r) for r in result) + "]"
        else:
            assert False

    def _format_subgraph(self, target, inputs=None, nvars=0, indexed=False):
        if isinstance(target, _nd.Tensor):
            return self._format_value(target)
        elif self._inline_subgraphs:
            if len(target.operations) == 0:
                if len(target.inputs) == 1:
                    return self._format_value(target.inputs[0])
                else:
                    return '{ yield ' + ', '.join(self._format_value(input)
                                                  for input in target.inputs) + '; }'
            else:
                label = self._make_id(target.name)
                text = label + ': {\n'
                for op in target.operations:
                    text += "\t\t\t"
                    text += self._format_operation(op.outputs, op.name, op.dtypes.values(),
                                                   op.attribs, op.inputs) + ";\n"
                text += '\t\t\tyield ' + ', '.join(self._make_id(output.name) for output in target.outputs) + ';\n'
                text += '\t\t}'
                return text
        else:
            args = list(target.inputs)
            if inputs is not None:
                last = len(target.inputs) - int(indexed)
                args[nvars:last] = inputs[nvars:last]
            args = [item for item in args
                    if (isinstance(item, _nd.TensorPack) and len(item) > 0)
                    or (isinstance(item, _nd.Tensor) and not self._can_inline(item))]
            name = self._make_id(target.name)
            return self._format_invocation(name, args)

    def _format_invocation(self, name, args, dtypes=None, attribs=None, alias=None, label=None):
        text = ""
        if label:
            text += label + ": "
        text += name
        if dtypes:
            text += "<" + ",".join(dtype.name.lower() for dtype in dtypes) + ">"
        if attribs:
            text += "{" + ", ".join(k + "=" + self._format_value(v) for k, v in attribs.items()) + "}"
        text += "(" + ", ".join(self._format_value(a) for a in args) + ")"
        if alias:
            text += " as " + alias
        return text

    def _format_decls(self, ids, tensors, sep):
        return ", ".join("{id} {sep} {value}".format(id=id, sep=sep, value=self._format_value(tensor))
                         for id, tensor in zip(ids, tensors))

    def _format_operation(self, results, name, dtypes, attribs, args, alias=None):
        if name == 'do':
            nvars = attribs['nvars']
            iters = attribs.get('max-iters')
            repeats = '..(' + self._format_value(iters) + ')' if iters else ''
            text = self._format_result(results[:nvars])
            for result in results[nvars:]:
                if len(text) != 0:
                    text += ", "
                if isinstance(result, _nd.Tensor):
                    text += self._make_id(result.name) + repeats
                else:
                    text += self._format_result(result)
            text += " = "
        else:
            text = self._format_result(results) + " = "

        if name == 'if':
            conditions = attribs['conditions']
            branches = attribs['branches']
            for i, (condition, branch) in enumerate(zip(conditions, branches)):
                text += ('if ' if i == 0 else ' elif ') + self._format_subgraph(condition) + ' then ' + \
                        self._format_subgraph(branch)
            text += ' else ' + self._format_subgraph(branches[-1])
        elif name == 'do':
            condition = attribs.get('condition')
            index = attribs.get('index')
            body = attribs['body']
            nvars = attribs['nvars']
            nscans = attribs['nscans']
            iters = attribs.get('iters') or attribs.get('max-iters')
            indexed = index is not None
            pretest = attribs.get('pretest', False)

            if nvars > 0:
                ids = [self._make_id(tensor.name) for tensor in body.inputs[:nvars]]
                initials = args[:nvars]
                text += 'with ' + self._format_decls(ids, initials, '=')

            if nscans > 0:
                ids = [self._make_id(tensor.name) for tensor in body.inputs[nvars:nvars+nscans]]
                scans = args[nvars:nvars+nscans]
                if nvars > 0:
                    text += ' '
                text += 'for ' + self._format_decls(ids, scans, ':')

            if condition and pretest:
                if nvars > 0 or nscans > 0:
                    text += ' '
                text += 'while ' + self._format_subgraph(condition)

            if nvars > 0 or nscans > 0 or (condition and pretest):
                text += ' '
            text += 'do'

            if indexed or iters:
                text += '..('
                if indexed:
                    text += self._make_id(index.name) + ' -> '
                if iters:
                    text += self._format_value(iters)
                text += ')'

            text += ' ' + self._format_subgraph(body, args, nvars, indexed)

            if condition and not pretest:
                text += ' while ' + self._format_subgraph(condition)

        elif name == '':    # assignment
            text += self._format_value(args[0])
        else:
            text += self._format_invocation(name, args, dtypes, attribs, alias)

        return text

    def _print_graph(self, model, graph, idx, file=None):
        self._block_scope = graph.name + '.'

        print("graph " + self._make_id(graph.name) + " {", file=file)

        print("\t@input {", file=file)
        for input in graph.inputs:
            if isinstance(input, _nd.TensorPack):
                print("\t\t" + self._format_param(input.name, input.dtype, input.shape,
                                                  packed=True, repeats=len(input)) + ";", file=file)
            elif not self._can_inline(input):
                print("\t\t" + self._format_param(input.name, input.dtype, input.shape) + ";",
                      file=file)
        print("\t}", file=file)

        print("\t@output {", file=file)
        for output in graph.outputs:
            if isinstance(output, _nd.TensorPack):
                print("\t\t" + self._format_param(output.name, output.dtype, output.shape,
                                                  packed=True, repeats=len(output)) + ";", file=file)
            else:
                print("\t\t" + self._format_param(output.name, output.dtype, output.shape) + ";",
                      file=file)
        print("\t}", file=file)

        if self._inline_subgraphs and idx == 0:
            variables = model.variables
            constants = model.constants
        else:
            variables = graph.variables
            constants = graph.constants

        variables = list(variables)
        constants = list(tensor for tensor in constants if not self._can_inline(tensor))

        if len(constants) > 0:
            print("\t@constant {", file=file)
            for tensor in constants:
                print("\t\t" + self._format_param(tensor.name, tensor.dtype, tensor.shape,
                                                  default=tensor.value) + ";", file=file)
            print("\t}", file=file)

        if len(variables) > 0:
            print("\t@variable {", file=file)
            for tensor in variables:
                print("\t\t" + self._format_param(tensor.name, tensor.dtype, tensor.shape) + ";", file=file)
            print("\t}", file=file)

        print("\t@compose {", file=file)
        for op in graph.operations:
            print("\t\t" + self._format_operation(op.outputs, op.name, op.dtypes.values(), op.attribs, op.inputs)
                  + ";", file=file)
        print("\t}", file=file)

        print("}\n", file=file)


def print_model(model, file=None, inline_subgraphs=False):
    printer = Printer(inline_subgraphs=inline_subgraphs)
    printer(model, file)
