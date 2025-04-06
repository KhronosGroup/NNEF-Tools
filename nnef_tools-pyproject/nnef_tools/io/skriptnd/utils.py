import skriptnd as nd


def remap_tensor(item, tensor_map):
    def fetch(item):
        return tensor_map[item.name] if item is not None else None

    return [fetch(x) for x in item] if type(item) is list else fetch(item)


def remap_tensors_in_expr(expr, tensor_map):
    if isinstance(expr, (list, tuple)):
        for item in expr:
            remap_tensors_in_expr(item, tensor_map)
    elif isinstance(expr, nd.Expr):
        for x in nd.recursive_enumerate_expr(expr):
            if isinstance(x, nd.ShapeAccess):
                x.tensor = remap_tensor(x.tensor, tensor_map)
            if isinstance(x, nd.SizeAccess):
                x.pack = remap_tensor(x.pack, tensor_map)
    return expr
