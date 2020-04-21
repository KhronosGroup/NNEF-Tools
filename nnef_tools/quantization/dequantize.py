from nnef_tools.io.nnef.nnef_graph import *
import numpy as np

def _const(graph, value):
    assert not isinstance(value, np.ndarray)
    return NNEFTensor(graph=graph,
               data=[value],
               shape=[],
               name="constant_{}".format(int(value)),
               dtype='scalar',
               )

def _var(graph, name, value, **kwargs):
    if not isinstance(value, np.ndarray):
        value = np.array(value, dtype=np.float32)
    return NNEFTensor(graph=graph,
               data=value,
               shape=list(value.shape),
               name=name,
               label=name,
               dtype='scalar',
               **kwargs
               )

def _op(graph, name, idx, output_quant=None, **kwargs):
    o = NNEFTensor(
        graph=graph,
        name="{}_output_{}".format(name, idx),
        dtype='scalar',
        quantization=output_quant
    )
    op = NNEFOperation(
        graph=graph,
        name=name,
        outputs=o,
        **kwargs
    )
    return o, op

def _repalce_tensor(t, r, exclude_ops=tuple()):
    r._consumers = []
    for consumer in t.consumers:
        if consumer not in exclude_ops:
            is_tuple = isinstance(consumer._inputs, tuple)
            inps = list(consumer._inputs)
            for i in range(len(inps)):
                if inps[i] == t:
                    inps[i] = r
            consumer._inputs = tuple(inps) if is_tuple else inps
            r._consumers.append(consumer)
    t._consumers = list([op for op in exclude_ops if op is not None])

    if t in t.graph.outputs:
        outps = list(t.graph._outputs)
        for i in range(len(outps)):
            if outps[i] == t:
                outps[i] = r
        t.graph._outputs = tuple(outps)



def dequantize(graph):
    gen_idx = 0
    c0 = _const(graph, 0.0)
    c255 = _const(graph, 255.0)
    for t in graph.tensors:
        if t.quantization:
            zp = t.quantization.attribs.get('zero_point', 0)
            sc = t.quantization.attribs.get('scale', 0)
            if t.is_variable:
                t.data = (t.data.astype(np.float32) - zp) * sc
            else:
                zp_t = _var(graph, "q_zp_{}".format(gen_idx), zp) if zp != 0 else None
                sc_t = _var(graph, "q_scale_{}".format(gen_idx), sc)
                m_op = s2_op = None
                if t not in graph.inputs:
                    m, m_op = _op(graph, "div", gen_idx, inputs=(t, sc_t))
                    s, s_op = _op(graph, "add", gen_idx, inputs=(m, zp_t)) if zp != 0 else (m, None)
                    r, r_op = _op(graph, "round", gen_idx, inputs=(s,))
                    c, c_op = _op(graph, "clamp", gen_idx, inputs=(r, c0, c255))
                else:
                    c = t
                if t not in graph.outputs:
                    s2, s2_op = _op(graph, "sub", gen_idx, inputs=(c, zp_t)) if zp != 0 else (c, None)
                    m2, m2_op = _op(graph, "mul", gen_idx, inputs=(s2, sc_t))
                else:
                    m2 = c
                _repalce_tensor(t, m2, exclude_ops=(m_op, s2_op))
                gen_idx += 1
            t.quantization = None
    graph.sort()
