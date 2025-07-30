/*
 * Copyright (c) 2017 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Python.h"
#include "numpy/arrayobject.h"
#include "nnef/flat/flat_parser.h"
#include "nnef/comp/comp_parser.h"
#include "nnef/flat/quant_parser.h"
#include "nnef.h"
#include <initializer_list>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <locale>
#include <memory>


static PyObject* NNEF_Error;


#if PY_MAJOR_VERSION >= 3
#define PY_STRING_OBJECT PyUnicodeObject
#define PY_STRING_TYPE PyUnicode_Type
#define PY_STRING_CHECK PyUnicode_Check
#define PY_STRING_FROM_CSTR PyUnicode_FromString
#define PY_STRING_AS_CSTR PyUnicode_AsUTF8
#define PY_INTEGER_CHECK PyLong_Check
#define PY_INTEGER_AS_LONG PyLong_AsLong
#else
#define PY_STRING_OBJECT PyStringObject
#define PY_STRING_TYPE PyString_Type
#define PY_STRING_CHECK PyString_Check
#define PY_STRING_FROM_CSTR PyString_FromString
#define PY_STRING_AS_CSTR PyString_AsString
#define PY_INTEGER_CHECK PyInt_Check
#define PY_INTEGER_AS_LONG PyInt_AsLong
#endif


struct NNEF_Identifier
{
    PY_STRING_OBJECT str;
};

static PyTypeObject NNEF_Identifier_Type =
{
    PyVarObject_HEAD_INIT(NULL, 0)
    "_nnef.Identifier",      /* tp_name */
    sizeof(NNEF_Identifier), /* tp_basicsize */
    0,                       /* tp_itemsize */
    0,                       /* tp_dealloc */
    0,                       /* tp_print */
    0,                       /* tp_getattr */
    0,                       /* tp_setattr */
    0,                       /* tp_reserved */
    0,                       /* tp_repr */
    0,                       /* tp_as_number */
    0,                       /* tp_as_sequence */
    0,                       /* tp_as_mapping */
    0,                       /* tp_hash */
    0,                       /* tp_call */
    0,                       /* tp_str */
    0,                       /* tp_getattro */
    0,                       /* tp_setattro */
    0,                       /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
};


static PyObject* OrderedDict;
static PyObject* NamedTuple;

static PyObject* Tensor;
static PyObject* Operation;
static PyObject* Graph;


// make tuple by STEALING references to args
template<typename... Args>
static PyObject* makePyTuple( Args&& ...args )
{
    PyObject* tuple = PyTuple_Pack(sizeof...(args), args...);
    for ( auto& arg : { args... } )
    {
        Py_DECREF(arg);
    }
    return tuple;
}

// make object by STEALING references to args
template<typename... Args>
static PyObject* makePyObject( PyObject* type, Args&& ...args )
{
    PyObject* argsTuple = makePyTuple(std::forward<Args>(args)...);
    PyObject* obj = PyObject_CallObject(type, argsTuple);
    Py_DECREF(argsTuple);
    return obj;
}

static PyObject* makeNamedTuple( const char* name, std::initializer_list<const char*> fields )
{
    PyObject* pyName = PY_STRING_FROM_CSTR(name);

    PyObject* pyFields = PyList_New(fields.size());
    size_t i = 0;
    for ( auto& field : fields )
    {
        PyList_SetItem(pyFields, i++, PY_STRING_FROM_CSTR(field));
    }

    return makePyObject(NamedTuple, pyName, pyFields);
}


static PyObject* buildPyBoolean( bool value )
{
    if ( value )
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}

static PyObject* buildPyNone()
{
    Py_RETURN_NONE;
}

static PyObject* buildPyObjectFromValue( const nnef::Value& value )
{
    switch ( value.kind() )
    {
        case nnef::Value::None:
        {
            return buildPyNone();
        }
        case nnef::Value::Integer:
        {
            return Py_BuildValue("i", value.integer());
        }
        case nnef::Value::Scalar:
        {
            return Py_BuildValue("f", value.scalar());
        }
        case nnef::Value::Logical:
        {
            return buildPyBoolean(value.logical());
        }
        case nnef::Value::String:
        {
            return PY_STRING_FROM_CSTR(value.string().c_str());
        }
        case nnef::Value::Identifier:
        {
            PyObject* arg = PY_STRING_FROM_CSTR(value.identifier().c_str());
            return makePyObject((PyObject*)&NNEF_Identifier_Type, arg);
        }
        case nnef::Value::Array:
        {
            PyObject* list = PyList_New(value.size());
            for ( size_t i = 0; i < value.size(); ++i )
            {
                PyList_SetItem(list, i, buildPyObjectFromValue(value[i]));
            }
            return list;
        }
        case nnef::Value::Tuple:
        {
            PyObject* tuple = PyTuple_New(value.size());
            for ( size_t i = 0; i < value.size(); ++i )
            {
                PyTuple_SetItem(tuple, i, buildPyObjectFromValue(value[i]));
            }
            return tuple;
        }
    }
    return nullptr;
}

static int numpy_type_num( const nnef::Typename& dtype )
{
    switch ( dtype )
    {
        case nnef::Typename::Scalar:
            return NPY_FLOAT32;
        case nnef::Typename::Integer:
            return NPY_INT32;
        case nnef::Typename::Logical:
            return NPY_BOOL;
        default:
            return NPY_VOID;
    }
}

static PyArray_Descr* numpy_dtype( const nnef::Typename& dtype )
{
    switch ( dtype )
    {
        case nnef::Typename::Scalar:
            return PyArray_DescrFromType(NPY_FLOAT32);
        case nnef::Typename::Integer:
            return PyArray_DescrFromType(NPY_INT32);
        case nnef::Typename::Logical:
            return PyArray_DescrFromType(NPY_BOOL);
        default:
            return NULL;
    }
}

static std::string buildErrorString( nnef::Error e )
{
    std::string str = "Parse error in '" + std::string(e.position().filename) + "' [" + std::to_string(e.position().line) + ":" + std::to_string(e.position().column) + "] " + e.what();

    auto origin = e.position().origin;
    while ( origin )
    {
        str += "\n... evaluated from '" + std::string(e.position().filename) + "' [" + std::to_string(e.position().line) + ":" + std::to_string(e.position().column) + "]";
        origin = origin->origin;
    }

    return str;
}


struct GraphCallback : public nnef::Parser::Callback
{
    GraphCallback( std::istream& qis, const char* qfn )
    : qis(qis), qfn(qfn), tensors(NULL), operations(NULL), graph(NULL), version(NULL), extensions(NULL)
    {
    }

    ~GraphCallback()
    {
        if ( tensors )
            Py_DECREF(tensors);
        if ( operations )
            Py_DECREF(operations);
        if ( graph )
            Py_DECREF(graph);
        if ( version )
            Py_DECREF(version);
        if ( extensions )
            Py_DECREF(extensions);
    }

    virtual void beginDocument( const std::string& filename, const nnef::Parser::version_t& version )
    {
        this->version = makePyTuple(Py_BuildValue("i", version.first), Py_BuildValue("i", version.second));
        this->extensions = PyList_New(0);
    }

    virtual bool handleExtension( const std::string& ext )
    {
        PyObject* pyStr = PY_STRING_FROM_CSTR(ext.c_str());
        PyList_Append(this->extensions, pyStr);
        Py_DECREF(pyStr);
        return false;
    }

    virtual void beginGraph( const nnef::Prototype& proto, const nnef::Dictionary<nnef::Prototype>& fragments )
    {
        PyObject* name = PY_STRING_FROM_CSTR(proto.name().c_str());

        this->protos = &fragments;
        this->tensors = PyDict_New();
        this->operations = PyList_New(0);
        
        PyObject* inputs = PyList_New(proto.paramCount());
        for ( size_t i = 0; i < proto.paramCount(); ++i )
        {
            PyList_SetItem(inputs, i, PY_STRING_FROM_CSTR(proto.param(i).name().c_str()));
        }
        
        PyObject* outputs = PyList_New(proto.resultCount());
        for ( size_t i = 0; i < proto.resultCount(); ++i )
        {
            PyList_SetItem(outputs, i, PY_STRING_FROM_CSTR(proto.result(i).name().c_str()));
        }

        Py_INCREF(this->tensors);
        Py_INCREF(this->operations);
        this->graph = makePyObject(Graph, name, tensors, operations, inputs, outputs);
        
        if ( qis )
        {
            quant = nnef::QuantParser::parse(qis, qfn, fragments);
        }
    }

    virtual void endGraph( const nnef::Prototype& proto, const nnef::Dictionary<nnef::Typename>& dtypes )
    {
        for ( auto& it : dtypes )
        {
            PyObject* name = PY_STRING_FROM_CSTR(it.first.c_str());
            PyObject* shape = buildPyNone();
            PyObject* dtype = PY_STRING_FROM_CSTR(nnef::toString(it.second));
            PyObject* data = buildPyNone();
            PyObject* quantization = PyDict_New();
            if ( quant.count(it.first) )
            {
                auto& attribs = quant.at(it.first);
                auto& op_name = attribs.at("op-name").string();
                auto& op_proto = protos->at(op_name);
                for ( auto& qit : attribs )
                {
                    auto obj = buildPyObjectFromValue(qit.second);
                    auto param = op_proto.param(qit.first.c_str());
                    if ( param && param->type()->kind() == nnef::Type::Tensor )
                    {
                        auto tensor_type = (const nnef::TensorType*)param->type();
                        auto data_type = (const nnef::PrimitiveType*)tensor_type->dataType();
                        PyArray_Descr* array_dtype = numpy_dtype(data_type->name());
                        PyObject* array = PyArray_FromAny(obj, array_dtype, 0, 0, 0, NULL); // steals reference to dtype
                        Py_DECREF(obj);
                        obj = array;
                    }
                    PyDict_SetItemString(quantization, qit.first.c_str(), obj);
                    Py_DECREF(obj);
                }
            }
            
            PyObject* tensor = makePyObject(Tensor, name, dtype, shape, data, quantization);
            PyDict_SetItemString(tensors, it.first.c_str(), tensor);
            Py_DECREF(tensor);
        }
    }

    virtual void operation( const nnef::Prototype& proto, const nnef::Dictionary<nnef::Value>& args,
                           const nnef::Dictionary<nnef::Typename>& dtypes )
    {
        PyObject* attribs = PyList_New(0);
        PyObject* inputs = PyList_New(0);
        PyObject* outputs = PyList_New(0);
        PyObject* dtype = args.count("?") ? PY_STRING_FROM_CSTR(args.at("?").string().c_str()) : buildPyNone();
        
        for ( size_t i = 0; i < proto.paramCount(); ++i )
        {
            auto& param = proto.param(i);
            auto& value = args.at(param.name());
            PyObject* item = makePyTuple(PY_STRING_FROM_CSTR(param.name().c_str()), buildPyObjectFromValue(value));
            PyList_Append(param.type()->isAttribute() ? attribs : inputs, item);
            Py_DECREF(item);
        }
        for ( size_t i = 0; i < proto.resultCount(); ++i )
        {
            auto& result = proto.result(i);
            auto& value = args.at(result.name());
            PyObject* item = makePyTuple(PY_STRING_FROM_CSTR(result.name().c_str()), buildPyObjectFromValue(value));
            PyList_Append(outputs, item);
            Py_DECREF(item);
        }

        PyObject* name = PY_STRING_FROM_CSTR(proto.name().c_str());
        attribs = makePyObject(OrderedDict, attribs);
        inputs = makePyObject(OrderedDict, inputs);
        outputs = makePyObject(OrderedDict, outputs);

        PyObject* operation = makePyObject(Operation, name, attribs, inputs, outputs, dtype);
        PyList_Append(operations, operation);
        Py_DECREF(operation);
    }

    std::istream& qis;
    const char* qfn;
    nnef::Dictionary<nnef::Dictionary<nnef::Value>> quant;
    const nnef::Dictionary<nnef::Prototype>* protos;

    PyObject* tensors;
    PyObject* operations;
    PyObject* graph;
    PyObject* version;
    PyObject* extensions;
};


static PyObject* parse( PyObject* self, PyObject* args, PyObject* kwargs, bool isFile )
{
	const char* input = nullptr;
    const char* quant = nullptr;
    const char* stdlib = nullptr;
    PyObject* lower = nullptr;

    static const char* kwlist[] = { "", "quantization", "stdlib", "lowered", NULL };

	if ( !PyArg_ParseTupleAndKeywords(args, kwargs, "s|zzO!", (char**)kwlist, &input, &quant, &stdlib, &PyList_Type, &lower) )
    {
        return NULL;
    }

    if ( !stdlib )
    {
        stdlib = "";
    }

    std::ifstream gfs, qfs;
    std::stringstream gss, qss;

    if ( isFile )
    {
        gfs.open(input);
        if ( !gfs )
        {
            const std::string message = "Could not open file: " + std::string(input);
            PyErr_SetString(NNEF_Error, message.c_str());
            return NULL;
        }

        if ( quant )
        {
            qfs.open(quant);
            if ( !qfs )
            {
                const std::string message = "Could not open file: " + std::string(quant);
                PyErr_SetString(NNEF_Error, message.c_str());
                return NULL;
            }
        }
    }
    else
    {
        gss.str(input);
        if ( quant )
        {
            qss.str(quant);
        }
    }

    std::istream& gis = isFile ? (std::istream&)gfs : (std::istream&)gss;
    std::istream& qis = isFile ? (std::istream&)qfs : (std::istream&)qss;
    
    std::set<std::string> lowered;
    if ( lower )
    {
        for ( Py_ssize_t i = 0; i < PyList_Size(lower); ++i )
        {
            PyObject* item = PyList_GetItem(lower, i);
            if ( !PY_STRING_CHECK(item) )
            {
                const std::string message = "Paremeter 'lowered' must be a list of strings";
                PyErr_SetString(NNEF_Error, message.c_str());
                return NULL;
            }
            lowered.insert(PY_STRING_AS_CSTR(item));
        }
    }
    
    nnef::CompParser parser(stdlib, lowered);

    GraphCallback callback(qis, isFile ? quant : "quantization");

	try
    {
        parser.parse(gis, isFile ? input : "input", callback);
        Py_INCREF(callback.graph);
        return callback.graph;
    }
    catch ( const nnef::Error& e )
    {
        PyErr_SetString(NNEF_Error, buildErrorString(e).c_str());
		return NULL;
    }
    catch ( const std::invalid_argument& e )
    {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
    catch ( const std::exception& e )
    {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

static PyObject* parseFile( PyObject* self, PyObject* args, PyObject* kwargs )
{
    return parse(self, args, kwargs, true);
}

static PyObject* parseString( PyObject* self, PyObject* args, PyObject* kwargs )
{
    return parse(self, args, kwargs, false);
}

static PyObject* createSession( PyObject* self, PyObject* args, PyObject* kwargs )
{
    static const char* kwlist[] = { "", "stdlib", "lowered", NULL };

    const char* path = nullptr;
    const char* stdlib = nullptr;
    PyObject* lower = nullptr;

	if ( !PyArg_ParseTupleAndKeywords(args, kwargs, "s|zO!", (char**)kwlist, &path, &stdlib, &PyList_Type, &lower) )
    {
        return NULL;
    }

    if ( !stdlib )
    {
        stdlib = "";
    }

    std::set<std::string> lowered;
    if ( lower )
    {
        for ( Py_ssize_t i = 0; i < PyList_Size(lower); ++i )
        {
            PyObject* item = PyList_GetItem(lower, i);
            if ( !PY_STRING_CHECK(item) )
            {
                const std::string message = "Paremeter 'lowered' must be a list of strings";
                PyErr_SetString(NNEF_Error, message.c_str());
                return NULL;
            }
            lowered.insert(PY_STRING_AS_CSTR(item));
        }
    }

    std::unique_ptr<nnef::Graph> graph(new nnef::Graph());
    std::string error;

    if ( !nnef::load_graph(path, *graph, error, stdlib, lowered) )
    {
        PyErr_SetString(PyExc_ValueError, error.c_str());
        return NULL;
    }

    if ( !nnef::infer_shapes(*graph, error) )
    {
        PyErr_SetString(PyExc_ValueError, error.c_str());
        return NULL;
    }

    if ( !nnef::allocate_buffers(*graph, error) )
    {
        PyErr_SetString(PyExc_ValueError, error.c_str());
        return NULL;
    }

    const size_t handle = reinterpret_cast<size_t>(graph.release());
    return PyLong_FromSize_t(handle);
}

static PyObject* cleanupSession( PyObject* self, PyObject* args, PyObject* kwargs )
{
    PyObject* handle;
    static const char* kwlist[] = { "", NULL };

	if ( !PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist, &handle) )
    {
        return NULL;
    }

    nnef::Graph* graph = reinterpret_cast<nnef::Graph*>(PyLong_AsSize_t(handle));
    delete graph;

    Py_RETURN_NONE;
}

static PyObject* executeSession( PyObject* self, PyObject* args, PyObject* kwargs )
{
    PyObject* handle;
    PyObject* inputs;
    static const char* kwlist[] = { "", "", NULL };

	if ( !PyArg_ParseTupleAndKeywords(args, kwargs, "OO!", (char**)kwlist, &handle, &PyTuple_Type, &inputs) )
    {
        return NULL;
    }

    nnef::Graph* graph = reinterpret_cast<nnef::Graph*>(PyLong_AsSize_t(handle));

    if ( PyTuple_Size(inputs) != graph->inputs.size() )
    {
        PyErr_Format(PyExc_ValueError, "number of inputs (%d) does not match number of graph inputs (%d)",
                     (int)PyTuple_Size(inputs), (int)graph->inputs.size());
        return NULL;
    }

    for ( size_t i = 0; i < PyTuple_Size(inputs); ++i )
    {
        PyObject* input = PyTuple_GetItem(inputs, i);
        if ( !PyArray_Check(input) )
        {
            PyErr_SetString(PyExc_ValueError, "inputs must be numpy arrays");
            return NULL;
        }
        PyArrayObject* array = (PyArrayObject*)input;

        nnef::Tensor& tensor = graph->tensors.at(graph->inputs[i]);
        nnef::Typename dtype = nnef::fromString(tensor.dtype);

        if ( PyArray_TYPE(array) != numpy_type_num(dtype) )
        {
            PyErr_Format(PyExc_ValueError, "dtype of input %d does not match input dtype in graph", (int)i+1);
            return NULL;
        }

        if ( PyArray_NDIM(array) != tensor.shape.size() || !std::equal(tensor.shape.begin(), tensor.shape.end(), PyArray_SHAPE(array)) )
        {
            PyErr_Format(PyExc_ValueError, "shape of input %d does not match input shape in graph", (int)i+1);
            return NULL;
        }

        std::copy_n(PyArray_BYTES(array), tensor.data.size(), tensor.data.data());
    }

    std::string error;
    if ( !nnef::execute(*graph, error) )
    {
        PyErr_SetString(PyExc_ValueError, error.c_str());
        return NULL;
    }

    PyObject* outputs = PyTuple_New(graph->outputs.size());
    for ( size_t i = 0; i < graph->outputs.size(); ++i )
    {
        nnef::Tensor& tensor = graph->tensors.at(graph->outputs[i]);
        std::vector<npy_intp> shape(tensor.shape.begin(), tensor.shape.end());
        nnef::Typename dtype = nnef::fromString(tensor.dtype);

        PyObject* output = PyArray_SimpleNew(shape.size(), shape.data(), numpy_type_num(dtype));
        std::copy_n(tensor.data.data(), tensor.data.size(), PyArray_BYTES((PyArrayObject*)output));

        PyTuple_SetItem(outputs, i, output);
    }

    return outputs;
}


static PyMethodDef NNEF_Methods[] = 
{
    { "parse_file", (PyCFunction)parseFile, METH_VARARGS | METH_KEYWORDS, "Parse the contents of a file" },
    { "parse_string", (PyCFunction)parseString, METH_VARARGS | METH_KEYWORDS, "Parse the contents of a string" },
    { "create_session", (PyCFunction)createSession, METH_VARARGS | METH_KEYWORDS, "Create session for executing a graph" },
    { "cleanup_session", (PyCFunction)cleanupSession, METH_VARARGS | METH_KEYWORDS, "Cleanup session" },
    { "execute_session", (PyCFunction)executeSession, METH_VARARGS | METH_KEYWORDS, "Execute graph in a session" },
 	{ NULL, NULL, 0, NULL }
};


#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef nnef_module = 
{
    PyModuleDef_HEAD_INIT,
    "_nnef",
    "_nnef module",
    -1,
    NNEF_Methods,
};

#endif


#if PY_MAJOR_VERSION >= 3
#define INIT_FUNC_NAME PyInit__nnef
#define RETURN_ERROR return NULL
#else
#define INIT_FUNC_NAME init_nnef
#define RETURN_ERROR return
#endif

PyMODINIT_FUNC INIT_FUNC_NAME(void)
{
    NNEF_Identifier_Type.tp_base = &PY_STRING_TYPE;
    if ( PyType_Ready(&NNEF_Identifier_Type) < 0 )
    {
        RETURN_ERROR;
    }

#if PY_MAJOR_VERSION >= 3
	PyObject* module = PyModule_Create(&nnef_module);
#else
    PyObject* module = Py_InitModule("_nnef", NNEF_Methods);
#endif
	if ( module == NULL )
	{
		RETURN_ERROR;
	}

	NNEF_Error = PyErr_NewException((char*)"_nnef.Error", NULL, NULL);
	PyModule_AddObject(module, "Error", NNEF_Error);

    PyModule_AddObject(module, "Identifier", (PyObject*)&NNEF_Identifier_Type);

    PyObject* collections = PyImport_ImportModule("collections");
    PyObject* dict = PyModule_GetDict(collections);
    OrderedDict = PyDict_GetItemString(dict, "OrderedDict");
    NamedTuple = PyDict_GetItemString(dict, "namedtuple");
    Py_DECREF(collections);

    Tensor = makeNamedTuple("Tensor", { "name", "dtype", "shape", "data", "quantization" });
    PyModule_AddObject(module, "Tensor", Tensor);

    Operation = makeNamedTuple("Operation", { "name", "attribs", "inputs", "outputs", "dtype" });
    PyModule_AddObject(module, "Operation", Operation);
    
    Graph = makeNamedTuple("Graph", { "name", "tensors", "operations", "inputs", "outputs" });
    PyModule_AddObject(module, "Graph", Graph);

    import_array();
    
#if PY_MAJOR_VERSION >= 3
	return module;
#endif
}
