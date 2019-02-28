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
#include "include/flat/flat_parser.h"
#include "include/comp/comp_parser.h"
#include "include/flat/quant_parser.h"
#include <initializer_list>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <locale>


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


static PyObject* makeNamedTuple( const char* name, std::initializer_list<const char*> fields )
{
    PyObject* pyName = PY_STRING_FROM_CSTR(name);

    PyObject* pyFields = PyList_New(0);
    for ( auto& field : fields )
    {
        PyList_Append(pyFields, PY_STRING_FROM_CSTR(field));
    }

    return PyObject_CallObject(NamedTuple, PyTuple_Pack(2, pyName, pyFields));
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
            return PyObject_CallObject((PyObject*)&NNEF_Identifier_Type, PyTuple_Pack(1, arg));
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

/*static nnef::Value buildValueFromPyObject( PyObject* object )
{
    if ( object == Py_None )
    {
        return nnef::Value::none();
    }
    else if ( PY_INTEGER_CHECK(object) )
    {
        auto value = PY_INTEGER_AS_LONG(object);
        return nnef::Value::integer(value);
    }
    else if ( PyFloat_Check(object) )
    {
        auto value = PyFloat_AsDouble(object);
        return nnef::Value::scalar(value);
    }
    else if ( PyBool_Check(object) )
    {
        auto value = object == Py_True;
        return nnef::Value::logical(value);
    }
    else if ( PY_STRING_CHECK(object) )
    {
        auto value = PY_STRING_AS_CSTR(object);
        return nnef::Value::string(value);
    }
    else if ( PyList_Check(object) )
    {
        const size_t size = PyList_Size(object);

        nnef::Value::items_t items(size);
        for ( size_t i = 0; i < size; ++i )
        {
            auto item = PyList_GetItem(object, i);
            items[i] = buildValueFromPyObject(item);
        }
        return nnef::Value::array(items);
    }
    else if ( PyTuple_Check(object) )
    {
        const size_t size = PyTuple_Size(object);

        nnef::Value::items_t items(size);
        for ( size_t i = 0; i < size; ++i )
        {
            auto item = PyTuple_GetItem(object, i);
            items[i] = buildValueFromPyObject(item);
        }
        return nnef::Value::tuple(items);
    }
    else
    {
        const PyTypeObject* type = object->ob_type;
        throw std::invalid_argument(type->tp_name);
    }
}*/

static PyObject* buildPyListFromShape( const nnef::Shape& shape )
{
    PyObject* list = PyList_New(shape.size());
    for ( size_t i = 0; i < shape.size(); ++i )
    {
        PyList_SetItem(list, i, Py_BuildValue("i", shape[i]));
    }
    return list;
}

static nnef::Shape shapeFromPyList( PyObject* list )
{
    const size_t size = PyList_Size(list);

    nnef::Shape shape(size);
    for ( size_t i = 0; i < size; ++i )
    {
        auto item = PyList_GetItem(list, i);
        shape[i] = (nnef::Shape::value_type)PY_INTEGER_AS_LONG(item);
    }
    return shape;
}

static PyObject* buildValuePyDict( const nnef::Dictionary<nnef::Value>& args )
{
    PyObject* dict = PyDict_New();
    for ( auto& arg : args )
    {
        PyDict_SetItemString(dict, arg.first.c_str(), buildPyObjectFromValue(arg.second));
    }
    return dict;
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
    : qis(qis), qfn(qfn)
    {
    }

    virtual void beginDocument( const std::string& filename, const nnef::Parser::version_t& version )
    {
        this->version = PyTuple_Pack(2, Py_BuildValue("i", version.first), Py_BuildValue("i", version.second));
        this->extensions = PyList_New(0);
    }

    virtual bool handleExtension( const std::string& ext )
    {
        PyList_Append(this->extensions, PY_STRING_FROM_CSTR(ext.c_str()));
        return false;
    }

    virtual void beginGraph( const nnef::Prototype& proto, const nnef::Dictionary<nnef::Prototype>& fragments )
    {
        PyObject* name = PY_STRING_FROM_CSTR(proto.name().c_str());
        
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
        
        this->graph = PyObject_CallObject(Graph, PyTuple_Pack(5, name, tensors, operations, inputs, outputs));
        
        if ( qis )
        {
            quant = nnef::QuantParser::parse(qis, qfn, fragments);
        }
    }

    virtual void endGraph( const nnef::Prototype& proto, const nnef::Dictionary<nnef::Typename>& dtypes, const nnef::Dictionary<nnef::Shape>& shapes )
    {
        for ( auto& it : shapes )
        {
            PyObject* name = PY_STRING_FROM_CSTR(it.first.c_str());
            PyObject* shape = buildPyListFromShape(it.second);
            PyObject* dtype = PY_STRING_FROM_CSTR(nnef::toString(dtypes.at(it.first)));
            PyObject* data = buildPyNone();
            PyObject* compression = buildPyNone();
            PyObject* quantization = PyDict_New();
            if ( quant.count(it.first) )
            {
                for ( auto& qit : quant.at(it.first) )
                {
                    PyDict_SetItemString(quantization, qit.first.c_str(), buildPyObjectFromValue(qit.second));
                }
            }
            
            PyObject* tensor = PyObject_CallObject(Tensor, PyTuple_Pack(6, name, dtype, shape, data, compression, quantization));
            PyDict_SetItemString(tensors, it.first.c_str(), tensor);
        }
    }

    virtual void operation( const nnef::Prototype& proto, const nnef::Dictionary<nnef::Value>& args,
                           const nnef::Dictionary<nnef::Typename>& dtypes, const nnef::Dictionary<nnef::Shape>& shapes )
    {
        PyObject* attribs = PyList_New(0);
        PyObject* inputs = PyList_New(0);
        PyObject* outputs = PyList_New(0);
        PyObject* dtype = args.count("?") ? PY_STRING_FROM_CSTR(args.at("?").string().c_str()) : buildPyNone();
        
        for ( size_t i = 0; i < proto.paramCount(); ++i )
        {
            auto& param = proto.param(i);
            auto& value = args.at(param.name());
            if ( param.type()->isAttribute() )
            {
                PyList_Append(attribs, PyTuple_Pack(2, PY_STRING_FROM_CSTR(param.name().c_str()), buildPyObjectFromValue(value)));
            }
            else
            {
                PyList_Append(inputs, PyTuple_Pack(2, PY_STRING_FROM_CSTR(param.name().c_str()), buildPyObjectFromValue(value)));
            }
        }
        for ( size_t i = 0; i < proto.resultCount(); ++i )
        {
            auto& result = proto.result(i);
            auto& value = args.at(result.name());
            PyList_Append(outputs, PyTuple_Pack(2, PY_STRING_FROM_CSTR(result.name().c_str()), buildPyObjectFromValue(value)));
        }
        
        attribs = PyObject_CallObject(OrderedDict, PyTuple_Pack(1, attribs));
        inputs = PyObject_CallObject(OrderedDict, PyTuple_Pack(1, inputs));
        outputs = PyObject_CallObject(OrderedDict, PyTuple_Pack(1, outputs));

        PyObject* operation = PyObject_CallObject(Operation, PyTuple_Pack(5, PY_STRING_FROM_CSTR(proto.name().c_str()), attribs, inputs, outputs, dtype));
        PyList_Append(operations, operation);
    }

    std::istream& qis;
    const char* qfn;
    nnef::Dictionary<nnef::Dictionary<nnef::Value>> quant;

    PyObject* tensors;
    PyObject* operations;
    PyObject* graph;
    PyObject* version;
    PyObject* extensions;
};


static PyObject* custom_shapes = PyDict_New();
static nnef::ShapeFuncs shape_funcs = nnef::standardShapeFuncs();


static void shapesToPy( const nnef::Value& value, const nnef::Dictionary<nnef::Shape>& shapes, PyObject* pyShapes )
{
    if ( value.kind() == nnef::Value::Identifier )
    {
        if ( shapes.count(value.identifier()) )
        {
            PyDict_SetItemString(pyShapes, value.identifier().c_str(), buildPyListFromShape(shapes.at(value.identifier())));
        }
    }
    else if ( value.kind() == nnef::Value::Array || value.kind() == nnef::Value::Tuple )
    {
        for ( auto& item : value.items() )
        {
            shapesToPy(item, shapes, pyShapes);
        }
    }
}

static void shapesFromPy( const nnef::Value& value, PyObject* pyShapes, nnef::Dictionary<nnef::Shape>& shapes, const char* prop_fn )
{
    if ( value.kind() == nnef::Value::Identifier )
    {
        if ( !shapes.count(value.identifier()) )
        {
            PyObject* shape = PyDict_GetItemString(pyShapes, value.identifier().c_str());
            if ( !shape )
            {
                throw nnef::Error("custom shape not set for tensor '%s' by function '%s'", value.identifier().c_str(), prop_fn);
            }
            shapes[value.identifier()] = shapeFromPyList(shape);
        }
    }
    else if ( value.kind() == nnef::Value::Array || value.kind() == nnef::Value::Tuple )
    {
        for ( auto& item : value.items() )
        {
            shapesFromPy(item, pyShapes, shapes, prop_fn);
        }
    }
}

static void customShapeFunc( const std::string& op, const std::map<std::string,nnef::Value>& args, std::map<std::string,nnef::Shape>& shapes )
{
    PyObject* pyShapes = PyDict_New();
    
    for ( auto& it : args )
    {
        shapesToPy(it.second, shapes, pyShapes);
    }
    
    PyObject* pyArgs = buildValuePyDict(args);
    
    auto func = PyDict_GetItemString(custom_shapes, op.c_str());
    if ( !PyObject_Call(func, PyTuple_Pack(3, PY_STRING_FROM_CSTR(op.c_str()), pyArgs, pyShapes), NULL) )
    {
        throw nnef::Error("call to custom shape propagator failed");
    }
    
    PyObject* name = PyObject_GetAttrString(func, "__name__");
    
    for ( auto& it : args )
    {
        shapesFromPy(it.second, pyShapes, shapes, PY_STRING_AS_CSTR(name));
    }
}


static nnef::CompParser parser(shape_funcs);


static PyObject* parse( PyObject* self, PyObject* args, PyObject* kwargs, bool isFile )
{
	const char* input = nullptr;
    const char* quant = nullptr;

    static const char* kwlist[] = { "", "quantization", NULL };

	if ( !PyArg_ParseTupleAndKeywords(args, kwargs, "s|z", (char**)kwlist, &input, &quant) )
    {
        return NULL;
    }

    std::ifstream fs, qfs;
    std::stringstream ss, qss;

    if ( isFile )
    {
        fs.open(input);
        if ( !fs )
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
        ss.str(input);
        if ( quant )
        {
            qss.str(quant);
        }
    }

    std::istream& is = isFile ? (std::istream&)fs : (std::istream&)ss;
    std::istream& qis = isFile ? (std::istream&)qfs : (std::istream&)qss;

    GraphCallback callback(qis, isFile ? quant : "quantization");

	try
    {
        parser.parse(is, isFile ? input : "input", callback);
        return callback.graph;
    }
    catch ( nnef::Error e )
    {
        PyErr_SetString(NNEF_Error, buildErrorString(e).c_str());
		return NULL;
    }
    catch ( std::invalid_argument e )
    {
        PyErr_SetString(PyExc_ValueError, e.what());
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

static PyObject* registerOps( const char* filename, const char* text )
{
    try
    {
        parser.import(filename, text);
    }
    catch ( nnef::Error e )
    {
        PyErr_SetString(NNEF_Error, buildErrorString(e).c_str());
        return NULL;
    }
    return buildPyNone();
}

static PyObject* unregisterOps( const char* filename )
{
    return buildPyBoolean(parser.unimport(filename));
}

static PyObject* registerCustomOps( PyObject* self, PyObject* args )
{
    const char* key;
    const char* text;
    if ( !PyArg_ParseTuple(args, "ss", &key, &text) )
    {
        return NULL;
    }

    return registerOps(key, text);
}

static PyObject* unregisterCustomOps( PyObject* self, PyObject* args )
{
    const char* key;
    if ( !PyArg_ParseTuple(args, "s", &key) )
    {
        return NULL;
    }

    return unregisterOps(key);
}

static PyObject* registerCustomShapes( PyObject* self, PyObject* args )
{
    PyObject* shapes;
    if ( !PyArg_ParseTuple(args, "O!", &PyDict_Type, &shapes) )
    {
        return NULL;
    }

    auto items = PyDict_Items(shapes);
    const size_t count = PyList_Size(items);
    for ( size_t i = 0; i < count; ++i )
    {
        auto item = PyList_GetItem(items, i);
        auto key = PyTuple_GetItem(item, 0);
        auto value = PyTuple_GetItem(item, 1);

        if ( !PY_STRING_CHECK(key) || (value != Py_None && !PyObject_HasAttrString(value, "__call__")) )
        {
            PyErr_SetString(NNEF_Error, "custom shapes must be a dict with string keys and callable values");
            return NULL;
        }

        if ( value != Py_None )
        {
            PyDict_SetItem(custom_shapes, key, value);
            shape_funcs[PY_STRING_AS_CSTR(key)] = customShapeFunc;
        }
        else
        {
            shape_funcs.erase(PY_STRING_AS_CSTR(key));
        }
    }
    return buildPyNone();
}


static PyMethodDef NNEF_Methods[] = 
{
    { "register_custom_ops", (PyCFunction)registerCustomOps, METH_VARARGS | METH_KEYWORDS, "Register custom operations to the parser" },
    { "unregister_custom_ops", (PyCFunction)unregisterCustomOps, METH_VARARGS | METH_KEYWORDS, "Unregister custom operations from the parser" },
    { "register_custom_shapes", (PyCFunction)registerCustomShapes, METH_VARARGS | METH_KEYWORDS, "Register custom shape propagation to the parser" },
	{ "parse_file", (PyCFunction)parseFile, METH_VARARGS | METH_KEYWORDS, "Parse the contents of a file" },
    { "parse_string", (PyCFunction)parseString, METH_VARARGS | METH_KEYWORDS, "Parse the contents of a string" },
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
	Py_INCREF(NNEF_Error);
	PyModule_AddObject(module, "Error", NNEF_Error);

    Py_INCREF(&NNEF_Identifier_Type);
    PyModule_AddObject(module, "Identifier", (PyObject*)&NNEF_Identifier_Type);

    auto collections = PyImport_ImportModule("collections");
    auto dict = PyModule_GetDict(collections);
    OrderedDict = PyDict_GetItemString(dict, "OrderedDict");
    NamedTuple = PyDict_GetItemString(dict, "namedtuple");

    Tensor = makeNamedTuple("Tensor", { "name", "dtype", "shape", "data", "compression", "quantization" });
    Py_INCREF(Tensor);
    PyModule_AddObject(module, "Tensor", Tensor);

    Operation = makeNamedTuple("Operation", { "name", "attribs", "inputs", "outputs", "dtype" });
    Py_INCREF(Operation);
    PyModule_AddObject(module, "Operation", Operation);
    
    Graph = makeNamedTuple("Graph", { "name", "tensors", "operations", "inputs", "outputs" });
    Py_INCREF(Graph);
    PyModule_AddObject(module, "Graph", Graph);
    
#if PY_MAJOR_VERSION >= 3
	return module;
#endif
}
