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
#include "nnef/flat/flat_parser.h"
#include "nnef/comp/comp_parser.h"
#include "nnef/flat/quant_parser.h"
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
                for ( auto& qit : quant.at(it.first) )
                {
                    PyDict_SetItemString(quantization, qit.first.c_str(), buildPyObjectFromValue(qit.second));
                }
            }
            
            PyObject* tensor = PyObject_CallObject(Tensor, PyTuple_Pack(5, name, dtype, shape, data, quantization));
            PyDict_SetItemString(tensors, it.first.c_str(), tensor);
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
    
    std::string stdlib_source;
    if ( stdlib )
    {
        stdlib_source = stdlib;
    }
    
    std::set<std::string> lowered;
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
    
    nnef::CompParser parser(stdlib_source, lowered);

    GraphCallback callback(qis, isFile ? quant : "quantization");

	try
    {
        parser.parse(gis, isFile ? input : "input", callback);
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


static PyMethodDef NNEF_Methods[] = 
{
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

    Tensor = makeNamedTuple("Tensor", { "name", "dtype", "shape", "data", "quantization" });
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
