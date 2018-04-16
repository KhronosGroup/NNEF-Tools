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
#include <fstream>
#include <sstream>
#include <string>
#include <locale>


static PyObject* NNEF_Error;


#if PY_MAJOR_VERSION >= 3
#define PY_STRING_OBJECT PyUnicodeObject
#define PY_STRING_TYPE PyUnicode_Type
#define PY_STRING_CHECK PyUnicode_Check
#define PY_STRING_AS_CSTR PyUnicode_AsUTF8
#define PY_STRING_FROM_CSTR PyUnicode_FromString
#define PY_INTEGER_AS_LONG PyLong_AsLong
#else
#define PY_STRING_OBJECT PyStringObject
#define PY_STRING_TYPE PyString_Type
#define PY_STRING_CHECK PyString_Check
#define PY_STRING_AS_CSTR PyString_AsString
#define PY_STRING_FROM_CSTR PyString_FromString
#define PY_INTEGER_AS_LONG PyInt_AsLong
#endif


struct NNEF_Tensor
{
    PY_STRING_OBJECT str;
};

static PyTypeObject NNEF_Tensor_Type =
{
    PyVarObject_HEAD_INIT(NULL, 0)
    "_nnef.Tensor",          /* tp_name */
    sizeof(NNEF_Tensor),     /* tp_basicsize */
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


static PyObject* buildPyObjectFromValue( const nnef::Value& value )
{
    switch ( value.kind() )
    {
        case nnef::Value::None:
        {
            Py_RETURN_NONE;
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
            if ( value.logical() )
            {
                Py_RETURN_TRUE;
            }
            else
            {
                Py_RETURN_FALSE;
            }
        }
        case nnef::Value::String:
        {
            return PY_STRING_FROM_CSTR(value.string().c_str());
        }
        case nnef::Value::Tensor:
        {
            PyObject* arg = PY_STRING_FROM_CSTR(value.tensor().c_str());
            PyObject* obj = PyObject_CallObject((PyObject*)&NNEF_Tensor_Type, PyTuple_Pack(1, arg));
            Py_DECREF(arg);
            return obj;
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
}

static PyObject* buildPyListFromShape( const nnef::Shape& shape )
{
    PyObject* list = PyList_New(nnef::Shape::MaxRank);
    for ( size_t i = 0; i < nnef::Shape::MaxRank; ++i )
    {
        PyList_SetItem(list, i, Py_BuildValue("i", shape[i]));
    }
    return list;
}

static nnef::Shape shapeFromPyList( PyObject* list )
{
    nnef::Shape shape(1);
    const size_t size = PyList_Size(list);
    for ( size_t i = 0; i < size; ++i )
    {
        auto item = PyList_GetItem(list, i);
        shape[i] = (nnef::Shape::extent_type)PY_INTEGER_AS_LONG(item);
    }
    return shape;
}

static PyObject* buildParamDict( const nnef::Prototype& proto, const nnef::Dictionary<nnef::Value>& args )
{
    PyObject* params = PyList_New(proto.paramCount());
    for ( size_t i = 0; i < proto.paramCount(); ++i )
    {
        auto& param = proto.param(i).name();
        auto key = PY_STRING_FROM_CSTR(param.c_str());
        auto value = buildPyObjectFromValue(args[param]);
        PyList_SetItem(params, i, PyTuple_Pack(2, key, value));
    }

    return PyObject_CallObject(OrderedDict, PyTuple_Pack(1, params));
}

static PyObject* buildResultDict( const nnef::Prototype& proto, const nnef::Dictionary<nnef::Value>& args )
{
    PyObject* results = PyList_New(proto.resultCount());
    for ( size_t i = 0; i < proto.resultCount(); ++i )
    {
        auto& result = proto.result(i).name();
        auto key = PY_STRING_FROM_CSTR(result.c_str());
        auto value = buildPyObjectFromValue(args[result]);
        PyList_SetItem(results, i, PyTuple_Pack(2, key, value));
    }

    return PyObject_CallObject(OrderedDict, PyTuple_Pack(1, results));
}


struct GraphCallback : public nnef::Parser::Callback
{
    GraphCallback( const std::set<std::string>& atomics, PyObject* shape_prop )
    : atomics(atomics), shape_prop(shape_prop)
    {
    }

    virtual void beginGraph( const nnef::Prototype& proto )
    {
        name = PY_STRING_FROM_CSTR(proto.name().c_str());

        inputs = PyList_New(proto.paramCount());
        for ( size_t i = 0; i < proto.paramCount(); ++i )
        {
            PyList_SetItem(inputs, i, PY_STRING_FROM_CSTR(proto.param(i).name().c_str()));
        }

        outputs = PyList_New(proto.resultCount());
        for ( size_t i = 0; i < proto.resultCount(); ++i )
        {
            PyList_SetItem(outputs, i, PY_STRING_FROM_CSTR(proto.result(i).name().c_str()));
        }

        operations = PyList_New(0);
        shapes = PyDict_New();
    }

    virtual void operation( const nnef::Prototype& proto, const nnef::Dictionary<nnef::Value>& args,
                           const nnef::Dictionary<nnef::Shape>& shapes )
    {
        PyObject* name = PY_STRING_FROM_CSTR(proto.name().c_str());
        PyObject* params = buildParamDict(proto, args);
        PyObject* results = buildResultDict(proto, args);

        PyList_Append(operations, PyTuple_Pack(3, name, params, results));

        for ( size_t i = 0; i < proto.resultCount(); ++i )
        {
            auto& result = args[proto.result(i).name()];
            if ( result.kind() == nnef::Value::Tensor )
            {
                PyDict_SetItemString(this->shapes, result.tensor().c_str(), buildPyListFromShape(shapes[result.tensor()]));
            }
        }
    }

    virtual bool isAtomic( const nnef::Prototype& proto, const nnef::Dictionary<nnef::Value>& args )
    {
        return atomics.find(proto.name()) != atomics.end();
    }

    virtual bool propagate( const nnef::Prototype& proto, const nnef::Dictionary<nnef::Value>& args, nnef::Dictionary<nnef::Shape>& shapes )
    {
        if ( nnef::Parser::Callback::propagate(proto, args, shapes) )
        {
            return true;
        }
        auto prop = PyDict_GetItemString(shape_prop, proto.name().c_str());
        if ( prop )
        {
            PyObject* params = buildParamDict(proto, args);
            PyObject* results = buildResultDict(proto, args);

            PyObject_Call(prop, PyTuple_Pack(3, params, results, this->shapes), NULL);

            for ( size_t i = 0; i < proto.resultCount(); ++i )
            {
                auto& result = args[proto.result(i).name()];
                shapes[result.tensor()] = shapeFromPyList(PyDict_GetItemString(this->shapes, result.tensor().c_str()));
            }
            return true;
        }
        return false;
    }

    const std::set<std::string>& atomics;

    PyObject* name;
    PyObject* inputs;
    PyObject* outputs;
    PyObject* operations;
    PyObject* shapes;
    PyObject* shape_prop;
};


static std::string buildErrorString( nnef::Error e )
{
    std::string str = "Parse error: [" + std::to_string(e.position().line) + ":" + std::to_string(e.position().column) + "] " + e.what();

    auto origin = e.position().origin;
    while ( origin )
    {
        str += "\n... evaluated from [" + std::to_string(e.position().line) + ":" + std::to_string(e.position().column) + "]";
        origin = origin->origin;
    }

    return str;
}


static PyObject* parse( PyObject* self, PyObject* args, PyObject* kwargs, bool isFile )
{
	const char* input;
    bool flat = false;
    bool layers = false;
    PyObject* atomics = PyList_New(0);
    PyObject* shape_prop = PyDict_New();

    static const char* kwlist[] = { "", "flat", "layers", "atomics", "shape_prop", NULL };

	if ( !PyArg_ParseTupleAndKeywords(args, kwargs, "s|bbO!O!", (char**)kwlist, &input, &flat, &layers, &PyList_Type, &atomics, &PyDict_Type, &shape_prop) )
    {
        return NULL;
    }

    std::ifstream fs;
    std::stringstream ss;

    if ( isFile )
    {
        fs.open(input);
        if ( !fs )
        {
            const std::string message = "Could not open file: " + std::string(input);
            PyErr_SetString(NNEF_Error, message.c_str());
            return NULL;
        }
    }
    else
    {
        ss.str(input);
    }

    std::istream& is = isFile ? (std::istream&)fs : (std::istream&)ss;

    std::unique_ptr<nnef::Parser> parser(flat ? (nnef::Parser*)new nnef::FlatParser() : (nnef::Parser*)new nnef::CompParser(layers));

    std::set<std::string> atoms;
    const size_t atomCount = PyList_Size(atomics);
    for ( size_t i = 0; i < atomCount; ++i )
    {
        PyObject* item = PyList_GetItem(atomics, i);
        if ( PY_STRING_CHECK(item) )
        {
            atoms.insert(PY_STRING_AS_CSTR(item));
        }
        else
        {
            PyErr_SetString(NNEF_Error, "parameter 'atomics' must be a list of strings");
            return NULL;
        }
    }

    auto items = PyDict_Items(shape_prop);
    const size_t itemCount = PyList_Size(items);
    for ( size_t i = 0; i < itemCount; ++i )
    {
        auto item = PyList_GetItem(items, i);
        auto key = PyTuple_GetItem(item, 0);
        auto value = PyTuple_GetItem(item, 1);
        if ( !PY_STRING_CHECK(key) || !PyObject_HasAttrString(value, "__call__") )
        {
            PyErr_SetString(NNEF_Error, "parameter 'shape_prop' must be a dict with string keys to callable values");
            return NULL;
        }
    }

    GraphCallback callback(atoms, shape_prop);

	try
    {
        parser->parse(is, callback);

        PyObject* dict = PyDict_New();
        PyDict_SetItemString(dict, "name", callback.name);
        PyDict_SetItemString(dict, "inputs", callback.inputs);
        PyDict_SetItemString(dict, "outputs", callback.outputs);

        return PyTuple_Pack(3, dict, callback.operations, callback.shapes);
    }
    catch ( nnef::Error e )
    {
        PyErr_SetString(NNEF_Error, buildErrorString(e).c_str());
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
    NNEF_Tensor_Type.tp_base = &PY_STRING_TYPE;
    if ( PyType_Ready(&NNEF_Tensor_Type) < 0 )
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

	NNEF_Error = PyErr_NewException((char*)"_nnef.error", NULL, NULL);
	Py_INCREF(NNEF_Error);
	PyModule_AddObject(module, "error", NNEF_Error);

    Py_INCREF(&NNEF_Tensor_Type);
    PyModule_AddObject(module, "Tensor", (PyObject*)&NNEF_Tensor_Type);

    auto collections = PyImport_ImportModule("collections");
    auto dict = PyModule_GetDict(collections);
    OrderedDict = PyDict_GetItemString(dict, "OrderedDict");

#if PY_MAJOR_VERSION >= 3
	return module;
#endif
}
