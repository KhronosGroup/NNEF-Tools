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
#include "include/comp/layers_source.h"
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


static const size_t MaxDims = 8;


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

static PyObject* Prototype;
static PyObject* TensorType;
static PyObject* ArrayType;
static PyObject* TupleType;

static PyObject* ShapeOf;


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
        case nnef::Value::ShapeOf:
        {
            PyObject* arg = PY_STRING_FROM_CSTR(value.shape_of().id.c_str());
            return PyObject_CallObject(ShapeOf, PyTuple_Pack(1, arg));
        }
    }
}

static nnef::Value buildValueFromPyObject( PyObject* object )
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
}

static PyObject* buildPyListFromShape( const nnef::Shape& shape )
{
    PyObject* list = PyList_New(shape.rank());
    for ( size_t i = 0; i < shape.rank(); ++i )
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
        shape[i] = (nnef::Shape::extent_type)PY_INTEGER_AS_LONG(item);
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

static PyObject* buildPyTypespec( const nnef::Type* type )
{
    switch ( type->kind() )
    {
        case nnef::Type::Primitive:
        {
            auto primitiveType = dynamic_cast<const nnef::PrimitiveType*>(type);
            return PY_STRING_FROM_CSTR(nnef::toString(primitiveType->name()));
        }
        case nnef::Type::Tensor:
        {
            auto tensorType = dynamic_cast<const nnef::TensorType*>(type);
            auto dataType = tensorType->dataType() ? buildPyTypespec(tensorType->dataType()) : buildPyNone();
            return PyObject_CallObject(TensorType, PyTuple_Pack(1, dataType));
        }
        case nnef::Type::Array:
        {
            auto arrayType = dynamic_cast<const nnef::ArrayType*>(type);
            auto itemType = buildPyTypespec(arrayType->itemType());
            return PyObject_CallObject(ArrayType, PyTuple_Pack(1, itemType));
        }
        case nnef::Type::Tuple:
        {
            auto tupleType = dynamic_cast<const nnef::TupleType*>(type);
            PyObject* itemTypes = PyList_New(tupleType->size());
            for ( size_t i = 0; i < tupleType->size(); ++i )
            {
                PyList_SetItem(itemTypes, i, buildPyTypespec(tupleType->itemType(i)));
            }
            return PyObject_CallObject(TupleType, PyTuple_Pack(1, itemTypes));
        }
    }
}

static PyObject* getPyTypespec( const nnef::Type* type )
{
    static std::map<const nnef::Type*,PyObject*> typespecs;

    auto& typespec = typespecs[type];
    if ( !typespec )
    {
        typespec = buildPyTypespec(type);
    }
    return typespec;
}

static PyObject* buildPyPrototype( const nnef::Prototype& proto )
{
    PyObject* name = PY_STRING_FROM_CSTR(proto.name().c_str());

    PyObject* params = PyList_New(proto.paramCount());
    for ( size_t i = 0; i < proto.paramCount(); ++i )
    {
        auto& param = proto.param(i);
        auto key = PY_STRING_FROM_CSTR(param.name().c_str());
        auto value = getPyTypespec(param.type());
        PyList_SetItem(params, i, PyTuple_Pack(2, key, value));
    }
    params = PyObject_CallObject(OrderedDict, PyTuple_Pack(1, params));

    PyObject* defaults = PyDict_New();
    for ( size_t i = 0; i < proto.paramCount(); ++i )
    {
        auto& param = proto.param(i);
        if ( param.defaultValue() )
        {
            PyDict_SetItemString(defaults, param.name().c_str(), buildPyObjectFromValue(param.defaultValue()));
        }
    }
    if ( proto.genericParamDefault() )
    {
        PyDict_SetItemString(defaults, "?", PY_STRING_FROM_CSTR(nnef::toString(proto.genericParamDefault()->name())));
    }

    PyObject* results = PyList_New(proto.resultCount());
    for ( size_t i = 0; i < proto.resultCount(); ++i )
    {
        auto& result = proto.result(i);
        auto key = PY_STRING_FROM_CSTR(result.name().c_str());
        auto value = getPyTypespec(result.type());
        PyList_SetItem(results, i, PyTuple_Pack(2, key, value));
    }
    results = PyObject_CallObject(OrderedDict, PyTuple_Pack(1, results));

    PyObject* generic = buildPyBoolean(proto.isGeneric());

    return PyObject_CallObject(Prototype, PyTuple_Pack(5, name, params, results, defaults, generic));
}

static PyObject* getPyPrototype( const nnef::Prototype& proto )
{
    static std::map<std::string,PyObject*> protos;

    auto& dict = protos[proto.name()];
    if ( !dict )
    {
        dict = buildPyPrototype(proto);
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
    GraphCallback( const std::set<std::string>& atomics, std::istream& qis, const char* qfn )
    : atomics(atomics), qis(qis), qfn(qfn)
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
        quantizations = PyDict_New();
        if ( qis )
        {
            auto quant = nnef::QuantParser::parse(qis, qfn, fragments);

            for ( auto& it : quant )
            {
                PyDict_SetItemString(quantizations, it.first.c_str(), buildValuePyDict(it.second));
            }
        }

        graph = getPyPrototype(proto);

        operations = PyList_New(0);

        declarations = PyDict_New();
        for ( auto& it : fragments )
        {
            auto& proto = it.second;
            PyDict_SetItemString(declarations, proto.name().c_str(), getPyPrototype(proto));
        }
    }

    virtual void endGraph( const nnef::Prototype& proto, const nnef::Dictionary<nnef::Typename>& dtypes, const nnef::Dictionary<nnef::Shape>& shapes )
    {
        this->dtypes = PyDict_New();
        for ( auto& it : dtypes )
        {
            PyDict_SetItemString(this->dtypes, it.first.c_str(), PY_STRING_FROM_CSTR(nnef::toString(it.second)));
        }
    }

    virtual void operation( const nnef::Prototype& proto, const nnef::Dictionary<nnef::Value>& args,
                           const nnef::Dictionary<nnef::Typename>& dtypes, const nnef::Dictionary<nnef::Shape>& shapes )
    {
        PyObject* decl = getPyPrototype(proto);
        PyObject* dict = buildValuePyDict(args);

        PyList_Append(operations, PyTuple_Pack(2, decl, dict));
    }

    virtual bool isAtomic( const nnef::Prototype& proto, const nnef::Dictionary<nnef::Value>& args )
    {
        return atomics.find(proto.name()) != atomics.end();
    }

    const std::set<std::string>& atomics;
    std::istream& qis;
    const char* qfn;

    PyObject* graph;
    PyObject* operations;
    PyObject* dtypes;
    PyObject* version;
    PyObject* extensions;
    PyObject* declarations;
    PyObject* quantizations;
};


static PyObject* custom_shapes = PyDict_New();
static PyObject* defer_shapes = PyDict_New();


struct CustomPropagation : nnef::Propagation
{
    CustomPropagation() : Propagation(MaxDims)
    {
        shapes = PyDict_New();
    }

    virtual bool shouldDeferShapeOf( const nnef::Prototype& proto, const std::string& param ) const
    {
        auto defer = PyDict_GetItemString(defer_shapes, proto.name().c_str());
        if ( defer )
        {
            if ( PyList_Check(defer) )
            {
                const size_t size = PyList_Size(defer);
                for ( size_t i = 0; i < size; ++i )
                {
                    auto item = PyList_GetItem(defer, i);
                    const char* str = PY_STRING_AS_CSTR(item);
                    if ( param == str )
                    {
                        return true;
                    }
                }
                return false;
            }
            else
            {
                const char* str = PY_STRING_AS_CSTR(defer);
                return param == str;
            }
        }
        return false;
    }

    virtual void propagateShapes( const nnef::Prototype& proto, const nnef::Dictionary<nnef::Value>& args, nnef::Dictionary<nnef::Shape>& shapes )
    {
        auto prop = PyDict_GetItemString(custom_shapes, proto.name().c_str());
        if ( prop )
        {
            PyObject* pyProto = getPyPrototype(proto);
            PyObject* pyArgs = buildValuePyDict(args);

            if ( !PyObject_Call(prop, PyTuple_Pack(3, pyProto, pyArgs, this->shapes), NULL) )
            {
                throw nnef::Error("call to custom shape propagator failed");
            }

            PyObject* name = PyObject_GetAttrString(prop, "__name__");

            for ( size_t i = 0; i < proto.resultCount(); ++i )
            {
                auto& result = args[proto.result(i).name()];
                shapesFromPy(shapes, result, PY_STRING_AS_CSTR(name));
            }
        }
        else
        {
            nnef::Propagation::propagateShapes(proto, args, shapes);

            for ( size_t i = 0; i < proto.resultCount(); ++i )
            {
                auto& result = args[proto.result(i).name()];
                shapesToPy(shapes, result);
            }
        }
    }

    void shapesFromPy( nnef::Dictionary<nnef::Shape>& shapes, const nnef::Value& value, const char* prop_fn )
    {
        if ( value.kind() == nnef::Value::Identifier )
        {
            PyObject* shape = PyDict_GetItemString(this->shapes, value.identifier().c_str());
            if ( !shape )
            {
                throw nnef::Error("custom shape not set for tensor '%s' by function '%s'", value.identifier().c_str(), prop_fn);
            }
            shapes[value.identifier()] = shapeFromPyList(shape);
        }
        else if ( value.kind() == nnef::Value::Array || value.kind() == nnef::Value::Tuple )
        {
            for ( auto& item : value.items() )
            {
                shapesFromPy(shapes, item, prop_fn);
            }
        }
    }

    void shapesToPy( nnef::Dictionary<nnef::Shape>& shapes, const nnef::Value& value )
    {
        if ( value.kind() == nnef::Value::Identifier )
        {
            PyDict_SetItemString(this->shapes, value.identifier().c_str(), buildPyListFromShape(shapes[value.identifier()]));
        }
        else if ( value.kind() == nnef::Value::Array || value.kind() == nnef::Value::Tuple )
        {
            for ( auto& item : value.items() )
            {
                shapesToPy(shapes, item);
            }
        }
    }

    PyObject* shapes;
};


static CustomPropagation propagation;
static nnef::CompParser parser(propagation);


static PyObject* parse( PyObject* self, PyObject* args, PyObject* kwargs, bool isFile )
{
	const char* input = nullptr;
    const char* quant = nullptr;
    PyObject* atomics = PyList_New(0);

    static const char* kwlist[] = { "", "quantization", "atomics", NULL };

	if ( !PyArg_ParseTupleAndKeywords(args, kwargs, "s|zO!", (char**)kwlist, &input, &quant, &PyList_Type, &atomics) )
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

    GraphCallback callback(atoms, qis, isFile ? quant : "quantization");

	try
    {
        parser.parse(is, isFile ? input : "input", callback);

        PyObject* dict = PyDict_New();
        PyDict_SetItemString(dict, "graph", callback.graph);
        PyDict_SetItemString(dict, "dtypes", callback.dtypes);
        PyDict_SetItemString(dict, "shapes", propagation.shapes);
        PyDict_SetItemString(dict, "version", callback.version);
        PyDict_SetItemString(dict, "extensions", callback.extensions);
        PyDict_SetItemString(dict, "declarations", callback.declarations);
        PyDict_SetItemString(dict, "quantizations", callback.quantizations);

        return PyTuple_Pack(2, dict, callback.operations);
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

static PyObject* registerLayerOps( PyObject* self )
{
    return registerOps("layers", nnef::layers_source());
}

static PyObject* unregisterLayerOps( PyObject* self )
{
    return unregisterOps("layers");
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
        }
        else
        {
            PyDict_DelItem(custom_shapes, key);
        }
    }
    return buildPyNone();
}

static PyObject* registerDeferredShapes( PyObject* self, PyObject* args )
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

        if ( !PY_STRING_CHECK(key) )
        {
            PyErr_SetString(NNEF_Error, "deferred shapes must be a dict with string keys");
            return NULL;
        }

        if ( PyList_Check(value) )
        {
            const size_t listSize = PyList_Size(value);
            for ( size_t j = 0; j < listSize; ++j )
            {
                auto listItem = PyList_GetItem(value, j);
                if ( !PY_STRING_CHECK(listItem) )
                {
                    PyErr_SetString(NNEF_Error, "deferred shapes must be a dict with (list of) string values");
                    return NULL;
                }
            }
        }
        else if ( !PY_STRING_CHECK(value) )
        {
            PyErr_SetString(NNEF_Error, "deferred shapes must be a dict with (list of) string values");
            return NULL;
        }

        PyDict_SetItem(defer_shapes, key, value);
    }
    return buildPyNone();
}

static PyObject* enumerateStandardOperations()
{
    const std::vector<nnef::Prototype> prototypes = nnef::stdlibPrototypes();

    PyObject* list = PyList_New(prototypes.size());
    for ( size_t i = 0; i < prototypes.size(); ++i )
    {
        PyList_SetItem(list, i, PY_STRING_FROM_CSTR(prototypes[i].name().c_str()));
    }
    return list;
}


static PyObject* StandardOperations = enumerateStandardOperations();


static PyMethodDef NNEF_Methods[] = 
{
    { "register_layer_ops", (PyCFunction)registerLayerOps, METH_VARARGS | METH_KEYWORDS, "Register layer operations to the parser" },
    { "unregister_layer_ops", (PyCFunction)unregisterLayerOps, METH_VARARGS | METH_KEYWORDS, "Unregister layer operations from the parser" },
    { "register_custom_ops", (PyCFunction)registerCustomOps, METH_VARARGS | METH_KEYWORDS, "Register custom operations to the parser" },
    { "unregister_custom_ops", (PyCFunction)unregisterCustomOps, METH_VARARGS | METH_KEYWORDS, "Unregister custom operations from the parser" },
    { "register_custom_shapes", (PyCFunction)registerCustomShapes, METH_VARARGS | METH_KEYWORDS, "Register custom shape propagation to the parser" },
    { "register_deferred_shapes", (PyCFunction)registerDeferredShapes, METH_VARARGS | METH_KEYWORDS, "Register deferred shapes to the parser" },
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

    Py_INCREF(StandardOperations);
    PyModule_AddObject(module, "StandardOperations", StandardOperations);

    auto collections = PyImport_ImportModule("collections");
    auto dict = PyModule_GetDict(collections);
    OrderedDict = PyDict_GetItemString(dict, "OrderedDict");
    NamedTuple = PyDict_GetItemString(dict, "namedtuple");

    Prototype = makeNamedTuple("Prototype", { "name", "params", "results", "defaults", "generic" });
    Py_INCREF(Prototype);
    PyModule_AddObject(module, "Prototype", Prototype);

    TensorType = makeNamedTuple("TensorType", { "dataType" });
    Py_INCREF(TensorType);
    PyModule_AddObject(module, "TensorType", TensorType);

    ArrayType = makeNamedTuple("ArrayType", { "itemType" });
    Py_INCREF(ArrayType);
    PyModule_AddObject(module, "ArrayType", ArrayType);

    TupleType = makeNamedTuple("TupleType", { "itemTypes" });
    Py_INCREF(TupleType);
    PyModule_AddObject(module, "TupleType", TupleType);

    ShapeOf = makeNamedTuple("ShapeOf", { "id" });
    Py_INCREF(ShapeOf);
    PyModule_AddObject(module, "ShapeOf", ShapeOf);

#if PY_MAJOR_VERSION >= 3
	return module;
#endif
}
