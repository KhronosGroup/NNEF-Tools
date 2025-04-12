#include "Python.h"
#include "structmember.h"
#include "numpy/arrayobject.h"
#include "skriptnd.h"
#include <initializer_list>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <locale>
#include <cassert>

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


static PyObject* OrderedDict;
static PyObject* DataClass;
static PyObject* DataClassField;
static PyObject* Enum;
static PyObject* Signature;

static PyObject* Dtype;
static PyObject* Position;
static PyObject* Contraction;
static PyObject* Tensor;
static PyObject* TensorPack;
static PyObject* Assertion;
static PyObject* Operation;
static PyObject* Graph;
static PyObject* Model;

static PyObject* Expr;
static PyObject* SizeAccess;
static PyObject* ShapeAccess;
static PyObject* TensorAccess;
static PyObject* PlaceholderExpr;
static PyObject* IdentifierExpr;
static PyObject* IdentifierKind;
static PyObject* ReferenceExpr;
static PyObject* CastExpr;
static PyObject* UnaryExpr;
static PyObject* BinaryExpr;
static PyObject* SelectExpr;
static PyObject* FoldExpr;
static PyObject* ListExpr;
static PyObject* BoundedExpr;
static PyObject* ConcatExpr;
static PyObject* SliceExpr;
static PyObject* SubscriptExpr;
static PyObject* UniformExpr;
static PyObject* RangeExpr;


static PyObject* EmptyTupleDefault = PyTuple_New(0);
static PyObject* EmptyListDefault = (PyObject*)&PyList_Type;
static PyObject* EmptyDictDefault = (PyObject*)&PyDict_Type;


struct BuildContext
{
    std::map<const sknd::Tensor*,PyObject*> tensors;
    std::map<const sknd::TensorPack*,PyObject*> packs;
    std::map<const sknd::ValueExpr*,PyObject*> subexprs;
};

static BuildContext EmptyBuildContext = BuildContext();


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

// make tuple by STEALING references to items
static PyObject* makePyTuple( std::initializer_list<PyObject*> items )
{
    PyObject* tuple = PyTuple_New(items.size());

    size_t i = 0;
    for ( auto& item : items )
    {
        PyTuple_SetItem(tuple, i++, item);
    }
    return tuple;
}

// make dict by STEALING references to items
static PyObject* makePyDict( std::initializer_list<std::pair<const char*,PyObject*>> items )
{
    PyObject* dict = PyDict_New();

    for ( auto& [key, value] : items )
    {
        PyDict_SetItemString(dict, key, value);
        Py_DECREF(value);
    }
    return dict;
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

static PyObject* callPyFunc( PyObject* func, std::initializer_list<PyObject*> args,
                             std::initializer_list<std::pair<const char*,PyObject*>> kwargs = {} )
{
    PyObject* pyArgs = makePyTuple(args);
    PyObject* pyKwargs = makePyDict(kwargs);
    PyObject* result = PyObject_Call(func, pyArgs, pyKwargs);
    Py_DECREF(pyArgs);
    Py_DECREF(pyKwargs);
    return result;
}

static PyObject* makeDataClass( PyObject* module, PyObject* base, const char* name,
                                std::initializer_list<const char*> fields,
                                std::initializer_list<PyObject*> defaults = {} )
{
    const size_t firstDefault = fields.size() - defaults.size();

    PyObject* pyName = PY_STRING_FROM_CSTR(name);

    PyObject* pyFields = PyList_New(fields.size());
    size_t i = 0;
    for ( auto& field : fields )
    {
        PyObject* fieldName = PY_STRING_FROM_CSTR(field);
        if ( i < firstDefault )
        {
            PyList_SetItem(pyFields, i++, fieldName);
        }
        else
        {
            PyObject* fieldType = PY_STRING_FROM_CSTR("typing.Any");
            PyObject* defaultValue = defaults.begin()[i - firstDefault];
            const char* key = defaultValue == EmptyListDefault || defaultValue == EmptyDictDefault ?
                              "default_factory" : "default";
            PyObject* fieldInfo = callPyFunc(DataClassField, {}, { {key, defaultValue} });
            PyObject* tuple = makePyTuple(fieldName, fieldType, fieldInfo);
            PyList_SetItem(pyFields, i++, tuple);
        }
    }

    PyObject* pyBases = base ? PyTuple_Pack(1, base) : PyTuple_Pack(0);

    PyObject* obj = callPyFunc(DataClass, { pyName, pyFields }, { {"bases", pyBases} });
    PyObject_SetAttrString(obj, "__module__", PyModule_GetNameObject(module));
    PyModule_AddObject(module, name, obj);
    return obj;
}

static PyObject* makeDataClass( PyObject* module, const char* name,
                                std::initializer_list<const char*> fields,
                                std::initializer_list<PyObject*> defaults = {} )
{
    return makeDataClass(module, nullptr, name, fields, defaults);
}

static PyObject* makeEnum( PyObject* module, const char* name, std::initializer_list<const char*> fields,
                           const size_t first_value = 0 )
{
    PyObject* pyName = PY_STRING_FROM_CSTR(name);

    PyObject* pyFields = PyList_New(fields.size());
    size_t i = 0;
    for ( auto& field : fields )
    {
        PyObject* key = PY_STRING_FROM_CSTR(field);
        PyObject* value = Py_BuildValue("i", (int)(first_value + i));
        PyObject* pair = makePyTuple(key, value);
        PyList_SetItem(pyFields, i++, pair);
    }

    PyObject* obj = makePyObject(Enum, pyName, pyFields);
    PyObject_SetAttrString(obj, "__module__", PyModule_GetNameObject(module));
    PyModule_AddObject(module, name, obj);
    return obj;
}

static PyObject* makeEmptyNumpyArray( int dtype = NPY_FLOAT32 )
{
    npy_intp dim = 0;
    return PyArray_SimpleNewFromData(1, &dim, dtype, (void*)nullptr);
}


static PyObject* buildPyBoolean( const sknd::bool_t& value )
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

static PyObject* buildPyInt( const sknd::int_t& value )
{
    return Py_BuildValue("i", (int)value);
}

static PyObject* buildPyReal( const sknd::real_t& value )
{
    return Py_BuildValue("f", (float)value);
}

static PyObject* buildPyStr( const sknd::str_t& value )
{
    return PY_STRING_FROM_CSTR(value.c_str());
}

static PyObject* buildPyNone()
{
    Py_RETURN_NONE;
}

static PyObject* buildPyDtype( const sknd::Typename dtype )
{
    PyObject* value = buildPyInt((int)dtype);
    return makePyObject(Dtype, value);
}

static PyObject* buildPyIdentifierKind( const sknd::ValueExpr::IdentifierKind kind )
{
    PyObject* value = buildPyInt((int)kind);
    return makePyObject(IdentifierKind, value);
}

static sknd::Typename dtypeFromPyObject( PyObject* obj )
{
    PyObject* value = PyObject_GetAttrString(obj, "value");
    auto dtype = (sknd::Typename)PY_INTEGER_AS_LONG(value);
    Py_DECREF(value);
    return dtype;
}

static PyObject* buildPyTensorRef( const sknd::TensorRef& ref, const BuildContext& context )
{
    if ( ref == nullptr )
    {
        return buildPyNone();
    }
    else if ( ref.packed() )
    {
        PyObject* pack = context.packs.at(ref.as<sknd::TensorPack*>());
        Py_INCREF(pack);
        return pack;
    }
    else
    {
        PyObject* tensor = context.tensors.at(ref.as<sknd::Tensor*>());
        Py_INCREF(tensor);
        return tensor;
    }
}

static PyObject* buildPyValueExpr( const sknd::ValueExpr& expr, const BuildContext& context )
{
    switch ( expr.kind() )
    {
        case sknd::ValueExpr::Null:
        {
            return buildPyNone();
        }
        case sknd::ValueExpr::Literal:
        {
            switch ( expr.dtype() )
            {
                case sknd::Typename::Type:
                case sknd::Typename::Arith:
                case sknd::Typename::Num:
                {
                    assert(false);
                    return buildPyNone();
                }
                case sknd::Typename::Int:
                {
                    return buildPyInt(expr.as_int());
                }
                case sknd::Typename::Real:
                {
                    return buildPyReal(expr.as_real());
                }
                case sknd::Typename::Bool:
                {
                    return buildPyBoolean(expr.as_bool());
                }
                case sknd::Typename::Str:
                {
                    return buildPyStr(expr.as_str());
                }
            }
        }
        case sknd::ValueExpr::Placeholder:
        {
            auto& placeholder = expr.as_placeholder();
            PyObject* id = buildPyStr(placeholder.id);
            PyObject* max_value = buildPyValueExpr(placeholder.max_value, context);
            return makePyObject(PlaceholderExpr, id, max_value);
        }
        case sknd::ValueExpr::Identifier:
        {
            auto& identifier = expr.as_identifier();
            PyObject* name = buildPyStr(identifier.name);
            PyObject* kind = buildPyIdentifierKind(identifier.kind);
            PyObject* dtype = buildPyDtype(expr.dtype());
            return makePyObject(IdentifierExpr, name, kind, dtype);
        }
        case sknd::ValueExpr::Reference:
        {
            auto& reference = expr.as_reference();
            PyObject* name = buildPyStr(reference.name);
            PyObject* target = context.subexprs.at(reference.target); Py_INCREF(target);
            PyObject* dtype = buildPyDtype(expr.dtype());
            return makePyObject(ReferenceExpr, name, target, dtype);
        }
        case sknd::ValueExpr::SizeAccess:
        {
            auto& access = expr.as_size_access();
            PyObject* pack = buildPyTensorRef(access.pack, context);
            return makePyObject(SizeAccess, pack);
        }
        case sknd::ValueExpr::ShapeAccess:
        {
            auto& access = expr.as_shape_access();
            PyObject* tensor = buildPyTensorRef(access.tensor, context);
            PyObject* dim = buildPyValueExpr(access.dim, context);
            PyObject* item = buildPyValueExpr(access.item, context);
            PyObject* max_size = expr.packed() ? buildPyInt(expr.max_size()) : buildPyNone();
            return makePyObject(ShapeAccess, tensor, dim, item, max_size);
        }
        case sknd::ValueExpr::TensorAccess:
        {
            auto& access = expr.as_tensor_access();
            PyObject* tensor = buildPyTensorRef(access.tensor, context);
            PyObject* item = buildPyValueExpr(access.item, context);
            PyObject* indices = PyList_New(access.indices.size());

            size_t i = 0;
            for ( auto& index : access.indices )
            {
                PyObject* item = buildPyValueExpr(index, context);
                PyList_SetItem(indices, i++, item);
            }
            return makePyObject(TensorAccess, tensor, indices, item);
        }
        case sknd::ValueExpr::Unary:
        {
            auto& unary = expr.as_unary();
            PyObject* op = buildPyStr(unary.op);
            PyObject* arg = buildPyValueExpr(unary.arg, context);
            PyObject* dtype = buildPyDtype(expr.dtype());
            PyObject* max_size = expr.packed() ? buildPyInt(expr.max_size()) : buildPyNone();
            return makePyObject(UnaryExpr, op, arg, dtype, max_size);
        }
        case sknd::ValueExpr::Binary:
        {
            auto& binary = expr.as_binary();
            PyObject* op = buildPyStr(binary.op);
            PyObject* left = buildPyValueExpr(binary.left, context);
            PyObject* right = buildPyValueExpr(binary.right, context);
            PyObject* dtype = buildPyDtype(expr.dtype());
            PyObject* max_size = expr.packed() ? buildPyInt(expr.max_size()) : buildPyNone();
            return makePyObject(BinaryExpr, op, left, right, dtype, max_size);
        }
        case sknd::ValueExpr::Select:
        {
            auto& select = expr.as_select();
            PyObject* cond = buildPyValueExpr(select.cond, context);
            PyObject* left = buildPyValueExpr(select.left, context);
            PyObject* right = buildPyValueExpr(select.right, context);
            PyObject* dtype = buildPyDtype(expr.dtype());
            PyObject* max_size = expr.packed() ? buildPyInt(expr.max_size()) : buildPyNone();
            return makePyObject(SelectExpr, cond, left, right, dtype, max_size);
        }
        case sknd::ValueExpr::Fold:
        {
            auto& fold = expr.as_fold();
            PyObject* op = buildPyStr(fold.op);
            PyObject* pack = buildPyValueExpr(fold.pack, context);
            PyObject* dtype = buildPyDtype(expr.dtype());
            PyObject* max_size = expr.packed() ? buildPyInt(expr.max_size()) : buildPyNone();
            return makePyObject(FoldExpr, op, pack, dtype, max_size);
        }
        case sknd::ValueExpr::Cast:
        {
            auto& cast = expr.as_cast();
            PyObject* arg = buildPyValueExpr(cast.arg, context);
            PyObject* dtype = buildPyDtype(expr.dtype());
            PyObject* max_size = expr.packed() ? buildPyInt(expr.max_size()) : buildPyNone();
            return makePyObject(CastExpr, arg, dtype, max_size);
        }
        case sknd::ValueExpr::List:
        {
            PyObject* items = PyList_New(expr.max_size());
            for ( size_t i = 0; i < expr.max_size(); ++i )
            {
                PyList_SetItem(items, i, buildPyValueExpr(expr[i], context));
            }
            PyObject* dtype = buildPyDtype(expr.dtype());
            return makePyObject(ListExpr, items, dtype);
        }
        case sknd::ValueExpr::Bounded:
        {
            auto& bounded = expr.as_bounded();
            PyObject* arg = buildPyValueExpr(bounded.arg, context);
            PyObject* lower = buildPyValueExpr(bounded.lower, context);
            PyObject* upper = buildPyValueExpr(bounded.upper, context);
            PyObject* dtype = buildPyDtype(expr.dtype());
            return makePyObject(BoundedExpr, arg, lower, upper, dtype);
        }
        case sknd::ValueExpr::Concat:
        {
            auto& concat = expr.as_concat();
            PyObject* items = PyList_New(concat.items.size());
            for ( size_t i = 0; i < concat.items.size(); ++i )
            {
                PyList_SetItem(items, i, buildPyValueExpr(concat.items[i], context));
            }
            PyObject* dtype = buildPyDtype(expr.dtype());
            PyObject* size = buildPyValueExpr(expr.size(), context);
            PyObject* max_size = expr.packed() ? buildPyInt(expr.max_size()) : buildPyNone();
            return makePyObject(ConcatExpr, items, dtype, size, max_size);
        }
        case sknd::ValueExpr::Slice:
        {
            auto& slice = expr.as_slice();
            PyObject* pack = buildPyValueExpr(slice.pack, context);
            PyObject* first = buildPyValueExpr(slice.first, context);
            PyObject* last = buildPyValueExpr(slice.last, context);
            PyObject* stride = buildPyValueExpr(slice.stride, context);
            PyObject* dtype = buildPyDtype(expr.dtype());
            PyObject* size = buildPyValueExpr(expr.size(), context);
            PyObject* max_size = expr.packed() ? buildPyInt(expr.max_size()) : buildPyNone();
            return makePyObject(SliceExpr, pack, first, last, stride, dtype, size, max_size);
        }
        case sknd::ValueExpr::Subscript:
        {
            auto& subscript = expr.as_subscript();
            PyObject* pack = buildPyValueExpr(subscript.pack, context);
            PyObject* index = buildPyValueExpr(subscript.index, context);
            PyObject* dtype = buildPyDtype(expr.dtype());
            PyObject* max_size = expr.packed() ? buildPyInt(expr.max_size()) : buildPyNone();
            return makePyObject(SubscriptExpr, pack, index, dtype, max_size);
        }
        case sknd::ValueExpr::Uniform:
        {
            auto& uniform = expr.as_uniform();
            PyObject* value = buildPyValueExpr(uniform.value, context);
            PyObject* size = buildPyValueExpr(uniform.size, context);
            PyObject* dtype = buildPyDtype(expr.dtype());
            PyObject* max_size = expr.packed() ? buildPyInt(expr.max_size()) : buildPyNone();
            return makePyObject(UniformExpr, value, size, dtype, max_size);
        }
        case sknd::ValueExpr::Range:
        {
            auto& range = expr.as_range();
            PyObject* first = buildPyValueExpr(range.first, context);
            PyObject* last = buildPyValueExpr(range.last, context);
            PyObject* stride = buildPyValueExpr(range.stride, context);
            PyObject* dtype = buildPyDtype(expr.dtype());
            PyObject* size = buildPyValueExpr(expr.size(), context);
            PyObject* max_size = expr.packed() ? buildPyInt(expr.max_size()) : buildPyNone();
            return makePyObject(RangeExpr, first, last, stride, dtype, size, max_size);
        }
    }
    assert(false);
    return buildPyNone();
}

static sknd::ValueExpr valueFromPyObject( PyObject* obj )
{
    if ( PyList_Check(obj) )
    {
        size_t size = PyList_Size(obj);
        if ( !size )
        {
            return sknd::ValueExpr::empty();
        }
        std::vector<sknd::ValueExpr> items(size);
        for ( int i = 0; i < size; ++i )
        {
            auto item = PyList_GetItem(obj, i);
            items[i] = valueFromPyObject(item);
        }
        auto dtype = items.front().dtype();
        return sknd::ValueExpr::list(std::move(items), dtype);
    }
    else if ( PY_INTEGER_CHECK(obj) )
    {
        return sknd::ValueExpr((sknd::int_t)PY_INTEGER_AS_LONG(obj));
    }
    else if ( PyFloat_Check(obj) )
    {
        return sknd::ValueExpr((sknd::real_t)PyFloat_AsDouble(obj));
    }
    else if ( PyBool_Check(obj) )
    {
        return sknd::ValueExpr((sknd::bool_t)(obj == Py_True));
    }
    else if ( PY_STRING_CHECK(obj) )
    {
        return sknd::ValueExpr((sknd::str_t)PY_STRING_AS_CSTR(obj));
    }
    else if ( obj == Py_None )
    {
        return sknd::ValueExpr(nullptr);
    }
    else if ( PyObject_TypeCheck(obj, (PyTypeObject*)IdentifierExpr) )
    {
        PyObject* pyName = PyObject_GetAttrString(obj, "name");
        PyObject* pyDtype = PyObject_GetAttrString(obj, "dtype");

        auto name = (sknd::str_t)PY_STRING_AS_CSTR(pyName);
        auto dtype = dtypeFromPyObject(pyDtype);

        Py_DECREF(pyName);
        Py_DECREF(pyDtype);

        return sknd::ValueExpr(sknd::ValueExpr::IdentifierExpr{ name }, dtype);
    }
    else
    {
        assert(false);
        return sknd::ValueExpr(nullptr);
    }
}

static PyObject* buildPyPosition( const sknd::Position& position )
{
    return makePyObject(Position,
        buildPyStr(position.module),
        buildPyInt(position.line),
        buildPyInt(position.column)
    );
}

static PyObject* buildPyList( const std::vector<size_t>& values )
{
    PyObject* list = PyList_New(values.size());
    for ( size_t i = 0; i < values.size(); ++i )
    {
        PyList_SetItem(list, i, Py_BuildValue("l", (long)values[i]));
    }
    return list;
}

static PyObject* buildPyValueExprs( const std::vector<sknd::ValueExpr>& exprs, const BuildContext& context )
{
    PyObject* items = PyList_New(exprs.size());
    for ( size_t i = 0; i < exprs.size(); ++i )
    {
        PyObject* item = buildPyValueExpr(exprs[i], context);
        PyList_SetItem(items, i, item);
    }
    return items;
}

static PyObject* buildPyShape( const std::vector<sknd::int_t>& shape )
{
    PyObject* tuple = PyTuple_New(shape.size());
    for ( size_t i = 0; i < shape.size(); ++i )
    {
        PyTuple_SetItem(tuple, i, buildPyInt(shape[i]));
    }
    return tuple;
}

static PyObject* buildPyShape( const std::vector<sknd::ValueExpr>& shape, const BuildContext& context,
                               const bool static_only = false )
{
    PyObject* items = PyTuple_New(shape.size());
    for ( size_t i = 0; i < shape.size(); ++i )
    {
        PyObject* item = static_only && shape[i].is_dynamic() ? buildPyNone() : buildPyValueExpr(shape[i], context);
        PyTuple_SetItem(items, i, item);
    }
    return items;
}

static PyObject* buildPyAttribs( const std::map<std::string,sknd::ValueExpr>& items, const BuildContext& context,
                                 const bool static_only = false )
{
    PyObject* dict = PyDict_New();
    for ( auto& [key, value] : items )
    {
        PyObject* obj = static_only && value.is_dynamic() ? buildPyNone() : buildPyValueExpr(value, context);
        PyDict_SetItemString(dict, key.c_str(), obj);
        Py_DECREF(obj);
    }
    return dict;
}

static PyObject* buildPySubexprs( const sknd::OrderedDict<sknd::ValueExpr>& items, BuildContext& context )
{
    PyObject* dict = PyDict_New();
    for ( auto& [key, value] : items )
    {
        PyObject* obj = buildPyValueExpr(value, context);
        PyDict_SetItemString(dict, key.c_str(), obj);
        Py_DECREF(obj);
        context.subexprs[&value] = obj;
    }
    return dict;
}

static PyObject* buildPyDtypes( const std::map<std::string,sknd::Typename>& dtypes )
{
    PyObject* dict = PyDict_New();
    for ( auto& [key, dtype] : dtypes )
    {
        PyObject* obj = buildPyDtype(dtype);
        PyDict_SetItemString(dict, key.c_str(), obj);
        Py_DECREF(obj);
    }
    return dict;
}

static PyObject* buildPyShapes( const std::vector<sknd::TensorRef>& tensors, const BuildContext& context,
                                const bool static_only = false )
{
    PyObject* shapes = PyList_New(tensors.size());
    for ( size_t i = 0; i < tensors.size(); ++i )
    {
        PyObject* shape = tensors[i] == nullptr ? buildPyNone() : buildPyShape(tensors[i].shape(), context, static_only);
        PyList_SetItem(shapes, i, shape);
    }
    return shapes;
}

static PyObject* buildPyTensor( const sknd::Tensor& tensor, const BuildContext& context )
{
    PyObject* name = buildPyStr(tensor.name);
    PyObject* dtype = buildPyDtype(tensor.dtype);
    PyObject* shape = Py_None; // deferred
    PyObject* max_shape = buildPyShape(tensor.max_shape);
    PyObject* quant = buildPyAttribs(tensor.quant, context);
    PyObject* value = buildPyValueExpr(tensor.value, context);
    PyObject* variable = buildPyBoolean(tensor.variable);

    return makePyObject(Tensor, name, dtype, shape, max_shape, quant, value, variable);
}

static PyObject* buildPyTensorPack( const sknd::TensorPack& pack, const BuildContext& context )
{
    PyObject* name = buildPyStr(pack.name);
    PyObject* dtype = buildPyDtype(pack.dtype);
    PyObject* shape = Py_None;  // deferred
    PyObject* max_shape = buildPyShape(pack.max_shape);
    PyObject* size = Py_None;   // deferred

    PyObject* items = PyList_New(pack.items.size());
    for ( size_t i = 0; i < pack.items.size(); ++i )
    {
        PyObject* item = context.tensors.at(pack.items[i]);
        Py_INCREF(item);
        PyList_SetItem(items, i, item);
    }

    return makePyObject(TensorPack, name, dtype, shape, max_shape, size, items);
}

static PyObject* buildPyContraction( const sknd::Contraction& contraction, const BuildContext& context )
{
    PyObject* left = buildPyValueExpr(contraction.left, context);
    PyObject* right = buildPyValueExpr(contraction.right, context);
    PyObject* cond = buildPyValueExpr(contraction.condition, context);
    PyObject* assignment = buildPyStr(contraction.assignment);
    PyObject* subscripts = buildPyList(contraction.subscripts);
    PyObject* axes = buildPyList(contraction.axes);

    PyObject* locals = PyList_New(contraction.locals.size());
    for ( size_t i = 0; i < contraction.locals.size(); ++i )
    {
        PyObject* id = buildPyStr(contraction.locals[i].first);
        PyObject* expr = buildPyValueExpr(contraction.locals[i].second, context);
        PyList_SetItem(locals, i, makePyTuple(id, expr));
    }

    PyObject* bounds = PyList_New(contraction.bounds.size());
    for ( size_t i = 0; i < contraction.bounds.size(); ++i )
    {
        PyObject* id = buildPyStr(contraction.bounds[i].first);
        PyObject* bound = buildPyValueExpr(contraction.bounds[i].second, context);
        PyList_SetItem(bounds, i, makePyTuple(id, bound));
    }

    return makePyObject(Contraction, left, right, cond, assignment, locals, bounds, subscripts, axes);
}

static PyObject* buildPyAssertion( const sknd::Assertion& assert, BuildContext& context )
{
    PyObject* condition = buildPyValueExpr(assert.condition, context);
    PyObject* message = buildPyStr(assert.message);
    PyObject* args = buildPyValueExprs(assert.args, context);

    return makePyObject(Assertion, condition, message, args);
}

static PyObject* buildPyOperation( const sknd::Operation& op, BuildContext& context )
{
    PyObject* name = buildPyStr(op.name);
    PyObject* dtypes = buildPyDtypes(op.dtypes);
    PyObject* subexprs = buildPySubexprs(op.subexprs, context);
    PyObject* attribs = buildPyAttribs(op.attribs, context);

    PyObject* inputs = PyTuple_New(op.inputs.size());
    for ( size_t i = 0; i < op.inputs.size(); ++i )
    {
        PyTuple_SetItem(inputs, i, buildPyTensorRef(op.inputs[i], context));
    }

    PyObject* outputs = PyTuple_New(op.outputs.size());
    for ( size_t i = 0; i < op.outputs.size(); ++i )
    {
        PyTuple_SetItem(outputs, i, buildPyTensorRef(op.outputs[i], context));
    }

    PyObject* internals = PyList_New(op.internals.size());
    for ( size_t i = 0; i < op.internals.size(); ++i )
    {
        PyList_SetItem(internals, i, buildPyTensorRef(op.internals[i], context));
    }

    PyObject* contractions = PyList_New(op.contractions.size());
    for ( size_t i = 0; i < op.contractions.size(); ++i )
    {
        PyList_SetItem(contractions, i, buildPyContraction(op.contractions[i], context));
    }

    PyObject* asserts = PyList_New(op.asserts.size());
    for ( size_t i = 0; i < op.asserts.size(); ++i )
    {
        PyList_SetItem(asserts, i, buildPyAssertion(op.asserts[i], context));
    }

    return makePyObject(Operation, name, dtypes, attribs, inputs, outputs, internals, contractions, asserts, subexprs);
}

static PyObject* buildPyGraph( const sknd::Graph& graph )
{
    BuildContext context;

    PyObject* name = buildPyStr(graph.name);

    PyObject* tensors = PyList_New(graph.tensors.size());
    for ( size_t i = 0; i < graph.tensors.size(); ++i )
    {
        PyObject* tensor = buildPyTensor(*graph.tensors[i], context);
        PyList_SetItem(tensors, i, tensor);
        context.tensors[graph.tensors[i].get()] = tensor;
    }

    PyObject* packs = PyList_New(graph.packs.size());
    for ( size_t i = 0; i < graph.packs.size(); ++i )
    {
        PyObject* pack = buildPyTensorPack(*graph.packs[i], context);
        PyList_SetItem(packs, i, pack);
        context.packs[graph.packs[i].get()] = pack;
    }

    PyObject* operations = PyList_New(graph.operations.size());
    for ( size_t i = 0; i < graph.operations.size(); ++i )
    {
        PyList_SetItem(operations, i, buildPyOperation(graph.operations[i], context));
    }

    // deferred setting of shapes
    for ( auto& [c_tensor, py_tensor] : context.tensors )
    {
        PyObject* py_shape = buildPyShape(c_tensor->shape, context);
        PyObject_SetAttrString(py_tensor, "shape", py_shape);
    }
    for ( auto& [c_pack, py_pack] : context.packs )
    {
        PyObject* py_shape = buildPyShape(c_pack->shape, context);
        PyObject_SetAttrString(py_pack, "shape", py_shape);
        PyObject* py_size = buildPyValueExpr(c_pack->size, context);
        PyObject_SetAttrString(py_pack, "size", py_size);
    }

    PyObject* inputs = PyTuple_New(graph.inputs.size());
    for ( size_t i = 0; i < graph.inputs.size(); ++i )
    {
        PyTuple_SetItem(inputs, i, buildPyTensorRef(graph.inputs[i], context));
    }

    PyObject* outputs = PyTuple_New(graph.outputs.size());
    for ( size_t i = 0; i < graph.outputs.size(); ++i )
    {
        PyTuple_SetItem(outputs, i, buildPyTensorRef(graph.outputs[i], context));
    }

    PyObject* asserts = PyList_New(graph.asserts.size());
    for ( size_t i = 0; i < graph.asserts.size(); ++i )
    {
        PyList_SetItem(asserts, i, buildPyAssertion(graph.asserts[i], context));
    }

    return makePyObject(Graph, name, operations, inputs, outputs, tensors, packs, asserts);
}

static PyObject* buildPyModel( const sknd::Model& model )
{
    PyObject* name = buildPyStr(model.name);

    PyObject* graphs = PyList_New(model.graphs.size());
    for ( size_t i = 0; i < model.graphs.size(); ++i )
    {
        PyList_SetItem(graphs, i, buildPyGraph(model.graphs[i]));
    }

    return makePyObject(Model, name, graphs);
}

static std::string module_from_path( const std::string& path )
{
    auto beg = path.find_last_of("\\/") + 1;
    auto end = path.find_last_of(".");
    return path.substr(beg, end - beg);
}

static std::set<std::string> make_string_set_from_iterable( PyObject* obj )
{
    std::set<std::string> set;

    PyObject* iter = PyObject_GetIter(obj);
    PyObject* item;
    while ( (item = PyIter_Next(iter)) )
    {
        set.insert(PY_STRING_AS_CSTR(item));
        Py_DECREF(item);
    }
    Py_DECREF(iter);
    return set;
}

static size_t function_arg_count( PyObject* pyFunc )
{
    Py_INCREF(pyFunc);
    PyObject* sig = callPyFunc(Signature, { pyFunc });
    PyObject* params = PyObject_GetAttrString(sig, "parameters");
    auto length = (size_t)PyObject_Length(params);
    Py_DECREF(params);
    return length;
}

static sknd::OperationCallback make_operation_callback()
{
}

static sknd::OperationCallback make_operation_callback( PyObject* obj, const std::string& key )
{
    if ( obj == Py_None )
    {
        return sknd::FalseOperationCallback;
    }
    else if ( PyBool_Check(obj) )
    {
        return (bool)PyObject_IsTrue(obj) ? sknd::TrueOperationCallback : sknd::FalseOperationCallback;
    }
    else if ( PyList_Check(obj) || PyTuple_Check(obj) || PySet_Check(obj) )
    {
        return sknd::make_operation_callback(make_string_set_from_iterable(obj));
    }
    else if ( PyDict_Check(obj) )
    {
        return [=]( const std::string& name,
                    const std::map<std::string,sknd::Typename>& dtypes,
                    const std::map<std::string,sknd::ValueExpr>& attribs,
                    const std::vector<sknd::TensorRef>& inputs )
        {
            PyObject* func = PyDict_GetItemString(obj, name.c_str());
            if ( !func )
            {
                return false;
            }
            if ( PyBool_Check(func) )
            {
                return func == Py_True;
            }
            auto argc = function_arg_count(func);
            if ( argc == 1 )
            {
                PyObject* pyName = buildPyStr(name.c_str());
                PyObject* ret = callPyFunc(func, { pyName });
                return (bool)PyObject_IsTrue(ret);
            }
            else if ( argc == 4 )
            {
                PyObject* pyName = buildPyStr(name.c_str());
                PyObject* pyDtypes = buildPyDtypes(dtypes);
                PyObject* pyAttribs = buildPyAttribs(attribs, EmptyBuildContext, true);
                PyObject* pyShapes = buildPyShapes(inputs, EmptyBuildContext, true);
                PyObject* ret = callPyFunc(func, { pyName, pyDtypes, pyAttribs, pyShapes });
                return (bool)PyObject_IsTrue(ret);
            }
            else
            {
                return false;
            }
        };
    }
    else if ( PyFunction_Check(obj) )
    {
        auto argc = function_arg_count(obj);
        if ( argc == 1 )
        {
            return [=]( const std::string& name,
                        const std::map<std::string,sknd::Typename>& dtypes,
                        const std::map<std::string,sknd::ValueExpr>& attribs,
                        const std::vector<sknd::TensorRef>& inputs )
            {
                PyObject* pyName = buildPyStr(name.c_str());
                PyObject* ret = callPyFunc(obj, { pyName });
                return (bool)PyObject_IsTrue(ret);
            };
        }
        else if ( argc == 4 )
        {
            return [=]( const std::string& name,
                        const std::map<std::string,sknd::Typename>& dtypes,
                        const std::map<std::string,sknd::ValueExpr>& attribs,
                        const std::vector<sknd::TensorRef>& inputs )
            {
                PyObject* pyName = buildPyStr(name.c_str());
                PyObject* pyDtypes = buildPyDtypes(dtypes);
                PyObject* pyAttribs = buildPyAttribs(attribs, EmptyBuildContext, true);
                PyObject* pyShapes = buildPyShapes(inputs, EmptyBuildContext, true);
                PyObject* ret = callPyFunc(obj, { pyName, pyDtypes, pyAttribs, pyShapes });
                return (bool)PyObject_IsTrue(ret);
            };
        }
        else
        {
            const std::string message = "Paremeter '" + key + "' must take 1 or 4 arguments";
            PyErr_SetString(PyExc_TypeError, message.c_str());
            return NULL;
        }
    }
    else
    {
        const std::string message = "Paremeter '" + key + "' must be either bool, list, tuple, set, dict or a callable";
        PyErr_SetString(PyExc_TypeError, message.c_str());
        return NULL;
    }
}


static PyObject* parse( PyObject* self, PyObject* args, PyObject* kwargs, bool isFile )
{
	const char* input = nullptr;
	const char* stdlib = nullptr;
    PyObject* atomic = nullptr;
    PyObject* unroll = nullptr;
    PyObject* attribs = nullptr;
    PyObject* error = nullptr;

    static const char* kwlist[] = { "", "stdlib", "attribs", "atomic_callback", "unroll_callback", "error_callback", NULL };

	if ( !PyArg_ParseTupleAndKeywords(args, kwargs, "s|sO!OOO!", (char**)kwlist, &input, &stdlib,
	    &PyDict_Type, &attribs, &atomic, &unroll, &PyFunction_Type, &error) )
    {
        return NULL;
    }

    auto atomic_callback = make_operation_callback(atomic, "atomic_callback");
    if ( !atomic_callback )
    {
        return NULL;
    }

    auto unroll_callback = make_operation_callback(unroll, "unroll_callback");
    if ( !unroll_callback )
    {
        return NULL;
    }

    auto error_callback = [&]( const sknd::Position& position, const std::string& message, const sknd::StackTrace& trace,
                                const bool warning )
    {
        PyObject* py_position = buildPyPosition(position);
        PyObject* py_message = buildPyStr(message);
        PyObject* py_trace = PyList_New(trace.size());
        PyObject* py_warning = buildPyBoolean(warning);

        size_t i = 0;
        for ( auto& [op, pos] : trace )
        {
            PyList_SetItem(py_trace, i++, makePyTuple(buildPyStr(op), buildPyPosition(pos)));
        }

        makePyObject(error, py_position, py_message, py_trace, py_warning);
    };

    std::map<std::string, sknd::ValueExpr> attributes;

    PyObject *key, *value;
    for ( Py_ssize_t ppos = 0; PyDict_Next(attribs, &ppos, &key, &value); )
    {
        if ( !PY_STRING_CHECK(key) )
        {
            const std::string message = "Paremeter 'attribs' must be a dict with string keys";
            PyErr_SetString(PyExc_TypeError, message.c_str());
            return NULL;
        }
        attributes[PY_STRING_AS_CSTR(key)] = valueFromPyObject(value);
    }

    std::optional<sknd::Model> model;

    if ( isFile )
    {
        const std::string& path = input;
        bool isFolder = path.back() == '\\' || path.back() == '/';
        const std::string import_path = isFolder ? path : "";
        const std::string filename = isFolder ? path + "main.sknd" : path;
        const std::string module = isFolder ? "main" : module_from_path(path);

        std::ifstream fs(filename);
        if ( !fs )
        {
            const std::string message = "Could not open file: " + std::string(filename);
            PyErr_SetString(PyExc_FileNotFoundError, message.c_str());
            return NULL;
        }
        model = sknd::read_model(fs, module, "", stdlib, import_path, error_callback, atomic_callback, unroll_callback, attributes);
        if ( model )
        {
            model->name = sknd::model_name_from_path(path);
        }
    }
    else
    {
        std::stringstream ss(input);
        model = sknd::read_model(ss, "main", "", stdlib, "", error_callback, atomic_callback, unroll_callback, attributes);
    }

    return model ? buildPyModel(*model) : buildPyNone();
}

static PyObject* parseFile( PyObject* self, PyObject* args, PyObject* kwargs )
{
    return parse(self, args, kwargs, true);
}

static PyObject* parseString( PyObject* self, PyObject* args, PyObject* kwargs )
{
    return parse(self, args, kwargs, false);
}


static PyMethodDef skriptnd_methods[] =
{
    { "parse_file", (PyCFunction)parseFile, METH_VARARGS | METH_KEYWORDS, "Parse the contents of a file" },
    { "parse_string", (PyCFunction)parseString, METH_VARARGS | METH_KEYWORDS, "Parse the contents of a string" },
 	{ NULL, NULL, 0, NULL }
};


#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef skriptnd_module =
{
    PyModuleDef_HEAD_INIT,
    "_skriptnd",
    "_skriptnd module",
    -1,
    skriptnd_methods,
};

#endif


#if PY_MAJOR_VERSION >= 3
#define INIT_FUNC_NAME PyInit__skriptnd
#define RETURN_ERROR return NULL
#else
#define INIT_FUNC_NAME init_skriptnd
#define RETURN_ERROR return
#endif

PyMODINIT_FUNC INIT_FUNC_NAME(void)
{
#if PY_MAJOR_VERSION >= 3
	PyObject* module = PyModule_Create(&skriptnd_module);
#else
    PyObject* module = Py_InitModule("_skriptnd", skriptnd_methods);
#endif
	if ( module == NULL )
	{
		RETURN_ERROR;
	}

	import_array();

    auto collections = PyImport_ImportModule("collections");
    auto collections_dict = PyModule_GetDict(collections);
    OrderedDict = PyDict_GetItemString(collections_dict, "OrderedDict");
    Py_DECREF(collections);

    auto _enum = PyImport_ImportModule("enum");
    auto enum_dict = PyModule_GetDict(_enum);
    Enum = PyDict_GetItemString(enum_dict, "Enum");
    Py_DECREF(_enum);

    auto dataclasses = PyImport_ImportModule("dataclasses");
    auto dataclasses_dict = PyModule_GetDict(dataclasses);
    DataClass = PyDict_GetItemString(dataclasses_dict, "make_dataclass");
    DataClassField = PyDict_GetItemString(dataclasses_dict, "field");
    Py_DECREF(dataclasses);

    auto inspect = PyImport_ImportModule("inspect");
    auto inspect_dict = PyModule_GetDict(inspect);
    Signature = PyDict_GetItemString(inspect_dict, "signature");
    Py_DECREF(inspect);

    Dtype = makeEnum(module, "Dtype", { "Type", "Arith", "Num", "Int", "Real", "Bool", "Str" });
    Position = makeDataClass(module, "Position", { "module", "line", "column" });

    Expr = makeDataClass(module, "Expr", {});
    SizeAccess = makeDataClass(module, Expr, "SizeAccess", { "pack" });
    ShapeAccess = makeDataClass(module, Expr, "ShapeAccess", { "tensor", "dim", "item", "max_size" });
    TensorAccess = makeDataClass(module, Expr, "TensorAccess", { "tensor", "indices", "item" });
    PlaceholderExpr = makeDataClass(module, Expr, "PlaceholderExpr", { "id", "max_value" });
    IdentifierExpr = makeDataClass(module, Expr, "IdentifierExpr", { "name", "kind", "dtype" });
    IdentifierKind = makeEnum(module, "IdentifierKind", { "LoopIndex", "LoopLocal" });
    ReferenceExpr = makeDataClass(module, Expr, "ReferenceExpr", { "name", "target", "dtype" });
    UnaryExpr = makeDataClass(module, Expr, "UnaryExpr", { "op", "arg", "dtype", "max_size" });
    BinaryExpr = makeDataClass(module, Expr, "BinaryExpr", { "op", "left", "right", "dtype", "max_size" });
    SelectExpr = makeDataClass(module, Expr, "SelectExpr", { "cond", "left", "right", "dtype", "max_size" });
    ListExpr = makeDataClass(module, Expr, "ListExpr", { "items", "dtype" });
    FoldExpr = makeDataClass(module, Expr, "FoldExpr", { "op", "pack", "dtype", "max_size" });
    CastExpr = makeDataClass(module, Expr, "CastExpr", { "arg", "dtype", "max_size" });
    BoundedExpr = makeDataClass(module, Expr, "BoundedExpr", { "arg", "lower", "upper", "dtype" });
    ConcatExpr = makeDataClass(module, Expr, "ConcatExpr", { "items", "dtype", "size", "max_size" });
    SliceExpr = makeDataClass(module, Expr, "SliceExpr", { "pack", "first", "last", "stride", "dtype", "size", "max_size" });
    SubscriptExpr = makeDataClass(module, Expr, "SubscriptExpr", { "pack", "index", "dtype", "max_size" });
    UniformExpr = makeDataClass(module, Expr, "UniformExpr", { "value", "size", "dtype", "max_size" });
    RangeExpr = makeDataClass(module, Expr, "RangeExpr", { "first", "last", "stride", "dtype", "size", "max_size" });

    Contraction = makeDataClass(module, "Contraction", { "left", "right", "condition", "assignment", "locals", "bounds", "subscripts", "axes" });

    Tensor = makeDataClass(module, "Tensor", { "name", "dtype", "shape", "max_shape", "quant", "value", "variable" },
                           { buildPyNone(), buildPyNone(), buildPyBoolean(false) });
    TensorPack = makeDataClass(module, "TensorPack", { "name", "dtype", "shape", "max_shape", "size", "items" },
                               { buildPyInt(0), EmptyListDefault });
    Assertion = makeDataClass(module, "Assertion", { "condition", "message", "args" });
    Operation = makeDataClass(module, "Operation", { "name", "dtypes", "attribs", "inputs", "outputs", "internals", "contractions", "asserts", "subexprs" },
                              { EmptyDictDefault, EmptyDictDefault, EmptyTupleDefault, EmptyTupleDefault, EmptyListDefault, EmptyListDefault, EmptyListDefault, EmptyListDefault});
    Graph = makeDataClass(module, "Graph", { "name", "operations", "inputs", "outputs", "tensors", "packs", "asserts" },
                          { EmptyListDefault, EmptyTupleDefault, EmptyTupleDefault, EmptyListDefault, EmptyListDefault, EmptyListDefault });
    Model = makeDataClass(module, "Model", { "name", "graphs" }, { EmptyListDefault });

#if PY_MAJOR_VERSION >= 3
	return module;
#endif
}
