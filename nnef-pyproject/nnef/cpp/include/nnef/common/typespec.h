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

#ifndef _NNEF_TYPESPEC_H_
#define _NNEF_TYPESPEC_H_

#include <map>
#include <memory>
#include <vector>
#include <utility>
#include <iostream>
#include <algorithm>
#include <initializer_list>


namespace nnef
{
    
    enum class Typename { Integer, Scalar, Logical, String, Generic };
    
    inline const char* toString( const Typename& name )
    {
        static const char* strings[] =
        {
            "integer", "scalar", "logical", "string", "?"
        };
        return strings[(size_t)name];
    }

    inline Typename fromString( const std::string& str )
    {
        static const std::map<std::string,Typename> typenames =
        {
            { "integer", Typename::Integer },
            { "scalar", Typename::Scalar },
            { "logical", Typename::Logical },
            { "string", Typename::String },
        };
        return typenames.at(str);
    }
    

    
    class Type
    {
    public:

        enum Kind { Primitive, Tensor, Array, Tuple };

    public:
        
        virtual ~Type() {}

        virtual Kind kind() const = 0;

        virtual bool isAttribute() const = 0;
        virtual bool isGeneric() const = 0;

        virtual std::string toString() const = 0;
    };


    class PrimitiveType : public Type
    {
    public:

        PrimitiveType( const Typename name )
        : _name(name)
        {
        }

        Typename name() const
        {
            return _name;
        }

        virtual Kind kind() const
        {
            return Primitive;
        }

        virtual bool isAttribute() const
        {
            return true;
        }

        virtual bool isGeneric() const
        {
            return _name == Typename::Generic;
        }

        virtual std::string toString() const
        {
            return nnef::toString(_name);
        }

    private:

        Typename _name;
    };


    class TensorType : public Type
    {
    public:

        TensorType( const Type* dataType )
        : _dataType(dataType)
        {
        }

        const Type* dataType() const
        {
            return _dataType;
        }

        virtual Kind kind() const
        {
            return Tensor;
        }

        virtual std::string toString() const
        {
            return _dataType ? "tensor<" + _dataType->toString() + ">" : "tensor<>";
        }

        virtual bool isAttribute() const
        {
            return false;
        }
        
        virtual bool isGeneric() const
        {
            return _dataType && _dataType->isGeneric();
        }

    private:

        const Type* _dataType;
    };


    class ArrayType : public Type
    {
    public:

        ArrayType( const Type* itemType )
        : _itemType(itemType)
        {
        }

        const Type* itemType() const
        {
            return _itemType;
        }

        virtual Kind kind() const
        {
            return Array;
        }

        virtual std::string toString() const
        {
            return _itemType ? _itemType->toString() + "[]" : "[]";
        }

        virtual bool isAttribute() const
        {
            return _itemType && _itemType->isAttribute();
        }

        virtual bool isGeneric() const
        {
            return _itemType && _itemType->isGeneric();
        }
        
    private:

        const Type* _itemType;
    };


    class TupleType : public Type
    {
    public:

        TupleType( const std::vector<const Type*>& itemTypes )
        : _itemTypes(itemTypes)
        {
        }
        
        TupleType( const std::initializer_list<const Type*>& itemTypes )
        : _itemTypes(itemTypes)
        {
        }

        size_t size() const
        {
            return _itemTypes.size();
        }

        const Type* itemType( const size_t i ) const
        {
            return _itemTypes[i];
        }

        virtual Kind kind() const
        {
            return Tuple;
        }

        virtual bool isAttribute() const
        {
            return std::all_of(_itemTypes.begin(), _itemTypes.end(), []( const Type* type ){ return type->isAttribute(); });
        }

        virtual bool isGeneric() const
        {
            return std::any_of(_itemTypes.begin(), _itemTypes.end(), []( const Type* type ){ return type->isGeneric(); });
        }

        virtual std::string toString() const
        {
            std::string str;
            str += '(';
            for ( size_t i = 0; i < _itemTypes.size(); ++i )
            {
                if ( i )
                {
                    str += ',';
                }
                str += _itemTypes[i]->toString();
            }
            str += ')';
            return str;
        }

    private:

        std::vector<const Type*> _itemTypes;
    };

    
    inline const PrimitiveType* primitiveType( const Typename name )
    {
        static const PrimitiveType types[] =
        {
            PrimitiveType(Typename::Integer),
            PrimitiveType(Typename::Scalar),
            PrimitiveType(Typename::Logical),
            PrimitiveType(Typename::String),
            PrimitiveType(Typename::Generic),
        };
        return &types[(size_t)name];
    }

    inline const TensorType* tensorType( const Typename name )
    {
        static const TensorType types[] =
        {
            TensorType(primitiveType(Typename::Integer)),
            TensorType(primitiveType(Typename::Scalar)),
            TensorType(primitiveType(Typename::Logical)),
            TensorType(primitiveType(Typename::String)),
            TensorType(primitiveType(Typename::Generic)),
        };
        return &types[(size_t)name];
    }

    inline const TensorType* tensorType()
    {
        static const TensorType type(nullptr);
        return &type;
    }

    inline const Type* arrayType( const Type* itemType )
    {
        static std::map<const Type*,ArrayType> types;
        
        auto it = types.lower_bound(itemType);
        if ( it == types.end() || it->first != itemType )
        {
            it = types.emplace_hint(it, itemType, itemType);
        }
        return &it->second;
    }

    inline const Type* tupleType( const std::vector<const Type*>& itemTypes )
    {
        static std::map<std::vector<const Type*>,TupleType> types;

        auto it = types.lower_bound(itemTypes);
        if ( it == types.end() || it->first != itemTypes )
        {
            it = types.emplace_hint(it, itemTypes, itemTypes);
        }
        return &it->second;
    }
    
}   // namespace nnef


#endif
