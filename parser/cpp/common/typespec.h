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

#include <utility>
#include <iostream>
#include <algorithm>
#include <memory>
#include <vector>
#include <map>


namespace nnef
{
    
    enum class Typename { Extent, Scalar, Logical, String };
    
    inline const char* toString( const Typename& name )
    {
        static const char* strings[] =
        {
            "extent", "scalar", "logical", "string",
        };
        return strings[(size_t)name];
    }
    

    
    class Type
    {
    public:

        virtual bool isPrimitive() const { return false; }
        virtual bool isArray() const { return false; }
        virtual bool isTuple() const { return false; }
        
        virtual bool isTensor() const = 0;

        virtual std::string toString() const = 0;
    };


    class PrimitiveType : public Type
    {
    public:

        PrimitiveType( const Typename name, bool tensor )
        : _name(name), _tensor(tensor)
        {
        }

        Typename name() const
        {
            return _name;
        }

        virtual bool isPrimitive() const
        {
            return true;
        }
        
        virtual bool isTensor() const
        {
            return _tensor;
        }

        virtual std::string toString() const
        {
            const std::string str = nnef::toString(_name);
            return _tensor ? "tensor<" + str + ">" : str;
        }

    private:

        Typename _name;
        bool _tensor;
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

        virtual bool isArray() const
        {
            return true;
        }
        
        virtual bool isTensor() const
        {
            return _itemType->isTensor();
        }

        virtual std::string toString() const
        {
            return _itemType->toString() + "[]";
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

        virtual bool isTuple() const
        {
            return true;
        }
        
        virtual bool isTensor() const
        {
            return std::all_of(_itemTypes.begin(), _itemTypes.end(), []( const Type* type ){ return type->isTensor(); });
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
    
}   // namespace nnef


#endif
