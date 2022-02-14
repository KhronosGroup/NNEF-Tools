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

#ifndef _NNEF_TYPEUTILS_H_
#define _NNEF_TYPEUTILS_H_

#include "typespec.h"
#include "prototype.h"
#include "dictionary.h"
#include <cassert>


namespace nnef
{

    inline bool isCastable( const Type* type1, const Type* type2, bool allowPrimitiveToTensor = true, bool allowArrayToTensor = false )
    {
        if ( type1 == type2 )
        {
            return true;
        }

        if ( type1->kind() == type2->kind() )
        {
            switch ( type1->kind() )
            {
                case Type::Primitive:
                {
                    auto primitiveType1 = static_cast<const PrimitiveType*>(type1);
                    auto primitiveType2 = static_cast<const PrimitiveType*>(type2);
                    return primitiveType1->name() == primitiveType2->name() || primitiveType2->name() == Typename::Generic;
                }
                case Type::Tensor:
                {
                    auto tensorType1 = static_cast<const TensorType*>(type1);
                    auto tensorType2 = static_cast<const TensorType*>(type2);
                    if ( tensorType1->dataType() && tensorType2->dataType() )
                    {
                        return isCastable(tensorType1->dataType(), tensorType2->dataType(), allowPrimitiveToTensor, allowArrayToTensor);
                    }
                    else
                    {
                        return !tensorType2->dataType();
                    }
                }
                case Type::Array:
                {
                    auto arrayType1 = static_cast<const ArrayType*>(type1);
                    auto arrayType2 = static_cast<const ArrayType*>(type2);
                    if ( arrayType1->itemType() && arrayType2->itemType() )
                    {
                        return isCastable(arrayType1->itemType(), arrayType2->itemType(), allowPrimitiveToTensor, allowArrayToTensor);
                    }
                    else
                    {
                        return !arrayType1->itemType();
                    }
                }
                case Type::Tuple:
                {
                    auto tupleType1 = static_cast<const TupleType*>(type1);
                    auto tupleType2 = static_cast<const TupleType*>(type2);
                    if ( tupleType1->size() != tupleType2->size() )
                    {
                        return false;
                    }
                    for ( size_t i = 0; i < tupleType1->size(); ++i )
                    {
                        if ( !isCastable(tupleType1->itemType(i), tupleType2->itemType(i), allowPrimitiveToTensor, allowArrayToTensor) )
                        {
                            return false;
                        }
                    }
                    return true;
                }
            }
        }
        else if ( type1->kind() == Type::Primitive && type2->kind() == Type::Tensor && allowPrimitiveToTensor )
        {
            auto tensorType = static_cast<const TensorType*>(type2);
            return !tensorType->dataType() || isCastable(type1, tensorType->dataType());
        }
        else if ( type1->kind() == Type::Array && type2->kind() == Type::Tensor && allowArrayToTensor )
        {
            auto arrayType = static_cast<const ArrayType*>(type1);
            auto itemType = arrayType->itemType();
            while ( itemType->kind() != Type::Primitive )
            {
                if ( itemType->kind() != Type::Array )
                {
                    return false;
                }
                itemType = static_cast<const ArrayType*>(itemType)->itemType();
            }
            auto tensorType = static_cast<const TensorType*>(type2);
            return !tensorType->dataType() || isCastable(itemType, tensorType->dataType());
        }

        return false;
    }

    inline const Type* commonType( const Type* type1, const Type* type2 )
    {
        if ( isCastable(type1, type2) )
        {
            return type2;
        }
        else if ( isCastable(type2, type1) )
        {
            return type1;
        }
        return nullptr;
    }
    
    inline const Type* bindDataType( const Type* paramType, const PrimitiveType* dataType )
    {
        if ( !paramType->isGeneric() || dataType == primitiveType(Typename::Generic) )
        {
            return paramType;
        }
        
        switch ( paramType->kind() )
        {
            case Type::Primitive:
            {
                return paramType == primitiveType(Typename::Generic) ? dataType : paramType;
            }
            case Type::Tensor:
            {
                auto tensor = static_cast<const TensorType*>(paramType);
                return tensor->dataType() == primitiveType(Typename::Generic) ? tensorType(dataType->name()) : paramType;
            }
            case Type::Array:
            {
                auto array = static_cast<const ArrayType*>(paramType);
                return array->itemType() ? arrayType(bindDataType(array->itemType(), dataType)) : paramType;
            }
            case Type::Tuple:
            {
                auto tuple = static_cast<const TupleType*>(paramType);
                
                std::vector<const Type*> itemTypes(tuple->size());
                for ( size_t i = 0; i < tuple->size(); ++i )
                {
                    itemTypes[i] = bindDataType(tuple->itemType(i), dataType);
                }
                return tupleType(itemTypes);
            }
        }
        assert(false);
        return nullptr;
    }

    inline void deduceDataType( const Type* paramType, const Type* argType, const PrimitiveType*& dataType )
    {
        if ( paramType->kind() == argType->kind() )
        {
            switch ( paramType->kind() )
            {
                case Type::Primitive:
                {
                    if ( paramType->isGeneric() )
                    {
                        auto primitiveType = static_cast<const PrimitiveType*>(argType);
                        if ( !dataType )
                        {
                            dataType = primitiveType;
                        }
                        else if ( dataType != argType )
                        {
                            throw std::make_pair(dataType->name(), primitiveType->name());
                        }
                    }
                    break;
                }
                case Type::Tensor:
                {
                    auto tensorType1 = static_cast<const TensorType*>(paramType);
                    auto tensorType2 = static_cast<const TensorType*>(argType);
                    if ( tensorType1->dataType() && tensorType2->dataType() )
                    {
                        deduceDataType(tensorType1->dataType(), tensorType2->dataType(), dataType);
                    }
                    break;
                }
                case Type::Array:
                {
                    auto arrayType1 = static_cast<const ArrayType*>(paramType);
                    auto arrayType2 = static_cast<const ArrayType*>(argType);
                    if ( arrayType1->itemType() && arrayType2->itemType() )
                    {
                        deduceDataType(arrayType1->itemType(), arrayType2->itemType(), dataType);
                    }
                    break;
                }
                case Type::Tuple:
                {
                    auto tupleType1 = static_cast<const TupleType*>(paramType);
                    auto tupleType2 = static_cast<const TupleType*>(argType);
                    assert(tupleType1->size() == tupleType2->size());

                    for ( size_t i = 0; i < tupleType1->size(); ++i )
                    {
                        deduceDataType(tupleType1->itemType(i), tupleType2->itemType(i), dataType);
                    }
                    break;
                }
            }
        }
        else if ( paramType->kind() == Type::Tensor && argType->kind() == Type::Primitive )
        {
            auto tensorType = static_cast<const TensorType*>(paramType);
            deduceDataType(tensorType->dataType(), argType, dataType);
        }
    }

    inline bool deduceDataType( const Prototype& proto, const Dictionary<const Type*>& types, const PrimitiveType*& dataType )
    {
        for ( size_t i = 0; i < proto.paramCount(); ++i )
        {
            auto& param = proto.param(i);
            if ( param.type()->isGeneric() )
            {
                auto argType = types.at(param.name());
                deduceDataType(param.type(), argType, dataType);
            }
        }
        return dataType != nullptr;
    }

}   // namespace nnef


#endif
