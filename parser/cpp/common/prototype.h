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

#ifndef _NNEF_PROTOTYPE_H_
#define _NNEF_PROTOTYPE_H_

#include "typespec.h"
#include "value.h"
#include <vector>
#include <string>
#include <initializer_list>


namespace nnef
{

    class Typed
    {
    public:

        Typed( const std::string& name, const Type* type )
        : _name(name), _type(type)
        {
        }

        const std::string& name() const
        {
            return _name;
        }

        const Type* type() const
        {
            return _type;
        }

    private:

        std::string _name;
        const Type* _type;
    };


    class Param : public Typed
    {
    public:

        Param( const std::string& name, const Type* type, const Value& defaultValue = Value::none() )
        : Typed(name,type), _default(defaultValue)
        {
        }

        const Value& defaultValue() const
        {
            return _default;
        }

    private:

        Value _default;
    };


    typedef Typed Result;


    class Prototype
    {
    private:

        void initGeneric()
        {
            auto isGeneric = []( const Typed& typed ){ return typed.type()->isGeneric(); };
            _hasGenericParams = std::any_of(_params.begin(), _params.end(), isGeneric);
            _hasGenericResults = std::any_of(_results.begin(), _results.end(), isGeneric);
        }

    public:

        Prototype( const std::string& name, std::initializer_list<Param> params, std::initializer_list<Result> results,
                  const PrimitiveType* genericParamDefault = nullptr )
        : _name(name), _params(params), _results(results), _genericParamDefault(genericParamDefault)
        {
            initGeneric();
        }
        
        Prototype( const std::string& name, std::vector<Param>& params, std::vector<Result>& results,
                  const PrimitiveType* genericParamDefault = nullptr )
        : _name(name), _params(std::move(params)), _results(std::move(results)), _genericParamDefault(genericParamDefault)
        {
            initGeneric();
        }

        const std::string& name() const
        {
            return _name;
        }
        
        const PrimitiveType* genericParamDefault() const
        {
            return _genericParamDefault;
        }

        size_t paramCount() const
        {
            return _params.size();
        }

        const Param& param( const size_t i ) const
        {
            return _params[i];
        }

        const Param* param( const std::string& name ) const
        {
            for ( auto& param : _params )
            {
                if ( param.name() == name )
                {
                    return &param;
                }
            }
            return nullptr;
        }

        size_t resultCount() const
        {
            return _results.size();
        }

        const Result& result( const size_t i ) const
        {
            return _results[i];
        }

        const Result* result( const std::string& name ) const
        {
            for ( auto& result : _results )
            {
                if ( result.name() == name )
                {
                    return &result;
                }
            }
            return nullptr;
        }

        bool hasGenericParams() const
        {
            return _hasGenericParams;
        }

        bool hasGenericResults() const
        {
            return _hasGenericResults;
        }

        bool isGeneric() const
        {
            return _hasGenericParams || _hasGenericResults;
        }

    private:

        std::string _name;
        std::vector<Param> _params;
        std::vector<Result> _results;

        bool _hasGenericParams;
        bool _hasGenericResults;
        const PrimitiveType* _genericParamDefault;
    };
    
    
    
    inline std::ostream& operator<<( std::ostream& os, const Typed& typed )
    {
        os << typed.name() << ": " << typed.type()->toString();
        return os;
    }
    
    inline std::ostream& operator<<( std::ostream& os, const Prototype& proto )
    {
        os << proto.name();
        
        if ( proto.isGeneric() )
        {
            os << "<?";
            if ( proto.genericParamDefault() )
            {
                os << " = " << proto.genericParamDefault()->toString();
            }
            os << ">";
        }
        
        os << "( ";
        for ( size_t i = 0; i < proto.paramCount(); ++i )
        {
            if ( i )
            {
                os << ", ";
            }
            os << proto.param(i);
        }
        os << " )";
        
        os << " -> ";
        
        os << "( ";
        for ( size_t i = 0; i < proto.resultCount(); ++i )
        {
            if ( i )
            {
                os << ", ";
            }
            os << proto.result(i);
        }
        os << " )";
        
        return os;
    }

}   // namespace nnef


#endif
