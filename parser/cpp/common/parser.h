/*
 * Copyright (c) 2012-2017 The Khronos Group Inc.
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

#ifndef _NNEF_PARSER_H_
#define _NNEF_PARSER_H_

#include "value.h"
#include "shape.h"
#include "prototype.h"
#include "dictionary.h"
#include "propagation.h"


namespace nnef
{

    struct Parser
    {
        struct Callback : private Propagation
        {
            using Propagation::variableShapes;

            virtual void beginGraph( const Prototype& proto ) {}
            virtual void endGraph( const Prototype& proto ) {}

            virtual void operation( const Prototype& proto, const Dictionary<Value>& args, const Dictionary<Shape>& shapes ) {}

            virtual bool isAtomic( const Prototype& proto, const Dictionary<Value>& args )
            {
                return false;
            }

            virtual bool propagate( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
            {
                return Propagation::propagateShapes(proto, args, shapes);
            }

            virtual size_t resultArrayLength( const Prototype& proto, const Dictionary<Value>& args, const size_t idx )
            {
                return Propagation::resultArrayLength(proto, args, idx);
            }
        };


        virtual void parse( std::istream& is, Callback& callback ) = 0;
    };

}   // namespace nnef


#endif
