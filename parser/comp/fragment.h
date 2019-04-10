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

#ifndef _NNEF_FRAGMENT_H_
#define _NNEF_FRAGMENT_H_

#include "../common/prototype.h"
#include "expression.h"
#include <iostream>


namespace nnef
{

    class Assignment
    {
    public:
        
        Assignment( const Shared<Expr>& lhs, const Shared<Expr>& rhs )
        : _lhs(lhs), _rhs(rhs)
        {
        }
        
        const Expr& lhs() const
        {
            return *_lhs;
        }
        
        const Expr& rhs() const
        {
            return *_rhs;
        }
        
    private:
        
        const Shared<Expr> _lhs;
        const Shared<Expr> _rhs;
    };

    
    class Fragment
    {
    public:

        Fragment( const Prototype& prototype )
        : _prototype(prototype)
        {
        }

        Fragment( const Prototype& prototype, std::vector<Assignment>& assignments )
        : _prototype(prototype), _assignments(std::move(assignments))
        {
        }
        
        const Prototype& prototype() const
        {
            return _prototype;
        }
        
        size_t assignmentCount() const
        {
            return _assignments.size();
        }
        
        const Assignment& assignment( const size_t i ) const
        {
            return _assignments[i];
        }
        
    private:
        
        const Prototype& _prototype;
        const std::vector<Assignment> _assignments;
    };


    inline std::ostream& operator<<( std::ostream& os, const Assignment& assignment )
    {
        os << assignment.lhs() << " = " << assignment.rhs();
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const Fragment& fragment )
    {
        os << fragment.prototype() << std::endl;

        if ( fragment.assignmentCount() )
        {
            os << '{' << std::endl;
            for ( size_t i = 0; i < fragment.assignmentCount(); ++i )
            {
                os << '\t' << fragment.assignment(i) << std::endl;
            }
            os << '}' << std::endl;
        }

        return os;
    }

}   // namespace nnef


#endif
