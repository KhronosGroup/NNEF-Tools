#ifndef _TS_POSITION_H_
#define _TS_POSITION_H_

#include <string>


namespace ts
{
    
    struct Position
    {
        std::string module;
        unsigned line = 0;
        unsigned column = 0;
    };
    
}   // namespace ts


#endif
