#ifndef _SKND_POSITION_H_
#define _SKND_POSITION_H_

#include <string>


namespace sknd
{
    
    struct Position
    {
        std::string module;
        unsigned line = 0;
        unsigned column = 0;
    };
    
}   // namespace sknd


#endif
