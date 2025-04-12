#ifndef _TS_RESULT_H_
#define _TS_RESULT_H_

#include <string>
#include "error.h"


namespace sknd
{
    
    template<typename T>
    class Result
    {
        typedef Error error_type;
        
    public:
        
        typedef T value_type;
        
    public:
        
        Result( const value_type& value = value_type() )
        : _value(value), _valid(true)
        {
        }
        
        Result( value_type&& value )
        : _value(std::forward<value_type>(value)), _valid(true)
        {
        }
        
        Result( const Error& error )
        : _error(error), _valid(false)
        {
        }
        
        Result( Error&& error )
        : _error(std::forward<error_type>(error)), _valid(false)
        {
        }
        
        Result( const Result& other )
        {
            construct(other);
        }
        
        Result( Result&& other )
        {
            move(std::forward<Result>(other));
        }
        
        ~Result()
        {
            destruct();
        }
        
        Result& operator=( const Result& other )
        {
            destruct();
            construct(other);
            return *this;
        }
        
        Result& operator=( Result&& other )
        {
            destruct();
            move(std::forward<Result>(other));
            return *this;
        }
        
        explicit operator bool() const
        {
            return _valid;
        }
        
        const value_type& operator*() const
        {
            return _value;
        }
        
        value_type& operator*()
        {
            return _value;
        }
        
        const value_type* operator->() const
        {
            return &_value;
        }
        
        value_type* operator->()
        {
            return &_value;
        }
        
        const error_type& error() const
        {
            return _error;
        }
        
        error_type& error()
        {
            return _error;
        }
        
    private:
        
        void construct( const Result& other )
        {
            _valid = other._valid;
            if ( _valid )
            {
                new(&_value) value_type(other._value);
            }
            else
            {
                new(&_error) error_type(other._error);
            }
        }
        
        void destruct()
        {
            if ( _valid )
            {
                _value.~value_type();
            }
            else
            {
                _error.~error_type();
            }
        }
        
        void move( Result&& other )
        {
            _valid = other._valid;
            if ( _valid )
            {
                new (&_value) value_type(std::move(other._value));
            }
            else
            {
                new (&_error) error_type(std::move(other._error));
            }
        }
        
    private:
        
        union
        {
            value_type _value;
            error_type _error;
        };
        bool _valid;
    };
    
    
    template<>
    class Result<void>
    {
        typedef Error error_type;
        
    public:
        
        Result()
        {
        }
        
        Result( const Error& error )
        : _error(error)
        {
        }
        
        Result( Error&& error )
        : _error(std::forward<error_type>(error))
        {
        }
        
        explicit operator bool() const
        {
            return !_error;
        }
        
        const error_type& error() const
        {
            return _error;
        }
        
        error_type& error()
        {
            return _error;
        }
        
    private:
        
        error_type _error;
    };

}   // namespace sknd


#define _CAT(X,Y) _CAT_(X,Y)
#define _CAT_(X,Y) X##Y

#ifdef _WIN32
    #define EXPAND(MACRO, ARGS) MACRO ARGS
    #define _ARGC_2(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, N, ...) N
    #define _ARGC_1(...) EXPAND(_ARGC_2, (__VA_ARGS__))
    #define _ARGC(...) _ARGC_1(__VA_ARGS__, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
#else
    #define _ARGC(...) _ARGC_(__VA_ARGS__,9,8,7,6,5,4,3,2,1,)
    #define _ARGC_(_9,_8,_7,_6,_5,_4,_3,_2,_1,_C,...) _C
#endif

#define _RES_1(_1) $##_1
#define _RES_2(_1,_2) $##_1##_2
#define _RES_3(_1,_2,_3) $##_1##_2##_3
#define _RES_4(_1,_2,_3,_4) $##_1##_2##_3##_4
#define _RES(...) _CAT(_RES_,_ARGC(__VA_ARGS__))(__VA_ARGS__)

#define _TRY_DECL_N(expr,...) auto _RES(__VA_ARGS__) = (expr); if ( !_RES(__VA_ARGS__) ) return _RES(__VA_ARGS__).error(); auto& [__VA_ARGS__] = *_RES(__VA_ARGS__);
#define _TRY_DECL_1(var,expr) auto _RES(var) = (expr); if ( !_RES(var) ) return _RES(var).error(); auto& var = *_RES(var);
#define _TRY_DECL_2(var1,var2,expr) _TRY_DECL_N(expr,var1,var2)
#define _TRY_DECL_3(var1,var2,var3,expr) _TRY_DECL_N(expr,var1,var2,var3)
#define _TRY_DECL_4(var1,var2,var3,var4,expr) _TRY_DECL_N(expr,var1,var2,var3,var4)

#define TRY_DECL(first,...) _CAT(_TRY_DECL_,_ARGC(__VA_ARGS__))(first,__VA_ARGS__)
#define TRY_CALL(expr) { const auto result = (expr); if ( !result ) return result.error(); }
#define TRY_MOVE(var,expr) { auto _RES(var) = (expr); if ( !_RES(var) ) return _RES(var).error(); var = std::move(*_RES(var)); }

#endif
