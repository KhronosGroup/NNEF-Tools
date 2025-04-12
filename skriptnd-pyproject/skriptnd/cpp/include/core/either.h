#ifndef _SKND_EITHER_H_
#define _SKND_EITHER_H_

#include <typeinfo>


namespace sknd
{

    template<typename Left, typename Right>
    class Either
    {
    public:
        
        typedef Left left_type;
        typedef Right right_type;
        
        static_assert(!std::is_same<left_type,right_type>::value, "types in either must be distinct");
        
    public:
        
        Either( const left_type& left )
        : _left(left), _which(false)
        {
        }
        
        Either( const right_type& right )
        : _right(right), _which(true)
        {
        }
        
        Either( left_type&& left )
        : _left(std::forward<left_type>(left)), _which(false)
        {
        }
        
        Either( right_type&& right )
        : _right(std::forward<right_type>(right)), _which(true)
        {
        }
        
        Either( const Either& other )
        {
            construct(other);
        }
        
        Either( Either&& other )
        {
            move(std::forward<Either>(other));
        }
        
        Either& operator=( const Either& other )
        {
            destruct();
            construct(other);
            return *this;
        }
        
        Either& operator=( Either&& other )
        {
            destruct();
            move(std::forward<Either>(other));
            return *this;
        }
        
        Either& operator=( const left_type& left )
        {
            destruct();
            _which = false;
            new (&_left) left_type(left);
            return *this;
        }
        
        Either& operator=( const right_type& right )
        {
            destruct();
            _which = true;
            new (&_right) right_type(right);
            return *this;
        }
        
        Either& operator=( left_type&& left )
        {
            destruct();
            _which = false;
            new (&_left) left_type(std::move(left));
            return *this;
        }
        
        Either& operator=( right_type&& right )
        {
            destruct();
            _which = true;
            new (&_right) right_type(std::move(right));
            return *this;
        }
        
        ~Either()
        {
            destruct();
        }
        
        template<typename T>
        typename std::enable_if<std::is_same<T,left_type>::value,bool>::type is() const
        {
            return !_which;
        }
        
        template<typename T>
        typename std::enable_if<std::is_same<T,right_type>::value,bool>::type is() const
        {
            return _which;
        }
        
        template<typename T>
        const typename std::enable_if<std::is_same<T,left_type>::value,left_type>::type& as() const
        {
            if ( _which )
            {
                throw std::bad_cast();
            }
            return _left;
        }
        
        template<typename T>
        const typename std::enable_if<std::is_same<T,right_type>::value,right_type>::type& as() const
        {
            if ( !_which )
            {
                throw std::bad_cast();
            }
            return _right;
        }
        
        template<typename T>
        typename std::enable_if<std::is_same<T,left_type>::value,left_type>::type& as()
        {
            if ( _which )
            {
                throw std::bad_cast();
            }
            return _left;
        }
        
        template<typename T>
        typename std::enable_if<std::is_same<T,right_type>::value,right_type>::type& as()
        {
            if ( !_which )
            {
                throw std::bad_cast();
            }
            return _right;
        }
        
        explicit operator const left_type&() const
        {
            return _left;
        }
        
        explicit operator const right_type&() const
        {
            return _right;
        }
        
        explicit operator left_type&()
        {
            return _left;
        }
        
        explicit operator right_type&()
        {
            return _right;
        }
        
        bool operator==( const Either& other ) const
        {
            return is<left_type>() ? other == _left : other == _right;
        }
        
        bool operator==( const left_type& other ) const
        {
            return is<left_type>() && _left == other;
        }
        
        bool operator==( const right_type& other ) const
        {
            return is<right_type>() && _right == other;
        }
        
        bool operator!=( const Either& other ) const
        {
            return !(*this == other);
        }
        
        bool operator!=( const left_type& other ) const
        {
            return !(*this == other);
        }
        
        bool operator!=( const right_type& other ) const
        {
            return !(*this == other);
        }
        
    private:
        
        void construct( const Either& other )
        {
            _which = other._which;
            if ( _which )
            {
                new(&_right) right_type(other._right);
            }
            else
            {
                new(&_left) left_type(other._left);
            }
        }
        
        void destruct()
        {
            if ( _which )
            {
                _right.~right_type();
            }
            else
            {
                _left.~left_type();
            }
        }
        
        void move( Either&& other )
        {
            _which = other._which;
            if ( _which )
            {
                new (&_right) right_type(std::move(other._right));
            }
            else
            {
                new (&_left) left_type(std::move(other._left));
            }
        }
        
    private:
        
        union
        {
            left_type _left;
            right_type _right;
        };
        bool _which;
    };
    
    
    template<typename L, typename R>
    std::ostream& operator<<( std::ostream& os, const Either<L,R>& either )
    {
        if ( either.template is<L>() )
        {
            os << either.template as<L>();
        }
        else
        {
            os << either.template as<R>();
        }
        return os;
    }

}   // namespace sknd


template<typename L, typename R>
struct std::hash<sknd::Either<L,R>>
{
    std::size_t operator()( const sknd::Either<L,R>& x ) const noexcept
    {
        if ( x.template is<L>() )
        {
            std::hash<L> hasher;
            return hasher(x.template as<L>());
        }
        else
        {
            std::hash<R> hasher;
            return hasher(x.template as<R>());
        }
    }
};


#endif
