#ifndef _SKND_MAYBE_H_
#define _SKND_MAYBE_H_


namespace sknd
{

    template<typename T>
    class Maybe
    {
    public:
        
        typedef T value_type;
        
    public:
        
        Maybe( std::nullptr_t = nullptr )
        : _value(nullptr)
        {
        }
        
        Maybe( const value_type& value )
        : _value(new value_type(value))
        {
        }
        
        Maybe( value_type&& value )
        : _value(new value_type(std::forward<value_type>(value)))
        {
        }
        
        Maybe( const Maybe& other )
        : _value(other._value ? new value_type(*other._value) : nullptr)
        {
        }
        
        Maybe( Maybe&& other )
        : _value(nullptr)
        {
            swap(other);
        }
        
        ~Maybe()
        {
            destruct();
        }
        
        void reset() const
        {
            destruct();
            _value = nullptr;
        }
        
        operator bool() const
        {
            return (bool)_value;
        }
        
        const value_type& operator*() const
        {
            return *_value;
        }
        
        value_type& operator*()
        {
            return *_value;
        }
        
        const value_type* operator->() const
        {
            return _value;
        }
        
        value_type* operator->()
        {
            return _value;
        }
        
        Maybe& operator=( const Maybe& other )
        {
            destruct();
            _value = other._value ? new value_type(*other._value) : nullptr;
            return *this;
        }
        
        Maybe& operator=( Maybe&& other )
        {
            swap(other);
            return *this;
        }
        
        bool operator==( const Maybe& other ) const
        {
            return _value == other._value;
        }
        
        bool operator!=( const Maybe& other ) const
        {
            return _value != other._value;
        }
        
        bool operator==( std::nullptr_t ) const
        {
            return _value == nullptr;
        }
        
        bool operator!=( std::nullptr_t ) const
        {
            return _value != nullptr;
        }
        
        void swap( Maybe& other )
        {
            std::swap(_value, other._value);
        }
        
    private:
        
        void destruct()
        {
            if ( _value )
            {
                delete _value;
            }
        }
        
    private:
        
        value_type* _value;
    };

}   // namespace sknd


#endif
