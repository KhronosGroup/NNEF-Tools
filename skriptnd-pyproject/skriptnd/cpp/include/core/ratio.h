#ifndef _TS_RATIO_H_
#define _TS_RATIO_H_

#include <numeric>


namespace ts
{

    template<typename T>
    class ratio
    {
    public:
        
        typedef T value_type;
        
    private:
        
        struct unnormalized_tag {};
        
        ratio( const value_type& nom, const value_type& denom, const unnormalized_tag& )
        : _nominator(nom), _denominator(denom)
        {
        }
        
    public:
        
        static ratio unnormalized( const value_type& nom, const value_type& denom )
        {
            return ratio(nom, denom, unnormalized_tag());
        }
        
    public:
        
        ratio( const value_type& value = value_type() )
        : _nominator(value), _denominator(1)
        {
        }
        
        ratio( const value_type& nom, const value_type& denom )
        : _nominator(nom), _denominator(denom)
        {
            normalize();
        }
        
        ratio( const ratio& other )
        : _nominator(other._nominator), _denominator(other._denominator)
        {
        }
        
        ratio& operator=( const ratio& other )
        {
            _nominator = other._nominator;
            _denominator = other._denominator;
            return *this;
        }
        
        ratio& operator=( const value_type& value )
        {
            _nominator = value;
            _denominator = (value_type)1;
            return *this;
        }
        
        const value_type& nominator() const
        {
            return _nominator;
        }
        
        const value_type& denominator() const
        {
            return _denominator;
        }
        
        void normalize()
        {
            auto gcd = std::gcd(_nominator, _denominator);
            if ( gcd != (value_type)1 )
            {
                _nominator /= gcd;
                _denominator /= gcd;
            }
            if ( _denominator < (value_type)0 )
            {
                _nominator = -_nominator;
                _denominator = -_denominator;
            }
        }
        
        bool is_integer() const
        {
            return _denominator == (value_type)1;
        }
        
        value_type as_integer() const
        {
            assert(is_integer());
            return _nominator;
        }
        
        operator value_type() const
        {
            return _nominator / _denominator;
        }
        
        ratio& operator+=( const ratio& other );
        ratio& operator-=( const ratio& other );
        ratio& operator*=( const ratio& other );
        ratio& operator/=( const ratio& other );
        
        ratio& operator+=( const value_type& value );
        ratio& operator-=( const value_type& value );
        ratio& operator*=( const value_type& value );
        ratio& operator/=( const value_type& value);
        
    private:
        
        value_type _nominator;
        value_type _denominator;
    };

    
    template<typename T>
    ts::ratio<T> operator+( const ts::ratio<T>& p, const ts::ratio<T>& q )
    {
        if ( p.denominator() == q.denominator() )
        {
            return ts::ratio(p.nominator() + q.nominator(), p.denominator());
        }
        else
        {
            return ts::ratio(p.nominator() * q.denominator() + q.nominator() * p.denominator(), p.denominator() * q.denominator());
        }
    }

    template<typename T>
    ts::ratio<T> operator+( const ts::ratio<T>& p, const T& v )
    {
        if ( p.is_integer() )
        {
            return ts::ratio(p.nominator() + v);
        }
        else
        {
            return ts::ratio(p.nominator() + p.denominator() * v, p.denominator());
        }
    }

    template<typename T>
    ts::ratio<T> operator+( const T& v, const ts::ratio<T>& p )
    {
        if ( p.is_integer() )
        {
            return ts::ratio(p.nominator() + v);
        }
        else
        {
            return ts::ratio(p.nominator() + p.denominator() * v, p.denominator());
        }
    }

    template<typename T>
    ts::ratio<T> operator-( const ts::ratio<T>& p, const ts::ratio<T>& q )
    {
        if ( p.denominator() == q.denominator() )
        {
            return ts::ratio(p.nominator() - q.nominator(), p.denominator());
        }
        else
        {
            return ts::ratio(p.nominator() * q.denominator() - q.nominator() * p.denominator(), p.denominator() * q.denominator());
        }
    }

    template<typename T>
    ts::ratio<T> operator-( const ts::ratio<T>& p, const T& v )
    {
        if ( p.is_integer() )
        {
            return ts::ratio(p.nominator() - v);
        }
        else
        {
            return ts::ratio(p.nominator() - p.denominator() * v, p.denominator());
        }
    }

    template<typename T>
    ts::ratio<T> operator-( const T& v, const ts::ratio<T>& p )
    {
        if ( p.is_integer() )
        {
            return ts::ratio(v - p.nominator());
        }
        else
        {
            return ts::ratio(p.denominator() * v - p.nominator(), p.denominator());
        }
    }

    template<typename T>
    ts::ratio<T> operator*( const ts::ratio<T>& p, const ts::ratio<T>& q )
    {
        return ts::ratio(p.nominator() * q.nominator(), p.denominator() * q.denominator());
    }

    template<typename T>
    ts::ratio<T> operator*( const ts::ratio<T>& p, const T& v )
    {
        return ts::ratio(p.nominator() * v, p.denominator());
    }

    template<typename T>
    ts::ratio<T> operator*( const T& v, const ts::ratio<T>& p )
    {
        return ts::ratio(p.nominator() * v, p.denominator());
    }

    template<typename T>
    ts::ratio<T> operator/( const ts::ratio<T>& p, const ts::ratio<T>& q )
    {
        return ts::ratio(p.nominator() * q.denominator(), p.denominator() * q.nominator());
    }

    template<typename T>
    ts::ratio<T> operator/( const ts::ratio<T>& p, const T& v )
    {
        return ts::ratio(p.nominator(), p.denominator() * v);
    }

    template<typename T>
    ts::ratio<T> operator/( const T& v, const ts::ratio<T>& p )
    {
        return ts::ratio(v * p.denominator(), p.nominator());
    }

    template<typename T>
    ts::ratio<T>& ts::ratio<T>::operator+=( const ts::ratio<T>& other )
    {
        *this = *this + other;
        return *this;
    }

    template<typename T>
    ts::ratio<T>& ts::ratio<T>::operator-=( const ts::ratio<T>& other )
    {
        *this = *this - other;
        return *this;
    }

    template<typename T>
    ts::ratio<T>& ts::ratio<T>::operator*=( const ts::ratio<T>& other )
    {
        *this = *this * other;
        return *this;
    }

    template<typename T>
    ts::ratio<T>& ts::ratio<T>::operator/=( const ts::ratio<T>& other )
    {
        *this = *this / other;
        return *this;
    }

    template<typename T>
    ts::ratio<T>& ts::ratio<T>::operator+=( const T& other )
    {
        *this = *this + other;
        return *this;
    }

    template<typename T>
    ts::ratio<T>& ts::ratio<T>::operator-=( const T& other )
    {
        *this = *this - other;
        return *this;
    }

    template<typename T>
    ts::ratio<T>& ts::ratio<T>::operator*=( const T& other )
    {
        *this = *this * other;
        return *this;
    }

    template<typename T>
    ts::ratio<T>& ts::ratio<T>::operator/=( const T& other )
    {
        *this = *this / other;
        return *this;
    }

    template<typename T>
    ts::ratio<T> operator+( const ts::ratio<T>& r )
    {
        return r;
    }

    template<typename T>
    ts::ratio<T> operator-( const ts::ratio<T>& r )
    {
        return ts::ratio<T>::unnormalized(-r.nominator(), r.denominator());
    }

    template<typename T>
    bool operator==( const ts::ratio<T>& p, const ts::ratio<T>& q )
    {
        return p.nominator() == q.nominator() && p.denominator() == q.denominator();
    }

    template<typename T>
    bool operator==( const ts::ratio<T>& p, const T& v )
    {
        return p.is_integer() && p.nominator() == v;
    }

    template<typename T>
    bool operator!=( const ts::ratio<T>& p, const ts::ratio<T>& q )
    {
        return !(p == q);
    }

    template<typename T>
    bool operator!=( const ts::ratio<T>& p, const T& v )
    {
        return !(p == v);
    }

    template<typename T>
    bool operator<( const ts::ratio<T>& p, const ts::ratio<T>& q )
    {
        return p.nominator() * q.denominator() < q.nominator() * p.denominator();
    }

    template<typename T>
    bool operator<( const ts::ratio<T>& p, const T& v )
    {
        return p.nominator() < v * p.denominator();
    }

    template<typename T>
    bool operator<=( const ts::ratio<T>& p, const ts::ratio<T>& q )
    {
        return p.nominator() * q.denominator() <= q.nominator() * p.denominator();
    }

    template<typename T>
    bool operator<=( const ts::ratio<T>& p, const T& v )
    {
        return p.nominator() <= v * p.denominator();
    }

    template<typename T>
    bool operator>( const ts::ratio<T>& p, const ts::ratio<T>& q )
    {
        return p.nominator() * q.denominator() > q.nominator() * p.denominator();
    }

    template<typename T>
    bool operator>( const ts::ratio<T>& p, const T& v )
    {
        return p.nominator() > v * p.denominator();
    }

    template<typename T>
    bool operator>=( const ts::ratio<T>& p, const ts::ratio<T>& q )
    {
        return p.nominator() * q.denominator() >= q.nominator() * p.denominator();
    }

    template<typename T>
    bool operator>=( const ts::ratio<T>& p, const T& v )
    {
        return p.nominator() >= v * p.denominator();
    }


    template<typename T>
    bool is_divisible( const ts::ratio<T>& x, const T& y ) { return true; }

}   // namespace ts


template<typename T>
std::ostream& operator<<( std::ostream& os, const ts::ratio<T>& r )
{
    if ( r.is_integer() )
    {
        os << r.nominator();
    }
    else
    {
        os << r.nominator() << "/" << r.denominator();
    }
    return os;
}


namespace std
{

    template<typename T>
    ts::ratio<T> abs( const ts::ratio<T>& x )
    {
        return ts::ratio<T>::unnormalized(std::abs(x.nominator()), x.denominator());
    }

}   // namespace std


#endif
