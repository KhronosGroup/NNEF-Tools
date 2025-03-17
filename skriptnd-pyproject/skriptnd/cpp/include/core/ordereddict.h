#ifndef _TS_ORDERED_DICT_H_
#define _TS_ORDERED_DICT_H_

#include <list>
#include <iostream>
#include <string>


namespace nd
{

    template<typename T>
    class OrderedDict
    {
    public:
        
        typedef T value_type;
        typedef std::string key_type;
        typedef std::pair<const key_type,value_type> item_type;
        typedef std::list<item_type> container_type;
        typedef typename container_type::size_type size_type;
        typedef typename container_type::iterator iterator;
        typedef typename container_type::const_iterator const_iterator;
        
    public:
        
        OrderedDict()
        {
        }
        
        OrderedDict( std::initializer_list<item_type> il )
        : _items(il)
        {
        }
        
        OrderedDict( const OrderedDict& other )
        : _items(other._items)
        {
        }
        
        OrderedDict( OrderedDict&& other ) noexcept
        : _items(std::forward<container_type>(other._items))
        {
        }
        
        OrderedDict& operator=( const OrderedDict& other )
        {
            _items = other._items;
            return *this;
        }
        
        OrderedDict& operator=( OrderedDict&& other ) noexcept
        {
            _items.swap(other._items);
            return *this;
        }
        
        size_type size() const
        {
            return _items.size();
        }
        
        iterator begin()
        {
            return _items.begin();
        }
        
        iterator end()
        {
            return _items.end();
        }
        
        const_iterator begin() const
        {
            return _items.begin();
        }
        
        const_iterator end() const
        {
            return _items.end();
        }
        
        size_type count( const key_type& key ) const
        {
            return find(key) != end() ? 1 : 0;
        }
        
        iterator find( const key_type& key )
        {
            for ( auto it = _items.begin(); it != _items.end(); ++it )
            {
                if ( it->first == key )
                {
                    return it;
                }
            }
            return _items.end();
        }
        
        const_iterator find( const key_type& key ) const
        {
            for ( auto it = _items.begin(); it != _items.end(); ++it )
            {
                if ( it->first == key )
                {
                    return it;
                }
            }
            return _items.end();
        }
        
        value_type& at( const key_type& key )
        {
            auto it = find(key);
            if ( it == end() )
            {
                throw std::out_of_range("key not found: " + key);
            }
            return it->second;
        }
        
        const value_type& at( const key_type& key ) const
        {
            auto it = find(key);
            if ( it == end() )
            {
                throw std::out_of_range("key not found: " + key);
            }
            return it->second;
        }
        
        value_type& operator[]( const key_type& key )
        {
            auto it = find(key);
            if ( it != end() )
            {
                return it->second;
            }
            else
            {
                _items.emplace_back(item_type(key,value_type()));
                return _items.back().second;
            }
        }
        
        std::pair<iterator,bool> insert( const item_type& item )
        {
            auto it = find(item.key);
            if ( it != end() )
            {
                return std::make_pair(it, false);
            }
            it = _items.emplace(_items.end(), item);
            return std::make_pair(it, true);
        }
        
        std::pair<iterator,bool> emplace( const key_type& key, const value_type& value )
        {
            auto it = find(key);
            if ( it != end() )
            {
                return std::make_pair(it, false);
            }
            it = _items.emplace(_items.end(), item_type(key, value));
            return std::make_pair(it, true);
        }
        
        iterator erase( iterator pos )
        {
            return _items.erase(pos);
        }
        
        size_type erase( const key_type& key )
        {
            auto it = find(key);
            if ( it == end() )
            {
                return 0;
            }
            erase(it);
            return 1;
        }
        
        void clear()
        {
            _items.clear();
        }
        
    private:
        
        container_type _items;
    };



    template<typename T>
    inline std::string str( const OrderedDict<T>& dict, const char* item_sep = ", ", const char* entry_sep = ": " )
    {
        std::string str;
        
        str += '{';
        for ( auto it = dict.begin(); it != dict.end(); ++it )
        {
            if ( it != dict.begin() )
            {
                str += item_sep;
            }
            str += it->first;
            str += entry_sep;
            str += nd::str(it->second);
        }
        str += '}';
        
        return str;
    }

}   // namespace nd


template<typename T>
std::ostream& operator<<( std::ostream& os, const nd::OrderedDict<T>& dict )
{
    os << "{ ";
    for ( auto it = dict.begin(); it != dict.end(); ++it )
    {
        if ( it != dict.begin() )
        {
            os << ", ";
        }
        os << '"' << it->first << '"' << ": " << it->second;
    }
    os << " }";
    return os;
}


#endif
