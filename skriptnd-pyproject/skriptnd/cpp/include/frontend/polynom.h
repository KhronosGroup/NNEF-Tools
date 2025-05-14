/*
 * Copyright (c) 2017-2025 The Khronos Group Inc.
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

#ifndef _SKND_POLYNOM_H_
#define _SKND_POLYNOM_H_

#include <functional>
#include <iostream>
#include <string>
#include <map>
#include <set>


namespace sknd
{
    
    template<typename T, typename S>
    class polynom_ratio;
    
    
    template<typename T, typename S = char>
    class polynom
    {
    public:
        
        typedef S symbol_type;
        typedef T value_type;
        
    private:
        
        typedef std::basic_string<symbol_type> monom_type;
        typedef std::map<monom_type,value_type> monom_list;
        
        polynom( monom_list&& monoms, const value_type& constant )
        : _monoms(std::move(monoms)), _constant(constant)
        {
            prune();
        }
        
        static bool is_divisible( const value_type& x, const value_type& y )
        {
            if constexpr( std::is_integral_v<value_type> )
            {
                return x % y == 0;
            }
            else
            {
                return true;
            }
        }
        
    public:
        
        typedef typename monom_list::const_iterator const_iterator;
        
    public:
        
        static polynom constant( const value_type& value )
        {
            return polynom(value);
        }
        
        static polynom monomial( const symbol_type& symbol )
        {
            return polynom(symbol);
        }
        
    public:
        
        explicit polynom( const value_type& constant = 0 )
        : _constant(constant)
        {
        }
        
        explicit polynom( const symbol_type& symbol )
        : _constant(0)
        {
            _monoms.emplace(monom_type(1, symbol), 1);
        }
        
        polynom& operator=( const polynom& ) = default;
        
        polynom& operator=( const value_type& constant )
        {
            _constant = constant;
            _monoms.clear();
            return *this;
        }
        
        polynom& operator+=( const polynom& other )
        {
            _constant += other._constant;
            for ( auto& monom : other._monoms )
            {
                _monoms[monom.first] += monom.second;
            }
            return prune();
        }
        
        polynom& operator+=( const value_type& constant )
        {
            _constant += constant;
            return *this;
        }
        
        polynom operator+( const polynom& other ) const
        {
            polynom poly = *this;
            return poly += other;
        }
        
        polynom operator+( const value_type& constant ) const
        {
            polynom poly = *this;
            return poly += constant;
        }
        
        polynom& operator-=( const polynom& other )
        {
            _constant -= other._constant;
            for ( auto& monom : other._monoms )
            {
                _monoms[monom.first] -= monom.second;
            }
            return prune();
        }
        
        polynom& operator-=( const value_type& constant )
        {
            _constant -= constant;
            return *this;
        }
        
        polynom operator-( const polynom& other ) const
        {
            polynom poly = *this;
            return poly -= other;
        }
        
        polynom operator-( const value_type& constant ) const
        {
            polynom poly = *this;
            return poly -= constant;
        }
        
        polynom& operator*=( const polynom& other )
        {
            return *this = *this * other;
        }
        
        polynom& operator*=( const value_type& constant )
        {
            for ( auto& monom : _monoms )
            {
                monom.second *= constant;
            }
            return *this;
        }
        
        polynom operator*( const polynom& other ) const
        {
            return polynom(combine(_monoms, other._monoms, _constant, other._constant), _constant * other._constant);
        }
        
        polynom operator*( const value_type& constant ) const
        {
            polynom poly = *this;
            return poly *= constant;
        }
        
        polynom_ratio<T,S> operator/( const polynom& other ) const
        {
            return polynom_ratio<T,S>(*this,other);
        }
        
        polynom_ratio<T,S> operator/( const value_type& constant ) const
        {
            return polynom_ratio<T,S>(*this, polynom(constant));
        }
        
        polynom& negate()
        {
            _constant = -_constant;
            for ( auto& monom : _monoms )
            {
                monom.second = -monom.second;
            }
            return *this;
        }
        
        polynom operator-() const
        {
            polynom poly = *this;
            return poly.negate();
        }
        
        bool operator==( const polynom& other ) const
        {
            return _constant == other._constant && _monoms == other._monoms;
        }
        
        bool operator!=( const polynom& other ) const
        {
            return _constant != other._constant || _monoms != other._monoms;
        }
        
        bool operator<( const polynom& other ) const
        {
            return _constant < other._constant || (_constant == other._constant && _monoms < other._monoms);
        }
        
        bool operator==( const value_type& constant ) const
        {
            return _constant == constant && _monoms.empty();
        }
        
        bool operator!=( const value_type& constant ) const
        {
            return _constant != constant || !_monoms.empty();
        }
        
        bool is_constant() const
        {
            return _monoms.empty();
        }
        
        const value_type& constant_value() const
        {
            return _constant;
        }
        
        const_iterator begin() const
        {
            return _monoms.begin();
        }
        
        const_iterator end() const
        {
            return _monoms.end();
        }
        
        void swap( polynom& other )
        {
            _monoms.swap(other._monoms);
            std::swap(_constant, other._constant);
        }
        
        bool is_divisible( const value_type& value ) const
        {
            for ( auto& monom : _monoms )
            {
                if ( !is_divisible(monom.second, value) )
                {
                    return false;
                }
            }
            return is_divisible(_constant, value);
        }
        
        void const_divide( const value_type& value )
        {
            _constant /= value;
            for ( auto& monom : _monoms )
            {
                monom.second /= value;
            }
        }
        
    private:
        
        static monom_list combine( const monom_list& monoms1, const monom_list& monoms2, const value_type constant1, const value_type constant2 )
        {
            monom_list monoms;
            if ( constant1 != (value_type)0 )
            {
                for ( auto& monom : monoms2 )
                {
                    monoms[monom.first] += monom.second * constant1;
                }
            }
            if ( constant2 != (value_type)0 )
            {
                for ( auto& monom : monoms1 )
                {
                    monoms[monom.first] += monom.second * constant2;
                }
            }
            for ( auto& monom1 : monoms1 )
            {
                for ( auto& monom2 : monoms2 )
                {
                    auto key = merge(monom1.first, monom2.first);
                    monoms[key] += monom1.second * monom2.second;
                }
            }
            return monoms;
        }
        
        static monom_type merge( const monom_type& m1, const monom_type& m2 )
        {
            monom_type monom(m1.length() + m2.length(), 0);
            std::merge(m1.begin(), m1.end(), m2.begin(), m2.end(), monom.begin());
            return monom;
        }
        
        polynom& prune()
        {
            for ( auto it = _monoms.begin(); it != _monoms.end(); )
            {
                it = it->second == 0 ? _monoms.erase(it) : ++it;
            }
            return *this;
        }
        
    private:
        
        monom_list _monoms;
        value_type _constant;
    };
    
    
    template<typename T, typename S>
    inline polynom<T,S> operator+( const T& x, const polynom<T,S>& p )
    {
        return p + x;
    }
    
    template<typename T, typename S>
    inline polynom<T,S> operator-( const T& x, const polynom<T,S>& p )
    {
        return polynom<T,S>(x) - p;
    }
    
    template<typename T, typename S>
    inline polynom<T,S> operator*( const T& x, const polynom<T,S>& p )
    {
        return p * x;
    }
    
    template<typename T, typename S>
    inline polynom_ratio<T,S> operator/( const T& x, const polynom<T,S>& p )
    {
        return polynom<T,S>(x) / p;
    }
    
    
    template<typename S>
    class polynom<bool,S>
    {
    public:
        
        typedef S symbol_type;
        typedef bool value_type;
        
    private:
        
        typedef std::basic_string<symbol_type> monom_type;
        typedef std::set<monom_type> monom_list;
        
    public:
        
        typedef typename monom_list::const_iterator const_iterator;
        
    public:
        
        static polynom constant( const value_type& value )
        {
            return polynom(value);
        }
        
        static polynom monomial( const symbol_type& symbol )
        {
            return polynom(symbol);
        }
        
    public:
        
        explicit polynom( const value_type& constant = false )
        : _constant(constant)
        {
        }
        
        explicit polynom( const symbol_type& symbol )
        : _monoms({ monom_type(1,symbol) }), _constant(false)
        {
        }
        
        polynom& operator=( const polynom& ) = default;
        
        polynom& operator=( const value_type& constant )
        {
            _constant = constant;
            _monoms.clear();
            return *this;
        }
        
        polynom& operator^=( const polynom& other )
        {
            _monoms = diff(_monoms, other._monoms);
            _constant ^= other._constant;
            return *this;
        }
        
        polynom& operator^=( const bool constant )
        {
            _constant ^= constant;
            return *this;
        }
        
        polynom operator^( const polynom& other ) const
        {
            polynom poly;
            poly._monoms = diff(_monoms, other._monoms);
            poly._constant = _constant ^ other._constant;
            return poly;
        }
        
        polynom operator^( const value_type& constant ) const
        {
            polynom poly = *this;
            poly._constant ^= constant;
            return poly;
        }
        
        polynom operator!() const
        {
            return *this ^ true;
        }
        
        polynom& negate()
        {
            _constant = !_constant;
            return *this;
        }
        
        polynom operator&&( const polynom& other ) const
        {
            polynom poly;
            poly._constant = _constant && other._constant;
            poly._monoms = combine(_monoms, other._monoms, _constant, other._constant);
            return poly;
        }
        
        polynom& operator&=( const polynom& other )
        {
            return *this = *this && other;
        }
        
        polynom operator||( const polynom& other ) const
        {
            polynom poly;
            poly._constant = _constant || other._constant;
            poly._monoms = combine(_monoms, other._monoms, !_constant, !other._constant);
            return poly;
        }
        
        polynom& operator|=( const polynom& other )
        {
            return *this = *this || other;
        }
        
        bool operator==( const polynom& other ) const
        {
            return _constant == other._constant && _monoms == other._monoms;
        }
        
        bool operator!=( const polynom& other ) const
        {
            return _constant != other._constant || _monoms != other._monoms;
        }
        
        bool operator<( const polynom& other ) const
        {
            return _constant < other._constant || (_constant == other._constant && _monoms < other._monoms);
        }
        
        bool operator==( const value_type& constant ) const
        {
            return _constant == constant && _monoms.empty();
        }
        
        bool operator!=( const value_type& constant ) const
        {
            return _constant != constant || !_monoms.empty();
        }
        
        bool is_constant() const
        {
            return _monoms.empty();
        }
        
        const value_type& constant_value() const
        {
            return _constant;
        }
        
        const_iterator begin() const
        {
            return _monoms.begin();
        }
        
        const_iterator end() const
        {
            return _monoms.end();
        }
        
        void swap( polynom& other )
        {
            _monoms.swap(other._monoms);
            std::swap(_constant, other._constant);
        }
        
    private:
        
        static monom_list diff( const monom_list& monoms1, const monom_list& monoms2 )
        {
            monom_list monoms;
            std::set_symmetric_difference(monoms1.begin(), monoms1.end(), monoms2.begin(), monoms2.end(), std::inserter(monoms, monoms.end()));
            return monoms;
        }
        
        static monom_list combine( const monom_list& monoms1, const monom_list& monoms2, bool constant1, bool constant2 )
        {
            monom_list monoms;
            if ( constant1 && constant2 )
            {
                monoms = diff(monoms1, monoms2);
            }
            else if ( constant1 )
            {
                monoms = monoms2;
            }
            else if ( constant2 )
            {
                monoms = monoms1;
            }
            for ( auto& monom1 : monoms1 )
            {
                for ( auto& monom2 : monoms2 )
                {
                    auto monom = unify(monom1, monom2);
                    auto it = std::lower_bound(monoms.begin(), monoms.end(), monom);
                    if ( it != monoms.end() && *it == monom )
                    {
                        monoms.erase(it);
                    }
                    else
                    {
                        monoms.insert(it, monom);
                    }
                }
            }
            return monoms;
        }
        
        static monom_type unify( const monom_type& m1, const monom_type& m2 )
        {
            monom_type monom;
            std::set_union(m1.begin(), m1.end(), m2.begin(), m2.end(), std::back_inserter(monom));
            return monom;
        }
        
    private:
        
        monom_list _monoms;
        bool _constant;
    };
    
    
    template<typename S>
    inline polynom<bool,S> operator&&( const bool x, const polynom<bool,S>& p )
    {
        return p && x;
    }
    
    template<typename S>
    inline polynom<bool,S> operator||( const bool x, const polynom<bool,S>& p )
    {
        return p || x;
    }
    
    template<typename S>
    inline polynom<bool,S> operator^( const bool x, const polynom<bool,S>& p )
    {
        return p ^ x;
    }
    
    
    template<typename T, typename S = char>
    class polynom_ratio
    {
    public:
        
        typedef polynom<T,S> polynom_type;
        typedef typename polynom_type::value_type value_type;
        typedef typename polynom_type::symbol_type symbol_type;
        
    public:
        
        explicit polynom_ratio( const value_type constant = 0 )
        : _nominator(constant), _denominator(1)
        {
        }
        
        explicit polynom_ratio( const symbol_type symbol )
        : _nominator(symbol), _denominator(1)
        {
        }
        
        explicit polynom_ratio( const polynom_type& nominator )
        : _nominator(nominator), _denominator(1)
        {
        }
        
        polynom_ratio( const polynom_type& nominator, const polynom_type& denominator )
        : _nominator(nominator), _denominator(denominator)
        {
        }
        
        polynom_ratio& operator=( const polynom_ratio& ) = default;
        
        polynom_ratio& operator=( const value_type& constant )
        {
            _nominator = constant;
            _denominator = 1;
            return *this;
        }
        
        polynom_ratio& operator+=( const polynom_ratio& other )
        {
            if ( _denominator == other._denominator )
            {
                _nominator += other._nominator;
            }
            else
            {
                _nominator *= other._denominator;
                _nominator += _denominator * other._nominator;
                _denominator *= other._denominator;
            }
            return *this;
        }
        
        polynom_ratio& operator+=( const value_type constant )
        {
            _nominator += _denominator * constant;
            return *this;
        }
        
        polynom_ratio operator+( const polynom_ratio& other ) const
        {
            polynom_ratio poly = *this;
            return poly += other;
        }
        
        polynom_ratio operator+( const value_type constant ) const
        {
            polynom_ratio poly = *this;
            return poly += constant;
        }
        
        polynom_ratio& operator-=( const polynom_ratio& other )
        {
            if ( _denominator == other._denominator )
            {
                _nominator -= other._nominator;
            }
            else
            {
                _nominator *= other._denominator;
                _nominator -= _denominator * other._nominator;
                _denominator *= other._denominator;
            }
            return *this;
        }
        
        polynom_ratio& operator-=( const value_type constant )
        {
            _nominator -= _denominator * constant;
            return *this;
        }
        
        polynom_ratio operator-( const polynom_ratio& other ) const
        {
            polynom_ratio poly = *this;
            return poly -= other;
        }
        
        polynom_ratio operator-( const value_type constant ) const
        {
            polynom_ratio poly = *this;
            return poly -= constant;
        }
        
        polynom_ratio& operator*=( const polynom_ratio& other )
        {
            _nominator *= other._nominator;
            _denominator *= other._denominator;
            return *this;
        }
        
        polynom_ratio& operator*=( const value_type constant )
        {
            _nominator *= constant;
            return *this;
        }
        
        polynom_ratio operator*( const polynom_ratio& other ) const
        {
            polynom_ratio poly = *this;
            return poly *= other;
        }
        
        polynom_ratio operator*( const value_type constant ) const
        {
            polynom_ratio poly = *this;
            return poly *= constant;
        }
        
        polynom_ratio& operator/=( const polynom_ratio& other )
        {
            _nominator *= other._denominator;
            _denominator *= other._nominator;
            return *this;
        }
        
        polynom_ratio& operator/=( const value_type constant )
        {
            _denominator *= constant;
            return *this;
        }
        
        polynom_ratio operator/( const polynom_ratio& other ) const
        {
            polynom_ratio poly = *this;
            return poly /= other;
        }
        
        polynom_ratio operator/( const value_type constant ) const
        {
            polynom_ratio poly = *this;
            return poly /= constant;
        }
        
        polynom_ratio& negate()
        {
            _nominator.negate();
            return *this;
        }
        
        polynom_ratio operator-() const
        {
            polynom_ratio poly = *this;
            return poly.negate();
        }
        
        bool operator==( const polynom_ratio& other ) const
        {
            if ( _denominator == other._denominator )
            {
                return _nominator == other._nominator;
            }
            else
            {
                return _nominator * other._denominator == other._nominator * _denominator;
            }
        }
        
        bool operator!=( const polynom_ratio& other ) const
        {
            return !(*this == other);
        }
        
        bool operator==( const polynom<T,S>& poly ) const
        {
            return _nominator == poly * _denominator;
        }
        
        bool operator!=( const polynom<T,S>& poly ) const
        {
            return !(*this == poly);
        }
        
        bool operator==( const value_type& value ) const
        {
            return _nominator == value && _denominator == 1;
        }
        
        bool operator!=( const value_type& value ) const
        {
            return !(*this == value);
        }
        
        bool is_constant() const
        {
            return _nominator.is_constant() && _denominator.is_constant();
        }
        
        value_type constant_value() const
        {
            return _nominator.constant_value() / _denominator.constant_value();
        }
        
        const polynom_type& nominator() const
        {
            return _nominator;
        }
        
        const polynom_type& denominator() const
        {
            return _denominator;
        }
        
        polynom_type& nominator()
        {
            return _nominator;
        }
        
        polynom_type& denominator()
        {
            return _denominator;
        }
        
        void swap( polynom_ratio& other )
        {
            _nominator.swap(other._nominator);
            _denominator.swap(other._denominator);
        }
        
    private:
        
        polynom_type _nominator;
        polynom_type _denominator;
    };
    
    
    template<typename T, typename S>
    inline polynom_ratio<T,S> operator+( const T& x, const polynom_ratio<T,S>& p )
    {
        return p + x;
    }
    
    template<typename T, typename S>
    inline polynom_ratio<T,S> operator-( const T& x, const polynom_ratio<T,S>& p )
    {
        return polynom_ratio<T,S>(x) - p;
    }
    
    template<typename T, typename S>
    inline polynom_ratio<T,S> operator*( const T& x, const polynom_ratio<T,S>& p )
    {
        return p * x;
    }
    
    template<typename T, typename S>
    inline polynom_ratio<T,S> operator/( const T& x, const polynom_ratio<T,S>& p )
    {
        return polynom_ratio<T,S>(x) / p;
    }
    
    
    template<typename P>
    class polynom_cases
    {
    public:
        
        typedef P polynom_type;
        typedef typename polynom_type::value_type value_type;
        typedef typename polynom_type::symbol_type symbol_type;
        typedef polynom<bool,symbol_type> condition_type;
        
    private:
        
        typedef std::map<condition_type,polynom_type> case_list;
        
        explicit polynom_cases( case_list&& cases )
        : _cases(std::move(cases))
        {
        }
        
    public:
        
        typedef typename case_list::iterator iterator;
        typedef typename case_list::const_iterator const_iterator;
        
    public:
        
        explicit polynom_cases( const value_type value = 0 )
        {
            _cases.emplace(condition_type(true), polynom_type(value));
        }
        
        explicit polynom_cases( const symbol_type symbol )
        {
            _cases.emplace(condition_type(true), polynom_type(symbol));
        }
        
        explicit polynom_cases( const polynom_type& poly )
        {
            _cases.emplace(condition_type(true), poly);
        }
        
        polynom_cases( const condition_type& condition, const polynom_type& then_poly, const polynom_type& else_poly )
        {
            _cases.emplace(condition, then_poly);
            _cases.emplace(!condition, else_poly);
        }
        
        polynom_cases( const condition_type& condition, const polynom_cases& then_poly, const polynom_cases& else_poly )
        {
            insert(condition, then_poly);
            insert(!condition, else_poly);
        }
        
        polynom_cases& negate()
        {
            for ( auto& item : _cases )
            {
                item.second.negate();
            }
            return *this;
        }
        
        polynom_cases operator-() const
        {
            polynom_cases poly = *this;
            return poly.negate();
        }
        
        polynom_cases& operator+=( const polynom_cases& other )
        {
            _cases = combine<std::plus>(_cases, other._cases);
            return *this;
        }
        
        polynom_cases operator+( const polynom_cases& other ) const
        {
            return polynom_cases(combine<std::plus>(_cases, other._cases));
        }
        
        polynom_cases& operator-=( const polynom_cases& other )
        {
            _cases = combine<std::minus>(_cases, other._cases);
            return *this;
        }
        
        polynom_cases operator-( const polynom_cases& other ) const
        {
            return polynom_cases(combine<std::minus>(_cases, other._cases));
        }
        
        polynom_cases& operator*=( const polynom_cases& other )
        {
            _cases = combine<std::multiplies>(_cases, other._cases);
            return *this;
        }
        
        polynom_cases operator*( const polynom_cases& other ) const
        {
            return polynom_cases(combine<std::multiplies>(_cases, other._cases));
        }
        
        polynom_cases& operator/=( const polynom_cases& other )
        {
            _cases = combine<std::divides>(_cases, other._cases);
            return *this;
        }
        
        polynom_cases operator/( const polynom_cases& other ) const
        {
            return polynom_cases(combine<std::divides>(_cases, other._cases));
        }
        
        polynom_cases& operator&=( const polynom_cases& other )
        {
            _cases = combine<std::logical_and>(_cases, other._cases);
            return *this;
        }
        
        polynom_cases operator&&( const polynom_cases& other ) const
        {
            return polynom_cases(combine<std::logical_and>(_cases, other._cases));
        }
        
        polynom_cases& operator|=( const polynom_cases& other )
        {
            _cases = combine<std::logical_or>(_cases, other._cases);
            return *this;
        }
        
        polynom_cases operator||( const polynom_cases& other ) const
        {
            return polynom_cases(combine<std::logical_or>(_cases, other._cases));
        }
        
        polynom_cases& operator^=( const polynom_cases& other )
        {
            _cases = combine<std::bit_xor>(_cases, other._cases);
            return *this;
        }
        
        polynom_cases operator^( const polynom_cases& other ) const
        {
            return polynom_cases(combine<std::bit_xor>(_cases, other._cases));
        }
        
        bool operator==( const polynom_cases& other ) const
        {
            return _cases == other._cases;
        }
        
        bool operator!=( const polynom_cases& other ) const
        {
            return !(*this == other);
        }
        
        bool operator==( const polynom_type& poly ) const
        {
            return is_unique() && unique_value() == poly;
        }
        
        bool operator!=( const polynom_type& poly ) const
        {
            return !(*this == poly);
        }
        
        bool operator==( const value_type& value ) const
        {
            return is_unique() && unique_value() == value;
        }
        
        bool operator!=( const value_type& value ) const
        {
            return !(*this == value);
        }
        
        bool is_unique() const
        {
            return _cases.size() == 1;
        }
        
        polynom_type unique_value() const
        {
            return _cases.begin()->second;
        }
        
        const_iterator begin() const
        {
            return _cases.begin();
        }
        
        const_iterator end() const
        {
            return _cases.end();
        }
        
        iterator begin()
        {
            return _cases.begin();
        }
        
        iterator end()
        {
            return _cases.end();
        }
        
    private:
        
        void insert( const condition_type& condition, const polynom_cases& poly )
        {
            for ( auto& item : poly._cases )
            {
                auto cond = item.first && condition;
                if ( cond != false )
                {
                    _cases.emplace(cond, item.second);
                }
            }
        }
        
        template<template<typename> class Op>
        static case_list combine( const case_list& lhs, const case_list& rhs )
        {
            Op<polynom_type> op;
            case_list cases;
            for ( auto lit = lhs.begin(); lit != lhs.end(); ++lit )
            {
                for ( auto rit = rhs.begin(); rit != rhs.end(); ++rit )
                {
                    auto cond = lit->first && rit->first;
                    if ( cond != false )
                    {
                        cases.emplace(cond, op(lit->second, rit->second));
                    }
                }
            }
            return cases;
        }
        
    private:
        
        case_list _cases;
    };
    
    
    
    inline std::string to_string( const std::string& str )
    {
        return str;
    }

    template<typename Ch>
    inline std::string to_string( const std::basic_string<Ch>& wstr )
    {
        size_t len = 0;
        
        char buf[16];
        for ( Ch ch : wstr )
        {
            len += snprintf(buf, 16, "\\x%x", (int)ch);
        }
        
        std::string str;
        str.reserve(len);
        
        for ( Ch ch : wstr )
        {
            snprintf(buf, 16, "\\x%x", (int)ch);
            str += buf;
        }
        
        return str;
    }
    
    
    template<typename T, typename S>
    inline std::ostream& operator<<( std::ostream& os, const sknd::polynom<T,S>& poly )
    {
        if ( poly.constant_value() || poly.is_constant() )
        {
            os << poly.constant_value();
        }
        for ( auto it = poly.begin(); it != poly.end(); ++it )
        {
            if ( it != poly.begin() || it->second < 0 || poly.constant_value() )
            {
                os << (it->second > 0 ? '+' : '-');
            }
            auto coeff = std::abs(it->second);
            if ( coeff != 1 )
            {
                os << coeff;
            }
            os << sknd::to_string(it->first);
        }
        return os;
    }
    
    template<typename S>
    inline std::ostream& operator<<( std::ostream& os, const sknd::polynom<bool,S>& poly )
    {
        if ( poly.constant_value() || poly.is_constant() )
        {
            os << poly.constant_value();
        }
        for ( auto it = poly.begin(); it != poly.end(); ++it )
        {
            if ( it != poly.begin() || poly.constant_value() )
            {
                os << '^';
            }
            os << sknd::to_string(*it);
        }
        return os;
    }
    
    template<typename T, typename S>
    inline std::ostream& operator<<( std::ostream& os, const sknd::polynom_ratio<T,S>& ratio )
    {
        if ( ratio.denominator() != 1 )
        {
            os << '(' << ratio.nominator() << ")/(" << ratio.denominator() << ')';
        }
        else
        {
            os << ratio.nominator();
        }
        return os;
    }
    
    template<typename P>
    inline std::ostream& operator<<( std::ostream& os, const sknd::polynom_cases<P>& cases )
    {
        if ( cases.is_unique() )
        {
            return os << cases.unique_value();
        }
        else
        {
            os << "{ ";
            for ( auto it = cases.begin(); it != cases.end(); ++it )
            {
                os << (it == cases.begin() ? "" : ", ") << it->first << ": " << it->second;
            }
            os << " }";
            return os;
        }
    }
    
    
    template<typename T, typename S>
    inline std::string str( const polynom<T,S>& poly )
    {
        std::ostringstream os;
        os << poly;
        return os.str();
    }
    
    template<typename S>
    inline std::string str( const polynom<bool,S>& poly )
    {
        std::ostringstream os;
        os << poly;
        return os.str();
    }
    
    template<typename T, typename S>
    inline std::string str( const polynom_ratio<T,S>& ratio )
    {
        std::ostringstream os;
        os << ratio;
        return os.str();
    }
    
    template<typename P>
    inline std::string str( const polynom_cases<P>& cases )
    {
        std::ostringstream os;
        os << cases;
        return os.str();
    }
    
}   // namespace sknd


namespace std
{
    
    template<typename T, typename S>
    void swap( sknd::polynom<T,S>& lhs, sknd::polynom<T,S>& rhs )
    {
        lhs.swap(rhs);
    }
    
    template<typename T, typename S>
    void swap( sknd::polynom_ratio<T,S>& lhs, sknd::polynom_ratio<T,S>& rhs )
    {
        lhs.swap(rhs);
    }
    
}   // namespace std


#endif

