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

#ifndef _NNEF_LEXER_H_
#define _NNEF_LEXER_H_

#include "error.h"
#include <iostream>
#include <cctype>
#include <string>
#include <map>


namespace nnef
{
    
    class Lexer
    {
    public:
        
        typedef Error::Position Position;
        
    public:
        
        enum Token
        {
            Eof,
            Version,
            Extension,
            Identifier,
            Characters,
            Decimal,
            Fractional,
            Graph,
            Fragment,
            Tensor,
            Integer,
            Scalar,
            Logical,
            String,
            True,
            False,
            For,
            In,
            If,
            Else,
            Yield,
            LengthOf,
            ShapeOf,
            RangeOf,
            Arrow,
            And,
            Or,
            Le,
            Ge,
            Eq,
            Ne,
        };
        
        static std::string tokenString( int token )
        {
            static const std::string strings[] =
            {
                "eof",
                "version",
                "extension",
                "identifier",
                "literal",
                "decimal",
                "fractional",
                "graph",
                "fragment",
                "tensor",
                "integer",
                "scalar",
                "logical",
                "string",
                "true",
                "false",
                "for",
                "in",
                "if",
                "else",
                "yield",
                "length_of",
                "shape_of",
                "range_of",
                "->",
                "&&",
                "||",
                "<=",
                ">=",
                "==",
                "!=",
            };
            
            char ch = (char)token;
            return token <= Ne ? strings[token] : std::string(&ch, 1);
        }
        
        static bool isType( int token )
        {
            return token >= Tensor && token <= String;
        }

        static bool isKeyword( int token )
        {
            return token >= Fragment && token <= False;
        }
        
        static bool isOperator( int token )
        {
            return token >= LengthOf;
        }
        
    public:
        
        Lexer( std::istream& input, const char* filename )
        : _input(input), _position({1,1,filename,nullptr}), _token(Eof)
        {
        }
        
        void next()
        {
            _position.column += (unsigned)_string.length() + 2 * (_token == Characters);
            
            skipSpace();
            skipComment();

            _string.clear();

            if ( _input.peek() == EOF )
            {
                _token = Eof;
            }
            else if ( _input.peek() == '\'' || _input.peek() == '\"' )
            {
                _token = getCharacters();
            }
            else if ( std::isalpha(_input.peek()) || _input.peek() == '_' )
            {
                _token = getIdentifier();
            }
            else if ( std::isdigit(_input.peek()) )
            {
                _token = getNumber();
            }
            else
            {
                _token = getOperator();
            }
        }
        
        int token() const
        {
            return _token;
        }
        
        const std::string& string() const
        {
            return _string;
        }
        
        const Position& position() const
        {
            return _position;
        }

        void readToken( int token )
        {
            if ( _token != token )
            {
                throw Error(_position, "expected token '%s', found '%s'", tokenString(token).c_str(), tokenString(_token).c_str());
            }
            next();
        }

        bool readIfToken( int token )
        {
            if ( _token == token )
            {
                next();
                return true;
            }
            return false;
        }
        
    private:
        
        Token getCharacters()
        {
            char delim = _input.get();
            while ( _input.peek() != delim && _input.peek() != EOF )
            {
                _string += (char)_input.get();
            }
            if ( _input.peek() == EOF )
            {
                const Position position = { _position.line, _position.column + (unsigned)_string.length() + 1, _position.filename, nullptr };
                throw Error(position, "expected %c", delim);
            }
            _input.get();
            return Token::Characters;
        }
        
        Token getIdentifier()
        {
            static const std::map<std::string,Token> keywords =
            {
                std::make_pair("version", Token::Version),
                std::make_pair("extension", Token::Extension),
                std::make_pair("graph", Token::Graph),
                std::make_pair("fragment", Token::Fragment),
                std::make_pair("tensor", Token::Tensor),
                std::make_pair("integer", Token::Integer),
                std::make_pair("scalar", Token::Scalar),
                std::make_pair("logical", Token::Logical),
                std::make_pair("string", Token::String),
                std::make_pair("true", Token::True),
                std::make_pair("false", Token::False),
                std::make_pair("for", Token::For),
                std::make_pair("in", Token::In),
                std::make_pair("if", Token::If),
                std::make_pair("else", Token::Else),
                std::make_pair("yield", Token::Yield),
                std::make_pair("length_of", Token::LengthOf),
                std::make_pair("shape_of", Token::ShapeOf),
                std::make_pair("range_of", Token::RangeOf),
            };
            
            do
            {
                _string += _input.get();
            }
            while ( std::isalnum(_input.peek()) || _input.peek() == '_' );
            
            auto it = keywords.find(_string);
            return it == keywords.end() ? Token::Identifier : it->second;
        }
        
        Token getNumber()
        {
            bool real = false;
            
            do
            {
                _string += _input.get();
                
                if ( _input.peek() == '.' && !real )
                {
                    _string += _input.get();
                    real = true;
                }
            }
            while ( std::isdigit(_input.peek()) );
            
            if ( _input.peek() == 'e' || _input.peek() == 'E' )
            {
                _string += _input.get();
                if ( _input.peek() == '+' || _input.peek() == '-' )
                {
                    _string += _input.get();
                }
                if ( !std::isdigit(_input.peek()) )
                {
                    const Position position = { _position.line, _position.column + (unsigned)_string.length(), _position.filename, nullptr };
                    throw Error(position, "expected digit");
                }
                while ( std::isdigit(_input.peek()) )
                {
                    _string += _input.get();
                }
                real = true;
            }
            
            return real ? Token::Fractional : Token::Decimal;
        }
        
        int getOperator()
        {
            int token = _input.get();
            _string += (char)token;
            
            if ( _input.peek() == '=' )
            {
                if ( token == '<' )
                {
                    _string += (char)_input.get();
                    token = Le;
                }
                else if ( token == '>' )
                {
                    _string += (char)_input.get();
                    token = Ge;
                }
                else if ( token == '=' )
                {
                    _string += (char)_input.get();
                    token = Eq;
                }
                else if ( token == '!' )
                {
                    _string += (char)_input.get();
                    token = Ne;
                }
            }
            if ( token == '&' && _input.peek() == '&' )
            {
                _string += (char)_input.get();
                token = And;
            }
            else if ( token == '|' && _input.peek() == '|' )
            {
                _string += (char)_input.get();
                token = Or;
            }
            else if ( token == '-' && _input.peek() == '>' )
            {
                _string += (char)_input.get();
                token = Arrow;
            }
            
            return token;
        }
        
        void skipSpace()
        {
            while ( std::isspace(_input.peek()) )
            {
                ++_position.column;
                
                char ch = _input.get();
                if ( ch == '\r' || ch == '\n' )
                {
                    ++_position.line;
                    _position.column = 1;
                }
                if ( ch == '\r' && _input.peek() == '\n' )
                {
                    _input.get();
                }
            }
        }
        
        void skipComment()
        {
            while ( _input.peek() == '#' )
            {
                while ( _input.peek() != '\n' && _input.peek() != '\r' && _input.peek() != EOF )
                {
                    _input.get();
                    ++_position.column;
                }
                
                skipSpace();
            }
        }
        
    private:
        
        std::istream& _input;
        std::string _string;
        Position _position;
        int _token;
    };


    inline float getScalarValue( Lexer& lexer )
    {
        return (float)std::atof(lexer.string().c_str());
    }

    inline int getIntegerValue( Lexer& lexer )
    {
        return std::atoi(lexer.string().c_str());
    }
    
}   // namespace nnef


#endif
