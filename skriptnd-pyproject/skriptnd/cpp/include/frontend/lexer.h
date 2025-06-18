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

#ifndef _SKND_LEXER_H_
#define _SKND_LEXER_H_

#include "result.h"
#include <iostream>
#include <exception>
#include <string>
#include <array>
#include <unordered_map>


namespace sknd
{
    
    class Lexer
    {
    public:
        
        enum class Category
        {
            Invalid,
            Keyword,
            Identifier,
            Number,
            String,
            Block,
            Operator,
        };
        
        static constexpr const char* const CategoryStrings[] =
        {
            "invalid",
            "keyword",
            "identifier",
            "number",
            "string",
            "block",
            "operator",
        };
        
        static constexpr const size_t CategoryCount = sizeof(CategoryStrings) / sizeof(const char*);
        
        static constexpr const char* str( const Category category )
        {
            return CategoryStrings[(size_t)category];
        }
        
    public:
        
        enum class Keyword
        {
            Import,
            Graph,
            Public,
            Operator,
            Optional,
            Type,
            Arith,
            Num,
            Int,
            Real,
            Bool,
            Str,
            Inf,
            Pi,
            True,
            False,
            As,
            Is,
            In,
            If,
            Then,
            Else,
            Elif,
            Do,
            For,
            While,
            With,
            Yield,
            Unroll,
        };
        
        static constexpr const char* const KeywordStrings[] =
        {
            "import",
            "graph",
            "public",
            "operator",
            "optional",
            "type",
            "arith",
            "num",
            "int",
            "real",
            "bool",
            "str",
            "inf",
            "pi",
            "true",
            "false",
            "as",
            "is",
            "in",
            "if",
            "then",
            "else",
            "elif",
            "do",
            "for",
            "while",
            "with",
            "yield",
            "unroll",
        };
        
        static constexpr const size_t KeywordCount = sizeof(KeywordStrings) / sizeof(const char*);
        
        static constexpr const char* str( const Keyword keyword )
        {
            return KeywordStrings[(size_t)keyword];
        }
        
    public:
        
        enum class Operator
        {
            Plus,
            Minus,
            Multiply,
            Divide,
            CeilDivide,
            Modulo,
            Power,
            And,
            Or,
            Xor,
            Not,
            Imply,
            LeftArrow,
            RightArrow,
            Less,
            LessEqual,
            Greater,
            GreaterEqual,
            Equal,
            NotEqual,
            PlusEqual,
            MultiplyEqual,
            AndEqual,
            OrEqual,
            MakeEqual,
            Min,
            Max,
            ArgMin,
            ArgMax,
            Bounds,
            Assign,
            Question,
            Questions,
            Comma,
            Dot,
            Dots,
            Ellipsis,
            Colon,
            Semicolon,
            Tilde,
            LeftParen,
            RightParen,
            LeftBracket,
            RightBracket,
            LeftBrace,
            RightBrace,
            Bar,
        };
        
        static constexpr const char* const OperatorStrings[] =
        {
            "+",
            "-",
            "*",
            "/",
            "\\",
            "%",
            "**",
            "&&",
            "||",
            "^",
            "!",
            "=>",
            "<-",
            "->",
            "<",
            "<=",
            ">",
            ">=",
            "==",
            "!=",
            "+=",
            "*=",
            "&=",
            "|=",
            ":=",
            "<<",
            ">>",
            "<<!",
            ">>!",
            "<>",
            "=",
            "?",
            "??",
            ",",
            ".",
            "..",
            "...",
            ":",
            ";",
            "~",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            "|",
        };
        
        static constexpr const size_t OperatorCount = sizeof(OperatorStrings) / sizeof(const char*);
        
        static constexpr const char* str( const Operator op )
        {
            return OperatorStrings[(size_t)op];
        }
        
        static bool is_comparison( const Operator op )
        {
            return op == Operator::Less || op == Operator::Greater ||
                   op == Operator::LessEqual || op == Operator::GreaterEqual ||
                   op == Operator::Equal || op == Operator::NotEqual;
        }
        
        static bool is_fold( const Operator op, bool accumulate )
        {
            if ( accumulate )
            {
                return op == Operator::Plus || op == Operator::Multiply
                    || op == Operator::Min || op == Operator::Max
                    || op == Operator::And || op == Operator::Or;
            }
            else
            {
                return op == Operator::Plus || op == Operator::Multiply
                    || op == Operator::Min || op == Operator::Max
                    || op == Operator::ArgMin || op == Operator::ArgMax
                    || op == Operator::And || op == Operator::Or
                    || op == Operator::MakeEqual || is_comparison(op);
            }
        }
        
    public:
        
        enum class Block
        {
            Dtype,
            Attrib,
            Input,
            Output,
            Constant,
            Variable,
            Using,
            Assert,
            Lower,
            Compose,
            Update,
            Quantize,
        };
        
        static constexpr const char* const BlockStrings[] =
        {
            "@dtype",
            "@attrib",
            "@input",
            "@output",
            "@constant",
            "@variable",
            "@using",
            "@assert",
            "@lower",
            "@compose",
            "@update",
            "@quantize",
        };
        
        static constexpr const size_t BlockCount = sizeof(BlockStrings) / sizeof(const char*);
        
        static constexpr const char* str( const Block block )
        {
            return BlockStrings[(size_t)block];
        }
        
    public:
        
        Lexer( std::istream& is, const Position& position )
        : _input(is), _categ(Category::Invalid), _position(position)
        {
            next();
        }
        
        Lexer( std::istream& is, const std::string& module )
        : Lexer(is, Position{ module, 1, 1 })
        {
        }
        
        bool empty() const
        {
            return _token.empty();
        }
        
        const std::string& token() const
        {
            return _token;
        }
        
        Category category() const
        {
            return _categ;
        }
        
        size_t index() const
        {
            return _index;
        }
        
        bool is_token( const Category& categ )
        {
            return _categ == categ;
        }
        
        bool is_token( const Keyword& keyword )
        {
            return _categ == Category::Keyword && _index == (size_t)keyword;
        }
        
        bool is_token( const Operator& opertor )
        {
            return _categ == Category::Operator && _index == (size_t)opertor;
        }
        
        bool is_token( const Block& block )
        {
            return _categ == Category::Block && _index == (size_t)block;
        }
        
        template<typename... Args>
        bool is_oneof( const Args&... args )
        {
            return (is_token(args) || ...);
        }
        
        Result<void> expect( const Category categ )
        {
            if ( _categ != categ )
            {
                return Error(_position, "expected '%s'; got '%s'", str(categ), _token.c_str());
            }
            return {};
        }
        
        Result<void> accept()
        {
            return next();
        }
        
        Result<void> accept( const Category categ )
        {
            if ( _categ != categ )
            {
                return Error(_position, "expected '%s'; got '%s'", str(categ), _token.c_str());
            }
            return next();
        }
        
        Result<void> accept( const Keyword keyword )
        {
            if ( _categ != Category::Keyword || _index != (size_t)keyword )
            {
                return Error(_position, "expected keyword '%s'; got '%s'", str(keyword), _token.c_str());
            }
            return next();
        }
        
        Result<void> accept( const Operator opertor )
        {
            if ( _categ != Category::Operator || _index != (size_t)opertor )
            {
                return Error(_position, "expected operator '%s'; got '%s'", str(opertor), _token.c_str());
            }
            return next();
        }
        
        Result<void> accept( const Block block )
        {
            if ( _categ != Category::Block || _index != (size_t)block )
            {
                return Error(_position, "expected block '%s'; got '%s'", str(block), _token.c_str());
            }
            return next();
        }
        
        Result<bool> accept_if( const Category categ )
        {
            if ( _categ == categ )
            {
                TRY_CALL(next())
                return true;
            }
            return false;
        }
        
        Result<bool> accept_if( const Keyword keyword )
        {
            if ( _categ == Category::Keyword && _index == (size_t)keyword )
            {
                TRY_CALL(next())
                return true;
            }
            return false;
        }
        
        Result<bool> accept_if( const Operator opertor )
        {
            if ( _categ == Category::Operator && _index == (size_t)opertor )
            {
                TRY_CALL(next())
                return true;
            }
            return false;
        }
        
        Result<bool> accept_if( const Block block )
        {
            if ( _categ == Category::Block && _index == (size_t)block )
            {
                TRY_CALL(next())
                return true;
            }
            return false;
        }
        
        template<typename... Args>
        void skip_until( const Args&... args )
        {
            while ( !_token.empty() && !(is_token(args) || ...) )
            {
                next();
            }
        }
        
        const Position& position() const
        {
            return _position;
        }

    private:
        
        Result<void> next()
        {
            _position.column += (unsigned)_token.length();
            _token.clear();
            
            skip_to_next();
            
            if ( _input.peek() == EOF )
            {
                _categ = Category::Invalid;
                return {};
            }
            
            _categ = category(_input.peek());
            switch ( _categ )
            {
                case Category::Keyword:
                case Category::Identifier:
                {
                    read_identifier();
                    
                    _index = keyword_index(_token);
                    if ( _index != KeywordCount )
                    {
                        _categ = Category::Keyword;
                    }
                    return {};
                }
                case Category::Number:
                {
                    return read_number();
                }
                case Category::String:
                {
                    return read_string();
                }
                case Category::Block:
                {
                    return read_block();
                }
                case Category::Operator:
                {
                    return read_operator();
                }
                case Category::Invalid:
                default:
                {
                    return Error(_position, "illegal symbol '%c'", (char)_input.peek());
                }
            }
        }
        
        Result<void> read_identifier()
        {
            read_while([]( char ch ){ return is_identifier(ch) || is_number(ch); });
            return {};
        }
        
        Result<void> read_number()
        {
            read_while(is_number);
            if ( read_maybe('.') )
            {
                read_while(is_number);
            }
            if ( read_maybe('e') || read_maybe('E') )
            {
                read_maybe('+') || read_maybe('-');
                if ( !is_number(_input.peek()) )
                {
                    const Position position = { _position.module, _position.line, _position.column + (unsigned)_token.length() };
                    return Error(position, "expected digit in number exponent; got symbol '%c'", (char)_input.peek());
                }
                read_while(is_number);
            }
            return {};
        }
        
        Result<void> read_string()
        {
            char delim = _input.get();
            char ch = delim;
            _token += delim;
            while ( _input.peek() != EOF && (_input.peek() != delim || ch == '\\') )
            {
                ch = _input.get();
                _token += ch;
            }
            if ( _input.peek() == EOF )
            {
                const Position position = { _position.module, _position.line, _position.column + (unsigned)_token.length() };
                return Error(position, "expected closing %c at the end of string literal", delim);
            }
            _token += _input.get();
            return {};
        }
        
        Result<void> read_block()
        {
            _token += _input.get();
            if ( !is_identifier(_input.peek()) )
            {
                const Position position = { _position.module, _position.line, _position.column + 1 };
                return Error(position, "expected block identifier after @; got symbol '%c'", (char)_input.peek());
            }
            read_while(is_identifier);
            
            _index = block_index(_token);
            if ( _index == BlockCount )
            {
                return Error(_position, "invalid block '%s'", _token.c_str());
            }
            return {};
        }
        
        Result<void> read_operator()
        {
            char ch = _input.get();
            _token += ch;
            
            if ( _input.peek() == ch )
            {
                if ( ch == '*' || ch == '&' || ch == '|' || ch == '.' || ch == '?' || ch == '=' || ch == '<' || ch == '>'  )
                {
                    _token += _input.get();
                    if ( ch == '=' && _input.peek() == '=' )
                    {
                        _token += _input.get();
                    }
                }
                if ( ch == '.' && _input.peek() == ch )
                {
                    _token += _input.get();
                }
            }
            else if ( _input.peek() == '=' )
            {
                if ( ch == '<' || ch == '>' || ch == '!' || ch == '+' || ch == '*' || ch == '&' || ch == '|' || ch == ':' )
                {
                    _token += _input.get();
                    if ( ch == '!' && _input.peek() == '=' )
                    {
                        _token += _input.get();
                    }
                }
            }
            else if ( _input.peek() == '>' )
            {
                if ( ch == '-' || ch == '=' || ch == '<' )
                {
                    _token += _input.get();
                }
            }
            else if ( _input.peek() == '!' )
            {
                if ( ch == '<' || ch == '>' )
                {
                    _token += _input.get();
                }
            }
            else if ( _input.peek() == '-' )
            {
                if ( ch == '<' )
                {
                    _token += _input.get();
                }
            }
            
            _index = operator_index(_token);
            if ( _index == OperatorCount )
            {
                return Error(_position, "invalid operator '%s'", _token.c_str());
            }
            return {};
        }
        
        template<typename Pred>
        inline void read_while( Pred pred )
        {
            while ( pred(_input.peek()) )
            {
                _token += _input.get();
            }
        }
        
        inline bool read_maybe( char c )
        {
            if ( _input.peek() == c )
            {
                _token += _input.get();
                return true;
            }
            return false;
        }
        
        void skip_space()
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
        
        void skip_comment()
        {
            while ( _input.peek() != EOF && _input.peek() != '\r' && _input.peek() != '\n' )
            {
                _input.get();
            }
        }
        
        void skip_to_next()
        {
            skip_space();
            while ( _input.peek() == '#' )
            {
                skip_comment();
                skip_space();
            }
        }
        
    public:
        
        void register_type_alias( const std::string& name, const Typename type )
        {
            _type_aliases[name] = type;
        }
        
        void unregister_type_alias( const std::string& name )
        {
            _type_aliases.erase(name);
        }
        
        void unregister_type_aliases()
        {
            _type_aliases.clear();
        }
        
        const Typename* aliased_type( const std::string& name ) const
        {
            auto it = _type_aliases.find(name);
            return it != _type_aliases.end() ? &it->second : nullptr;
        }
        
    private:
        
        static const size_t CharCount = 256;
        
        static std::array<Category,CharCount> build_category_index()
        {
            std::array<Category,CharCount> types;
            std::fill(types.begin(), types.end(), Category::Invalid);
            
            for ( char c = 'a'; c <= 'z'; ++c )
            {
                types[c] = Category::Identifier;
            }
            for ( char c = 'A'; c <= 'Z'; ++c )
            {
                types[c] = Category::Identifier;
            }
            for ( char c = '0'; c <= '9'; ++c )
            {
                types[c] = Category::Number;
            }
            
            types['_'] = Category::Identifier;
            types['"'] = Category::String;
            types['\''] = Category::String;
            types['@'] = Category::Block;
            types['+'] = Category::Operator;
            types['-'] = Category::Operator;
            types['*'] = Category::Operator;
            types['/'] = Category::Operator;
            types['\\'] = Category::Operator;
            types['%'] = Category::Operator;
            types['^'] = Category::Operator;
            types['&'] = Category::Operator;
            types['|'] = Category::Operator;
            types[','] = Category::Operator;
            types['.'] = Category::Operator;
            types[':'] = Category::Operator;
            types[';'] = Category::Operator;
            types['?'] = Category::Operator;
            types['!'] = Category::Operator;
            types['<'] = Category::Operator;
            types['>'] = Category::Operator;
            types['='] = Category::Operator;
            types['~'] = Category::Operator;
            types['('] = Category::Operator;
            types[')'] = Category::Operator;
            types['['] = Category::Operator;
            types[']'] = Category::Operator;
            types['{'] = Category::Operator;
            types['}'] = Category::Operator;
            
            return types;
        }
        
        static std::unordered_map<std::string,size_t> build_index( const char* const strings[], const size_t count )
        {
            std::unordered_map<std::string,size_t> index;
            for ( size_t i = 0; i < count; ++i )
            {
                index.emplace(strings[i], i);
            }
            return index;
        }
        
        static Category category( char ch )
        {
            static const auto categories = build_category_index();
            return categories[ch];
        }
        
        static bool is_number( char ch )
        {
            return category(ch) == Category::Number;
        }
        
        static bool is_identifier( char ch )
        {
            return category(ch) == Category::Identifier;
        }
        
        static size_t keyword_index( const std::string& str )
        {
            static const auto keywords = build_index(KeywordStrings, KeywordCount);
            auto it = keywords.find(str);
            return it != keywords.end() ? it->second : KeywordCount;
        }
        
        static size_t operator_index( const std::string& str )
        {
            static const auto operators = build_index(OperatorStrings, OperatorCount);
            auto it = operators.find(str);
            return it != operators.end() ? it->second : OperatorCount;
        }
        
        static size_t block_index( const std::string& str )
        {
            static const auto blocks = build_index(BlockStrings, BlockCount);
            auto it = blocks.find(str);
            return it != blocks.end() ? it->second : BlockCount;
        }
        
    public:
        
        static Keyword keyword_value( const std::string& str )
        {
            return (Keyword)keyword_index(str);
        }
        
        static Operator operator_value( const std::string& str )
        {
            return (Operator)operator_index(str);
        }
        
        static Block block_value( const std::string& str )
        {
            return (Block)block_index(str);
        }
        
    private:
        
        std::istream& _input;
        std::string _token;
        Category _categ;
        size_t _index;
        Position _position;
        std::map<std::string,Typename> _type_aliases;
    };
    
    
#if __cplusplus < 201703L
    constexpr const char* const Lexer::CategoryStrings[];
    constexpr const char* const Lexer::KeywordStrings[];
    constexpr const char* const Lexer::OperatorStrings[];
    constexpr const char* const Lexer::BlockStrings[];
#endif
    

}   // namespace sknd


#endif
