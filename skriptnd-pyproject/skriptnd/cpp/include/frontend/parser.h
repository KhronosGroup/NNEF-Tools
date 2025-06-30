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

#ifndef _SKND_PARSER_H_
#define _SKND_PARSER_H_

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif

#include "lexer.h"
#include "operator.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cassert>
#include <limits>
#include <math.h>
#include <map>
#include <set>


namespace sknd
{

    class Parser
    {
        using Category = Lexer::Category;
        using Keyword = Lexer::Keyword;
        using Operator = Lexer::Operator;
        using Block = Lexer::Block;
        
        enum Flags : unsigned
        {
            IsDecl = 0x01,
            IsUsing = 0x02,
            IsIndex = 0x04,
            AllowTilde = 0x10,
        };

        static constexpr const char* FileExtension = ".sknd";
        
    public:
        
        Parser( const std::string& stdlib_path, const std::string& import_path, const ErrorCallback error )
        : _stdlib_path(stdlib_path), _import_path(import_path), _error(error)
        {
        }
        
        std::pair<std::map<std::string,sknd::Operator>,std::vector<std::string>> operator()( std::istream& is, const std::string& module )
        {
            std::set<std::string> imports;
            std::map<std::string,sknd::Operator> operators;
            std::vector<std::string> graphs;
            
            Lexer lexer(is, module);
            
            parse_module(lexer, imports, operators, graphs, true);
            return std::make_pair(std::move(operators), std::move(graphs));
        }
        
    private:
        
        bool import_module( const std::string& module, std::set<std::string>& imports, std::map<std::string,sknd::Operator>& operators,
                           std::vector<std::string>& graphs )
        {
            auto insert = imports.insert(module);
            if ( insert.second )
            {
                std::ifstream is;
                is.open(_stdlib_path + module + FileExtension);
                if ( !is.is_open() && !_import_path.empty() )
                {
                    std::string module_path = module;
                    std::replace(module_path.begin(), module_path.end(), '.', '/');
                    is.open(_import_path + module_path + FileExtension);
                }
                if ( !is.is_open() )
                {
                    return false;
                }
                
                Lexer lexer(is, *insert.first);
                parse_module(lexer, imports, operators, graphs, false);
            }
            return true;
        }
        
        void parse_module( Lexer& lexer, std::set<std::string>& imports, std::map<std::string,sknd::Operator>& operators, 
                          std::vector<std::string>& graphs, const bool main )
        {
            while ( lexer.is_token(Keyword::Import) )
            {
                auto result = parse_import(lexer, imports, operators, graphs);
                if ( !result )
                {
                    report_error(result.error());
                    lexer.skip_until(Keyword::Import, Keyword::Public, Keyword::Operator, Keyword::Graph);
                }
            }
            
            while ( !lexer.empty() )
            {
                auto op = parse_operator(lexer, main);
                if ( op )
                {
                    const std::string qname = op->position.module + ("." + op->name);
                    auto [it, inserted] = operators.emplace(qname, std::move(*op));
                    if ( !inserted )
                    {
                        auto& position = it->second.position;
                        report_error(op->position, "%s '%s' is already defined at [%d:%d]",
                                     it->second.graph ? "graph" : "operator", qname.c_str(),
                                     (int)position.line, (int)position.column);
                    }
                    else if ( op->graph && main )
                    {
                        graphs.push_back(op->name);
                    }
                }
                else
                {
                    report_error(op.error());
                    lexer.skip_until(Keyword::Public, Keyword::Operator, Keyword::Graph);
                }
            }
        }
        
        Result<void> parse_import( Lexer& lexer, std::set<std::string>& imports, std::map<std::string,sknd::Operator>& operators,
                                  std::vector<std::string>& graphs )
        {
            TRY_CALL(lexer.accept())
            
            bool more;
            do
            {
                auto position = lexer.position();
                TRY_DECL(name, parse_identifier(lexer))
                if ( !import_module(name, imports, operators, graphs) )
                {
                    return Error(position, "could not import module '%s'", name.c_str());
                }
                TRY_DECL(is_comma, lexer.accept_if(Operator::Comma))
                more = is_comma;
            }
            while ( more );
            
            TRY_CALL(lexer.accept(Operator::Semicolon))
            
            return Result<void>();
        }
        
        Result<sknd::Operator> parse_operator( Lexer& lexer, bool allow_graph )
        {
            auto position = lexer.position();
            
            TRY_DECL(publish, lexer.accept_if(Keyword::Public))
            
            bool graph = lexer.is_token(Keyword::Graph);
            if ( graph )
            {
                if ( allow_graph )
                {
                    TRY_CALL(lexer.accept())
                }
                else
                {
                    return Error(position, "graph not allowed in this context");
                }
                publish = true;
            }
            else
            {
                TRY_CALL(lexer.accept(Keyword::Operator))
            }
            TRY_DECL(name, parse_identifier(lexer))
            
            std::vector<TypeParam> dtypes;
            std::vector<Param> attribs;
            std::vector<Param> inputs;
            std::vector<Param> outputs;
            std::vector<Param> constants;
            std::vector<Param> variables;
            std::vector<Using> usings;
            std::vector<Assert> asserts;
            std::vector<Lowering> lowerings;
            std::vector<Component> components;
            std::vector<Component> updates;
            std::vector<Lowering> initializers;
            std::vector<Quantization> quantizations;
            
            TRY_CALL(lexer.accept(Operator::LeftBrace))
            
            while ( lexer.is_token(Category::Block) )
            {
                auto block = (Block)lexer.index();
                TRY_CALL(lexer.accept());
                
                switch ( block )
                {
                    case Block::Dtype:
                    {
                        if ( !dtypes.empty() )
                        {
                            return Error(lexer.position(), "block %s already defined", Lexer::str(block));
                        }
                        TRY_MOVE(dtypes, parse_block(lexer, graph, parse_type_param))
                        for ( auto& type : dtypes )
                        {
                            lexer.register_type_alias(type.name, type.base_type);
                        }
                        break;
                    }
                    case Block::Attrib:
                    {
                        if ( !attribs.empty() )
                        {
                            return Error(lexer.position(), "block %s already defined", Lexer::str(block));
                        }
                        TRY_MOVE(attribs, parse_block(lexer, graph, parse_attrib_param))
                        break;
                    }
                    case Block::Input:
                    {
                        if ( !inputs.empty() )
                        {
                            return Error(lexer.position(), "block %s already defined", Lexer::str(block));
                        }
                        TRY_MOVE(inputs, parse_block(lexer, graph, parse_input_param))
                        break;
                    }
                    case Block::Output:
                    {
                        if ( !outputs.empty() )
                        {
                            return Error(lexer.position(), "block %s already defined", Lexer::str(block));
                        }
                        TRY_MOVE(outputs, parse_block(lexer, graph, parse_output_param))
                        break;
                    }
                    case Block::Constant:
                    {
                        if ( !constants.empty() )
                        {
                            return Error(lexer.position(), "block %s already defined", Lexer::str(block));
                        }
                        TRY_MOVE(constants, parse_block(lexer, graph, parse_constant_param))
                        break;
                    }
                    case Block::Variable:
                    {
                        if ( !variables.empty() )
                        {
                            return Error(lexer.position(), "block %s already defined", Lexer::str(block));
                        }
                        TRY_MOVE(variables, parse_block(lexer, graph, parse_variable_param))
                        break;
                    }
                    case Block::Assert:
                    {
                        if ( !asserts.empty() )
                        {
                            return Error(lexer.position(), "block %s already defined", Lexer::str(block));
                        }
                        TRY_MOVE(asserts, parse_block(lexer, graph, parse_assert))
                        break;
                    }
                    case Block::Using:
                    {
                        if ( !usings.empty() )
                        {
                            return Error(lexer.position(), "block %s already defined", Lexer::str(block));
                        }
                        TRY_MOVE(usings, parse_block(lexer, graph, parse_using))
                        break;
                    }
                    case Block::Lower:
                    {
                        if ( !lowerings.empty() )
                        {
                            return Error(lexer.position(), "block %s already defined", Lexer::str(block));
                        }
                        TRY_MOVE(lowerings, parse_block(lexer, graph, parse_lowering))
                        break;
                    }
                    case Block::Compose:
                    {
                        if ( !components.empty() )
                        {
                            return Error(lexer.position(), "block %s already defined", Lexer::str(block));
                        }
                        TRY_MOVE(components, parse_block(lexer, graph, [&]( Lexer& lexer, bool graph ) { return parse_component(lexer, graph); }))
                        break;
                    }
                    case Block::Update:
                    {
                        if ( !updates.empty() )
                        {
                            return Error(lexer.position(), "block %s already defined", Lexer::str(block));
                        }
                        TRY_MOVE(updates, parse_block(lexer, graph, [&]( Lexer& lexer, bool graph ) { return parse_component(lexer, graph); }))
                        break;
                    }
                    case Block::Quantize:
                    {
                        if ( !quantizations.empty() )
                        {
                            return Error(lexer.position(), "block %s already defined", Lexer::str(block));
                        }
                        TRY_MOVE(quantizations, parse_block(lexer, graph, parse_quantization))
                        break;
                    }
                }
            }
            
            TRY_CALL(lexer.accept(Operator::RightBrace))
            
            lexer.unregister_type_aliases();
            
            return sknd::Operator
            {
                position,
                graph,
                publish,
                name,
                std::move(dtypes),
                std::move(attribs),
                std::move(inputs),
                std::move(outputs),
                std::move(constants),
                std::move(variables),
                std::move(asserts),
                std::move(usings),
                std::move(lowerings),
                std::move(components),
                std::move(updates),
                std::move(quantizations),
            };
        }
        
        static Result<Typename> parse_typename( Lexer& lexer )
        {
            if ( !is_typename(lexer) )
            {
                return Error(lexer.position(), "expected type identifier; got '%s'", lexer.token().c_str());
            }
            
            const Typename type = (Typename)(lexer.index() - (size_t)Keyword::Type);
            TRY_CALL(lexer.accept())
            return type;
        }
        
        static Result<std::pair<std::string,Typename>> parse_type_identifier( Lexer& lexer )
        {
            if ( lexer.is_token(Category::Identifier) )
            {
                auto type = lexer.aliased_type(lexer.token());
                if ( !type )
                {
                    return Error(lexer.position(), "expected type identifier; got '%s'", lexer.token().c_str());
                }
                auto name = lexer.token();
                TRY_CALL(lexer.accept())
                return std::make_pair(name, *type);
            }
            else
            {
                TRY_DECL(type, parse_typename(lexer))
                return std::make_pair(str(type), type);
            }
        }
        
        static Result<TypeParam> parse_type_param( Lexer& lexer, bool graph )
        {
            auto position = lexer.position();
            
            TRY_DECL(name, parse_identifier(lexer))
            TRY_CALL(lexer.accept(Operator::Colon))
            TRY_DECL(type, parse_typename(lexer))
            
            std::optional<Typename> default_type;
            TRY_DECL(has_default, lexer.accept_if(Operator::Assign))
            if ( has_default )
            {
                TRY_DECL(type, parse_typename(lexer))
                default_type = std::make_optional(type);
            }
            
            return TypeParam{ position, name, type, default_type };
        }
        
        static Result<Param> parse_attrib_param( Lexer& lexer, bool graph )
        {
            return parse_param(lexer, graph, Lexer::Block::Attrib);
        }
        
        static Result<Param> parse_input_param( Lexer& lexer, bool graph )
        {
            return parse_param(lexer, graph, Lexer::Block::Input);
        }
        
        static Result<Param> parse_output_param( Lexer& lexer, bool graph )
        {
            return parse_param(lexer, graph, Lexer::Block::Output);
        }
        
        static Result<Param> parse_constant_param( Lexer& lexer, bool graph )
        {
            return parse_param(lexer, graph, Lexer::Block::Constant);
        }
        
        static Result<Param> parse_variable_param( Lexer& lexer, bool graph )
        {
            return parse_param(lexer, graph, Lexer::Block::Variable);
        }
        
        static Result<Param> parse_param( Lexer& lexer, bool graph, const Lexer::Block block )
        {
            auto position = lexer.position();
            
            TRY_DECL(name, parse_identifier(lexer))
            TRY_CALL(lexer.accept(Operator::Colon))
            
            TRY_DECL(optional, lexer.accept_if(Keyword::Optional))
            
            bool aliased = lexer.is_token(Category::Identifier);
            TRY_DECL(type_id, parse_type_identifier(lexer))
            auto type_alias = aliased ? type_id.first : std::string();
            
            unsigned flags = 0;
            if ( block == Lexer::Block::Input )
            {
                flags |= Flags::IsDecl;
            }
            if ( block == Lexer::Block::Input || block == Lexer::Block::Output )
            {
                flags |= Flags::AllowTilde;
            }
            
            TRY_DECL(ranked, lexer.accept_if(Operator::Xor))
            TRY_DECL(rank, ranked ? parse_paren(lexer, parse_iden_expr) : Shared<Expr>())
            
            Shared<Shapedef> shape;
            if ( lexer.is_token(Operator::LeftBracket) )
            {
                TRY_MOVE(shape, parse_shape(lexer, name, flags))
            }
            
            TRY_DECL(packed, lexer.accept_if(Operator::Dots))
            
            Shared<Expr> repeats;
            Shared<Expr> repeats_bound;
            if ( packed )
            {
                if ( lexer.is_token(Operator::LeftParen) )
                {
                    TRY_CALL(lexer.accept(Operator::LeftParen))
                    
                    TRY_DECL(tilde, (flags & Flags::AllowTilde) ? lexer.accept_if(Lexer::Operator::Tilde) : false);
                    if ( !tilde )
                    {
                        TRY_MOVE(repeats, parse_expr(lexer))
                    }
                    
                    TRY_DECL(bar, lexer.accept_if(Lexer::Operator::Bar))
                    if ( bar )
                    {
                        TRY_MOVE(repeats_bound, parse_expr(lexer))
                    }
                    
                    if ( !repeats && (repeats_bound || block != Lexer::Block::Output) )
                    {
                        repeats = std::make_shared<IdenfitierExpr>(position, "|" + name + "|");
                    }
                    
                    TRY_CALL(lexer.accept(Operator::RightParen))
                }
                else if ( block != Lexer::Block::Output )
                {
                    repeats = std::make_shared<IdenfitierExpr>(position, "|" + name + "|");
                }
            }
            
            Shared<Expr> default_value;
            TRY_DECL(has_default, lexer.accept_if(Operator::Assign))
            if ( has_default )
            {
                TRY_MOVE(default_value, (packed && block != Lexer::Block::Attrib) ? parse_list_expr(lexer) : parse_expr(lexer))
            }
            
            std::vector<std::pair<std::string,Shared<Expr>>> default_bounds;
            if ( has_default && default_value->kind != Expr::List && (block == Lexer::Block::Constant || block == Lexer::Block::Variable) )
            {
                while ( lexer.is_token(Operator::Comma) )
                {
                    TRY_CALL(lexer.accept())
                    TRY_DECL(id, parse_identifier(lexer))
                    TRY_CALL(lexer.accept(Operator::Less))
                    TRY_DECL(expr, parse_expr(lexer))
                    
                    default_bounds.emplace_back(id, expr);
                }
            }
            
            bool tensor = block != Lexer::Block::Attrib;
            const Type type = make_type(type_id.second, optional, tensor, packed);
            return Param{ position, name, type, type_alias, rank, shape, repeats, repeats_bound, default_value, default_bounds };
        }
        
        static Result<Shared<Shapedef>> parse_shape( Lexer& lexer, const std::string& name = std::string(), unsigned flags = 0 )
        {
            auto position = lexer.position();
            
            TRY_CALL(lexer.accept(Operator::LeftBracket))
            
            std::vector<Shared<Expr>> extents;
            std::vector<Shared<Expr>> bounds;
            size_t spreads = 0;
            
            if ( !lexer.is_token(Operator::RightBracket) )
            {
                bool more;
                do
                {
                    TRY_DECL(spread, lexer.accept_if(Operator::Dots))
                    if ( spread )
                    {
                        spreads |= 1 << extents.size();
                    }
                    
                    auto pos = lexer.position();
                    
                    Shared<Expr> extent;
                    TRY_DECL(tilde, (flags & Flags::AllowTilde) ? lexer.accept_if(Operator::Tilde) : false)
                    if ( !tilde )
                    {
                        TRY_MOVE(extent, parse_expr(lexer))
                    }
                    
                    Shared<Expr> bound;
                    TRY_DECL(bar, lexer.accept_if(Operator::Bar))
                    if ( bar )
                    {
                        TRY_MOVE(bound, parse_expr(lexer))
                    }
                    
                    if ( !extent && (bound || (flags & Flags::IsDecl)) )
                    {
                        extent = (Shared<Expr>)std::make_shared<IdenfitierExpr>(pos, name + ".shape:" + std::to_string(extents.size()));
                    }
                    
                    TRY_DECL(expand, lexer.accept_if(Operator::Dots))
                    if ( expand )
                    {
                        TRY_DECL(count, lexer.is_token(Operator::LeftParen) || !extent ? parse_paren(lexer, parse_expr) : Shared<Expr>())
                        if ( !count && extent->kind == Expr::Identifier && (flags & Flags::IsDecl) )
                        {
                            count = std::make_shared<IdenfitierExpr>(pos, "|" + as_identifier(*extent).name + "|");
                        }
                        extent = (Shared<Expr>)std::make_shared<ExpandExpr>(pos, extent, count);
                    }
                    
                    extents.emplace_back(extent);
                    bounds.emplace_back(bound);
                    
                    TRY_DECL(is_comma, lexer.accept_if(Operator::Comma))
                    more = is_comma;
                }
                while ( more );
            }
            
            TRY_CALL(lexer.accept(Operator::RightBracket))
            
            return (Shared<Shapedef>)std::make_shared<Shapedef>(Shapedef{ position, std::move(extents), std::move(bounds), spreads });
        }
        
        static Result<Shared<Expr>> parse_expr( Lexer& lexer )
        {
            if ( lexer.is_token(Operator::Bar) )
            {
                return parse_bounded_expr(lexer);
            }
            else
            {
                return parse_expr_with_prim(lexer, nullptr);
            }
        }
        
        static Result<Shared<Expr>> parse_expr_with_prim( Lexer& lexer, const Shared<Expr> prim )
        {
            TRY_DECL(expr, parse_binary_expr(lexer, prim))
            
            if ( lexer.is_token(Operator::Question) )
            {
                TRY_MOVE(expr, parse_select_expr(lexer, expr))
            }
            if ( lexer.is_token(Operator::Questions) )
            {
                TRY_MOVE(expr, parse_coalesce_expr(lexer, expr))
            }
            return expr;
        }
        
        static Result<Shared<Expr>> parse_bounded_expr( Lexer& lexer )
        {
            auto position = lexer.position();
            
            TRY_CALL(lexer.accept(Operator::Bar))
            TRY_DECL(index, parse_expr(lexer))
            
            Shared<Expr> lower, upper;
            if ( lexer.is_token(Operator::Bounds) )
            {
                TRY_CALL(lexer.accept())
                TRY_MOVE(lower, parse_expr(lexer))
                TRY_CALL(lexer.accept(Operator::Colon))
                TRY_MOVE(upper, parse_expr(lexer))
            }
            
            TRY_CALL(lexer.accept(Operator::Bar))
            
            return (Shared<Expr>)std::make_shared<BoundedExpr>(position, index, lower, upper);
        }
        
        static Result<Shared<Expr>> parse_prim_expr( Lexer& lexer )
        {
            auto position = lexer.position();
            switch ( lexer.category() )
            {
                case Category::Number:
                {
                    if ( std::all_of(lexer.token().begin(), lexer.token().end(), isdigit) )
                    {
                        auto value = (int_t)std::atoi(lexer.token().c_str());
                        TRY_CALL(lexer.accept());
                        return (Shared<Expr>)std::make_shared<IntExpr>(position, value);
                    }
                    else
                    {
                        auto value = (real_t)std::atof(lexer.token().c_str());
                        TRY_CALL(lexer.accept());
                        return (Shared<Expr>)std::make_shared<RealExpr>(position, value);
                    }
                }
                case Category::String:
                {
                    return parse_str_expr(lexer);
                }
                case Category::Identifier:
                {
                    TRY_DECL(name, parse_identifier(lexer))
                    auto aliased_type = lexer.aliased_type(name);
                    if ( aliased_type )
                    {
                        TRY_CALL(lexer.accept(Operator::LeftParen))
                        TRY_DECL(arg, !lexer.is_token(Operator::RightParen) ? parse_expr(lexer) : Shared<Expr>())
                        TRY_CALL(lexer.accept(Operator::RightParen))
                        return (Shared<Expr>)std::make_shared<CastExpr>(position, name, *aliased_type, arg);
                    }
                    
                    if ( lexer.is_token(Operator::LeftParen) )
                    {
                        TRY_CALL(lexer.accept(Operator::LeftParen))
                        TRY_DECL(arg, parse_expr(lexer))
                        TRY_CALL(lexer.accept(Operator::RightParen))
                        return (Shared<Expr>)std::make_shared<BuiltinExpr>(position, name, arg);
                    }
                    
                    Shared<Expr> expr;
                    if ( lexer.is_token(Operator::Dot) )
                    {
                        position = lexer.position();
                        TRY_CALL(lexer.accept())
                        
                        const std::string member = lexer.token();
                        if ( member != "shape" && member != "rank" && member != "size" )
                        {
                            return Error(lexer.position(), "expected identifier 'shape' or 'rank' or 'size'");
                        }
                        TRY_CALL(lexer.accept(Category::Identifier))
                        expr = (Shared<Expr>)std::make_shared<IdenfitierExpr>(position, name + "." + member);
                    }
                    else
                    {
                        expr = (Shared<Expr>)std::make_shared<IdenfitierExpr>(position, name);
                    }
                    
                    while ( lexer.is_token(Operator::LeftBracket) )
                    {
                        TRY_MOVE(expr, parse_index_expr(lexer, expr))
                    }
                    return expr;
                }
                case Category::Keyword:
                {
                    if ( lexer.is_token(Keyword::Inf) )
                    {
                        TRY_CALL(lexer.accept());
                        return (Shared<Expr>)std::make_shared<RealExpr>(position, std::numeric_limits<real_t>::infinity());
                    }
                    else if ( lexer.is_token(Keyword::Pi) )
                    {
                        TRY_CALL(lexer.accept());
                        return (Shared<Expr>)std::make_shared<RealExpr>(position, M_PI);
                    }
                    else if ( lexer.is_token(Keyword::True) )
                    {
                        TRY_CALL(lexer.accept());
                        return (Shared<Expr>)std::make_shared<BoolExpr>(position, true);
                    }
                    else if ( lexer.is_token(Keyword::False) )
                    {
                        TRY_CALL(lexer.accept());
                        return (Shared<Expr>)std::make_shared<BoolExpr>(position, false);
                    }
                    else if ( is_typename(lexer) )
                    {
                        TRY_DECL(type, parse_typename(lexer))
                        TRY_CALL(lexer.accept(Operator::LeftParen))
                        TRY_DECL(arg, !lexer.is_token(Operator::RightParen) ? parse_expr(lexer) : Shared<Expr>())
                        TRY_CALL(lexer.accept(Operator::RightParen))
                        Shared<Expr> expr = std::make_shared<CastExpr>(position, str(type), type, arg);
                        while ( lexer.is_token(Operator::LeftBracket) )
                        {
                            TRY_MOVE(expr, parse_index_expr(lexer, expr))
                        }
                        return expr;
                    }
                    break;
                }
                case Category::Operator:
                {
                    const auto op = (Operator)lexer.index();
                    switch ( op )
                    {
                        case Operator::LeftBracket:
                        {
                            TRY_DECL(expr, parse_list_expr(lexer))
                            while ( lexer.is_token(Operator::LeftBracket) )
                            {
                                TRY_MOVE(expr, parse_index_expr(lexer, expr))
                            }
                            return expr;
                        }
                        case Operator::LeftParen:
                        {
                            TRY_DECL(expr, parse_paren(lexer, parse_expr))
                            while ( lexer.is_token(Operator::LeftBracket) )
                            {
                                TRY_MOVE(expr, parse_index_expr(lexer, expr))
                            }
                            return expr;
                        }
                        case Operator::Plus:
                        case Operator::Minus:
                        case Operator::Not:
                        {
                            TRY_CALL(lexer.accept())
                            TRY_DECL(expr, parse_prim_expr(lexer))
                            while ( lexer.is_token(Operator::Power) )
                            {
                                auto position = lexer.position();
                                TRY_CALL(lexer.accept());
                                TRY_DECL(exponent, parse_prim_expr(lexer))
                                expr = std::make_shared<BinaryExpr>(position, expr, exponent, Operator::Power);
                            }
                            if ( op == Operator::Minus && expr->kind == Expr::Literal )
                            {
                                if ( as_literal(*expr).type == make_type(Typename::Int) )
                                {
                                    return (Shared<Expr>)std::make_shared<IntExpr>(position, -as_int(*expr).value);
                                }
                                else if ( as_literal(*expr).type == make_type(Typename::Real) )
                                {
                                    return (Shared<Expr>)std::make_shared<RealExpr>(position, -as_real(*expr).value);
                                }
                            }
                            return (Shared<Expr>)std::make_shared<UnaryExpr>(position, expr, op);
                        }
                        case Operator::Question:
                        {
                            TRY_CALL(lexer.accept())
                            TRY_DECL(expr, parse_iden_expr(lexer))
                            return (Shared<Expr>)std::make_shared<UnaryExpr>(position, expr, op);
                        }
                        default:
                        {
                            break;
                        }
                    }
                    break;
                }
                default:
                {
                    break;
                }
            }
            return Error(lexer.position(), "unexpected token: '%s'", lexer.token().c_str());
        }
        
        static Result<Shared<Expr>> parse_iden_expr( Lexer& lexer )
        {
            auto position = lexer.position();
            
            TRY_DECL(name, parse_identifier(lexer))
            return (Shared<Expr>)std::make_shared<IdenfitierExpr>(position, name);
        }
        
        static Result<Shared<Expr>> parse_str_expr( Lexer& lexer )
        {
            auto pos = lexer.position();
            
            TRY_DECL(str, parse_string(lexer))
            
            std::string fmt;
            std::map<size_t, Shared<Expr>> subs;
            for ( size_t i = 0; i < str.length(); ++i )
            {
                if ( str[i] == '{' && (i == 0 || str[i-1] != '\\') )
                {
                    const Position position = { pos.module, pos.line, pos.column + (unsigned)i + 2 };
                    
                    auto j = str.find('}', i+1);
                    if ( j == std::string::npos )
                    {
                        return Error(position, "expected '}' character to terminate string formatting");
                    }
                    std::stringstream ss(str.substr(i + 1, j - (i + 1)));
                    Lexer lexer(ss, position);
                    
                    TRY_DECL(sub, parse_expr(lexer))
                    subs.emplace(fmt.length(), sub);
                    
                    i = j;
                }
                else
                {
                    fmt += str[i];
                }
            }
            if ( subs.empty() )
            {
                return (Shared<Expr>)std::make_shared<StrExpr>(pos, str);
            }
            else
            {
                return (Shared<Expr>)std::make_shared<FormatExpr>(pos, fmt, std::move(subs));
            }
        }
        
        static Result<Shared<Expr>> parse_binary_expr( Lexer& lexer, const Shared<Expr> prim = nullptr, const int prec = 1 )
        {
            TRY_DECL(expr, prim ? prim : parse_prim_expr(lexer))
            
            while ( lexer.is_token(Category::Operator) || lexer.is_oneof(Keyword::Is, Keyword::In) )
            {
                const bool identity = lexer.is_token(Keyword::Is);
                const bool contains = lexer.is_token(Keyword::In);
                
                const Operator op = identity || contains ? Operator::Equal : (Operator)lexer.index();
                const int curr_prec = precedence(op);
                if ( std::abs(curr_prec) < prec )
                {
                    break;
                }
                const int next_prec = curr_prec > 0 ? curr_prec + 1 : -curr_prec;
                
                auto position = lexer.position();
                TRY_CALL(lexer.accept())
                
                if ( !identity && !contains && lexer.is_oneof(Operator::Dots, Operator::Ellipsis) )
                {
                    bool cumulative = lexer.is_token(Operator::Ellipsis);
                    TRY_CALL(lexer.accept())
                    
                    expr = std::make_shared<FoldExpr>(position, expr, op, cumulative);
                    
                    TRY_DECL(has_rhs, lexer.accept_if(op))
                    if ( !has_rhs )
                    {
                        break;
                    }
                }
                
                TRY_DECL(rhs, parse_binary_expr(lexer, nullptr, next_prec))
                if ( identity )
                {
                    expr = std::make_shared<IdentityExpr>(position, expr, rhs);
                }
                else if ( contains )
                {
                    expr = std::make_shared<ContainExpr>(position, expr, rhs);
                }
                else
                {
                    expr = std::make_shared<BinaryExpr>(position, expr, rhs, op);
                }
            }
            return expr;
        }
        
        static Result<Shared<Expr>> parse_list_expr( Lexer& lexer, unsigned flags = 0 )
        {
            auto position = lexer.position();
            
            auto parse_item_func = [&]( Lexer& lexer ){ return parse_list_item(lexer, flags); };
            
            TRY_CALL(lexer.accept(Operator::LeftBracket))
            TRY_DECL(items, !lexer.is_token(Operator::RightBracket) ? parse_items(lexer, parse_item_func) : std::vector<Shared<Expr>>())
            TRY_CALL(lexer.accept(Operator::RightBracket))
            
            return (Shared<Expr>)std::make_shared<ListExpr>(position, std::move(items));
        }
        
        static Result<Shared<Expr>> parse_list_item( Lexer& lexer, unsigned flags )
        {
            auto position = lexer.position();
            
            Shared<Expr> item;
            bool parenthesized = false;
            
            TRY_MOVE(parenthesized, lexer.accept_if(Operator::LeftParen))
            TRY_MOVE(item, (flags & Flags::IsUsing) ? parse_iden_expr(lexer) : parse_expr(lexer))
            
            if ( parenthesized )
            {
                bool zipped = lexer.is_token(Operator::Comma);
                if ( zipped )
                {
                    TRY_MOVE(item, parse_zipped_expr(lexer, item))
                }
                TRY_CALL(lexer.accept(Operator::RightParen))
                if ( !zipped && !(flags & Flags::IsUsing) )
                {
                    while ( lexer.is_token(Operator::LeftBracket) )
                    {
                        TRY_MOVE(item, parse_index_expr(lexer, item))
                    }
                    TRY_MOVE(item, parse_expr_with_prim(lexer, item))
                }
            }
            
            TRY_DECL(expand, lexer.accept_if(Operator::Dots))
            if ( expand )
            {
                TRY_DECL(count, lexer.is_token(Operator::LeftParen) ? parse_paren(lexer, parse_expr) : Shared<Expr>())
                if ( !count && !(flags & Flags::IsUsing) && item->kind == Expr::Identifier && (flags & Flags::IsDecl) )
                {
                    auto& iden = as_identifier(*item);
                    count = std::make_shared<IdenfitierExpr>(position, "|" + iden.name + "|");
                }
                return (Shared<Expr>)std::make_shared<ExpandExpr>(position, item, count);
            }
            
            if ( !(flags & Flags::IsDecl) && lexer.is_token(Operator::Colon) )
            {
                TRY_MOVE(item, parse_range_expr(lexer, item, (flags & Flags::IsIndex)))
            }
            
            return item;
        }
        
        static Result<Shared<Expr>> parse_zipped_expr( Lexer& lexer, const Shared<Expr> first )
        {
            std::vector<Shared<Expr>> items = { first };
            while ( lexer.is_token(Operator::Comma) )
            {
                TRY_CALL(lexer.accept())
                TRY_DECL(item, parse_expr(lexer))
                items.push_back(item);
            }
            return (Shared<Expr>)std::make_shared<ZipExpr>(first->position, std::move(items));
        }
        
        static Result<Shared<Expr>> parse_index_expr( Lexer& lexer, const Shared<Expr> array )
        {
            auto position = lexer.position();
            
            TRY_CALL(lexer.accept(Operator::LeftBracket))
            
            if ( lexer.is_token(Operator::RightBracket) )
            {
                TRY_CALL(lexer.accept())
                return (Shared<Expr>)std::make_shared<AccessExpr>(position, array, std::vector<Shared<Expr>>());
            }
            else if ( lexer.is_token(Operator::Colon) )
            {
                TRY_DECL(range, parse_range_expr(lexer, nullptr, true))
                TRY_CALL(lexer.accept(Operator::RightBracket))
                return (Shared<Expr>)std::make_shared<IndexExpr>(position, array, range);
            }
            
            TRY_DECL(index, parse_list_item(lexer, Flags::IsIndex))
            if ( index->kind == Expr::Expand || lexer.is_token(Operator::Comma) )
            {
                std::vector<Shared<Expr>> indices = { index };
                while ( lexer.is_token(Operator::Comma) )
                {
                    TRY_CALL(lexer.accept())
                    if ( lexer.is_token(Operator::RightBracket) )
                    {
                        break;
                    }
                    TRY_DECL(item, parse_list_item(lexer, Flags::IsIndex))
                    indices.emplace_back(std::move(item));
                }
                TRY_CALL(lexer.accept(Operator::RightBracket))
                return (Shared<Expr>)std::make_shared<AccessExpr>(position, array, std::move(indices));
            }
            else if ( lexer.is_token(Operator::Colon) )
            {
                TRY_DECL(range, parse_range_expr(lexer, index, true))
                TRY_CALL(lexer.accept(Operator::RightBracket))
                return (Shared<Expr>)std::make_shared<IndexExpr>(position, array, range);
            }
            else
            {
                TRY_CALL(lexer.accept(Operator::RightBracket))
                
                if ( lexer.is_token(Operator::LeftArrow) )
                {
                    TRY_CALL(lexer.accept())
                    TRY_DECL(substitution, parse_expr(lexer))
                    return (Shared<Expr>)std::make_shared<SubstituteExpr>(position, array, index, substitution);
                }
                
                return (Shared<Expr>)std::make_shared<IndexExpr>(position, array, index);
            }
        }
        
        static Result<Shared<Expr>> parse_range_expr( Lexer& lexer, const Shared<Expr> first, const bool indexing )
        {
            auto position = first ? first->position : lexer.position();
            
            TRY_CALL(lexer.accept(Operator::Colon))
            TRY_DECL(last, (indexing && lexer.is_token(Operator::RightBracket)) || lexer.is_token(Operator::Colon) ? Shared<Expr>() : parse_expr(lexer))
            TRY_DECL(strided, lexer.accept_if(Operator::Colon))
            TRY_DECL(stride, strided ? parse_expr(lexer) : Shared<Expr>())
            return (Shared<Expr>)std::make_shared<RangeExpr>(position, first, last, stride);
        }
        
        static Result<Shared<Expr>> parse_select_expr( Lexer& lexer, const Shared<Expr> cond )
        {
            const auto position = lexer.position();
            
            TRY_CALL(lexer.accept(Operator::Question))
            TRY_DECL(left, parse_expr(lexer))
            TRY_DECL(has_rhs, lexer.accept_if(Operator::Colon));
            TRY_DECL(right, has_rhs ? parse_expr(lexer) : Shared<Expr>())
            
            return (Shared<Expr>)std::make_shared<SelectExpr>(position, cond, left, right);
        }
        
        static Result<Shared<Expr>> parse_coalesce_expr( Lexer& lexer, const Shared<Expr> condition )
        {
            const auto position = lexer.position();
            
            TRY_CALL(lexer.accept())
            TRY_DECL(alternate, parse_expr(lexer))
            
            return (Shared<Expr>)std::make_shared<CoalesceExpr>(position, condition, alternate);
        }
        
        static Result<Using> parse_using( Lexer& lexer, bool graph )
        {
            auto position = lexer.position();
            
            TRY_DECL(identifier, lexer.is_token(Operator::LeftBracket) ? parse_list_expr(lexer, Flags::IsDecl | Flags::IsUsing) : parse_iden_expr(lexer))
            Shared<Expr> rank;
            if ( identifier->kind == Expr::Identifier && lexer.is_token(Operator::Dots) )
            {
                TRY_CALL(lexer.accept())
                TRY_MOVE(rank, parse_paren(lexer, parse_expr))
            }
            TRY_CALL(lexer.accept(Operator::Assign))
            TRY_DECL(expr, parse_expr(lexer))
            
            return Using{ position, identifier, expr, rank };
        }
        
        static Result<Assert> parse_assert( Lexer& lexer, bool graph )
        {
            auto position = lexer.position();
            
            TRY_DECL(expr, parse_expr(lexer))
            TRY_DECL(has_message, lexer.accept_if(Operator::Colon))
            TRY_DECL(message, has_message ? parse_str_expr(lexer) : Shared<Expr>());
            
            Pairs<std::string,Shared<Expr>> prints;
            if ( has_message )
            {
                while ( lexer.is_token(Operator::Comma) )
                {
                    TRY_CALL(lexer.accept())
                    
                    std::string label;
                    if ( lexer.is_token(Category::String) )
                    {
                        TRY_MOVE(label, parse_string(lexer))
                        TRY_CALL(lexer.accept(Operator::Colon))
                    }
                    TRY_DECL(expr, parse_expr(lexer))
                    prints.push_back(std::make_pair(label,expr));
                }
            }
            
            return Assert{ position, expr, message, prints };
        }
        
        static Result<Lowering> parse_lowering( Lexer& lexer, bool graph )
        {
            auto position = lexer.position();
            
            std::string unroll_index;
            Shared<Expr> unroll_count;
            TRY_DECL(unroll, lexer.accept_if(Lexer::Keyword::Unroll))
            if ( unroll )
            {
                TRY_CALL(lexer.accept(Operator::Dots))
                TRY_CALL(lexer.accept(Operator::LeftParen))
                TRY_MOVE(unroll_index, parse_identifier(lexer))
                TRY_CALL(lexer.accept(Operator::RightArrow))
                TRY_MOVE(unroll_count, parse_expr(lexer))
                TRY_CALL(lexer.accept(Operator::RightParen))
            }
            
            std::vector<std::pair<std::string,Shared<Expr>>> locals;
            
            if ( lexer.is_token(Keyword::With) )
            {
                do
                {
                    TRY_CALL(lexer.accept())
                    TRY_DECL(iden, parse_identifier(lexer))
                    TRY_CALL(lexer.accept(Operator::Assign))
                    TRY_DECL(expr, parse_expr(lexer))
                    locals.emplace_back(iden, expr);
                }
                while ( lexer.is_token(Operator::Comma) );
                TRY_CALL(lexer.accept(Operator::Colon))
            }
            
            TRY_DECL(left, parse_iden_expr(lexer))
            TRY_MOVE(left, parse_index_expr(lexer, left))
            if ( lexer.is_token(Operator::LeftBracket) )
            {
                TRY_MOVE(left, parse_index_expr(lexer, left))
            }
            
            auto op = (Operator)lexer.index();
            TRY_CALL(lexer.accept(Category::Operator))
            
            TRY_DECL(right, parse_expr(lexer))
            
            std::vector<std::pair<std::string,Shared<Expr>>> bounds;
            while ( lexer.is_token(Operator::Comma) )
            {
                TRY_CALL(lexer.accept())
                TRY_DECL(id, parse_identifier(lexer))
                TRY_CALL(lexer.accept(Operator::Less))
                TRY_DECL(expr, parse_expr(lexer))
                
                bounds.emplace_back(id, expr);
            }
            
            Shared<Expr> condition;
            if ( lexer.is_token(Operator::Bar) )
            {
                TRY_CALL(lexer.accept())
                TRY_MOVE(condition, parse_expr(lexer))
            }
            
            return Lowering{ position, left, right, op, locals, bounds, condition, unroll_index, unroll_count };
        }
        
        Result<Invocation> parse_invocation( Lexer& lexer, const std::string& label, const std::string& name, const Position& position )
        {
            bool qualified = name.find('.') != std::string::npos;
            std::string target = qualified ? name : lexer.position().module + ("." + name);
            
            TRY_DECL(types, parse_generic_types(lexer))
            TRY_DECL(attribs, parse_attribs(lexer))
            
            TRY_CALL(lexer.accept(Operator::LeftParen))
            TRY_DECL(args, !lexer.is_token(Operator::RightParen) ? parse_args(lexer) : std::vector<Shared<Expr>>())
            TRY_CALL(lexer.accept(Operator::RightParen))
            
            return Invocation{position, label, target, std::move(types), std::move(attribs), std::move(args)};
        }
        
        Result<Region> parse_region( Lexer& lexer, const std::string& label = {} )
        {
            auto position = lexer.position();
            
            TRY_CALL(lexer.accept(Operator::LeftBrace))
            
            std::vector<Component> components;
            while ( !lexer.is_oneof(Operator::RightBrace, Keyword::Yield) )
            {
                auto component = parse_component(lexer, false);
                if ( component )
                {
                    components.emplace_back(std::move(*component));
                }
                else
                {
                    report_error(component.error());
                    
                    lexer.skip_until(Operator::Semicolon, Operator::RightBrace, Keyword::Yield);
                }
                
                if ( !lexer.accept(Operator::Semicolon) )
                {
                    report_error(lexer.position(), "missing semicolon after expression");
                }
            }
            
            TRY_CALL(lexer.accept(Keyword::Yield))
            TRY_DECL(yields, parse_items(lexer, []( Lexer& lexer ) { return parse_expr(lexer); } ))
            TRY_CALL(lexer.accept(Operator::Semicolon))
            
            TRY_CALL(lexer.accept(Operator::RightBrace))
            
            return Region{ position, label, std::move(components), std::move(yields) };
        }
        
        Result<Callable> parse_callable( Lexer& lexer )
        {
            auto position = lexer.position();
            
            if ( lexer.is_token(Operator::LeftBrace) )
            {
                TRY_DECL(region, parse_region(lexer))
                return Callable(std::move(region));
            }
            else if ( !lexer.is_token(Category::Identifier) )
            {
                TRY_DECL(expr, parse_expr(lexer))
                return Callable(Region{ position, {}, {}, { expr } });
            }
            else
            {
                TRY_DECL(iden, parse_identifier(lexer, true))
                bool qualified = iden.find('.') != std::string::npos;
                if ( !qualified && lexer.is_token(Operator::Colon) )
                {
                    TRY_CALL(lexer.accept())
                    if ( lexer.is_token(Operator::LeftBrace) )
                    {
                        TRY_DECL(region, parse_region(lexer, iden))
                        return Callable(std::move(region));
                    }
                    else
                    {
                        TRY_DECL(name, parse_identifier(lexer, true))
                        TRY_DECL(invocation, parse_invocation(lexer, iden, name, position))
                        return Callable(std::move(invocation));
                    }
                }
                else if ( lexer.is_oneof(Operator::Less, Operator::LeftBrace, Operator::LeftParen) )
                {
                    TRY_DECL(invocation, parse_invocation(lexer, {}, iden, position))
                    return Callable(std::move(invocation));
                }
                else
                {
                    Shared<Expr> prim = std::make_shared<IdenfitierExpr>(position, iden);
                    while ( lexer.is_token(Operator::LeftBracket) )
                    {
                        TRY_DECL(index, parse_index_expr(lexer, prim))
                        prim = index;
                    }
                    TRY_DECL(expr, parse_expr_with_prim(lexer, prim))
                    return Callable(Region{ position, {}, {}, { expr } });
                }
            }
        }
        
        static Result<Quantization> parse_quantization( Lexer& lexer, bool graph )
        {
            auto position = lexer.position();
            
            TRY_DECL(tensor, parse_identifier(lexer, true))
            TRY_CALL(lexer.accept(Operator::Colon))
            
            auto pos = lexer.position();
            
            TRY_DECL(target, parse_identifier(lexer, true))
            const bool qualified = target.find('.') != std::string::npos;
            if ( !qualified )
            {
                target = lexer.position().module + ("." + target);
            }
            
            TRY_DECL(types, parse_generic_types(lexer))
            TRY_DECL(attribs, parse_attribs(lexer))
            
            return Quantization{position, tensor, Invocation{ pos, {}, target, types, attribs } };
        }
        
        static Result<Pairs<std::string,Shared<Expr>>> parse_loop_scans( Lexer& lexer, const Keyword keyword, const Operator separator )
        {
            Pairs<std::string,Shared<Expr>> decls;
            
            if ( lexer.is_token(keyword) )
            {
                TRY_CALL(lexer.accept())
                
                bool more;
                do
                {
                    auto position = lexer.position();
                    TRY_DECL(iden, parse_identifier(lexer))
                    TRY_CALL(lexer.accept(separator))
                    TRY_DECL(expr, parse_expr(lexer))
                    TRY_MOVE(more, lexer.accept_if(Operator::Comma))
                    decls.emplace_back(iden, expr);
                }
                while ( more );
            }
            
            return decls;
        }
        
        static Result<Pairs<Typed,Shared<Expr>>> parse_loop_carries( Lexer& lexer, const Keyword keyword, const Operator separator )
        {
            Pairs<Typed,Shared<Expr>> decls;
            
            if ( lexer.is_token(keyword) )
            {
                TRY_CALL(lexer.accept())
                
                bool more;
                do
                {
                    auto position = lexer.position();
                    TRY_DECL(iden, parse_identifier(lexer))
                    
                    Type type;
                    std::string type_alias;
                    Shared<Shapedef> shape;
                    if ( lexer.is_token(Operator::Colon) )
                    {
                        TRY_CALL(lexer.accept())
                        
                        bool aliased = lexer.is_token(Category::Identifier);
                        TRY_DECL(type_id, parse_type_identifier(lexer))
                        type_alias = aliased ? type_id.first : std::string();
                        type = make_type(type_id.second, false, true, false);
                        TRY_MOVE(shape, parse_shape(lexer))
                    }
                    TRY_CALL(lexer.accept(separator))
                    TRY_DECL(expr, parse_expr(lexer))
                    TRY_MOVE(more, lexer.accept_if(Operator::Comma))
                    
                    const Typed typed = { position, iden, type, type_alias, nullptr, shape, nullptr, nullptr };
                    decls.emplace_back(typed, expr);
                }
                while ( more );
            }
            
            return decls;
        }
        
        Result<Component> parse_component( Lexer& lexer, bool graph )
        {
            auto position = lexer.position();
            
            TRY_DECL(results, parse_results(lexer))
            TRY_CALL(lexer.accept(Operator::Assign))
            
            if ( lexer.is_token(Keyword::If) )
            {
                std::vector<Branch> branches;
                
                do
                {
                    TRY_CALL(lexer.accept())
                    TRY_DECL(condition, parse_callable(lexer))
                    TRY_CALL(lexer.accept(Keyword::Then))
                    TRY_DECL(consequent, parse_callable(lexer))
                    
                    branches.push_back(Branch{ condition, consequent });
                }
                while ( lexer.is_token(Keyword::Elif) );
                
                TRY_CALL(lexer.accept(Keyword::Else))
                TRY_DECL(alternative, parse_callable(lexer))
                
                return Component{ position, std::move(results), std::move(alternative), std::move(branches), nullptr };
            }
            else if ( lexer.is_oneof(Keyword::With, Keyword::For, Keyword::While, Keyword::Do) )
            {
                TRY_DECL(carries, parse_loop_carries(lexer, Keyword::With, Operator::Assign))
                TRY_DECL(scans, parse_loop_scans(lexer, Keyword::For, Operator::Colon))
                
                Shared<Callable> condition;
                if ( lexer.is_token(Keyword::While) )
                {
                    TRY_CALL(lexer.accept())
                    TRY_DECL(callable, parse_callable(lexer))
                    condition = std::make_shared<Callable>(std::move(callable));
                }
                
                TRY_DECL(unroll, lexer.accept_if(Keyword::Unroll))
                if ( !unroll )
                {
                    TRY_CALL(lexer.accept(Keyword::Do))
                }
                TRY_DECL(iter, count, parse_iter_count(lexer, true))
                
                TRY_DECL(body, parse_callable(lexer))
                
                bool pretest = condition != nullptr;
                if ( !pretest && lexer.is_token(Keyword::While) )
                {
                    TRY_CALL(lexer.accept())
                    TRY_DECL(callable, parse_callable(lexer))
                    condition = std::make_shared<Callable>(std::move(callable));
                }
                
                auto loop = std::make_shared<Loop>(Loop{ carries, scans, condition, count, iter, pretest, unroll });
                return Component{ position, std::move(results), std::move(body), {}, loop };
            }
            else
            {
                TRY_DECL(callable, parse_callable(lexer))
                return Component{ position, std::move(results), std::move(callable), {}, nullptr };
            }
        }
        
        static Result<std::pair<Shared<IdenfitierExpr>,Shared<Expr>>> parse_iter_count( Lexer& lexer, bool allow_omit_count )
        {
            Shared<IdenfitierExpr> index;
            Shared<Expr> count;
            
            TRY_DECL(bounded, lexer.accept_if(Operator::Dots))
            if ( bounded )
            {
                TRY_CALL(lexer.accept(Operator::LeftParen))
                TRY_DECL(expr, parse_expr(lexer))
                if ( expr->kind == Expr::Identifier && lexer.is_token(Operator::RightArrow) )
                {
                    index = std::dynamic_pointer_cast<const IdenfitierExpr>(expr);
                    TRY_CALL(lexer.accept())
                    if ( !allow_omit_count || !lexer.is_token(Operator::RightParen) )
                    {
                        TRY_MOVE(count, parse_expr(lexer))
                    }
                }
                else
                {
                    count = expr;
                }
                TRY_CALL(lexer.accept(Operator::RightParen))
            }
            return std::make_pair(index, count);
        }
        
        static Result<Packable<Typed>> parse_result( Lexer& lexer )
        {
            if ( lexer.is_token(Operator::LeftBracket) )
            {
                auto position = lexer.position();
                
                auto parse_item_func = [&]( Lexer& lexer) -> Result<Typed>
                {
                    return parse_result_item(lexer, false);
                };
                
                TRY_CALL(lexer.accept(Operator::LeftBracket))
                TRY_DECL(items, !lexer.is_token(Operator::RightBracket) ? parse_items(lexer, parse_item_func) : std::vector<Typed>())
                TRY_CALL(lexer.accept(Operator::RightBracket))
                
                return Packable<Typed>(items.data(), items.size());
            }
            else
            {
                TRY_DECL(item, parse_result_item(lexer, true))
                return Packable<Typed>(item);
            }
        }
        
        static Result<Typed> parse_result_item( Lexer& lexer, bool force_count = false )
        {
            auto position = lexer.position();
            
            std::string iden;
            if ( lexer.is_token(Operator::Tilde) )
            {
                TRY_CALL(lexer.accept())
            }
            else
            {
                TRY_MOVE(iden, parse_identifier(lexer))
            }
            
            Typename type_name = Typename::Type;
            std::string type_alias;
            Shared<Shapedef> shape;
            
            TRY_DECL(has_type, lexer.accept_if(Operator::Colon))
            if ( has_type )
            {
                bool aliased = lexer.is_token(Category::Identifier);
                TRY_DECL(type_id, parse_type_identifier(lexer))
                type_name = type_id.second;
                if ( aliased )
                {
                    type_alias = type_id.first;
                }
                TRY_MOVE(shape, parse_shape(lexer, iden, Flags::IsDecl))
            }
            
            TRY_DECL(packed, lexer.accept_if(Operator::Dots))
            
            Shared<Expr> count;
            if ( packed )
            {
                if ( lexer.is_token(Operator::LeftParen) || force_count )
                {
                    TRY_CALL(lexer.accept(Operator::LeftParen))
                    TRY_MOVE(count, parse_expr(lexer))
                    TRY_CALL(lexer.accept(Operator::RightParen))
                }
            }
            return Typed{ position, iden, make_type(type_name, false, true, packed), type_alias, nullptr, shape, count, nullptr };
        }
        
        static Result<std::map<std::string,Shared<Expr>>> parse_attribs( Lexer& lexer )
        {
            std::map<std::string,Shared<Expr>> items;
            
            if ( lexer.is_token(Operator::LeftBrace) )
            {
                TRY_CALL(lexer.accept(Operator::LeftBrace))
                
                while ( !lexer.is_token(Operator::RightBrace) )
                {
                    TRY_DECL(key, parse_identifier(lexer))
                    TRY_CALL(lexer.accept(Operator::Assign))
                    TRY_DECL(value, parse_expr(lexer))
                    
                    items.emplace(key, value);
                    
                    TRY_DECL(has_comma, lexer.accept_if(Operator::Comma))
                    if ( !has_comma )
                    {
                        break;
                    }
                }
                
                TRY_CALL(lexer.accept(Operator::RightBrace))
            }
            return items;
        }
        
        static Result<std::vector<std::pair<std::string,Typename>>> parse_generic_types( Lexer& lexer )
        {
            if ( !lexer.is_token(Operator::Less) )
            {
                return std::vector<std::pair<std::string,Typename>>();
            }
            TRY_CALL(lexer.accept(Operator::Less))
            TRY_DECL(types, parse_items(lexer, parse_type_identifier))
            TRY_CALL(lexer.accept(Operator::Greater))
            return types;
        }
        
    private:
        
        template<typename F>
        auto parse_block( Lexer& lexer, bool graph, F func ) ->
            Result<std::vector<typename decltype(func(lexer,graph))::value_type>>
        {
            std::vector<typename decltype(func(lexer,graph))::value_type> items;
            
            TRY_CALL(lexer.accept(Operator::LeftBrace))
            
            while ( !lexer.is_token(Operator::RightBrace) )
            {
                auto item = func(lexer, graph);
                if ( item )
                {
                    items.emplace_back(std::move(*item));
                }
                else
                {
                    report_error(item.error());
                    
                    lexer.skip_until(Operator::Semicolon, Category::Block, Keyword::Operator, Keyword::Graph);
                    if ( !lexer.is_token(Operator::Semicolon) )
                    {
                        return items;
                    }
                }
                
                if ( !lexer.accept(Operator::Semicolon) )
                {
                    report_error(lexer.position(), "missing semicolon after expression");
                }
            }
            
            TRY_CALL(lexer.accept(Operator::RightBrace))
            
            return items;
        }
        
        template<typename F>
        static auto parse_items( Lexer& lexer, F func ) ->
            Result<std::vector<typename decltype(func(lexer))::value_type>>
        {
            std::vector<typename decltype(func(lexer))::value_type> items;
            
            bool more;
            do
            {
                TRY_DECL(item, func(lexer))
                items.emplace_back(std::move(item));
                TRY_DECL(is_comma, lexer.accept_if(Operator::Comma))
                more = is_comma;
            }
            while ( more );
            
            return items;
        }
        
        static Result<std::vector<Shared<Expr>>> parse_args( Lexer& lexer )
        {
            std::vector<Shared<Expr>> items;
            
            bool more;
            do
            {
                TRY_DECL(placeholder, lexer.accept_if(Operator::Tilde))
                if ( placeholder )
                {
                    items.push_back(nullptr);
                }
                else
                {
                    TRY_DECL(item, parse_expr(lexer))
                    items.emplace_back(std::move(item));
                }
                TRY_DECL(is_comma, lexer.accept_if(Operator::Comma))
                more = is_comma;
            }
            while ( more );
            
            return items;
        }
        
        static Result<std::vector<Packable<Typed>>> parse_results( Lexer& lexer )
        {
            std::vector<Packable<Typed>> items;
            
            bool more;
            do
            {
                TRY_DECL(item, parse_result(lexer))
                items.emplace_back(std::move(item));
                TRY_DECL(is_comma, lexer.accept_if(Operator::Comma))
                more = is_comma;
            }
            while ( more );
            
            return items;
        }
        
        template<typename F>
        static auto parse_paren( Lexer& lexer, F func ) ->
            Result<typename decltype(func(lexer))::value_type>
        {
            TRY_CALL(lexer.accept(Operator::LeftParen))
            TRY_DECL(expr, func(lexer))
            TRY_CALL(lexer.accept(Operator::RightParen))
            return expr;
        }
        
        static Result<std::string> parse_identifier( Lexer& lexer, bool qualified = false )
        {
            std::string id = lexer.token();
            TRY_CALL(lexer.accept(Category::Identifier))
            if ( qualified )
            {
                while ( lexer.is_token(Operator::Dot) )
                {
                    TRY_CALL(lexer.accept())
                    id += "." + lexer.token();
                    TRY_CALL(lexer.accept(Category::Identifier))
                }
            }
            return id;
        }
        
        static Result<std::string> parse_string( Lexer& lexer )
        {
            TRY_CALL(lexer.expect(Category::String))
            
            std::string str;
            while ( lexer.is_token(Category::String) )
            {
                const size_t length = lexer.token().length();
                str += lexer.token().substr(1, length - 2);
                TRY_CALL(lexer.accept())
            }
            return str;
        }
        
        static bool is_typename( Lexer& lexer )
        {
            return lexer.is_oneof(Keyword::Type, Keyword::Num, Keyword::Arith, Keyword::Int, Keyword::Real, Keyword::Bool, Keyword::Str);
        }
        
    private:
        
        static int precedence( const Operator op )
        {
            static const std::map<Operator,int> precedences =
            {
                { Operator::Imply, 1 },
                { Operator::And, 2 },
                { Operator::Or, 2 },
                { Operator::Xor, 2 },
                { Operator::Less, 3 },
                { Operator::Greater, 3 },
                { Operator::LessEqual, 3 },
                { Operator::GreaterEqual, 3 },
                { Operator::Equal, 3 },
                { Operator::NotEqual, 3 },
                { Operator::MakeEqual, 3 },
                { Operator::Min, 4 },
                { Operator::Max, 4 },
                { Operator::ArgMin, 4 },
                { Operator::ArgMax, 4 },
                { Operator::Plus, 5 },
                { Operator::Minus, 5 },
                { Operator::Multiply, 6 },
                { Operator::Divide, 6 },
                { Operator::Modulo, 6 },
                { Operator::CeilDivide, 6 },
                { Operator::Power, 7 },
            };
            auto it = precedences.find(op);
            return it != precedences.end() ? it->second : 0;
        }
        
    private:
        
        void report_error( const Error& error )
        {
            _error(error.position, error.message, error.trace, false);
        }
        
        template<typename... Args>
        void report_error( const Position& position, const char* format, Args&&... args )
        {
            _error(position, Error::format_string(format, std::forward<Args>(args)...), {}, false);
        }
        
        template<typename... Args>
        void report_warning( const Position& position, const char* format, Args&&... args )
        {
            _error(position, Error::format_string(format, std::forward<Args>(args)...), {}, true);
        }
        
    private:
        
        const std::string _stdlib_path;
        const std::string _import_path;
        const ErrorCallback _error;
    };

}   // namespace sknd

#endif
