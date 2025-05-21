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

#ifndef _SKND_TYPING_H_
#define _SKND_TYPING_H_

#include "operator.h"
#include "function.h"
#include "symbolic.h"
#include "evaluation.h"
#include "result.h"
#include <optional>
#include <numeric>
#include <map>
#include <set>

#define ENABLE_SYMBOLIC_CHECKS 1



namespace sknd
{
    
    class Typing
    {
        enum Flags : unsigned
        {
            AllowTensorOperators = 0x01,
            AllowBounded = 0x04,
        };
        
        struct Declaration
        {
            static const unsigned Dtype = 0x01;
            static const unsigned Attrib = 0x02;
            static const unsigned Input = 0x04;
            static const unsigned Output = 0x08;
            static const unsigned Variable = 0x10;
            static const unsigned Constant = 0x20;
            static const unsigned TensorIndex = 0x40;
            static const unsigned Unrolled = 0x80;
            static const unsigned LoopLocal = 0x0100;
            static const unsigned Shape = 0x0200;
            static const unsigned Implicit = 0x0400;
            static const unsigned Result = 0x0800;
            static const unsigned Using = 0x1000;
            static const unsigned Inherited = 0x2000;
            static const unsigned AllowShadowing = 0x4000;
            
            Position position;
            Type type;
            Shared<Expr> repeats;
            Shared<Expr> cond;
            unsigned flags;
        };
        
        struct Definition
        {
            const Position position;
            bool update;
        };
        
        template<typename T>
        using Dict = std::map<std::string,T>;
        
    public:
        
        Typing( const ErrorCallback error )
        : _error(error)
        {
        }
        
        void check_operator( const Operator& op, const Dict<Operator>& operators, const bool main ) const
        {
            Dict<Declaration> decls;
            std::vector<bool> checked(op.asserts.size(), false);
            std::vector<bool> declared(op.usings.size(), false);
            
            for ( auto& param : op.dtypes )
            {
                check_type_param(param);
                declare_symbol(decls, param.position, param.name, make_type(param.base_type), nullptr, Declaration::Dtype);
            }
            
            for ( auto& param : op.attribs )
            {
                if ( !is_deferred_attrib(param) )
                {
                    declare_symbol(decls, param.position, param.name, param.type, param.repeats, Declaration::Attrib);
                }
            }
            
            check_asserts(op.asserts, decls, checked);
            
            for ( size_t i = 0; i < op.usings.size(); ++i )
            {
                auto& usage = op.usings[i];
                if ( !has_unknown_symbols(*usage.expr, decls) )
                {
                    declare_using(decls, usage);
                    check_asserts(op.asserts, decls, checked);
                    declared[i] = true;
                }
            }
            
            for ( auto& param : op.attribs )
            {
                if ( is_deferred_attrib(param) )
                {
                    declare_symbol(decls, param.position, param.name, param.type, param.repeats, Declaration::Attrib);
                    if ( param.repeats )
                    {
                        declare_repeats(decls, param);
                    }
                }
            }
            
            auto order = deduction_order(op.attribs, op.inputs, op.usings, op.name);
            if ( !order )
            {
                report_error(order.error());
                return;
            }
            
            for ( size_t i = 0; i < op.inputs.size(); ++i )
            {
                auto& param = op.inputs[(*order)[i]];
                declare_symbol(decls, param.position, param.name, param.type, param.repeats, Declaration::Input);
                if ( param.repeats && (!op.graph || param.repeats_bound) )
                {
                    if ( main )
                    {
                        check_repeat_bound(decls, param.repeats, param.repeats_bound);
                    }
                    declare_repeats(decls, param);
                }
                declare_shape_rank(decls, param.rank);
                if ( param.shape )
                {
                    auto& shape = *param.shape;
                    if ( main )
                    {
                        check_shape_component_bounds(decls, shape);
                    }
                    declare_shape_components(decls, shape, param.repeats, param.type.optional);
                    if ( uses_shape_symbols(shape) )
                    {
                        report_error(param.position, "implicitly defined shape/rank symbols are not allowed in input shape declaration");
                    }
                }
                else if ( !op.graph )
                {
                    report_error(param.position, "input shape can only be omitted for graphs");
                }
                if ( param.rank && main )
                {
                    report_error(param.rank->position, "capturing input rank in main graph is not allowed");
                }
            }
            for ( auto& [iden, decl] : decls )
            {
                if ( decl.cond && is_always_true(*decl.cond, decls) )
                {
                    decl.type.optional = false;
                    decl.cond = nullptr;
                }
            }
            
            for ( auto& param : op.attribs )
            {
                check_param(param, decls, Lexer::Block::Attrib);
            }
            for ( size_t i = 0; i < op.inputs.size(); ++i )
            {
                auto& param = op.inputs[(*order)[i]];
                check_param(param, decls, Lexer::Block::Input);
                if ( param.shape )
                {
                    check_shape_components(decls, *param.shape, param.repeats, false);
                }
                if ( main )
                {
                    if ( param.type.packed && !param.repeats )
                    {
                        report_error(param.position, "packed inputs of main graph must have their pack sizes defined");
                    }
                    if ( !param.shape )
                    {
                        report_error(param.position, "inputs of main graph must have their shape specified");
                    }
                    else
                    {
                        auto& shape = *param.shape;
                        if ( std::any_of(shape.extents.begin(), shape.extents.end(), []( const Shared<Expr>& item ){ return !item; }) )
                        {
                            report_error(shape.position, "inputs of main graph must have their shape components specified");
                        }
                    }
                }
            }
            for ( size_t i = 0; i < op.usings.size(); ++i )
            {
                if ( !declared[i] )
                {
                    declare_using(decls, op.usings[i]);
                    check_asserts(op.asserts, decls, checked);
                    declared[i] = true;
                }
            }
            check_asserts(op.asserts, decls, checked, true);
            for ( auto& param : op.constants )
            {
                check_param(param, decls, Lexer::Block::Constant);
                if ( param.shape )
                {
                    check_shape_components(decls, *param.shape, param.repeats, true);
                }
                else
                {
                    report_error(param.position, "constants must have their shape specified");
                }
                declare_symbol(decls, param.position, param.name, param.type, param.repeats, Declaration::Constant);
            }
            for ( auto& param : op.variables )
            {
                check_param(param, decls, Lexer::Block::Variable);
                if ( param.shape )
                {
                    check_shape_components(decls, *param.shape, param.repeats, true);
                }
                else
                {
                    report_error(param.position, "variables must have their shape specified");
                }
                declare_symbol(decls, param.position, param.name, param.type, param.repeats, Declaration::Variable);
            }
            for ( auto& param : op.outputs )
            {
                check_param(param, decls, Lexer::Block::Output);
                declare_symbol(decls, param.position, param.name, param.type, param.repeats, Declaration::Output);
                if ( param.shape )
                {
                    declare_dynamic_shape_components(decls, *param.shape);
                    check_shape_components(decls, *param.shape, param.repeats, true);
                }
                else if ( !op.graph )
                {
                    report_error(param.position, "output shape can only be omitted for graphs");
                }
            }
            
            Dict<Definition> defs;
            
            if ( op.graph && op.components.empty() )
            {
                report_error(op.position, "graph must have a @compose block");
            }
            if ( !op.components.empty() && !op.lowerings.empty() )
            {
                report_error(op.position, "operator must not have both @lower and @compose blocks");
            }
            if ( op.components.empty() && op.name.front() == '_' )
            {
                report_error(op.position, "module private operator must have @compose block");
            }
            
            
            for ( auto& component : op.components )
            {
                check_component(component, operators, decls, defs, false);
                check_invocation_access(op.position.module, component);
            }
            
            for ( auto& component : op.updates )
            {
                check_component(component, operators, decls, defs, true);
                check_invocation_access(op.position.module, component);
            }
            
            for ( auto& quantization : op.quantizations )
            {
                check_quantization(quantization, operators, decls);
            }
            
            for ( size_t i = 0; i < op.lowerings.size(); ++i )
            {
                auto& lowering = op.lowerings[i];
                if ( !has_initializer(op.lowerings, i) && !can_generate_implicit_initializer(lowering.op) )
                {
                    report_error(lowering.position, "contraction with operator '%s' must have a corresponding explicit initializer",
                                 Lexer::str(lowering.op));
                }
                check_lowering(lowering, decls, defs);
            }
            
            if ( !op.components.empty() || !op.lowerings.empty() )
            {
                for ( auto& param : op.outputs )
                {
                    if ( !defs.count(param.name) )
                    {
                        report_error(op.position, "output '%s' of operator '%s' must be defined",
                                     param.name.c_str(), op.name.c_str());
                    }
                }
            }
            
            check_labels(op, operators, decls);
        }
        
    private:
        
        void declare_symbol( Dict<Declaration>& decls, const Position& position, const std::string& name, const Type& type,
                            const Shared<Expr>& repeats, const unsigned flags, const Shared<Expr>& cond = nullptr ) const
        {
            auto ins = decls.emplace(name, Declaration{ position, type, repeats, cond, flags });
            if ( !ins.second )
            {
                Declaration& decl = ins.first->second;
                if ( (decl.flags & Declaration::Inherited) || (flags & Declaration::AllowShadowing) )
                {
                    decl = Declaration{ position, type, repeats, cond, flags };
                }
                else
                {
                    bool allow_redeclare = false;
                    if ( flags & Declaration::Result )
                    {
                        allow_redeclare = (decl.flags & Declaration::Output) || (decl.flags & Declaration::Variable);
                    }
                    else if ( flags & Declaration::Shape )
                    {
                        allow_redeclare = (decl.flags & Declaration::Shape) || (decl.flags & Declaration::Attrib);
                    }
                    if ( !allow_redeclare )
                    {
                        report_error(position, "identifier '%s' is already declared at [%d:%d]",
                                     name.c_str(), (int)decl.position.line, (int)decl.position.column);
                    }
                    else
                    {
                        if ( decl.type.optional )
                        {
                            if ( !type.optional )
                            {
                                decl.type.optional = false;
                                decl.cond = nullptr;
                            }
                            else if ( cond )
                            {
                                if ( !decl.cond )
                                {
                                    decl.cond = cond;
                                }
                                else
                                {
                                    decl.cond = std::make_shared<BinaryExpr>(cond->position, decl.cond, cond, Lexer::Operator::Or);
                                }
                            }
                        }
                        if ( as_non_optional(decl.type) != as_non_optional(type) && type.name != Typename::Type )
                        {
                            report_error(position, "identifer '%s' was previously declared as '%s' at [%d,%d]; would be redeclared as '%s'",
                                         name.c_str(), str(decl.type).c_str(), (int)decl.position.line, (int)decl.position.column, str(type).c_str());
                        }
                        if ( decl.repeats && repeats && !ranks_equal(*decl.repeats, *repeats, decls) )
                        {
                            report_error(position, "packed identifer '%s' was previously declared with pack size '%s' at [%d,%d]"
                                         "; would be redeclared with pack size '%s'",
                                         name.c_str(), str(*decl.repeats).c_str(), (int)decl.position.line, (int)decl.position.column,
                                         str(*repeats).c_str());
                        }
                    }
                }
            }
            else if ( type.tensor )
            {
                if ( type.packed )
                {
                    declare_symbol(decls, position, name + ".size", make_type(Typename::Int, type.optional, false, false), nullptr,
                                   Declaration::Shape | Declaration::Implicit);
                }
                else
                {
                    auto shape_rank = std::make_shared<IdenfitierExpr>(position, name + ".rank");
                    declare_symbol(decls, position, name + ".shape", make_type(Typename::Int, type.optional, false, true), shape_rank,
                                   Declaration::Shape | Declaration::Implicit);
                    declare_symbol(decls, position, name + ".rank", make_type(Typename::Int, type.optional, false, false), nullptr,
                                   Declaration::Shape | Declaration::Implicit);
                }
            }
        }
        
        void declare_shape_components( Dict<Declaration>& decls, const Shapedef& shape, const Shared<Expr>& repeats, bool optional ) const
        {
            size_t idx = 0;
            for ( auto item : shape.extents )
            {
                bool packed = false;
                bool optnal = optional;
                bool dynamic = shape.bounds[idx] != nullptr;
                bool spread = shape.spreads & (1 << idx++);
                
                Shared<Expr> count, cond;
                
                if ( spread )
                {
                    count = repeats;
                    packed = true;
                }
                else if ( item->kind == Expr::Expand )
                {
                    count = as_expand(*item).count;
                    item = as_expand(*item).item;
                    packed = true;
                    
                    if ( count )
                    {
                        auto type = eval_type(*count, decls);
                        if ( type && type->name == Typename::Bool )
                        {
                            cond = count;
                            count = nullptr;
                            packed = false;
                            optnal = true;
                        }
                        else
                        {
                            auto& iden = find_affine_id(*count);
                            if ( !iden.empty() )
                            {
                                auto type = make_type(Typename::Int, optional, false, false);
                                declare_symbol(decls, count->position, iden, type, nullptr, Declaration::Shape);
                            }
                        }
                    }
                }
                
                auto& iden = find_affine_id(*item);
                if ( !iden.empty() )
                {
                    if ( dynamic )
                    {
                        auto it = decls.find(iden);
                        if ( it != decls.end() && (it->second.flags & Declaration::Attrib) )
                        {
                            auto& pos = it->second.position;
                            report_error(item->position, "identifier '%s' is already declared as an attribute at [%d,%d]; "
                                                         "cannot be used as a dynamic shape component",
                                         iden.c_str(), (int)pos.line, (int)pos.column);
                        }
                    }
                    auto type = make_type(Typename::Int, optnal, false, packed);
                    declare_symbol(decls, item->position, iden, type, count, Declaration::Shape, cond);
                }
            }
        }
        
        void declare_dynamic_shape_components( Dict<Declaration>& decls, const Shapedef& shape ) const
        {
            for ( size_t i = 0; i < shape.extents.size(); ++i )
            {
                auto& extent = shape.extents[i];
                auto& bound = shape.bounds[i];
                if ( bound )
                {
                    if ( extent->kind == Expr::Identifier )
                    {
                        auto& iden = as_identifier(*extent).name;
                        declare_symbol(decls, extent->position, iden, make_type(Typename::Int), nullptr,
                                       Declaration::Shape | Declaration::Implicit);
                    }
                    else
                    {
                        report_error(extent->position, "expected identifier when upper bound is specified");
                    }
                }
            }
        }
        
        void declare_shape_rank( Dict<Declaration>& decls, const Shared<Expr>& rank ) const
        {
            if ( rank )
            {
                auto& iden = as_identifier(*rank).name;
                declare_symbol(decls, rank->position, iden, make_type(Typename::Int), nullptr, Declaration::Shape);
            }
        }
        
        void check_shape_components( const Dict<Declaration>& decls, const Shapedef& shape, const Shared<Expr>& repeats, bool enforce ) const
        {
            size_t idx = 0;
            for ( auto item : shape.extents )
            {
                bool spread = shape.spreads & (1 << idx++);
                if ( !item )
                {
                    continue;
                }
                if ( item->kind == Expr::Expand )
                {
                    auto& expand = as_expand(*item);
                    if ( expand.count )
                    {
                        if ( !is_affine_expr(*expand.count) || enforce )
                        {
                            check_repeat(*expand.count, decls, true);
                        }
                    }
                    item = expand.item;
                    if ( !item )
                    {
                        continue;
                    }
                    
                    if ( !is_affine_expr(*item) || enforce )
                    {
                        auto [type, rank] = check_extent(*item, decls);
                        if ( type && !type->packed && !expand.count )
                        {
                            report_error(item->position, "repeat count must be supplied if item is not a pack");
                        }
                    }
                }
                else
                {
                    if ( spread && !repeats )
                    {
                        report_error(item->position, "packed item can only be spread across a packed parameter");
                    }
                    if ( !is_affine_expr(*item) || enforce )
                    {
                        auto [type, rank] = check_extent(*item, decls, repeats);
                        if ( type && type->packed && !spread )
                        {
                            report_error(item->position, "packed item must be expanded or spread in shape definitions");
                        }
                    }
                }
            }
        }
        
        void check_shape_component_bounds( const Dict<Declaration>& decls, const Shapedef& shape ) const
        {
            for ( size_t i = 0; i < shape.extents.size(); ++i )
            {
                auto& extent = shape.extents[i];
                auto& bound = shape.bounds[i];
                
                auto& iden = find_affine_id(expanded(*extent));
                if ( !iden.empty() && !decls.count(iden) && !bound )
                {
                    report_error(extent->position, "upper bound must be specified in main graph for dynamic shape");
                }
                if ( extent->kind == Expr::Expand )
                {
                    auto count = as_expand(*extent).count;
                    if ( has_unknown_symbols(*count, decls) )
                    {
                        report_error(count->position, "shape components of main graph inputs must have fixed length");
                    }
                }
            }
        }
        
        void check_repeat_bound( const Dict<Declaration>& decls, const Shared<Expr>& repeats, const Shared<Expr>& bound ) const
        {
            if ( repeats )
            {
                auto& iden = find_affine_id(*repeats);
                if ( !iden.empty() && !decls.count(iden) && !bound )
                {
                    report_error(repeats->position, "upper bound must be specified in main graph for dynamic pack size");
                }
            }
        }
        
        static bool ends_with( const std::string& str, const std::string& suffix )
        {
            return str.length() > suffix.length() && str.substr(str.length() - suffix.length()) == suffix;
        }
        
        bool uses_shape_symbols( const Shapedef& shape ) const
        {
            for ( auto& item : shape.extents )
            {
                if ( item && uses_shape_symbols(*item) )
                {
                    return true;
                }
            }
            return false;
        }
        
        bool uses_shape_symbols( const Expr& expr ) const
        {
            return any_of(expr, [&]( const Expr& e )
            {
                if ( e.kind == Expr::Identifier )
                {
                    auto& id = as_identifier(e).name;
                    return ends_with(id, ".shape") || ends_with(id, ".rank");
                }
                return false;
            });
        }
        
        void declare_repeats( Dict<Declaration>& decls, const Typed& param ) const
        {
            auto& iden = find_affine_id(*param.repeats);
            if ( !iden.empty() )
            {
                auto type = make_type(Typename::Int, param.type.optional, false, false);
                declare_symbol(decls, param.repeats->position, iden, type, nullptr, Declaration::Shape);
                
                if ( param.repeats_bound )
                {
                    auto it = decls.find(iden);
                    if ( it != decls.end() && (it->second.flags & Declaration::Attrib) )
                    {
                        auto& pos = it->second.position;
                        report_error(param.repeats->position, "identifier '%s' is already declared as an attribute at [%d,%d]; "
                                                              "cannot be used as a dynamic pack size",
                                     iden.c_str(), (int)pos.line, (int)pos.column);
                    }
                }
            }
        }
        
        void declare_results( Dict<Declaration>& decls, const std::vector<Packable<Typed>>& results, const std::vector<Type>& types,
                             const std::vector<Shared<Expr>>& counts, const Position& position, const bool updates ) const
        {
            for ( size_t i = 0; i < results.size(); ++i )
            {
                auto& result = results[i];
                auto type = updates ? as_non_optional(types[i]) : types[i];
                auto repeats = counts[i];
                if ( result.packed() )
                {
                    if ( !type.packed )
                    {
                        report_error(position, "result %d cannot be a list for non-packed type '%s'",
                                     (int)(i + 1), str(type).c_str());
                    }
                    else
                    {
                        bool has_pack = false;
                        for ( size_t k = 0; k < result.size(); ++k )
                        {
                            auto& item = result[k];
                            
                            Shared<Expr> pack_size;
                            bool pack = item.type.packed;
                            if ( pack )
                            {
                                if ( has_pack )
                                {
                                    report_error(item.position, "result can have at most one packed item");
                                    break;
                                }
                                
                                pack_size = repeats;
                                if ( repeats && result.size() > 1 )
                                {
                                    auto diff = std::make_shared<IntExpr>(item.position, result.size() - 1);
                                    pack_size = std::make_shared<BinaryExpr>(item.position, repeats, diff, Lexer::Operator::Minus);
                                }
                                
                                auto count = item.repeats;
                                if ( count )
                                {
                                    if ( !check_repeat(*count, decls) )
                                    {
                                        continue;
                                    }
                                    if ( !repeats )
                                    {
                                        pack_size = count;
                                    }
                                    else if ( !ranks_equal(*count, *pack_size, decls) )
                                    {
                                        report_error(item.position, "mismatch between explicit pack length '%s' and repeat count, '%s'",
                                                     str(*count).c_str(), str(*pack_size).c_str());
                                    }
                                }
                            }
                            
                            if ( is_empty_pack(type) )
                            {
                                type.name = item.type.name;
                            }
                            if ( !is_compatible(item.type.name, type.name) )
                            {
                                report_error(item.position, "mismatch between declared type '%s' and derived type '%s'",
                                             str(item.type.name).c_str(), str(type.name).c_str());
                            }
                            
                            if ( !item.name.empty() )
                            {
                                if ( pack && !pack_size )
                                {
                                    report_error(item.position, "could not determine symbolic length of packed output '%s'",
                                                 item.name.c_str());
                                }
                                
                                declare_symbol(decls, item.position, item.name, pack ? as_packed(type) : as_non_packed(type), pack_size, Declaration::Result);
                            }
                            
                            if ( item.shape )
                            {
                                declare_shape_rank(decls, item.rank);
                                declare_shape_components(decls, *item.shape, item.repeats, item.type.optional);
                            }
                            
                            has_pack |= pack;
                        }
                    }
                }
                else
                {
                    auto& item = *result;
                    
                    Shared<Expr> count = item.repeats;
                    if ( count )
                    {
                        if ( !type.packed )
                        {
                            report_error(item.position, "non-packed result cannot be assigned to packed identifier");
                        }
                        if ( !check_repeat(*count, decls) )
                        {
                            continue;
                        }
                        if ( repeats && !ranks_equal(*count, *repeats, decls) )
                        {
                            report_error(item.position, "mismatch between result rank '%s' and repeat count '%s'",
                                         str(*count).c_str(), str(*repeats).c_str());
                        }
                    }
                    else if ( repeats )
                    {
                        count = repeats;
                    }
                    
                    if ( is_empty_pack(type) )
                    {
                        type.name = item.type.name;
                    }
                    if ( !is_compatible(item.type.name, type.name) )
                    {
                        report_error(item.position, "mismatch between declared type '%s' and derived type '%s'",
                                     str(item.type.name).c_str(), str(type.name).c_str());
                    }
                    
                    if ( item.shape )
                    {
                        declare_shape_rank(decls, item.rank);
                        declare_shape_components(decls, *item.shape, item.repeats, item.type.optional);
                    }
                    
                    if ( !item.name.empty() )
                    {
                        if ( type.packed && !count )
                        {
                            report_error(item.position, "could not determine symbolic length of packed output '%s'",
                                         item.name.c_str());
                        }
                        declare_symbol(decls, item.position, item.name, type, type.packed ? count : nullptr, Declaration::Result);
                    }
                }
            }
        }
        
        void declare_using( Dict<Declaration>& decls, const Using& usage ) const
        {
            auto [type, rank] = check_expr(*usage.expr, decls);
            if ( !type || !rank )
            {
                return;
            }
            
            if ( usage.rank )
            {
                if ( !*rank )
                {
                    report_error(usage.position, "identifier is declared packed but the right hand side expression is not packed");
                }
                else if ( check_repeat(*usage.rank, decls) )
                {
                    *rank = usage.rank; // overwrite with hint
                }
            }
            
            if ( usage.identifier->kind == Expr::List )
            {
                if ( !type->packed )
                {
                    report_error(usage.position, "left-hand-side must be a packed expression");
                }
                bool has_undeclared_rank = false;
                for ( auto item : as_list(*usage.identifier).items )
                {
                    Shared<Expr> count;
                    bool expand = item->kind == Expr::Expand;
                    if ( expand )
                    {
                        count = as_expand(*item).count;
                        if ( count )
                        {
                            check_repeat(*count, decls);
                        }
                        else
                        {
                            if ( has_undeclared_rank )
                            {
                                report_error(item->position, "declared list can have at most one flexible item");
                            }
                            count = flexible_item_rank(as_list(*usage.identifier), *rank);
                            assert(count);
                            has_undeclared_rank = true;
                        }
                        item = as_expand(*item).item;
                    }
                    auto item_type = expand ? as_packed(*type) : as_non_packed(*type);
                    
                    bool unzip = item->kind == Expr::Zip;
                    if ( unzip )
                    {
                        auto item_count = as_zip(*item).items.size();
                        auto size = std::make_shared<IntExpr>(item->position, item_count);
                        if ( !rank_divisible(*count, (int_t)item_count, decls) )
                        {
                            report_error(item->position, "unzipped pack size (%s) is not divisible by number of items in zip expression (%d)",
                                         str(*count).c_str(), (int)item_count);
                        }
                        count = std::make_shared<BinaryExpr>(item->position, count, size, Lexer::Operator::Divide);
                        for ( auto part : as_zip(*item).items )
                        {
                            declare_symbol(decls, part->position, as_identifier(*part).name, item_type, count, Declaration::Using);
                        }
                    }
                    else
                    {
                        declare_symbol(decls, item->position, as_identifier(*item).name, item_type, count, Declaration::Using);
                    }
                }
            }
            else
            {
                declare_symbol(decls, usage.position, as_identifier(*usage.identifier).name, *type, *rank, Declaration::Using);
            }
        }
        
        void define_results( Dict<Definition>& defs, const std::vector<Packable<Typed>>& results ) const
        {
            for ( size_t i = 0; i < results.size(); ++i )
            {
                auto& result = results[i];
                if ( result.packed() )
                {
                    for ( size_t k = 0; k < result.size(); ++k )
                    {
                        if ( !result[k].name.empty() )
                        {
                            define_symbol(defs, result[k].position, result[k].name);
                        }
                    }
                }
                else if ( !result->name.empty() )
                {
                    define_symbol(defs, result->position, result->name);
                }
            }
        }
        
        void define_symbol( Dict<Definition>& defs, const Position& position, const std::string& name, bool update = false ) const
        {
            auto ins = defs.emplace(name, Definition{ position, update });
            if ( !ins.second )
            {
                Definition& def = ins.first->second;
                if ( def.update || !update )
                {
                    report_error(position, "identifier '%s' is already defined at [%d:%d]",
                                 name.c_str(), (int)def.position.line, (int)def.position.column);
                }
                else
                {
                    def.update = true;
                }
            }
        }
        
        void check_param( const Param& param, const Dict<Declaration>& decls, const Lexer::Block block ) const
        {
            bool declares_repeats = block == Lexer::Block::Attrib || block == Lexer::Block::Input;
            if ( !declares_repeats && !param.repeats && param.type.packed )
            {
                report_error(param.position, "repeat count must be defined");
            }
            if ( param.repeats && (!declares_repeats || !is_affine_expr(*param.repeats)) )
            {
                check_repeat(*param.repeats, decls);
            }
            if ( !param.type_alias.empty() )
            {
                if ( !decls.count(param.type_alias) )
                {
                    report_error(param.position, "undefined generic type '%s'", param.type_alias.c_str());
                }
            }
            else if ( is_abstract(param.type.name) )
            {
                report_error(param.position, "type of parameter '%s' must not be abstract; found type '%s'",
                             param.name.c_str(), str(param.type).c_str());
            }
            if ( block == Lexer::Block::Constant && !param.default_value )
            {
                report_error(param.position, "constant '%s' must have a value assigned", param.name.c_str());
            }
            if ( block == Lexer::Block::Variable && param.default_value )
            {
                report_error(param.position, "variable '%s' must not have a value assigned", param.name.c_str());
            }
            if ( block == Lexer::Block::Output && param.type.optional )
            {
                report_error(param.position, "output must not be declared as optional", param.name.c_str());
            }
            if ( param.default_value )
            {
                check_default_value(param, decls, param.repeats, block == Lexer::Block::Constant);
            }
        }
        
        void check_type_param( const TypeParam& type ) const
        {
            if ( !is_abstract(type.base_type) )
            {
                report_error(type.position, "generic type must be abstract; found '%s'",
                             str(type.base_type).c_str());
            }
            if ( type.default_type && is_abstract(*type.default_type) )
            {
                report_error(type.position, "generic type default value must not be abstract; found '%s'",
                             str(*type.default_type).c_str());
            }
        }
        
        void check_component( const Component& component, const Dict<Operator>& operators, Dict<Declaration>& decls, Dict<Definition>& defs,
                             const bool updates ) const
        {
            Dict<Declaration> saved_decls;
            bool restore_decls = false;
            
            if ( component.loop )
            {
                saved_decls = decls;
                restore_decls = true;
                
                for ( auto& [iden, expr] : component.loop->carries )
                {
                    auto [type, rank] = check_expr(*expr, decls);
                    if ( type && rank )
                    {
                        if ( type->optional )
                        {
                            report_error(expr->position, "loop carried dependency declaration must not be of optional type");
                        }
                        if ( iden.type.name != Typename::Type && iden.type.name != type->name )
                        {
                            report_error(expr->position, "mismatch between declared and derived type of loop carried dependency (%s vs %s)",
                                         str(iden.type.name).c_str(), str(type->name).c_str());
                        }
                        declare_symbol(decls, expr->position, iden.name, as_tensor(*type), *rank, Declaration::LoopLocal | Declaration::AllowShadowing);
                    }
                }
                
                for ( auto& [iden, expr] : component.loop->scans )
                {
                    auto [type, rank] = check_expr(*expr, decls);
                    if ( type && rank )
                    {
                        if ( !type->packed )
                        {
                            report_error(expr->position, "scan input declaration must be of packed type");
                        }
                        declare_symbol(decls, expr->position, iden, as_non_packed(*type), nullptr, Declaration::LoopLocal | Declaration::AllowShadowing);
                    }
                }
                if ( component.loop->index )
                {
                    auto type = make_type(Typename::Int, false, !component.loop->unroll, false);
                    declare_symbol(decls, component.loop->index->position, component.loop->index->name, type, nullptr, Declaration::LoopLocal | Declaration::AllowShadowing);
                }
                if ( component.loop->condition )
                {
                    check_condition(*component.loop->condition, operators, decls);
                }
                if ( component.loop->count )
                {
                    auto [type, rank] = check_expr(*component.loop->count, decls);
                    if ( rank && *rank )
                    {
                        report_error(component.loop->count->position, "loop count must not be packed");
                    }
                    if ( type && (type->name != Typename::Int || type->packed) )
                    {
                        report_error(component.loop->count->position, "loop count must be of (optional) type 'int' or 'int[]', found '%s'",
                                     str(*type).c_str());
                    }
                    if ( type && type->tensor && component.loop->unroll )
                    {
                        report_error(component.loop->count->position, "loop count must not be of tensor type for unrolled loops, found '%s'",
                                     str(*type).c_str());
                    }
                    if ( type && type->optional && component.loop->scans.empty() && !component.loop->condition )
                    {
                        report_error(component.position, "loop without scan inputs and condition must not have optional loop count");
                    }
                }
                else if ( component.loop->scans.empty() )
                {
                    if ( !component.loop->condition )
                    {
                        report_error(component.position, "loop without scan inputs and condition must have its loop count explicitly defined");
                    }
                    if ( component.results.size() > component.loop->carries.size() )
                    {
                        report_error(component.position, "loop without scan inputs and loop count must not have scan outputs");
                    }
                }
            }
            else if ( component.branches.size() )
            {
                for ( auto& [condition, consequent] : component.branches )
                {
                    check_condition(condition, operators, decls);
                }
            }
            
            auto types = result_type(component, decls, operators);
            if ( !types.empty() )
            {
                if ( component.loop )
                {
                    if ( types.size() < component.loop->carries.size() )
                    {
                        report_error(position(component.operation),
                                     "loop body must have at least as many outputs as loop carried dependencies (%d); found %d",
                                     (int)component.loop->carries.size(), (int)types.size());
                    }
                    else
                    {
                        size_t i = 0;
                        for ( auto& [iden, expr] : component.loop->carries )
                        {
                            auto& type = types[i++];
                            auto it = decls.find(iden.name);
                            if ( it != decls.end() )
                            {
                                auto& decl_type = it->second.type;
                                if ( type != decl_type )
                                {
                                    report_error(position(component.operation),
                                                 "type of loop body output %d does not match that of loop carried dependency %d (%s vs %s)",
                                                 (int)i, (int)i, str(type).c_str(), str(decl_type).c_str());
                                }
                            }
                        }
                    }
                }
                
                const char* op_name = component.operation.is<Invocation>() ? component.operation.as<Invocation>().target.c_str() : nullptr;
                if ( check_argument_count(component.results.size(), min_output_count(types), types.size(), component.position, op_name, "results") )
                {
                    std::vector<Shared<Expr>> repeats(component.results.size());
                    if ( component.loop )
                    {
                        auto loop_repeats = static_repeats(component, decls, component.loop->carries.size());
                        repeats.assign(repeats.size(), loop_repeats);
                    }
                    else if ( component.branches.empty() && component.operation.is<Region>() )
                    {
                        auto& region = component.operation.as<Region>();
                        for ( size_t i = 0; i < region.yields.size(); ++i )
                        {
                            auto rank = eval_rank(*region.yields[i], decls);
                            if ( rank )
                            {
                                repeats[i] = *rank;
                            }
                        }
                    }
                    if ( restore_decls )
                    {
                        std::swap(decls, saved_decls);
                        restore_decls = false;
                    }
                    declare_results(decls, component.results, types, repeats, component.position, updates);
                }
            }
            
            if ( restore_decls )
            {
                std::swap(decls, saved_decls);
            }
            
            define_results(defs, component.results);
            check_updates(decls, component.results, updates);
        }
        
        void check_condition( const Callable& condition, const Dict<Operator>& operators, const Dict<Declaration>& decls ) const
        {
            auto cond_type = result_type(condition, decls, operators, true);
            if ( !cond_type.empty() )
            {
                if ( cond_type.size() == 1 )
                {
                    auto& type = cond_type.front();
                    if ( type.name != Typename::Bool || type.packed || type.optional )
                    {
                        report_error(position(condition), "condition must be of type 'bool' (found '%s')",
                                     str(type).c_str());
                    }
                }
                else
                {
                    report_error(position(condition), "condition must return a single result (found %d)",
                                 (int)cond_type.size());
                }
            }
        }
        
        void check_default_value( const Param& param, const Dict<Declaration>& decls, const Shared<Expr> repeats, bool check_rank ) const
        {
            auto& locals = const_cast<Dict<Declaration>&>(decls);
            
            for ( auto& [id, expr] : param.default_bounds )
            {
                auto [type, rank] = check_expr(*expr, locals);
                if ( type && rank )
                {
                    if ( type->optional )
                    {
                        report_error(expr->position, "default value bound expression cannot be of optional type");
                    }
                    declare_symbol(locals, expr->position, id, *type, *rank, Declaration::Unrolled);
                }
            }
            
            auto [type, rank] = check_expr(*param.default_value, locals);
            if ( type )
            {
                if ( !is_compatible(param.type.name, type->name) && type->name != Typename::Type )
                {
                    report_error(param.default_value->position, "default value type '%s' is not compatible with parameter type '%s'",
                                 str(*type).c_str(), str(param.type).c_str());
                }
                if ( type->optional && !param.default_bounds.empty() )
                {
                    report_error(param.default_value->position, "default value expression cannot be of optional type if it is defined by bounds");
                }
            }
            if ( rank && *rank && repeats )
            {
                if ( is_const_expr(*repeats) )
                {
                    if ( !is_const_expr(**rank) )
                    {
                        report_error(param.default_value->position, "default value length must be a constant if parameter pack length is constant");
                    }
                    else
                    {
                        auto repeats_value = Evaluation::eval(*repeats, {});
                        auto rank_value = Evaluation::eval(**rank, {});
                        if ( rank_value && repeats_value && (int_t)*rank_value != (int_t)*repeats_value )
                        {
                            report_error(param.default_value->position, "default value length (%s) does not match parameter pack length (%s)",
                                         str(**rank).c_str(), str(*repeats).c_str());
                        }
                    }
                }
                else if ( check_rank && !ranks_equal(**rank, *repeats, decls) )
                {
                    report_error(param.default_value->position, "default value length (%s) does not match parameter pack length (%s)",
                                 str(**rank).c_str(), str(*repeats).c_str());
                }
            }
            
            for ( auto& [id, expr] : param.default_bounds )
            {
                locals.erase(id);
            }
        }
        
        void check_quantization( const Quantization& quantization, const Dict<Operator>& operators, const Dict<Declaration>& decls ) const
        {
            auto types = check_invocation(quantization.invocation, decls, operators, true);
            if ( !types.empty() )
            {
                auto& op = operators.at(quantization.invocation.target);
                if ( op.inputs.size() != 1 || op.inputs.front().type.packed )
                {
                    report_error(quantization.invocation.position, "quantization operator must have exactly 1 non-packed input; found %d%s",
                                 (int)op.inputs.size(), op.inputs.size() == 1 ? " packed" : "");
                }
                else if ( op.outputs.size() != 1 || op.outputs.front().type.packed )
                {
                    report_error(quantization.invocation.position, "quantization operator must have exactly 1 non-packed output; found %d%s",
                                 (int)op.outputs.size(), op.outputs.size() == 1 ? " packed" : "");
                }
                else
                {
                    auto& input = op.inputs.front();
                    auto& output = op.outputs.front();
                    if ( input.type != output.type )
                    {
                        report_error(quantization.position, "quantization operator's output type must be the same as its input type (%s vs %s)",
                                     str(output.type).c_str(), str(input.type).c_str());
                    }
                }
            }
        }
        
        static const IdenfitierExpr& get_tensor_access_iden( Shared<Expr> expr )
        {
            while ( expr->kind == Expr::Index || expr->kind == Expr::Access )
            {
                expr = expr->kind == Expr::Index ? as_index(*expr).array : as_access(*expr).tensor;
            }
            assert(expr->kind == Expr::Identifier);
            return as_identifier(*expr);
        }
        
        static bool has_initializer( const std::vector<Lowering>& lowerings, const size_t idx )
        {
            if ( lowerings[idx].op == Lexer::Operator::Assign )
            {
                return true;
            }
            auto& iden = get_tensor_access_iden(lowerings[idx].left);
            for ( size_t i = 0; i < idx; ++i )
            {
                auto& lowering = lowerings[i];
                if ( lowering.op == Lexer::Operator::Assign && get_tensor_access_iden(lowering.left).name == iden.name )
                {
                    return true;
                }
            }
            return false;
        }
        
        static bool can_generate_implicit_initializer( const Lexer::Operator op )
        {
            return op != Lexer::Operator::MakeEqual;
        }
        
        void check_lowering( const Lowering& lowering, Dict<Declaration>& decls, Dict<Definition>& defs ) const
        {
            static const std::set<Lexer::Operator> ContractionOperators =
            {
                Lexer::Operator::Assign,
                Lexer::Operator::PlusEqual,
                Lexer::Operator::MultiplyEqual,
                Lexer::Operator::AndEqual,
                Lexer::Operator::OrEqual,
                Lexer::Operator::LessEqual,
                Lexer::Operator::GreaterEqual,
                Lexer::Operator::MakeEqual,
            };
            
            if ( !ContractionOperators.count(lowering.op) )
            {
                report_error(lowering.position, "invalid contraction operator '%s' in lowering", Lexer::str(lowering.op));
            }
            
            if ( lowering.left->kind != Expr::Access )
            {
                report_error(lowering.left->position, "left hand side of lowering must be a tensor access expression");
            }
            
            if ( lowering.unroll_count )
            {
                if ( check_repeat(*lowering.unroll_count, decls) )
                {
                    declare_symbol(decls, lowering.unroll_count->position, lowering.unroll_index, make_type(Typename::Int), nullptr, Declaration::LoopLocal);
                }
            }
            
            for ( auto& [iden, expr] : lowering.bounds )
            {
                auto [type, rank] = check_extent(*expr, decls);
                if ( type && rank )
                {
                    if ( type->optional )
                    {
                        report_error(expr->position, "expected non-optional expression for index bounds");
                    }
                    declare_symbol(decls, expr->position, iden, *type, *rank, Declaration::TensorIndex);
                    check_extent(*expr, decls);
                }
            }
            
            for ( auto& [iden, expr] : lowering.locals )
            {
                auto [type, rank] = check_expr(*expr, decls, AllowTensorOperators | AllowBounded);
                if ( type && rank )
                {
                    check_tensor_expr(*expr, decls);
                    declare_symbol(decls, expr->position, iden, *type, *rank, Declaration::LoopLocal);
                }
            }
            
            auto [left_type, left_rank] = check_expr(*lowering.left, decls, AllowTensorOperators);
            if ( left_type && left_rank )
            {
                if ( left_type->optional )
                {
                    report_error(lowering.position, "expected non-optional expression on left-hand-side");
                }
                auto tensor = check_tensor_access(*lowering.left, decls);
                if ( tensor )
                {
                    define_symbol(defs, tensor->position, as_identifier(*tensor).name, lowering.op != Lexer::Operator::Assign);
                }
            }
            
            auto [right_type, right_rank] = check_expr(*lowering.right, decls, AllowTensorOperators);
            if ( right_type && right_rank )
            {
                check_tensor_expr(*lowering.right, decls);
            }
            
            if ( lowering.condition )
            {
                auto [cond_type, cond_rank] = check_expr(*lowering.condition, decls, AllowTensorOperators);
                if ( cond_type && cond_rank )
                {
                    if ( cond_type->name != Typename::Bool )
                    {
                        report_error(lowering.condition->position, "expected expression of type bool in condition");
                    }
                    if ( *cond_rank )
                    {
                        report_error(lowering.condition->position, "expected non-packed expression in condition");
                    }
                    check_tensor_expr(*lowering.condition, decls);
                }
            }
            
            if ( lowering.op == Lexer::Operator::Assign && left_type && right_type && right_type->optional && !left_type->optional )
            {
                right_type->optional = false;
            }
            if ( left_type && right_type )
            {
                if ( left_type->name != right_type->name )
                {
                    report_error(lowering.position, "mismatch between lhs type and rhs type (%s vs %s)",
                                 str(left_type->name).c_str(), str(right_type->name).c_str());
                }
                if ( !left_type->packed && right_type->packed )
                {
                    report_error(lowering.position, "left hand side must be packed if right hand side is packed");
                }
            }
            
            for ( auto& [iden, expr] : lowering.bounds )
            {
                decls.erase(iden);
            }
            for ( auto& [iden, expr] : lowering.locals )
            {
                decls.erase(iden);
            }
        }
        
        void check_tensor_expr( const Expr& expr, Dict<Declaration>& decls ) const
        {
            if ( expr.kind == Expr::Fold )
            {
                auto& fold = as_fold(expr);
                check_has_no_tensor_access(*fold.pack);
            }
            return recurse(expr, [&]( const Expr& x ){ check_tensor_expr(x, decls); });
        }
        
        Result<std::string> check_tensor_iden( const Expr& expr, const Dict<Declaration>& decls ) const
        {
            if ( expr.kind != Expr::Identifier )
            {
                return Error(expr.position, "expected identifier");
            }
            auto tensor = as_identifier(expr).name;
            auto it = decls.find(tensor);
            if ( it == decls.end() )
            {
                return Error(expr.position, "undefined identifier '%s'", tensor.c_str());
            }
            if ( !it->second.type.tensor )
            {
                return Error(expr.position, "expected tensor identifier");
            }
            return tensor;
        }
        
        Shared<Expr> check_tensor_access( const Expr& expr, Dict<Declaration>& decls ) const
        {
            auto& access = as_access(expr);
            
            const Shared<Expr> tensor = access.tensor->kind == Expr::Index ? as_index(*access.tensor).array : access.tensor;
            
            auto iden = check_tensor_iden(*tensor, decls);
            if ( !iden )
            {
                report_error(iden.error());
                return nullptr;
            }
            
            for ( auto& index : access.indices )
            {
                check_tensor_index(*index, decls, true);
            }
            
            return tensor;
        }
        
        void check_tensor_index( const Expr& expr, const Dict<Declaration>& decls, bool allow_bounded ) const
        {
            if ( expr.kind == Expr::Fold )
            {
                auto& fold = as_fold(expr);
                check_has_no_tensor_access(*fold.pack);
            }
            else if ( expr.kind == Expr::Bounded )
            {
                if ( !allow_bounded )
                {
                    report_error(expr.position, "bounded expression not allowed in this context");
                }
            }
            
            allow_bounded &= expr.kind == Expr::Expand || expr.kind == Expr::Select || 
                             expr.kind == Expr::Coalesce || expr.kind == Expr::Substitute;
            
            return recurse(expr, [&]( const Expr& x ){ check_tensor_index(x, decls, allow_bounded); });
        }
        
        void check_has_no_tensor_access( const Expr& expr ) const
        {
            if ( expr.kind == Expr::Access )
            {
                report_error(expr.position, "tensor access not allowed in this context");
            }
            return recurse(expr, [&]( const Expr& x ){ check_has_no_tensor_access(x); });
        }
        
        void check_updates( const Dict<Declaration>& decls, const std::vector<Packable<Typed>>& results, const bool updates ) const
        {
            for ( size_t i = 0; i < results.size(); ++i )
            {
                auto& result = results[i];
                if ( result.packed() )
                {
                    for ( size_t k = 0; k < result.size(); ++k )
                    {
                        if ( !result[k].name.empty() )
                        {
                            check_update(decls, result[k].position, result[k].name, updates);
                        }
                    }
                }
                else
                {
                    if ( !result->name.empty() )
                    {
                        check_update(decls, result->position, result->name, updates);
                    }
                }
            }
        }
        
        void check_update( const Dict<Declaration>& decls, const Position& position, const std::string& name, const bool updates ) const
        {
            auto it = decls.find(name);
            if ( it != decls.end() )
            {
                auto& decl = it->second;
                if ( !updates && (decl.flags & Declaration::Variable) )
                {
                    report_error(position, "identifier '%s' declared as a variable at [%d,%d] cannot be assigned in this block",
                                 name.c_str(), (int)decl.position.line, (int)decl.position.column);
                }
                else if ( updates && !(decl.flags & Declaration::Variable) )
                {
                    report_error(position, "identifier '%s' cannot be assigned in this block because it was not declared as a variable",
                                 name.c_str());
                }
            }
        }
        
        void check_labels( const Operator& op, const Dict<Operator>& operators, const Dict<Declaration>& decls ) const
        {
            Dict<Position> labels;
            for ( auto& component : op.components )
            {
                auto label = auto_label(component, decls);
                check_label(component.operation, operators, label, labels, 0);
                if ( component.loop && component.loop->condition )
                {
                    check_label(*component.loop->condition, operators, label, labels, 1);
                }
                size_t repetition = 0;
                for ( auto& [condition, operation] : component.branches )
                {
                    check_label(condition, operators, label, labels, repetition++);
                    check_label(operation, operators, label, labels, repetition++);
                }
            }
        }
        
        void check_label( const Callable& callable, const Dict<Operator>& operators, const std::string& auto_label,
                         Dict<Position>& labels, const size_t repetition ) const
        {
            if ( callable.is<Invocation>() )
            {
                auto& invocation = callable.as<Invocation>();
                auto label = !invocation.label.empty() ? invocation.label : auto_label;
                if ( label.empty() )
                {
                    auto it = operators.find(invocation.target);
                    if ( it != operators.end() && !it->second.graph && has_variables(invocation, operators) )
                    {
                        report_error(invocation.position, "invoked operator '%s' must be labelled because it defines internal variables",
                                     invocation.target.c_str());
                    }
                }
                else if ( label != "~" )
                {
                    auto [it, inserted] = labels.emplace(label, invocation.position);
                    if ( !inserted && (!invocation.label.empty() || !repetition) )
                    {
                        report_error(invocation.position, "label '%s' is already used at [%d,%d]",
                                     label.c_str(), (int)it->second.line, (int)it->second.column);
                    }
                }
            }
        }
        
        bool has_variables( const Invocation& invocation, const Dict<Operator>& operators ) const
        {
            if ( invocation.target.empty() )
            {
                return false;
            }
            auto it = operators.find(invocation.target);
            if ( it == operators.end() )
            {
                return false;
            }
            auto& op = it->second;
            if ( !op.variables.empty() )
            {
                return true;
            }
            for ( auto& component : op.components )
            {
                if ( component.operation.is<Invocation>() && has_variables(component.operation.as<Invocation>(), operators) )
                {
                    return true;
                }
                if ( component.loop && component.loop->condition && component.loop->condition->is<Invocation>() &&
                    has_variables(component.loop->condition->as<Invocation>(), operators) )
                {
                    return true;
                }
                for ( auto& [condition, operation] : component.branches )
                {
                    if ( condition.is<Invocation>() && has_variables(condition.as<Invocation>(), operators) )
                    {
                        return true;
                    }
                    if ( operation.is<Invocation>() && has_variables(operation.as<Invocation>(), operators) )
                    {
                        return true;
                    }
                }
            }
            return false;
        }
        
        std::pair<Result<Type>,Result<Shared<Expr>>> check_extent( const Expr& expr, const Dict<Declaration>& decls,
                                                                  const Shared<Expr> repeats = nullptr ) const
        {
            auto type = eval_type(expr, decls);
            auto rank = eval_rank(expr, decls);
            if ( !type )
            {
                report_error(type.error());
            }
            else if ( !rank )
            {
                report_error(rank.error());
            }
            else if ( type->name != Typename::Int || type->tensor || type->optional )
            {
                report_error(expr.position, "extent must be of type 'int'; found '%s'",
                             str(*type).c_str());
            }
            
            check_has_no_tensor_access(expr);
            
            if ( repeats && *rank && !ranks_equal(**rank, *repeats, decls) )
            {
                report_error(expr.position, "extent pack size (%s) does not match parameter count (%s)",
                             str(**rank).c_str(), str(*repeats).c_str());
            }
            
            return std::make_pair(type, rank);
        }
        
        bool check_repeat( const Expr& expr, const Dict<Declaration>& decls, bool allow_bool = false ) const
        {
            auto type = eval_type(expr, decls);
            auto rank = eval_rank(expr, decls);
            if ( !type )
            {
                report_error(type.error());
                return false;
            }
            else if ( !rank )
            {
                report_error(rank.error());
                return false;
            }
            else if ( *type != make_type(Typename::Int) && !(allow_bool && *type == make_type(Typename::Bool)) )
            {
                report_error(expr.position, "repeat expression must be of type 'int'; found '%s'",
                             str(*type).c_str());
                return false;
            }
            return true;
        }
        
        void check_assert( const Assert& assert, const Dict<Declaration>& decls ) const
        {
            auto [type, rank] = check_expr(*assert.expression, decls);
            if ( type )
            {
                if ( type->name != Typename::Bool || type->tensor )
                {
                    report_error(assert.position, "assert expression must be of type 'bool'; found '%s'",
                                 str(*type).c_str());
                }
                if ( type->packed )
                {
                    const_cast<Shared<Expr>&>(assert.expression) = std::make_shared<FoldExpr>(assert.expression->position,
                                                                                              assert.expression,
                                                                                              Lexer::Operator::And);
                }
            }
            if ( assert.message )
            {
                auto [tp, r] = check_expr(*assert.message, decls);
                if ( tp->optional && !type->optional )
                {
                    report_error(assert.message->position, "assert message must not be optional (due to one of its substitutions being optional) "
                                                           "if condition is not optional");
                }
            }
            for ( auto& item : assert.prints )
            {
                check_expr(*item.second, decls);
            }
        }
        
        void check_asserts( const std::vector<Assert>& asserts, const Dict<Declaration>& decls, std::vector<bool>& checked, bool enforce = false ) const
        {
            for ( size_t i = 0; i < asserts.size(); ++i )
            {
                if ( !checked[i] && (enforce || can_check(asserts[i], decls)) )
                {
                    check_assert(asserts[i], decls);
                    checked[i] = true;
                }
            }
        }
        
        std::pair<Result<Type>,Result<Shared<Expr>>> check_expr( const Expr& expr, const Dict<Declaration>& decls, unsigned flags = 0 ) const
        {
            auto type = eval_type(expr, decls, flags);
            auto rank = eval_rank(expr, decls);
            if ( !type )
            {
                report_error(type.error());
            }
            else if ( !rank )
            {
                report_error(rank.error());
            }
            return std::make_pair(type, rank);
        }
        
        std::vector<Type> check_invocation( const Invocation& invocation, const Dict<Declaration>& decls, const Dict<Operator>& operators,
                                           const bool quantization, const bool subgraph = false, const bool repeated = false, const size_t nvars = 0 ) const
        {
            auto it = operators.find(invocation.target);
            if ( it == operators.end() )
            {
                report_error(invocation.position, "undefined operator '%s'", invocation.target.c_str());
                
                for ( auto& [key, value] : invocation.attribs )
                {
                    check_expr(*value, decls);
                }
                for ( auto& arg : invocation.args )
                {
                    if ( arg )
                    {
                        check_expr(*arg, decls);
                    }
                }
                return {};
            }
            
            auto& op = it->second;
            if ( op.graph && !subgraph )
            {
                report_error(invocation.position, "'%s' is defined as graph, which is not allowed to be invoked in this context", op.name.c_str());
                return {};
            }
            
            Dict<Typename> dtypes = check_dtypes(op, invocation);
            check_attribs(op, invocation.attribs, decls, dtypes, invocation.position);
            
            if ( !quantization )
            {
                const size_t min_args = min_input_count(op.inputs);
                const size_t max_args = op.inputs.size();
                if ( check_argument_count(invocation.args.size(), min_args, max_args, invocation.position, op.name.c_str(), "arguments") )
                {
                    for ( size_t i = 0; i < invocation.args.size(); ++i )
                    {
                        auto& param = op.inputs[i];
                        
                        auto& arg = invocation.args[i];
                        if ( arg )
                        {
                            auto [type, rank] = check_expr(*arg, decls);
                            if ( type )
                            {
                                if ( type->packed && !param.type.packed )
                                {
                                    report_error(arg->position,
                                                 "packed expression type '%s' incompatible with non-packed parameter type '%s' for argument %d",
                                                 str(*type).c_str(), str(param.type).c_str(), (int)(i+1));
                                }
                                if ( type->optional && !param.type.optional )
                                {
                                    report_error(arg->position,
                                                 "optional expression type '%s' incompatible with non-optional parameter type '%s' for argument %d",
                                                 str(*type).c_str(), str(param.type).c_str(), (int)(i+1));
                                }
                                if ( !type->packed && param.type.packed )
                                {
                                    report_error(arg->position,
                                                 "non-packed expression type '%s' incompatible with packed parameter type '%s' for argument %d",
                                                 str(*type).c_str(), str(param.type).c_str(), (int)(i+1));
                                }
                                if ( !param.type_alias.empty() )
                                {
                                    check_base_type(op, param.type_alias, type->name, arg->position);
                                    update_generic_type(dtypes, param.type_alias, type->name, arg->position);
                                }
                                else if ( !is_compatible(param.type.name, type->name) && type->name != Typename::Type )
                                {
                                    report_error(arg->position, "expression type '%s' incompatible with parameter type '%s' for argument %d",
                                                 str(*type).c_str(), str(param.type).c_str(), (int)(i+1));
                                }
                            }
                        }
                        else if ( !param.type.optional )
                        {
                            report_error(invocation.position, "placeholder '~' may only be used for optional arguments (found type '%s' for input '%s')",
                                         str(param.type).c_str(), param.name.c_str());
                        }
                    }
                }
            }
            
            if ( repeated )
            {
                for ( size_t i = 0; i < op.outputs.size(); ++i )
                {
                    if ( op.outputs[i].type.packed )
                    {
                        report_error(invocation.position, "repeated invocation can only be applied to operators with outputs of non-packed type (found '%s' for output %d)", str(op.outputs.front().type).c_str(), (int)i+1);
                        return {};
                    }
                }
            }
            
            for ( auto& type : op.dtypes )
            {
                auto it = dtypes.find(type.name);
                if ( it == dtypes.end() && type.default_type )
                {
                    it = dtypes.emplace(type.name, *type.default_type).first;
                }
                if ( it == dtypes.end() )
                {
                    report_error(invocation.position, "could not deduce generic type '%s'", type.name.c_str());
                    return {};
                }
            }
            
            std::vector<Type> types(op.outputs.size(), make_type(Typename::Type));
            for ( size_t i = 0; i < op.outputs.size(); ++i )
            {
                types[i] = op.outputs[i].type;
                auto& alias = op.outputs[i].type_alias;
                if ( is_abstract(types[i].name) && !alias.empty() )
                {
                    types[i].name = dtypes.at(alias);
                }
                types[i].packed |= repeated && i >= nvars;
            }
            return types;
        }
        
        std::vector<Type> check_region( const Region& region, const Dict<Declaration>& decls, const Dict<Operator>& operators,
                                       const bool repeated = false, const size_t nvars = 0 ) const
        {
            Dict<Declaration> _decls = decls;
            for ( auto& [name, decl] : _decls )
            {
                decl.flags |= Declaration::Inherited;
            }
            
            Dict<Definition> defs = {};
            for ( auto& component : region.components )
            {
                check_component(component, operators, _decls, defs, false);
            }
            
            std::vector<Type> types;
            for ( auto& yield : region.yields )
            {
                auto [type, rank] = check_expr(*yield, _decls);
                if ( !type || !rank )
                {
                    return {};
                }
                if ( type->packed && repeated )
                {
                    report_error(region.position, "repeated block can only yield results of non-packed type (found '%s' for result %d)", str(*type).c_str(), (int)types.size() + 1);
                    return {};
                }
                if ( type->optional )
                {
                    report_error(region.position, "yielded result must not be of optional type (found '%s' for result %d)", str(*type).c_str(), (int)types.size() + 1);
                    return {};
                }
                type->packed |= repeated && types.size() >= nvars;
                type->tensor = true;
                types.push_back(*type);
            }
            return types;
        }
        
        Dict<Typename> check_dtypes( const Operator& op, const Invocation& invocation ) const
        {
            if ( invocation.dtypes.size() > op.dtypes.size() )
            {
                report_error(invocation.position, "too many generic type arguments for operator '%s'; expected at most %d, found %d",
                             invocation.target.c_str(), (int)op.dtypes.size(), (int)invocation.dtypes.size());
            }
            
            Dict<Typename> dtypes;
            for ( size_t i = 0; i < std::min(op.dtypes.size(), invocation.dtypes.size()); ++i )
            {
                if ( !is_compatible(op.dtypes[i].base_type, invocation.dtypes[i].second) )
                {
                    report_error(invocation.position, "type parameter '%s' must be of type '%s'; found '%s'",
                                 op.dtypes[i].name.c_str(), str(op.dtypes[i].base_type).c_str(), str(invocation.dtypes[i].second).c_str());
                }
                else
                {
                    dtypes[op.dtypes[i].name] = invocation.dtypes[i].second;
                }
            }
            return dtypes;
        }
        
        void check_attribs( const Operator& op, const Dict<Shared<Expr>>& attribs, const Dict<Declaration>& decls,
                           Dict<Typename>& generic_types, const Position& position ) const
        {
            for ( auto& [key, value] : attribs )
            {
                auto [type, rank] = check_expr(*value, decls);
                if ( type )
                {
                    auto attrib = find_param(op.attribs, key);
                    if ( !attrib )
                    {
                        report_error(value->position, "operator '%s' has no attribute '%s'", qname(op).c_str(), key.c_str());
                    }
                    else if ( !attrib->type_alias.empty() )
                    {
                        check_base_type(op, attrib->type_alias, type->name, value->position);
                        update_generic_type(generic_types, attrib->type_alias, type->name, value->position);
                    }
                    else if ( (!is_compatible(attrib->type.name, type->name) && type->name != Typename::Type) ||
                             (type->optional && !attrib->type.optional) ||
                             (type->packed && !attrib->type.packed) )
                    {
                        report_error(value->position, "expression type '%s' is not compatible with attribute type '%s' for attribute '%s'",
                                     str(*type).c_str(), str(attrib->type).c_str(), key.c_str());
                    }
                    if ( type->tensor )
                    {
                        report_error(value->position, "tensor identifier is not allowed for attributes");
                    }
                }
            }
            
            for ( auto& attrib : op.attribs )
            {
                if ( !attrib.type.optional && !attrib.default_value && !attribs.count(attrib.name) )
                {
                    report_error(position, "attribute '%s' must be supplied for operator '%s'", attrib.name.c_str(), qname(op).c_str());
                }
            }
        }
        
        void check_base_type( const Operator& op, const std::string type_alias, const Typename type, const Position& position ) const
        {
            for ( size_t i = 0; i < op.dtypes.size(); ++i )
            {
                if ( op.dtypes[i].name == type_alias )
                {
                    if ( !is_compatible(op.dtypes[i].base_type, type) )
                    {
                        report_error(position, "type parameter '%s' must deduce to type '%s'; found '%s'",
                                     op.dtypes[i].name.c_str(), str(op.dtypes[i].base_type).c_str(), str(type).c_str());
                    }
                    break;
                }
            }
        }
        
        void check_invocation_access( const std::string& module, const Component& component ) const
        {
            if ( component.branches.size() )
            {
                for ( auto& item : component.branches )
                {
                    check_invocation_access(module, item.condition);
                    check_invocation_access(module, item.consequent);
                }
            }
            else if ( component.loop )
            {
                if ( component.loop->condition )
                {
                    check_invocation_access(module, *component.loop->condition);
                }
            }
            check_invocation_access(module, component.operation);
        }
        
        void check_invocation_access( const std::string& module, const Callable& callable ) const
        {
            if ( callable.is<Invocation>() )
            {
                auto& invocation = callable.as<Invocation>();
                if ( !invocation.target.empty() )
                {
                    auto pos = invocation.target.find_last_of('.');
                    auto target_module = invocation.target.substr(0, pos);
                    auto target_name = invocation.target.substr(pos + 1);
                    
                    if ( target_name[0] == '_' && target_module != module )
                    {
                        report_error(invocation.position, "cannot invoke operator '%s' from module '%s' because it is private in module '%s'",
                                     invocation.target.c_str(), module.c_str(), target_module.c_str());
                    }
                }
            }
            else
            {
                auto& region = callable.as<Region>();
                for ( auto& component : region.components )
                {
                    check_invocation_access(module, component);
                }
            }
        }
        
        size_t min_input_count( const std::vector<Param>& params ) const
        {
            size_t min = 0;
            while ( min < params.size() && !params[min].default_value && !params[min].type.optional )
            {
                ++min;
            }
            return min;
        }
        
        size_t min_output_count( const std::vector<Type>& types ) const
        {
            size_t min = 0;
            while ( min < types.size() && !types[min].optional )
            {
                ++min;
            }
            return min;
        }
        
        bool check_argument_count( const size_t args, const size_t min, const size_t max,
                                  const Position& position, const char* op_name, const char* arg_str ) const
        {
            if ( min == max && args != min )
            {
                if ( op_name )
                {
                    report_error(position, "operation '%s' expects %d %s, got %d",
                                 op_name, (int)min, arg_str, (int)args);
                }
                else
                {
                    report_error(position, "expression expects %d %s, got %d",
                                 (int)min, arg_str, (int)args);
                }
                return false;
            }
            if ( args < min )
            {
                report_error(position, "operation '%s' expects at least %d %s, got %d",
                             op_name, (int)min, arg_str, (int)args);
                return false;
            }
            if ( args > max )
            {
                report_error(position, "operation '%s' expects at most %d %s, got %d",
                             op_name, (int)max, arg_str, (int)args);
                return false;
            }
            return true;
        }
        
        const Param* find_param( const std::vector<Param>& params, const std::string& name ) const
        {
            auto it = std::find_if(params.begin(), params.end(), [&]( const Param& param ){ return param.name == name; });
            return it != params.end() ? &*it : nullptr;
        }
        
        void update_generic_type( Dict<Typename>& types, const std::string name, const Typename type, const Position& position ) const
        {
            auto it = types.find(name);
            if ( it == types.end() )
            {
                types[name] = type;
            }
            else if ( it->second != type )
            {
                report_error(position, "abiguous deduction of generic type '%s' as '%s'; previously deduced as '%s'",
                             name.c_str(), str(type).c_str(), str(it->second).c_str());
            }
        }
        
        std::vector<Type> result_type( const Callable& callable, const Dict<Declaration>& decls, const Dict<Operator>& operators,
                                      const bool subgraph = false, const bool repeated = false, const size_t nvars = 0 ) const
        {
            if ( callable.is<Invocation>() )
            {
                return check_invocation(callable.as<Invocation>(), decls, operators, false, subgraph, repeated, nvars);
            }
            else
            {
                return check_region(callable.as<Region>(), decls, operators, repeated, nvars);
            }
        }
        
        std::vector<Type> result_type( const Component& component, const Dict<Declaration>& decls, const Dict<Operator>& operators ) const
        {
            if ( component.branches.size() )
            {
                auto common_type = result_type(component.operation, decls, operators, true);
                if ( common_type.empty() )
                {
                    return {};
                }
                for ( size_t i = 0; i < component.branches.size(); ++i )
                {
                    std::vector<std::string> promoted;
                    Dict<Declaration> _decls;
                    
                    auto& condition = component.branches[i].condition;
                    if ( condition.is<Region>() && condition.as<Region>().components.empty() )
                    {
                        enum_promoted_optionals(*condition.as<Region>().yields.front(), promoted);
                        if ( !promoted.empty() )
                        {
                            _decls = decls;
                            for ( auto& id : promoted )
                            {
                                _decls.at(id).type.optional = false;
                            }
                        }
                    }
                    
                    auto item_type = result_type(component.branches[i].consequent, !promoted.empty() ? _decls : decls, operators, true);
                    if ( item_type.empty() )
                    {
                        return {};
                    }
                    for ( size_t k = 0; k < item_type.size(); ++k )
                    {
                        if ( common_type[k].name == Typename::Type )
                        {
                            common_type[k].name = item_type[k].name;
                        }
                        else if ( item_type[k].name == Typename::Type )
                        {
                            item_type[k].name = common_type[k].name;
                        }
                    }
                    if ( item_type != common_type )
                    {
                        report_error(component.position, "operation result type mismatch (%s vs %s)",
                                     types_str(common_type).c_str(), types_str(item_type).c_str());
                        return {};
                    }
                }
                return common_type;
            }
            else if ( component.loop )
            {
                return result_type(component.operation, decls, operators, true, true, component.loop->carries.size());
            }
            else
            {
                return result_type(component.operation, decls, operators);
            }
        }
        
        Shared<Expr> static_repeats( const Component& component, const Dict<Declaration>& decls, const size_t nvars ) const
        {
            bool dynamic = component.loop->count && eval_type(*component.loop->count, decls)->tensor;
            auto repeats = !dynamic ? component.loop->count : nullptr;
            for ( auto& [iden, expr] : component.loop->scans )
            {
                auto arg_repeats = eval_rank(*expr, decls);
                if ( arg_repeats && *arg_repeats )  // otherwise error has already been reported
                {
                    if ( !repeats )
                    {
                        repeats = *arg_repeats;
                    }
                    else if ( !ranks_equal(*repeats, **arg_repeats, decls) )
                    {
                        report_error(component.position, "mismatch between implied loop counts (%s vs %s)",
                                     str(*repeats).c_str(), str(**arg_repeats).c_str());
                    }
                }
            }
            return repeats;
        }
        
        static const std::string types_str( const std::vector<Type>& types )
        {
            std::string str;
            str += '(';
            for ( size_t i = 0; i < types.size(); ++i )
            {
                if ( i != 0 )
                {
                    str += ',';
                }
                str += sknd::str(types[i]);
            }
            str += ')';
            return str;
        }
        
        static const std::string qname( const Operator& op )
        {
            return op.position.module + ("." + op.name);
        }
        
    private:
        
        static Result<Type> eval_type( const Expr& expr, const Dict<Declaration>& decls, unsigned flags = 0 )
        {
            switch ( expr.kind )
            {
                case Expr::Literal:
                {
                    return eval_type(as_literal(expr), decls);
                }
                case Expr::Identifier:
                {
                    return eval_type(as_identifier(expr), decls);
                }
                case Expr::List:
                {
                    return eval_type(as_list(expr), decls, flags);
                }
                case Expr::Unary:
                {
                    return eval_type(as_unary(expr), decls, flags);
                }
                case Expr::Binary:
                {
                    return eval_type(as_binary(expr), decls, flags);
                }
                case Expr::Select:
                {
                    return eval_type(as_select(expr), decls, flags);
                }
                case Expr::Expand:
                {
                    return eval_type(as_expand(expr), decls, flags);
                }
                case Expr::Index:
                {
                    return eval_type(as_index(expr), decls, flags);
                }
                case Expr::Access:
                {
                    return eval_type(as_access(expr), decls, flags);
                }
                case Expr::Range:
                {
                    return eval_type(as_range(expr), decls);
                }
                case Expr::Zip:
                {
                    return eval_type(as_zip(expr), decls, flags);
                }
                case Expr::Coalesce:
                {
                    return eval_type(as_coalesce(expr), decls, flags);
                }
                case Expr::Identity:
                {
                    return eval_type(as_identity(expr), decls, flags);
                }
                case Expr::Contain:
                {
                    return eval_type(as_contain(expr), decls, flags);
                }
                case Expr::Fold:
                {
                    return eval_type(as_fold(expr), decls, flags);
                }
                case Expr::Cast:
                {
                    return eval_type(as_cast(expr), decls, flags);
                }
                case Expr::Builtin:
                {
                    return eval_type(as_builtin(expr), decls, flags);
                }
                case Expr::Format:
                {
                    return eval_type(as_format(expr), decls);
                }
                case Expr::Bounded:
                {
                    return eval_type(as_bounded(expr), decls, flags);
                }
                case Expr::Substitute:
                {
                    return eval_type(as_substitute(expr), decls, flags);
                }
            }
        }
        
        static Result<Type> eval_type( const LiteralExpr& literal, const Dict<Declaration>& decls )
        {
            return literal.type;
        }
        
        static Result<Type> eval_type( const IdenfitierExpr& iden, const Dict<Declaration>& decls )
        {
            auto& name = iden.name;
            auto it = decls.find(name);
            if ( it == decls.end() )
            {
                return Error(iden.position, "undeclared identifier '%s'", name.c_str());
            }
            return it->second.type;
        }
        
        static Result<Type> eval_type( const ListExpr& list, const Dict<Declaration>& decls, unsigned flags )
        {
            auto& items = list.items;
            if ( items.empty() )
            {
                return make_type(Typename::Type, false, false, true);
            }
            
            Type type;
            for ( size_t i = 0; i < items.size(); ++i )
            {
                auto& item = *items[i];
                TRY_DECL(item_type, eval_type(item, decls, flags))
                
                if ( item_type.packed && item.kind != Expr::Range )
                {
                    return Error(item.position, "packed items must be expanded in list expression (found type '%s')",
                                 str(item_type).c_str());
                }
                if ( i == 0 )
                {
                    type = item_type;
                }
                else if ( item_type.name != type.name || item_type.tensor != type.tensor )
                {
                    return Error(list.position, "item type mismatch in list expression ('%s' vs '%s')",
                                 str(type).c_str(), str(item_type).c_str());
                }
                type.optional |= item_type.optional;
            }
            type.packed = true;
            return type;
        }
        
        static Result<Type> eval_type( const UnaryExpr& unary, const Dict<Declaration>& decls, unsigned flags )
        {
            TRY_DECL(type, eval_type(*unary.arg, decls, flags))
            if ( type.tensor && !(flags & Flags::AllowTensorOperators) && unary.op != Lexer::Operator::Question )
            {
                return Error(unary.position, "operator '%s' is not allowed for tensors in this context", Lexer::str(unary.op));
            }
            
            bool valid;
            switch ( unary.op )
            {
                case Lexer::Operator::Plus:
                case Lexer::Operator::Minus:
                {
                    valid = (type.name == Typename::Real || type.name == Typename::Int || type.name == Typename::Num);
                    break;
                }
                case Lexer::Operator::Not:
                {
                    valid = (type.name == Typename::Bool);
                    break;
                }
                case Lexer::Operator::Question:
                {
                    if ( !type.optional )
                    {
                        return Error(unary.position, "argument of unary operator '?' must be of optional type; found '%s'",
                                     str(type).c_str());
                    }
                    return make_type(Typename::Bool);
                }
                default:
                {
                    return Error(unary.position, "invalid unary operator '%s'", Lexer::str(unary.op));
                }
            }
            if ( !valid )
            {
                return Error(unary.position, "invalid argument type '%s' for operator '%s'",
                             str(type.name).c_str(), Lexer::str(unary.op));
            }
            return type;
        }
        
        static Result<Type> eval_type( const BinaryExpr& binary, const Dict<Declaration>& decls, unsigned flags )
        {
            TRY_DECL(lhs_type, eval_type(*binary.left, decls, flags))
            TRY_DECL(rhs_type, eval_type(*binary.right, decls, flags))
            
            if ( lhs_type.name != rhs_type.name )
            {
                return Error(binary.position, "argument type mismatch in operator '%s' ('%s' vs '%s')",
                             Lexer::str(binary.op), str(lhs_type.name).c_str(), str(rhs_type.name).c_str());
            }
            auto name = lhs_type.name;
            auto type = make_type(Lexer::is_comparison(binary.op) ? Typename::Bool : name,
                                  lhs_type.optional || rhs_type.optional,
                                  lhs_type.tensor || rhs_type.tensor,
                                  lhs_type.packed || rhs_type.packed);
            
            if ( type.tensor && !(flags & Flags::AllowTensorOperators) )
            {
                return Error(binary.position, "operator '%s' is not allowed for tensors in this context", Lexer::str(binary.op));
            }
            
            bool valid;
            switch ( binary.op )
            {
                case Lexer::Operator::Plus:
                {
                    valid = (name == Typename::Real || name == Typename::Int || name == Typename::Num || name == Typename::Str);
                    break;
                }
                case Lexer::Operator::Minus:
                case Lexer::Operator::Multiply:
                case Lexer::Operator::Divide:
                case Lexer::Operator::CeilDivide:
                case Lexer::Operator::Modulo:
                case Lexer::Operator::Power:
                case Lexer::Operator::Min:
                case Lexer::Operator::Max:
                {
                    valid = (name == Typename::Real || name == Typename::Int || name == Typename::Num);
                    break;
                }
                case Lexer::Operator::And:
                case Lexer::Operator::Or:
                case Lexer::Operator::Xor:
                case Lexer::Operator::Imply:
                {
                    valid = (name == Typename::Bool);
                    break;
                }
                case Lexer::Operator::Less:
                case Lexer::Operator::Greater:
                case Lexer::Operator::LessEqual:
                case Lexer::Operator::GreaterEqual:
                {
                    valid = (name == Typename::Real || name == Typename::Int || name == Typename::Num);
                    break;
                }
                case Lexer::Operator::Equal:
                case Lexer::Operator::NotEqual:
                {
                    valid = true;
                    break;
                }
                default:
                {
                    return Error(binary.position, "invalid binary operator '%s'", Lexer::str(binary.op));
                }
            }
            if ( !valid )
            {
                return Error(binary.position, "invalid argument type '%s' for operator '%s'",
                             str(lhs_type).c_str(), Lexer::str(binary.op));
            }
            return type;
        }
        
        static Result<Type> eval_type( const SelectExpr& select, const Dict<Declaration>& decls, unsigned flags )
        {
            TRY_DECL(cond_type, eval_type(*select.cond, decls, flags))
            
            if ( cond_type.name != Typename::Bool )
            {
                return Error(select.position, "condition in operator '?:' must be of type 'bool'; found '%s'",
                             str(cond_type.name).c_str());
            }
            if ( cond_type.tensor && !(flags & Flags::AllowTensorOperators) )
            {
                return Error(select.position, "condition in operator '?:' must not be of tensor type");
            }
            
            std::vector<std::string> promoted;
            enum_promoted_optionals(*select.cond, promoted);
            
            auto& _decls = const_cast<Dict<Declaration>&>(decls);
            if ( !promoted.empty() )
            {
                for ( auto& item : promoted )
                {
                    _decls.at(item).type.optional = false;
                }
            }
            
            TRY_DECL(lhs_type, eval_type(*select.left, decls, flags))
            
            if ( !promoted.empty() )
            {
                for ( auto& item : promoted )
                {
                    _decls.at(item).type.optional = true;
                }
            }
            
            if ( !select.right )
            {
                if ( cond_type.packed )
                {
                    return Error(select.position, "condition in operator '?:' must not be of packed type when the right hand side is absent");
                }
                return Type{ lhs_type.name, true, lhs_type.tensor, lhs_type.packed };
            }
            
            TRY_DECL(rhs_type, eval_type(*select.right, decls, flags))
            
            if ( !is_empty_pack(lhs_type) && !is_empty_pack(rhs_type) )
            {
                if ( lhs_type.name != rhs_type.name || (lhs_type.tensor != rhs_type.tensor && !(flags & Flags::AllowTensorOperators)) )
                {
                    return Error(select.position, "argument type mismatch in operator '?:' ('%s' vs '%s')",
                                 str(lhs_type).c_str(), str(rhs_type).c_str());
                }
            }
            
            return make_type(!is_empty_pack(lhs_type) ? lhs_type.name : rhs_type.name,
                             cond_type.optional || lhs_type.optional || rhs_type.optional,
                             cond_type.tensor || lhs_type.tensor || rhs_type.tensor,
                             cond_type.packed || lhs_type.packed || rhs_type.packed);
        }
        
        static Result<Type> eval_type( const ExpandExpr& expand, const Dict<Declaration>& decls, unsigned flags )
        {
            if ( expand.count )
            {
                TRY_DECL(type, eval_type(*expand.count, decls))
                if ( type.name != Typename::Int && type.name != Typename::Bool )
                {
                    return Error(expand.position, "repeat count expression must be of type 'int'; found '%s'",
                                 str(type.name).c_str());
                }
                if ( type.tensor )
                {
                    return Error(expand.position, "repeat count expression must not be of tensor type");
                }
            }
            TRY_DECL(item_type, eval_type(*expand.item, decls, flags))
            if ( !item_type.packed && !expand.count )
            {
                return Error(expand.position, "repeat count must be supplied if item is not a pack");
            }
            if ( item_type.packed && expand.count )
            {
                return Error(expand.position, "repeat count must not be supplied if item is a pack");
            }
            return as_non_packed(item_type);
        }
        
        static Result<Type> eval_type( const IndexExpr& index, const Dict<Declaration>& decls, unsigned flags )
        {
            TRY_DECL(array_type, eval_type(*index.array, decls, flags))
            TRY_DECL(index_type, eval_type(*index.index, decls, flags))
            
            if ( index_type.name != Typename::Int && !(index_type.name == Typename::Bool && index_type.packed) )
            {
                return Error(index.position, "index expression must be of type 'int', 'int..' or 'bool..'; found '%s'",
                             str(index_type).c_str());
            }
            if ( index_type.tensor )
            {
                return Error(index.position, "index must not be of tensor type");
            }
            if ( !array_type.packed && array_type.name != Typename::Str )
            {
                return Error(index.position, "indexed expression must be of packed type or str; found '%s'",
                             str(array_type).c_str());
            }
            
            return Type{
                array_type.name,
                array_type.optional || index_type.optional,
                array_type.tensor,
                index_type.packed && array_type.name != Typename::Str,
            };
        }
        
        static Result<Type> eval_type( const AccessExpr& access, const Dict<Declaration>& decls, unsigned flags )
        {
            TRY_DECL(tensor_type, eval_type(*access.tensor, decls, flags))
            if ( !tensor_type.tensor )
            {
                return Error(access.position, "indexed expression must be of tensor type; found '%s'",
                             str(tensor_type).c_str());
            }
            
            bool packed = tensor_type.packed;
            bool optional = tensor_type.optional;
            for ( auto& index : access.indices )
            {
                TRY_DECL(index_type, eval_type(*index, decls, flags | Flags::AllowBounded))
                if ( index_type.name != Typename::Int )
                {
                    return Error(index->position, "index expression must be of type 'int'; found '%s'",
                                 str(index_type.name).c_str());
                }
                if ( index_type.tensor )
                {
                    return Error(index->position, "index must not be of tensor type");
                }
                if ( index->kind != Expr::Expand && index_type.packed )
                {
                    if ( tensor_type.packed )
                    {
                        return Error(index->position, "index must not be of packed type when indexing a packed tensor expression");
                    }
                    else if ( packed )
                    {
                        return Error(index->position, "at most one index may be of packed type when indexing a tensor");
                    }
                    packed = true;
                }
                optional |= index_type.optional;
            }
            
            return Type{ tensor_type.name, optional, false, packed };
        }
        
        static Result<Type> eval_type( const RangeExpr& range, const Dict<Declaration>& decls )
        {
            bool optional = false;
            if ( range.first )
            {
                TRY_DECL(type, eval_type(*range.first, decls))
                if ( type.name != Typename::Int || type.tensor || type.packed )
                {
                    return Error(range.position, "range begin must be of type 'int'; found '%s'", str(type).c_str());
                }
                optional |= type.optional;
            }
            if ( range.last )
            {
                TRY_DECL(type, eval_type(*range.last, decls))
                if ( type.name != Typename::Int || type.tensor || type.packed )
                {
                    return Error(range.position, "range end must be of type 'int'; found '%s'", str(type).c_str());
                }
                optional |= type.optional;
            }
            if ( range.stride )
            {
                TRY_DECL(type, eval_type(*range.stride, decls))
                if ( type.name != Typename::Int || type.tensor || type.packed )
                {
                    return Error(range.position, "range stride must be of type 'int'; found '%s'", str(type).c_str());
                }
                optional |= type.optional;
            }
            
            return Type{ Typename::Int, optional, false, true };
        }
        
        static Result<Type> eval_type( const ZipExpr& zip, const Dict<Declaration>& decls, unsigned flags )
        {
            TRY_DECL(first_type, eval_type(*zip.items[0], decls, flags))
            if ( !first_type.packed )
            {
                return Error(zip.items[0]->position, "items in zip expression must be of packed type; found '%s'",
                             str(first_type).c_str());
            }
            for ( size_t i = 1; i < zip.items.size(); ++i )
            {
                TRY_DECL(item_type, eval_type(*zip.items[i], decls, flags))
                if ( !item_type.packed )
                {
                    return Error(zip.items[i]->position, "items in zip expression must be of packed type; found '%s'",
                                 str(item_type).c_str());
                }
                else if ( item_type.name != first_type.name || item_type.tensor != first_type.tensor )
                {
                    return Error(zip.position, "item type mismatch in zip expression ('%s' vs '%s')",
                                 str(first_type.name).c_str(), str(item_type.name).c_str());
                }
            }
            return first_type;
        }
        
        static Result<Type> eval_type( const CoalesceExpr& coalesce, const Dict<Declaration>& decls, unsigned flags )
        {
            TRY_DECL(condition_type, eval_type(*coalesce.condition, decls, flags))
            TRY_DECL(alternate_type, eval_type(*coalesce.alternate, decls, flags))
            
            if ( !condition_type.optional )
            {
                return Error(coalesce.position, "expected optional type as first argument of operator '\?\?'; found '%s'",
                             str(condition_type).c_str());
            }
            if ( alternate_type.optional )
            {
                return Error(coalesce.position, "expected non-optional type as second argument of operator '\?\?'; found '%s'",
                             str(alternate_type).c_str());
            }
            if ( condition_type.name != alternate_type.name || condition_type.packed != alternate_type.packed ||
                (condition_type.tensor != alternate_type.tensor && !(flags & Flags::AllowTensorOperators)) )
            {
                return Error(coalesce.position, "argument type mismatch for operator '\?\?' ('%s' vs '%s')",
                             str(as_non_optional(condition_type)).c_str(), str(as_non_optional(alternate_type)).c_str());
            }
            
            return Type{ alternate_type.name, false, condition_type.tensor || alternate_type.tensor, alternate_type.packed };
        }
        
        static Result<Type> eval_type( const IdentityExpr& iden, const Dict<Declaration>& decls, unsigned flags )
        {
            TRY_DECL(left_type, eval_type(*iden.left, decls, flags))
            TRY_DECL(right_type, eval_type(*iden.right, decls, flags))
            
            if ( left_type.name != right_type.name || left_type.tensor != right_type.tensor )
            {
                return Error(iden.position, "argument type mismatch in operator 'is'; ('%s' vs '%s')",
                             str(left_type).c_str(), str(right_type).c_str());
            }
            
            return make_type(Typename::Bool, left_type.optional || left_type.optional, false, left_type.packed || right_type.packed);
        }
        
        static Result<Type> eval_type( const ContainExpr& contain, const Dict<Declaration>& decls, unsigned flags )
        {
            TRY_DECL(item_type, eval_type(*contain.item, decls, flags))
            TRY_DECL(pack_type, eval_type(*contain.pack, decls, flags))
            
            if ( item_type.name != pack_type.name || item_type.tensor != pack_type.tensor )
            {
                return Error(contain.position, "argument type mismatch in operator 'in'; ('%s' vs '%s')",
                             str(item_type).c_str(), str(pack_type).c_str());
            }
            if ( !pack_type.packed )
            {
                return Error(contain.position, "right argument of operator 'in' must be a packed type; found '%s'",
                             str(pack_type).c_str());
            }
            
            return make_type(Typename::Bool, item_type.optional || pack_type.optional, false, item_type.packed);
        }
        
        static Result<Type> eval_type( const FoldExpr& fold, const Dict<Declaration>& decls, unsigned flags )
        {
            if ( !Lexer::is_fold(fold.op, fold.cumulative) )
            {
                return Error(fold.position, "invalid %s fold operator '%s'", fold.cumulative ? "cumulative" : "", Lexer::str(fold.op));
            }
            
            TRY_DECL(pack_type, eval_type(*fold.pack, decls, flags))
            if ( !pack_type.packed )
            {
                return Error(fold.position, "expected packed type as argument of fold operator '%s'; found '%s'",
                             Lexer::str(fold.op), str(pack_type).c_str());
            }
            if ( pack_type.tensor && !(flags & Flags::AllowTensorOperators) )
            {
                return Error(fold.position, "operator '%s' is not allowed for tensors in this context", Lexer::str(fold.op));
            }
            
            bool is_arg = fold.op == Lexer::Operator::ArgMin || fold.op == Lexer::Operator::ArgMax;
            bool optional = (is_arg || fold.op == Lexer::Operator::Min || fold.op == Lexer::Operator::Max || fold.op == Lexer::Operator::MakeEqual) &&
                            !fold.cumulative;
            
            Typename name = Lexer::is_comparison(fold.op) ? Typename::Bool : is_arg ? Typename::Int : pack_type.name;
            return make_type(name, optional || pack_type.optional, pack_type.tensor, fold.cumulative);
        }
        
        static Result<Type> eval_type( const CastExpr& cast, const Dict<Declaration>& decls, unsigned flags )
        {
            if ( !cast.arg )
            {
                return Type{ cast.base, false, false, false };
            }
            
            TRY_DECL(arg_type, eval_type(*cast.arg, decls, flags))
            
            if ( !is_compatible(Typename::Arith, cast.base) || (!is_compatible(Typename::Arith, arg_type.name) && arg_type.name != Typename::Type) )
            {
                return Error(cast.position, "invalid cast from '%s' to '%s'",
                             str(arg_type.name).c_str(), str(cast.base).c_str());
            }
            if ( arg_type.tensor && !(flags & Flags::AllowTensorOperators) )
            {
                return Error(cast.position, "cast operator is not allowed for tensors in this context");
            }
            
            return Type{ cast.base, arg_type.optional, arg_type.tensor, arg_type.packed };
        }
        
        static Result<Type> eval_type( const BuiltinExpr& builtin, const Dict<Declaration>& decls, unsigned flags )
        {
            static const std::map<std::string,std::pair<Typename,Typename>> func_types =
            {
                { "abs", { Typename::Num, Typename::Num } },
                { "sign", { Typename::Num, Typename::Num } },
                { "sqrt", { Typename::Real, Typename::Real } },
                { "log", { Typename::Real, Typename::Real } },
                { "exp", { Typename::Real, Typename::Real } },
                { "sin", { Typename::Real, Typename::Real } },
                { "cos", { Typename::Real, Typename::Real } },
                { "tan", { Typename::Real, Typename::Real } },
                { "asin", { Typename::Real, Typename::Real } },
                { "acos", { Typename::Real, Typename::Real } },
                { "atan", { Typename::Real, Typename::Real } },
                { "sinh", { Typename::Real, Typename::Real } },
                { "cosh", { Typename::Real, Typename::Real } },
                { "tanh", { Typename::Real, Typename::Real } },
                { "asinh", { Typename::Real, Typename::Real } },
                { "acosh", { Typename::Real, Typename::Real } },
                { "atanh", { Typename::Real, Typename::Real } },
                { "erf", { Typename::Real, Typename::Real } },
                { "round", { Typename::Real, Typename::Real } },
                { "floor", { Typename::Real, Typename::Real } },
                { "ceil", { Typename::Real, Typename::Real } },
                { "frac", { Typename::Real, Typename::Real } },
            };
            
            TRY_DECL(arg_type, eval_type(*builtin.arg, decls, flags))
            
            auto it = func_types.find(builtin.func);
            if ( it == func_types.end() )
            {
                return Error(builtin.position, "undefined function '%s'", builtin.func.c_str());
            }
            auto& types = it->second;
            if ( !is_compatible(types.first, arg_type.name) )
            {
                return Error(builtin.position, "invalid argument type '%s' for function '%s'; expected '%s'",
                             str(arg_type.name).c_str(), builtin.func.c_str(), str(types.first).c_str());
            }
            if ( arg_type.tensor && !(flags & Flags::AllowTensorOperators) )
            {
                return Error(builtin.position, "function '%s' is not allowed for tensors in this context",
                             builtin.func.c_str());
            }
            
            return Type{ types.second == types.first ? arg_type.name : types.second, arg_type.optional, arg_type.tensor, arg_type.packed };
        }
        
        static Result<Type> eval_type( const FormatExpr& format, const Dict<Declaration>& decls )
        {
            bool optional = false;
            for ( auto& sub : format.subs )
            {
                TRY_DECL(type, eval_type(*sub.second, decls))
                if ( type.tensor )
                {
                    return Error(format.position, "arguments to string formatting must not be tensors");
                }
                optional |= type.optional;
            }
            return make_type(Typename::Str, optional, false, false);
        }
        
        static Result<Type> eval_type( const BoundedExpr& bounded, const Dict<Declaration>& decls, unsigned flags )
        {
            if ( !(flags & Flags::AllowBounded) )
            {
                return Error(bounded.position, "bounded expression not allowed in this context");
            }
            
            TRY_DECL(index_type, eval_type(*bounded.index, decls))
            
            if ( index_type.name != Typename::Int )
            {
                return Error(bounded.position, "index in bounded expression must be of type 'int', found '%s'",
                             str(index_type).c_str());
            }
            if ( bounded.lower_value )
            {
                TRY_DECL(lower_type, eval_type(*bounded.lower_value, decls))
                if ( lower_type.name != Typename::Int )
                {
                    return Error(bounded.position, "lower value in bounded expression must be of type 'int', found '%s'",
                                 str(lower_type).c_str());
                }
            }
            if ( bounded.upper_value )
            {
                TRY_DECL(upper_type, eval_type(*bounded.upper_value, decls))
                if ( upper_type.name != Typename::Int )
                {
                    return Error(bounded.position, "upper value in bounded expression must be of type 'int', found '%s'",
                                 str(upper_type).c_str());
                }
            }
            
            return index_type;
        }
        
        static Result<Type> eval_type( const SubstituteExpr& substitute, const Dict<Declaration>& decls, unsigned flags )
        {
            TRY_DECL(pack_type, eval_type(*substitute.pack, decls, flags & ~Flags::AllowTensorOperators))
            TRY_DECL(index_type, eval_type(*substitute.index, decls, flags & ~Flags::AllowTensorOperators))
            TRY_DECL(value_type, eval_type(*substitute.value, decls, flags & ~Flags::AllowTensorOperators))
            
            if ( !pack_type.packed )
            {
                return Error(substitute.position, "pack in substitution expression must be of packed type, found '%s'",
                             str(pack_type).c_str());
            }
            if ( index_type.name != Typename::Int )
            {
                return Error(substitute.position, "index in substitution expression must be of type 'int', found '%s'",
                             str(index_type).c_str());
            }
            if ( value_type.name != pack_type.name || value_type.tensor != pack_type.tensor )
            {
                return Error(substitute.position, "substituted value in substitution expression must match pack type, found '%s' vs '%s'",
                             str(value_type).c_str(), str(pack_type).c_str());
            }
            if ( index_type.packed && !value_type.packed )
            {
                return Error(substitute.position, "value in substitution expression must be packed if index is packed, found '%s' vs '%s'",
                             str(value_type).c_str(), str(index_type).c_str());
            }
            if ( !index_type.packed && value_type.packed )
            {
                return Error(substitute.position, "value in substitution expression must not be packed if index is not packed, found '%s' vs '%s'",
                             str(value_type).c_str(), str(index_type).c_str());
            }
            
            pack_type.optional |= index_type.optional || value_type.optional;
            return pack_type;
        }
        
    private:
        
        static Result<Shared<Expr>> eval_rank( const Expr& expr, const Dict<Declaration>& decls )
        {
            switch ( expr.kind )
            {
                case Expr::Literal:
                {
                    return Shared<Expr>();
                }
                case Expr::Identifier:
                {
                    auto& iden = as_identifier(expr);
                    auto it = decls.find(iden.name);
                    if ( it == decls.end() )
                    {
                        return Shared<Expr>();
                    }
                    return it->second.repeats;
                }
                case Expr::List:
                {
                    auto& list = as_list(expr);
                    return eval_items_rank(list.items, decls, list.position);
                }
                case Expr::Unary:
                {
                    auto& unary = as_unary(expr);
                    if ( unary.op == Lexer::Operator::Question )
                    {
                        return Shared<Expr>();
                    }
                    return eval_rank(*unary.arg, decls);
                }
                case Expr::Binary:
                {
                    auto& binary = as_binary(expr);
                    TRY_DECL(lhs_rank, eval_rank(*binary.left, decls))
                    TRY_DECL(rhs_rank, eval_rank(*binary.right, decls))
                    if ( lhs_rank && rhs_rank && !ranks_equal(*lhs_rank, *rhs_rank, decls) )
                    {
                        return Error(expr.position, "incompatible pack lengths in binary expression '%s' ('%s' vs '%s')",
                                     Lexer::str(binary.op), str(*lhs_rank).c_str(), str(*rhs_rank).c_str());
                    }
                    return lhs_rank ? lhs_rank : rhs_rank;
                }
                case Expr::Select:
                {
                    auto& select = as_select(expr);
                    TRY_DECL(cond_rank, eval_rank(*select.cond, decls))
                    TRY_DECL(lhs_rank, eval_rank(*select.left, decls))
                    TRY_DECL(rhs_rank, select.right ? eval_rank(*select.right, decls) : Shared<Expr>())
                    if ( cond_rank )
                    {
                        if ( lhs_rank && rhs_rank && !ranks_equal(*lhs_rank, *rhs_rank, decls) )
                        {
                            return Error(expr.position, "incompatible left and right pack lengths in '?:' expression ('%s' vs '%s')",
                                         str(*lhs_rank).c_str(), str(*rhs_rank).c_str());
                        }
                        auto value_rank = lhs_rank ? lhs_rank : rhs_rank;
                        if ( value_rank && !ranks_equal(*cond_rank, *value_rank, decls) )
                        {
                            return Error(expr.position, "incompatible condition and value pack lengths in '?:' expression ('%s' vs '%s')",
                                         str(*cond_rank).c_str(), str(*value_rank).c_str());
                        }
                        return cond_rank;
                    }
                    else
                    {
                        if ( !lhs_rank )
                        {
                            return rhs_rank;
                        }
                        else if ( !rhs_rank )
                        {
                            return lhs_rank;
                        }
                        else if ( ranks_equal(*lhs_rank, *rhs_rank, decls) )
                        {
                            return lhs_rank;
                        }
                        else
                        {
                            return (Shared<Expr>)std::make_shared<SelectExpr>(select.position, select.cond, lhs_rank, rhs_rank);
                        }
                    }
                }
                case Expr::Expand:
                {
                    auto& expand = as_expand(expr);
                    return expand.count ? expand.count : eval_rank(*expand.item, decls);
                }
                case Expr::Index:
                {
                    auto& index = as_index(expr);
                    TRY_DECL(array_rank, eval_rank(*index.array, decls))
                    if ( !array_rank )  // string indexing
                    {
                        return Shared<Expr>();
                    }
                    TRY_DECL(index_type, eval_type(*index.index, decls))
                    if ( index_type.name == Typename::Bool )
                    {
                        TRY_DECL(index_rank, eval_rank(*index.index, decls))
                        if ( !ranks_equal(*index_rank, *array_rank, decls) )
                        {
                            return Error(expr.position, "incompatible mask length and pack length (%s vs %s)",
                                         str(*index_rank).c_str(), str(*array_rank).c_str());
                        }
                        auto as_int = (Shared<Expr>)std::make_shared<CastExpr>(index.position, Typename::Int, index.index);
                        return (Shared<Expr>)std::make_shared<FoldExpr>(index.position, as_int, Lexer::Operator::Plus);
                    }
                    else if ( index.index->kind == Expr::Range )
                    {
                        auto& range = as_range(*index.index);
                        
                        const int_t stride_value = !range.stride ? 1 : range.stride->kind == Expr::Literal && as_literal(*range.stride).type.name == Typename::Int ? as_int(*range.stride).value : 0;
                        
                        if ( stride_value < 0 )
                        {
                            if ( !range.first && !range.last )
                            {
                                return array_rank;
                            }
                            auto minus_one = std::make_shared<IntExpr>(range.position, -1);
                            auto first = range.first ? range.first :
                                std::make_shared<BinaryExpr>(range.position, array_rank, minus_one, Lexer::Operator::Plus);
                            if ( range.first && is_const_expr(*first) )
                            {
                                TRY_DECL(value, Evaluation::eval(*first, {}))
                                if ( value.as_int() < 0 )
                                {
                                    first = std::make_shared<BinaryExpr>(range.position, array_rank, first, Lexer::Operator::Plus);
                                }
                            }
                            auto last = range.last ? range.last : minus_one;
                            if ( range.last && is_const_expr(*last) )
                            {
                                TRY_DECL(value, Evaluation::eval(*last, {}))
                                if ( value.as_int() < 0 )
                                {
                                    last = std::make_shared<BinaryExpr>(range.position, array_rank, last, Lexer::Operator::Plus);
                                }
                            }
                            auto expr = (Shared<Expr>)std::make_shared<BinaryExpr>(range.position, first, last, Lexer::Operator::Minus);
                            if ( stride_value == -1 )
                            {
                                return expr;
                            }
                            auto stride = (Shared<Expr>)std::make_shared<IntExpr>(range.position, -stride_value);
                            return (Shared<Expr>)std::make_shared<BinaryExpr>(range.position, expr, stride, Lexer::Operator::Divide);
                        }
                        else
                        {
                            auto first = range.first;
                            if ( range.first && is_const_expr(*first) )
                            {
                                TRY_DECL(value, Evaluation::eval(*first, {}))
                                if ( value.as_int() < 0 )
                                {
                                    first = std::make_shared<BinaryExpr>(range.position, first, array_rank, Lexer::Operator::Plus);
                                }
                            }
                            auto last = range.last ? range.last : array_rank;
                            if ( range.last && is_const_expr(*last) )
                            {
                                TRY_DECL(value, Evaluation::eval(*last, {}))
                                if ( value.as_int() < 0 )
                                {
                                    last = std::make_shared<BinaryExpr>(range.position, last, array_rank, Lexer::Operator::Plus);
                                }
                            }
                            auto expr = !first ? last :
                                std::make_shared<BinaryExpr>(range.position, last, first, Lexer::Operator::Minus);
                            return stride_value == 1 ? expr :
                                std::make_shared<BinaryExpr>(range.position, expr, range.stride, Lexer::Operator::Divide);
                        }
                    }
                    return eval_rank(*index.index, decls);
                }
                case Expr::Access:
                {
                    auto& access = as_access(expr);
                    TRY_DECL(tensor_rank, eval_rank(*access.tensor, decls))
                    if ( tensor_rank )
                    {
                        return tensor_rank;
                    }
                    
                    if ( access.tensor )
                    for ( auto& index : access.indices )
                    {
                        if ( index->kind != Expr::Expand )
                        {
                            TRY_DECL(index_rank, eval_rank(*index, decls))
                            if ( index_rank )
                            {
                                return index_rank;
                            }
                        }
                    }
                    return Shared<Expr>();
                }
                case Expr::Range:
                {
                    auto& range = as_range(expr);
                    if ( range.first )
                    {
                        TRY_CALL(eval_rank(*range.first, decls))
                    }
                    if ( range.last )
                    {
                        TRY_CALL(eval_rank(*range.last, decls))
                    }
                    if ( range.stride )
                    {
                        TRY_CALL(eval_rank(*range.stride, decls))
                    }
                    const Shared<Expr> expr = range.first && range.last ? std::make_shared<BinaryExpr>(range.position, range.last, range.first, Lexer::Operator::Minus) : range.last;
                    return range.stride && expr ? std::make_shared<BinaryExpr>(range.position, expr, range.stride, Lexer::Operator::Divide) : expr;
                }
                case Expr::Zip:
                {
                    auto& zip = as_zip(expr);
                    TRY_DECL(first_rank, eval_rank(*zip.items.front(), decls))
                    for ( size_t i = 1; i < zip.items.size(); ++i )
                    {
                        TRY_DECL(item_rank, eval_rank(*zip.items[i], decls))
                        if ( first_rank && item_rank && !ranks_equal(*first_rank, *item_rank, decls) )
                        {
                            return Error(expr.position, "incompatible pack lengths in zip expression ('%s' vs '%s')",
                                         str(*first_rank).c_str(), str(*item_rank).c_str());
                        }
                    }
                    auto count = std::make_shared<IntExpr>(zip.position, zip.items.size());
                    return (Shared<Expr>)std::make_shared<BinaryExpr>(zip.position, first_rank, count, Lexer::Operator::Multiply);
                }
                case Expr::Coalesce:
                {
                    auto& coalesce = as_coalesce(expr);
                    TRY_DECL(condition_rank, eval_rank(*coalesce.condition, decls))
                    TRY_DECL(alternate_rank, eval_rank(*coalesce.alternate, decls))
                    if ( !condition_rank )
                    {
                        return alternate_rank;
                    }
                    else if ( !alternate_rank )
                    {
                        return condition_rank;
                    }
                    else if ( ranks_equal(*condition_rank, *alternate_rank, decls) )
                    {
                        return condition_rank;
                    }
                    else
                    {
                        auto condition = std::make_shared<UnaryExpr>(coalesce.position, coalesce.condition, Lexer::Operator::Question);
                        return (Shared<Expr>)std::make_shared<SelectExpr>(coalesce.position, condition, condition_rank, alternate_rank);
                    }
                }
                case Expr::Identity:
                {
                    auto& identity = as_identity(expr);
                    TRY_DECL(lhs_rank, eval_rank(*identity.left, decls))
                    TRY_DECL(rhs_rank, eval_rank(*identity.right, decls))
                    if ( lhs_rank && rhs_rank && !ranks_equal(*lhs_rank, *rhs_rank, decls) )
                    {
                        return Error(expr.position, "incompatible pack lengths in 'is' expression ('%s' vs '%s')",
                                     str(*lhs_rank).c_str(), str(*rhs_rank).c_str());
                    }
                    return lhs_rank ? lhs_rank : rhs_rank;
                }
                case Expr::Contain:
                {
                    auto& contain = as_contain(expr);
                    TRY_DECL(item_rank, eval_rank(*contain.item, decls))
                    TRY_CALL(eval_rank(*contain.pack, decls))
                    return item_rank;
                }
                case Expr::Fold:
                {
                    auto& fold = as_fold(expr);
                    TRY_DECL(rank, eval_rank(*fold.pack, decls))
                    return fold.cumulative ? rank : Shared<Expr>();
                }
                case Expr::Cast:
                {
                    auto& cast = as_cast(expr);
                    return cast.arg ? eval_rank(*cast.arg, decls) : Shared<Expr>();
                }
                case Expr::Builtin:
                {
                    auto& builtin = as_builtin(expr);
                    return eval_rank(*builtin.arg, decls);
                }
                case Expr::Format:
                {
                    auto& format = as_format(expr);
                    for ( auto& sub : format.subs )
                    {
                        TRY_CALL(eval_rank(*sub.second, decls))
                    }
                    return Shared<Expr>();
                }
                case Expr::Bounded:
                {
                    auto& bounded = as_bounded(expr);
                    TRY_DECL(index_rank, eval_rank(*bounded.index, decls))
                    
                    if ( bounded.lower_value )
                    {
                        TRY_DECL(lower_rank, eval_rank(*bounded.lower_value, decls))
                        if ( index_rank && lower_rank && !ranks_equal(*index_rank, *lower_rank, decls) )
                        {
                            return Error(expr.position, "incompatible pack lengths in bounded expression for index and lower value ('%s' vs '%s')",
                                         str(*index_rank).c_str(), str(*lower_rank).c_str());
                        }
                    }
                    if ( bounded.upper_value )
                    {
                        TRY_DECL(upper_rank, eval_rank(*bounded.upper_value, decls))
                        if ( index_rank && upper_rank && !ranks_equal(*index_rank, *upper_rank, decls) )
                        {
                            return Error(expr.position, "incompatible pack lengths in bounded expression for index and upper value ('%s' vs '%s')",
                                         str(*index_rank).c_str(), str(*upper_rank).c_str());
                        }
                    }
                    return index_rank;
                }
                case Expr::Substitute:
                {
                    auto& substitute = as_substitute(expr);
                    TRY_DECL(pack_rank, eval_rank(*substitute.pack, decls))
                    TRY_DECL(index_rank, eval_rank(*substitute.index, decls))
                    TRY_DECL(value_rank, eval_rank(*substitute.value, decls))
                    
                    if ( index_rank && value_rank && !ranks_equal(*index_rank, *value_rank, decls) )
                    {
                        return Error(expr.position, "incompatible pack lengths in substitution expression for index and value ('%s' vs '%s')",
                                     str(*index_rank).c_str(), str(*value_rank).c_str());
                    }
                    
                    return pack_rank;
                }
            }
        }
        
        static Result<Shared<Expr>> eval_items_rank( const std::vector<Shared<Expr>>& items, const Dict<Declaration>& decls,
                                                    const Position& position )
        {
            int_t constant = 0;
            for ( auto& item : items )
            {
                if ( item->kind != Expr::Expand && item->kind != Expr::Range )
                {
                    ++constant;
                }
            }
            
            Shared<Expr> rank = constant ? std::make_shared<IntExpr>(position, constant) : nullptr;
            for ( auto& item : items )
            {
                if ( item->kind == Expr::Expand || item->kind == Expr::Range )
                {
                    TRY_DECL(item_rank, eval_rank(*item, decls))
                    if ( item_rank )
                    {
                        rank = rank ? std::make_shared<BinaryExpr>(position, rank, item_rank, Lexer::Operator::Plus) : item_rank;
                    }
                }
            }
            return rank ? rank : std::make_shared<IntExpr>(position, 0);
        }
        
        Shared<Expr> known_items_rank( const ListExpr& list ) const
        {
            int_t constant = 0;
            for ( auto& item : list.items )
            {
                if ( item->kind != Expr::Expand )
                {
                    ++constant;
                }
            }
            
            Shared<Expr> items_rank = constant ? std::make_shared<IntExpr>(list.position, constant) : nullptr;
            for ( auto& item : list.items )
            {
                if ( item->kind == Expr::Expand )
                {
                    auto count = as_expand(*item).count;
                    if ( count )
                    {
                        items_rank = items_rank ? std::make_shared<BinaryExpr>(list.position, items_rank, count, Lexer::Operator::Plus) : count;
                    }
                }
            }
            return items_rank;
        }
        
        Shared<Expr> flexible_item_rank( const ListExpr& list, const Shared<Expr> rank ) const
        {
            auto items_rank = known_items_rank(list);
            return items_rank ? std::make_shared<BinaryExpr>(list.position, rank, items_rank, Lexer::Operator::Minus) : rank;
        }
        
        static SymbolTypes symbol_types( const Dict<Declaration>& decls )
        {
            return [&]( const std::string& name )
            {
                return decls.at(name).type.name;
            };
        }
        
        static bool ranks_equal( const Expr& left, const Expr& right, const Dict<Declaration>& decls )
        {
#if ENABLE_SYMBOLIC_CHECKS
            Symbolic symbolic;
            SymbolTypes types = symbol_types(decls);
            return symbolic.eval_polynom<int_t>(left, types) == symbolic.eval_polynom<int_t>(right, types);
#else
            return true;
#endif
        }
        
        static bool rank_divisible( const Expr& rank, const int_t divisor, const Dict<Declaration>& decls )
        {
#if ENABLE_SYMBOLIC_CHECKS
            Symbolic symbolic;
            SymbolTypes types = symbol_types(decls);
            return symbolic.eval_polynom<int_t>(rank, types).is_divisible(divisor);
#else
            return true;
#endif
        }
        
        static bool is_always_true( const Expr& expr, const Dict<Declaration>& decls )
        {
            Symbolic symbolic;
            SymbolTypes types = symbol_types(decls);
            auto poly = symbolic.eval_polynom<bool_t>(expr, types);
            return poly == true;
        }
        
    public:
        
        static bool is_deferred_attrib( const Param& param )
        {
            return param.repeats || param.type.optional || (param.default_value && !is_const_expr(*param.default_value));
        }
        
        static Result<std::vector<size_t>> deduction_order( const std::vector<Param>& attribs, const std::vector<Param>& inputs,
                                                           const std::vector<Using>& usings, const std::string& op_name )
        {
            std::set<std::string> symbols;
            for ( auto& param : attribs )
            {
                if ( !is_deferred_attrib(param) )
                {
                    symbols.insert(param.name);
                }
            }
            
            for ( auto& usage : usings )
            {
                if ( !has_unknown_symbols(*usage.expr, symbols) )
                {
                    symbols.insert(as_identifier(*usage.identifier).name);
                }
            }
            
            std::vector<size_t> order(inputs.size());
            std::iota(order.begin(), order.end(), 0);
            
            size_t k = 0;
            for ( size_t i = 0; i < inputs.size(); ++i )
            {
                auto& input = inputs[order[i]];
                if ( input.shape )
                {
                    auto& shape = *input.shape;
                    if ( can_deduce(shape, symbols) )
                    {
                        add_symbols(shape, symbols);
                        rotate_right(order.begin() + k, order.begin() + i + 1);
                        i = k++;
                    }
                }
            }
            
            for ( size_t i = k; i < inputs.size(); ++i )
            {
                auto& input = inputs[order[i]];
                if ( input.shape )
                {
                    auto& shape = *input.shape;
                    if ( !can_deduce(shape, symbols) )
                    {
                        return Error(input.position, "deduction of shape components may not be possible for input '%s' of operator '%s'",
                                     input.name.c_str(), op_name.c_str());
                    }
                }
            }
            
            for ( auto& attrib : attribs )
            {
                if ( attrib.default_value && has_unknown_symbols(*attrib.default_value, symbols) )
                {
                    return Error(attrib.position, "deduction of default value may not be possible for attribue '%s' of operator '%s'",
                                 attrib.name.c_str(), op_name.c_str());
                }
                if ( attrib.type.optional )
                {
                    symbols.insert(attrib.name);
                }
            }
            
            for ( auto& attrib : attribs )
            {
                if ( !attrib.type.optional && attrib.repeats )
                {
                    auto& iden = find_affine_id(*attrib.repeats);
                    if ( !iden.empty() )
                    {
                        symbols.insert(iden);
                    }
                }
            }
            
            return order;
        }
        
        static const std::string& find_affine_id( const Expr& expr )
        {
            static std::string empty;
            if ( expr.kind == Expr::Identifier )
            {
                return as_identifier(expr).name;
            }
            if ( expr.kind == Expr::Binary )
            {
                auto& binary = as_binary(expr);
                if ( binary.op == Lexer::Operator::Multiply || binary.op == Lexer::Operator::Plus || binary.op == Lexer::Operator::Minus )
                {
                    if ( is_const_expr(*binary.left) )
                    {
                        return find_affine_id(*binary.right);
                    }
                    else if ( is_const_expr(*binary.right) )
                    {
                        return find_affine_id(*binary.left);
                    }
                }
            }
            return empty;
        }
        
        static bool is_affine_expr( const Expr& expr )
        {
            return !find_affine_id(expr).empty();
        }
        
        static bool is_static_expr( const Expr& expr, const Dict<Declaration>& decls )
        {
            if ( expr.kind == Expr::Identifier )
            {
                auto it = decls.find(as_identifier(expr).name);
                return it != decls.end() && !it->second.type.tensor &&
                    !(it->second.flags & Declaration::TensorIndex) &&
                    !(it->second.flags & Declaration::LoopLocal);
            }
            else if ( expr.kind == Expr::Unary )
            {
                if ( as_unary(expr).op == Lexer::Operator::Question )
                {
                    return true;
                }
            }
            return all_recurse(expr, [&]( const Expr& e ){ return is_static_expr(e, decls); });
        }
        
        template<typename Symbols, bool (*IsStatic)(const Expr&, const Symbols&) = is_static_expr>
        static std::string auto_label( const Component& component, const Symbols& symbols )
        {
            if ( component.results.size() == 1 && !component.results.front().packed() && !component.results.front()->name.empty() &&
                has_single_callable<Symbols,IsStatic>(component, symbols) )
            {
                const Packable<Typed>& result = component.results.front();
                return result->name;
            }
            else
            {
                return {};
            }
        }
        
        static bool is_callable( const Callable& callable )
        {
            return callable.is<Invocation>() || !callable.as<Region>().components.empty();
        }
        
        template<typename Symbols, bool (*IsStatic)(const Expr&, const Symbols&) = is_static_expr>
        static bool has_single_callable( const Component& component, const Symbols& symbols )
        {
            if ( component.loop )
            {
                return !component.loop->condition || !component.loop->condition->is<Invocation>();
            }
            else if ( !component.branches.empty() )
            {
                size_t dynamic_count = 0;
                size_t callable_count = is_callable(component.operation) ? 1 : 0;
                for ( auto& [condition, consequent] : component.branches )
                {
                    if ( is_callable(condition) || !IsStatic(*condition.template as<Region>().yields.front(), symbols) )
                    {
                        ++dynamic_count;
                    }
                    
                    if ( is_callable(condition) )
                    {
                        ++callable_count;
                    }
                    if ( is_callable(consequent) )
                    {
                        ++callable_count;
                    }
                }
                return dynamic_count == 0 || callable_count <= 1;
            }
            else
            {
                return true;
            }
        }
        
    private:
        
        static size_t flexible_items( const Shapedef& shape, const std::set<std::string>& symbols )
        {
            size_t flexibles = 0;
            for ( auto& item : shape.extents )
            {
                if ( item && item->kind == Expr::Expand )
                {
                    auto id = rank_id(as_expand(*item), symbols);
                    if ( !id.empty() && !symbols.count(id) )
                    {
                        ++flexibles;
                    }
                }
            }
            return flexibles;
        }
        
        static bool can_deduce( const Shapedef& shape, const std::set<std::string>& symbols )
        {
            if ( flexible_items(shape, symbols) > 1 )
            {
                return false;
            }
            
            auto local_symbols = symbols;
            for ( auto item : shape.extents )
            {
                if ( item )
                {
                    item = unwrapped(item);
                    auto& iden = find_affine_id(*item);
                    if ( !iden.empty() )
                    {
                        local_symbols.insert(iden);
                    }
                }
            }
            
            for ( auto item : shape.extents )
            {
                if ( item )
                {
                    item = unwrapped(item);
                    if ( !is_affine_expr(*item) && has_unknown_symbols(*item, local_symbols) )
                    {
                        return false;
                    }
                }
            }
            return true;
        }
        
        template<typename S>
        static bool has_unknown_symbols( const Expr& expr, const S& symbols )
        {
            return any_of(expr, [&]( const Expr& e )
            {
                return e.kind == Expr::Identifier && !symbols.count(as_identifier(e).name);
            });
        }
        
        static void add_symbols( const Shapedef& shape, std::set<std::string>& symbols )
        {
            for ( auto extent : shape.extents )
            {
                if ( extent )
                {
                    if ( extent->kind == Expr::Expand )
                    {
                        auto& expand = as_expand(*extent);
                        auto id = rank_id(expand, symbols);
                        if ( !id.empty() )
                        {
                            symbols.insert(id);
                        }
                        extent = expand.item;
                    }
                    if ( extent->kind == Expr::Identifier )
                    {
                        symbols.insert(as_identifier(*extent).name);
                    }
                }
            }
        }
        
        static std::string rank_id( const ExpandExpr& expand, const std::set<std::string>& symbols )
        {
            if ( expand.count )
            {
                return find_affine_id(*expand.count);
            }
            else
            {
                auto& iden = find_affine_id(*expand.item);
                return !iden.empty() ? "|" + iden + "|" : "";
            }
        }
        
        static bool can_check( const Assert& assert, const Dict<Declaration>& decls )
        {
            for ( auto& item : assert.prints )
            {
                if ( has_unknown_symbols(*item.second, decls) )
                {
                    return false;
                }
            }
            if ( assert.message && has_unknown_symbols(*assert.message, decls) )
            {
                return false;
            }
            return !has_unknown_symbols(*assert.expression, decls);
        }
        
        template<typename It>
        static void rotate_right( It first, It last )
        {
            std::rotate(std::make_reverse_iterator(last), std::make_reverse_iterator(last - 1), std::make_reverse_iterator(first));
        }
        
        static void enum_promoted_optionals( const Expr& expr, std::vector<std::string>& ids )
        {
            if ( expr.kind == Expr::Unary )
            {
                auto& unary = as_unary(expr);
                if ( unary.op == Lexer::Operator::Question )
                {
                    auto& iden = as_identifier(*unary.arg);
                    ids.push_back(iden.name);
                }
            }
            else if ( expr.kind == Expr::Binary )
            {
                auto& binary = as_binary(expr);
                if ( binary.op == Lexer::Operator::And )
                {
                    enum_promoted_optionals(*binary.left, ids);
                    enum_promoted_optionals(*binary.right, ids);
                }
            }
        }
        
    private:
        
        void report_error( const Error& error ) const
        {
            _error(error.position, error.message, error.trace, false);
        }
        
        template<typename... Args>
        void report_error( const Position& position, const char* format, Args&&... args ) const
        {
            _error(position, Error::format_string(format, std::forward<Args>(args)...), {}, false);
        }
        
        template<typename... Args>
        void report_warning( const Position& position, const char* format, Args&&... args ) const
        {
            _error(position, Error::format_string(format, std::forward<Args>(args)...), {}, true);
        }
        
    private:
        
        const ErrorCallback _error;
    };
    
}   // namespace sknd


#endif
