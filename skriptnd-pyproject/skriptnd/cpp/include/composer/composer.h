#ifndef _SKND_COMPOSER_H_
#define _SKND_COMPOSER_H_

#include <stack>
#include <unordered_set>
#include "evaluation.h"
#include "operator.h"
#include "typing.h"
#include "print.h"


template<typename T>
struct std::hash<std::vector<T>>
{
    std::size_t operator()( const std::vector<T>& items ) const noexcept
    {
        std::hash<T> hasher;
        size_t hash = 0;
        for ( auto& item : items )
        {
            hash ^= hasher(item) + 0x9e3779b9 + (hash<<6) + (hash>>2);
        }
        return hash;
    }
};


namespace sknd
{
    
    class Composer : private Evaluation
    {
        using Evaluation::has_undefined_symbols;
        
    public:
        
        enum Flags : unsigned
        {
            EliminateTrivialLoops = 0x01,
            EliminateTrivialLocals = 0x02,
            EliminateTrivialBounded = 0x04,
        };
        
        static const unsigned DefaultFlags = EliminateTrivialLoops | EliminateTrivialLocals | EliminateTrivialBounded;
        
    private:
        
        AsTensor as_tensor( Graph& graph )
        {
            return [&graph,this]( const ValueExpr& value, const Typename& type ) -> TensorRef
            {
                if ( is_literal(value) )
                {
                    return make_constant(graph, value, type);
                }
                else
                {
                    TensorRef output = make_tensor(graph, type, {}, {});
                    _contexts.top().exprs.emplace(output, value);
                    graph.operations.push_back(Operation{ "", {}, { std::make_pair("", value) }, {}, { output } });
                    return output;
                }
            };
        }
        
        AsTensorPack as_tensor_pack( Graph& graph )
        {
            return [&graph,this]( const Tensors& tensors, const Typename& dtype, const Shape& shape, const std::vector<int_t>& max_shape,
                                 const ValueExpr& size ) -> TensorRef
            {
                auto& packs = _contexts.top().packs;
                auto it = packs.find(tensors);
                if ( it != packs.end() )
                {
                    return TensorRef(it->second);
                }
                auto pack = make_tensor_pack(graph, dtype, tensors.size(), size, shape, max_shape);
                for ( size_t i = 0; i < tensors.size(); ++i )
                {
                    pack->items[i] = tensors[i];
                }
                cache_tensor_pack(pack);
                return TensorRef(pack);
            };
        }
        
    public:
        
        Composer( const ErrorCallback error, 
                 const OperationCallback atomic = nullptr,
                 const OperationCallback unroll = nullptr,
                 const unsigned flags = DefaultFlags )
        : _error(error), _flags(flags), _atomic(atomic ? atomic : FalseOperationCallback), _unroll(unroll ? unroll : FalseOperationCallback) {}
        
        Result<Model> operator()( const Dict<Operator>& operators, const std::string& graph_name,
                                 const Dict<ValueExpr>& attribs = {}, const Dict<Typename>& dtypes = {} )
        {
            reset();
            
            const Operator& main = operators.at(graph_name);
            
            Dict<Symbol> symbols;
            
            for ( auto& param : main.dtypes )
            {
                auto it = dtypes.find(param.name);
                if ( it != dtypes.end() )
                {
                    symbols.emplace(param.name, Symbol(ValueExpr(nullptr), it->second));
                }
                else if ( param.default_type )
                {
                    symbols.emplace(param.name, Symbol(ValueExpr(nullptr), *param.default_type));
                }
                else
                {
                    return Error(param.position, "value of generic type '%s' must be supplied", param.name.c_str());
                }
            }
            
            for ( auto& param : main.attribs )
            {
                auto it = attribs.find(param.name);
                if ( it != attribs.end() )
                {
                    auto type = resolve_type(param, symbols);
                    symbols.emplace(param.name, Symbol(it->second, type));
                }
                else if ( param.default_value )
                {
                    auto type = resolve_type(param, symbols);
                    TRY_DECL(value, eval(*param.default_value, {}))
                    symbols.emplace(param.name, Symbol(value, type));
                }
                else
                {
                    return Error(param.position, "value of attribute '%s' must be supplied", param.name.c_str());
                }
            }
            
            TRY_CALL(add_placeholder_symbols(main.inputs, symbols))
            
            const std::string scope = graph_name + ".";
            
            Model model = { main.name };
            new_graph(model, graph_name);
            
            for ( auto& param : main.inputs )
            {
                auto type = resolve_type(param, symbols);
                TRY_DECL(tensor, make_tensors_for_param(model.graphs.front(), param, symbols, type, scope, false))
                symbols.emplace(param.name, Symbol(tensor, type));
                add_shape_symbols(param.name, tensor.shape(), tensor.size_or_null(), symbols);
            }
            
            TRY_CALL(replace_placeholder_symbols(main.inputs, symbols))
            
            for ( auto& param : main.constants )
            {
                auto type = resolve_type(param, symbols);
                TRY_DECL(tensor, make_tensors_for_param(model.graphs.front(), param, symbols, type, scope, false))
                symbols.emplace(param.name, Symbol(tensor, type));
                add_shape_symbols(param.name, tensor.shape(), tensor.size_or_null(), symbols);
            }
            for ( auto& param : main.variables )
            {
                auto type = resolve_type(param, symbols);
                TRY_DECL(tensor, make_tensors_for_param(model.graphs.front(), param, symbols, type, scope, true))
                symbols.emplace(param.name, Symbol(tensor, type));
                add_shape_symbols(param.name, tensor.shape(), tensor.size_or_null(), symbols);
            }
            
            std::vector<Assertion> dynamic_asserts;
            std::vector<bool> checked(main.asserts.size(), false);
            for ( const Using& usage : main.usings )
            {
                TRY_CALL(check_asserts(main.asserts, symbols, main.position, checked, dynamic_asserts))
                TRY_CALL(eval_using(usage, symbols))
            }
            TRY_CALL(check_asserts(main.asserts, symbols, main.position, checked, dynamic_asserts, true))
            
            const bool propagate_label = can_propagate_label(main, symbols);
            for ( auto& component : main.components )
            {
                TRY_CALL(compose(component, operators, symbols, model, 0, scope, propagate_label))
            }
            
            TRY_CALL(add_placeholder_symbols(main.outputs, symbols))
            
            auto& graph = model.graphs.front();
            graph.inputs = list_tensors(main.inputs, symbols);
            graph.outputs = list_tensors(main.outputs, symbols);
            graph.asserts = std::move(dynamic_asserts);
            
            TRY_CALL(check_outputs(main.outputs, graph.outputs, symbols))
            
            if ( !main.quantizations.empty() )
            {
                TRY_CALL(eval_quantization(graph, main.quantizations, symbols, operators))
            }
            
            return model;
        }
        
        void reset()
        {
            _contexts = {};
            _contexts.push({});
            _subgraphs.clear();
            _dereferenced.clear();
            _trace.clear();
            _next_tensor_idx = 1;
            _next_graph_idx = 1;
            _next_pack_idx = 1;
            _next_local_idx = 1;
            _next_placeholder_idx = 1;
        }
        
    private:
        
        Graph& new_graph( Model& model, const std::string& name )
        {
            model.graphs.push_back(Graph{ name });
            return model.graphs.back();
        }
        
        Result<void> add_placeholder_symbols( const std::vector<Param>& params, Dict<Symbol>& symbols )
        {
            for ( auto& param : params )
            {
                if ( param.shape )
                {
                    for ( size_t i = 0; i < param.shape->extents.size(); ++i )
                    {
                        bool spread = param.shape->spreads & (1 << i);
                        auto& extent = param.shape->extents[i];
                        auto& bound = param.shape->bounds[i];
                        if ( bound )
                        {
                            TRY_CALL(add_placeholder_symbol(*extent, *bound, symbols, spread ? param.repeats : nullptr))
                        }
                    }
                }
                if ( param.repeats && param.repeats_bound )
                {
                    TRY_CALL(add_placeholder_symbol(*param.repeats, *param.repeats_bound, symbols))
                }
            }
            return {};
        }
        
        Result<void> add_placeholder_symbol( const Expr& extent, const Expr& bound, Dict<Symbol>& symbols, const Shared<Expr>& repeats = nullptr )
        {
            auto count = extent.kind == Expr::Expand ? as_expand(extent).count : repeats;
            auto& iden = as_identifier(expanded(extent)).name;
            auto it = symbols.find(iden);
            if ( it == symbols.end() )
            {
                if ( count )
                {
                    TRY_DECL(size, eval(*count, symbols))
                    std::vector<ValueExpr> items(size.as_int());
                    for ( size_t i = 0; i < items.size(); ++i )
                    {
                        TRY_DECL(max_value, eval(bound, symbols, i))
                        items[i] = ValueExpr::placeholder(next_placeholder_name(), max_value);
                    }
                    symbols.emplace(iden, Symbol(ValueExpr::list(std::move(items), Typename::Int), Typename::Int));
                }
                else
                {
                    TRY_DECL(max_value, eval(bound, symbols))
                    auto value = ValueExpr::placeholder(next_placeholder_name(), max_value);
                    symbols.emplace(iden, Symbol(value, Typename::Int));
                }
            }
            else
            {
                auto& symbol = it->second;
                assert(symbol.is<ValueExpr>());
                auto& value = symbol.as<ValueExpr>();
                if ( count )
                {
                    TRY_DECL(new_max_value, eval(bound, symbols))
                    if ( !match_max_value(value.as_list(), new_max_value) )
                    {
                        auto old_max_value = get_max_value(value.as_list());
                        return Error(extent.position, "symbol '%s' was previously defined with a different upper bound (%s vs %s)",
                                     iden.c_str(), str(old_max_value).c_str(), str(new_max_value).c_str());
                    }
                }
                else
                {
                    auto& old_max_value = value.as_placeholder().max_value;
                    TRY_DECL(new_max_value, eval(bound, symbols))
                    if ( new_max_value != old_max_value )
                    {
                        return Error(extent.position, "symbol '%s' was previously defined with a different upper bound (%s vs %s)",
                                     iden.c_str(), str(old_max_value).c_str(), str(new_max_value).c_str());
                    }
                }
            }
            return {};
        }
        
        bool match_max_value( const std::vector<ValueExpr>& placeholders, const ValueExpr& values )
        {
            if ( values.packed() && placeholders.size() != values.max_size() )
            {
                return false;
            }
            for ( size_t i = 0; i < placeholders.size(); ++i )
            {
                if ( placeholders[i].as_placeholder().max_value != values.at(i) )
                {
                    return false;
                }
            }
            return true;
        }
        
        ValueExpr get_max_value( const std::vector<ValueExpr>& placeholders )
        {
            if ( placeholders.size() > 0 )
            {
                auto& first_value = placeholders.front().as_placeholder().max_value;
                if ( std::all_of(placeholders.begin() + 1, placeholders.end(),
                                 [&]( const ValueExpr& x ){ return x.as_placeholder().max_value == first_value; }) )
                {
                    return first_value;
                }
            }
            
            std::vector<ValueExpr> items(placeholders.size());
            for ( size_t i = 0; i < items.size(); ++i )
            {
                items[i] = placeholders[i].as_placeholder().max_value;
            }
            return ValueExpr::list(std::move(items), Typename::Int);
        }
        
        Result<void> replace_placeholder_symbols( const std::vector<Param>& params, Dict<Symbol>& symbols )
        {
            for ( auto& param : params )
            {
                auto& tensor = symbols.at(param.name).as<TensorRef>();
                
                size_t dim = 0;
                for ( size_t i = 0; i < param.shape->extents.size(); ++i )
                {
                    bool spread = param.shape->spreads & (1 << i);
                    auto& extent = param.shape->extents[i];
                    auto& bound = param.shape->bounds[i];
                    if ( bound )
                    {
                        replace_placeholder_symbol(*extent, tensor, dim, symbols, spread);
                    }
                    if ( extent->kind == Expr::Expand )
                    {
                        auto count = as_expand(*extent).count;
                        TRY_DECL(size, eval(*count, symbols))
                        dim += size.as_int();
                    }
                    else
                    {
                        dim += 1;
                    }
                }
                if ( param.repeats && param.repeats_bound )
                {
                    replace_placeholder_symbol(*param.repeats, tensor, std::nullopt, symbols);
                }
            }
            return {};
        }
        
        void replace_placeholder_symbol( const Expr& extent, const TensorRef& tensor, const std::optional<size_t> dim, Dict<Symbol>& symbols,
                                        bool spread = false )
        {
            auto& iden = as_identifier(expanded(extent)).name;
            auto& value = symbols.at(iden).as<ValueExpr>();
            if ( value.is_placeholder() )
            {
                if ( dim )
                {
                    value = ValueExpr(ValueExpr::ShapeAccessExpr{ tensor, ValueExpr((int_t)*dim) });
                }
                else
                {
                    value = ValueExpr(ValueExpr::SizeAccessExpr{ tensor });
                }
            }
            else if ( value.is_list() )
            {
                if ( spread )
                {
                    size_t k = 0;
                    for ( auto& item : value.as_list() )
                    {
                        item = ValueExpr(ValueExpr::ShapeAccessExpr{ TensorRef((Tensor*)&tensor[k++]), ValueExpr((int_t)*dim) });
                    }
                }
                else
                {
                    size_t k = *dim;
                    for ( auto& item : value.as_list() )
                    {
                        item = ValueExpr(ValueExpr::ShapeAccessExpr{ tensor, ValueExpr((int_t)k++) });
                    }
                }
            }
        }
        
        Result<ValueExpr> loop_repeats( const Component& component, const Dict<Symbol>& symbols, Graph& graph, const Shared<Expr> loop_count )
        {
            ValueExpr repeats;
            ValueExpr remapped;
            bool dynamic = false;
            if ( loop_count )
            {
                TRY_DECL(is_null, eval_null(*loop_count, symbols))
                if ( !is_null )
                {
                    dynamic = is_tensor_expr(*loop_count, symbols);
                    if ( !dynamic )
                    {
                        TRY_MOVE(repeats, eval_shape_expr(*loop_count, symbols))
                        remapped = repeats;
                    }
                }
            }
            for ( auto& [iden, expr] : component.loop->scans )
            {
                TRY_DECL(arg_repeats, eval_dynamic_rank<true>(*expr, symbols))
                if ( arg_repeats != nullptr )
                {
                    if ( repeats == nullptr )
                    {
                        repeats = arg_repeats;
                        if ( arg_repeats.is_literal() )
                        {
                            remapped = arg_repeats;
                        }
                        else
                        {
                            TRY_DECL(value, eval(*expr, symbols, as_tensor(graph), as_tensor_pack(graph)))
                            remapped = SizeAccess{ value.as<TensorPack*>() };
                        }
                    }
                    else if ( repeats != arg_repeats )
                    {
                        return Error(expr->position, "mismatch between implied loop counts (%s vs %s)",
                                     str(repeats).c_str(), str(arg_repeats).c_str());
                    }
                }
            }
            return remapped;
        }
        
        Result<std::tuple<std::vector<TensorRef>,std::vector<TensorRef>>>
        compose( const Component& component, const Dict<Operator>& operators, Dict<Symbol>& symbols, Model& model,
                const size_t graph_idx, const std::optional<std::string>& scope, const bool propagate_label )
        {
            auto label = propagate_label ? "~" : auto_label(component, symbols);
            
            if ( !component.branches.empty() )
            {
                std::vector<ValueExpr> condition_graphs;
                std::vector<ValueExpr> branch_graphs;
                
                std::vector<TensorRef> inputs;
                std::vector<TensorRef> outputs;
                
                std::vector<TensorRef> cond_inputs, branch_inputs;
                
                bool shortcut = false;
                for ( size_t i = 0; i < component.branches.size() && !shortcut; ++i )
                {
                    const Callable& condition = component.branches[i].condition;
                    const Callable& consequent = component.branches[i].consequent;
                    
                    auto expr = condition.is<Region>() && condition.as<Region>().components.empty() ?
                                condition.as<Region>().yields.front() : nullptr;
                    if ( expr && !is_tensor_expr(*expr, symbols) )
                    {
                        TRY_DECL(value, eval(*expr, symbols))
                        if ( value.is_literal() )
                        {
                            if ( !value.as_bool() )             // skip this branch, it's never executed
                            {
                                continue;
                            }
                            
                            shortcut = true;                    // this becomes the `else` branch
                            
                            if ( condition_graphs.empty() )     // branching is completely eliminated
                            {
                                TRY_DECL(attribs, inputs, outputs, compose_callable(consequent, operators, symbols, model, graph_idx, graph_idx, scope, label))
                                rename_results(component.results, outputs, scope);
                                TRY_CALL(add_results_to_symbols(component.results, outputs, model.graphs[graph_idx], symbols, scope, component.position))
                                return std::make_tuple(inputs, outputs);
                            }
                        }
                    }
                    if ( !shortcut )
                    {
                        TRY_DECL(subgraph_idx, subgraph_inputs, compose_subgraph(condition, operators, symbols, model, graph_idx, scope, label))
                        
                        auto& result = *model.graphs[subgraph_idx].outputs.front();
                        if ( !is_singular(result.shape) )
                        {
                            return Error(position(condition), "condition must be a singular tensor, found tensor of shape %s",
                                         str(result.shape).c_str());
                        }
                        
                        condition_graphs.push_back((int_t)subgraph_idx);
                        cond_inputs.insert(cond_inputs.end(), subgraph_inputs.begin(), subgraph_inputs.end());
                        add_all(inputs, subgraph_inputs);
                    }
                    
                    TRY_DECL(subgraph_idx, subgraph_inputs, compose_subgraph(consequent, operators, symbols, model, graph_idx, scope, label))
                    branch_graphs.push_back((int_t)subgraph_idx);
                    auto& graph_outputs = model.graphs[subgraph_idx].outputs;
                    
                    if ( branch_graphs.size() == 1 )
                    {
                        outputs = make_tensors_like(model.graphs[graph_idx], graph_outputs, std::nullopt, {}, true);
                    }
                    else
                    {
                        TRY_CALL(update_branch_output_shapes(outputs, graph_outputs, model.graphs[graph_idx], component.position))
                    }
                    branch_inputs.insert(branch_inputs.end(), subgraph_inputs.begin(), subgraph_inputs.end());
                    add_all(inputs, subgraph_inputs);
                }
                
                if ( !shortcut )
                {
                    if ( condition_graphs.empty() )     // only else branch was not eliminated
                    {
                        TRY_DECL(attribs, inputs, outputs, compose_callable(component.operation, operators, symbols, model, graph_idx, graph_idx, scope, label))
                        rename_results(component.results, outputs, scope);
                        TRY_CALL(add_results_to_symbols(component.results, outputs, model.graphs[graph_idx], symbols, scope, component.position))
                        
                        return std::make_tuple(inputs, outputs);
                    }
                    else                                // add else branch as well
                    {
                        TRY_DECL(subgraph_idx, subgraph_inputs, compose_subgraph(component.operation, operators, symbols, model, graph_idx, scope, label))
                        branch_graphs.push_back((int_t)subgraph_idx);
                        
                        auto& graph_outputs = model.graphs[subgraph_idx].outputs;
                        TRY_CALL(update_branch_output_shapes(outputs, graph_outputs, model.graphs[graph_idx], component.position))
                        
                        branch_inputs.insert(branch_inputs.end(), subgraph_inputs.begin(), subgraph_inputs.end());
                        add_all(inputs, subgraph_inputs);
                    }
                }
                
                rename_results(component.results, outputs, scope);
                TRY_CALL(add_results_to_symbols(component.results, outputs, model.graphs[graph_idx], symbols, scope, component.position))
                
                const Dict<ValueExpr> attribs =
                {
                    { "cond_graphs", ValueExpr(condition_graphs.data(), condition_graphs.size(), Typename::Int) },
                    { "cond_inputs", subgraph_input_mapping(inputs, cond_inputs) },
                    { "branch_graphs", ValueExpr(branch_graphs.data(), branch_graphs.size(), Typename::Int) },
                    { "branch_inputs", subgraph_input_mapping(inputs, branch_inputs) },
                };
                
                model.graphs[graph_idx].operations.push_back(Operation{ "if", {}, attribs, inputs, outputs });
                return std::make_tuple(inputs, outputs);
            }
            else if ( component.loop && !component.loop->unroll )
            {
                auto graph = &model.graphs[graph_idx];
                
                Dict<Symbol> saved_symbols = symbols;
                
                const size_t nvars = component.loop->carries.size();
                
                std::vector<TensorRef> inputs;
                for ( auto& [iden, expr] : component.loop->carries )
                {
                    if ( !is_tensor_expr(*expr, symbols) && iden.shape )
                    {
                        TRY_DECL(value, eval(*expr, symbols))
                        TRY_DECL(shape, eval_shape(*iden.shape, symbols))
                        if ( std::any_of(shape.begin(), shape.end(), []( const ValueExpr& x ){ return !x.is_literal(); }) )
                        {
                            return Error(iden.position, "loop carried dependency must not have dynamic shape");
                        }
                        auto tensor = make_constant(*graph, value, value.dtype(), shape, eval_shape_max(shape));
                        inputs.push_back(tensor);
                    }
                    else
                    {
                        TRY_DECL(tensor, eval(*expr, symbols, as_tensor(*graph), as_tensor_pack(*graph)))
                        inputs.push_back(tensor);
                    }
                }
                for ( auto& [iden, expr] : component.loop->scans )
                {
                    TRY_DECL(tensor, eval(*expr, symbols, as_tensor(*graph), as_tensor_pack(*graph)))
                    inputs.push_back(tensor);
                }
                
                auto first_local = graph->tensors.size();
                
                std::vector<TensorRef> locals;
                for ( auto& [iden, expr] : component.loop->carries )
                {
                    auto tensor = inputs[locals.size()];
                    auto var = TensorRef(make_tensor(*graph, tensor.dtype(), tensor.shape(), tensor.max_shape(), {}, {}));
                    symbols.insert_or_assign(iden.name, Symbol(var, tensor.dtype()));
                    add_shape_symbols(iden.name, tensor.shape(), tensor.size_or_null(), symbols);
                    locals.push_back(var);
                }
                
                for ( auto& [iden, expr] : component.loop->scans )
                {
                    auto tensor = inputs[locals.size()];
                    if ( tensor != nullptr )
                    {
                        auto var = TensorRef(make_tensor(*graph, tensor.dtype(), tensor.shape(), tensor.max_shape(), {}, {}));
                        symbols.insert_or_assign(iden, Symbol(var, tensor.dtype()));
                        locals.push_back(var);
                    }
                    else
                    {
                        symbols.emplace(iden, Symbol(TensorRef(nullptr), Typename::Type));
                        locals.push_back(nullptr);
                    }
                    add_shape_symbols(iden, tensor.shape(), tensor.size_or_null(), symbols);
                }
                
                filter_null(inputs);
                filter_null(locals);
                
                const size_t nscans = locals.size() - nvars;
                
                std::string index;
                if ( component.loop->index )
                {
                    auto& iden = component.loop->index->name;
                    auto tensor = TensorRef(make_tensor(*graph, Typename::Int, {}, {}, {}, {}));
                    symbols.insert_or_assign(iden, Symbol(tensor, Typename::Int));
                    add_shape_symbols(iden, {}, symbols);
                    locals.push_back(tensor);
                    index = iden;
                }
                else
                {
                    locals.push_back(TensorRef(nullptr));
                }
                
                auto last_local = graph->tensors.size();
                
                TRY_DECL(repeats, loop_repeats(component, symbols, *graph, component.loop->count))
                
                TRY_DECL(count_is_null, component.loop->count ? eval_null(*component.loop->count, symbols) : true)
                bool count_is_tensor = !count_is_null && is_tensor_expr(*component.loop->count, symbols);
                if ( count_is_tensor )
                {
                    TRY_DECL(count, eval(*component.loop->count, symbols, as_tensor(*graph), as_tensor_pack(*graph)))
                    if ( !is_singular(count->shape) )
                    {
                        return Error(component.loop->count->position, "loop count must be a singular tensor, found tensor of shape %s",
                                     str(count->shape).c_str());
                    }
                    inputs.push_back(count);
                }
                else if ( repeats != nullptr && repeats.is_literal() )
                {
                    auto count = make_constant(*graph, repeats, Typename::Int);
                    inputs.push_back(count);
                }
                else
                {
                    inputs.push_back(TensorRef(nullptr));
                }
                
                TRY_DECL(body_graph_idx, body_inputs, compose_subgraph(component.operation, operators, symbols, model, graph_idx, scope, label))
                
                graph = &model.graphs[graph_idx];   // reset as it may have become invalid
                
                add_all(locals, body_inputs);
                
                auto& graph_outputs = model.graphs[body_graph_idx].outputs;
                
                for ( size_t i = 0; i < component.loop->carries.size(); ++i )
                {
                    if ( locals[i]->shape != graph_outputs[i].shape() )
                    {
                        return Error(position(component.operation),
                                     "shape %s of loop body output %d does not match shape %s of loop carried dependency %d",
                                     str(graph_outputs[i].shape()).c_str(), (int)i+1, str(locals[i]->shape).c_str(), (int)i+1);
                    }
                }
                
                std::vector<TensorRef> outputs(graph_outputs.size());
                for ( size_t i = 0; i < component.loop->carries.size(); ++i )
                {
                    auto& output = *graph_outputs[i];
                    outputs[i] = make_tensor(*graph, output.dtype, dereferenced(output.shape), output.max_shape, {}, {});
                }
                
                if ( repeats != nullptr )
                {
                    const int_t max_repeats = component.loop->count ? eval_shape_expr_max_checked(repeats, component.loop->count->position) :
                                                                       eval_shape_expr_max(repeats);
                    auto size = repeats;
                    if ( component.loop->condition || count_is_tensor )
                    {
                        size = ValueExpr::placeholder(next_placeholder_name(), ValueExpr(max_repeats));
                    }
                    
                    for ( size_t i = component.loop->carries.size(); i < outputs.size(); ++i )
                    {
                        auto& output = *graph_outputs[i];
                        auto pack = make_tensor_pack(*graph, output.dtype, max_repeats, size, dereferenced(output.shape), output.max_shape);
                        for ( size_t k = 0; k < (size_t)max_repeats; ++k )
                        {
                            pack->items[k] = make_tensor(*graph, output.dtype, dereferenced(output.shape), output.max_shape, {}, {});
                        }
                        cache_tensor_pack(pack);
                        outputs[i] = TensorRef(pack);
                    }
                }
                
                Dict<ValueExpr> attribs =
                {
                    { "nvars", ValueExpr((int_t)nvars) },
                    { "nscans", ValueExpr((int_t)nscans) },
                    { "body_graph", ValueExpr((int_t)body_graph_idx) },
                    { "body_inputs", subgraph_input_mapping(locals, body_inputs) },
                };
                
                if ( repeats != nullptr )
                {
                    attribs.emplace("iters", repeats);
                }
                if ( !index.empty() )
                {
                    attribs.emplace("index", ValueExpr((str_t)index));
                }
                
                if ( component.loop->condition )
                {
                    auto& condition = *component.loop->condition;
                    TRY_DECL(cond_graph_idx, cond_inputs, compose_subgraph(condition, operators, symbols, model, graph_idx, scope, label))
                    
                    graph = &model.graphs[graph_idx];   // rest as it may have become invalid
                    
                    add_all(locals, cond_inputs);
                    
                    auto& cond_result = *model.graphs[cond_graph_idx].outputs.front();
                    if ( !is_singular(cond_result.shape) )
                    {
                        return Error(position(condition), "condition must be a singular tensor, found tensor of shape %s",
                                     str(cond_result.shape).c_str());
                    }
                    
                    attribs.emplace("pretest", ValueExpr((bool_t)component.loop->pretest));
                    attribs.emplace("cond_graph", ValueExpr((int_t)cond_graph_idx));
                    attribs.emplace("cond_inputs", subgraph_input_mapping(locals, cond_inputs));
                }
                
                inputs.insert(inputs.end(), locals.begin() + inputs.size(), locals.end());
                
                check_loop_variable_shapes(inputs, outputs, component.loop->carries.size(), position(component.operation));
                
                std::swap(symbols, saved_symbols);
                
                rename_results(component.results, outputs, scope);
                TRY_CALL(add_results_to_symbols(component.results, outputs, *graph, symbols, scope, component.position))
                
                graph->operations.push_back(Operation{ "do", {}, attribs, inputs, outputs });
                
                // remove local placeholder tensors
                graph->tensors.erase(graph->tensors.begin() + first_local, graph->tensors.begin() + last_local);
                
                return std::make_tuple(inputs, outputs);
            }
            else if ( component.loop && component.loop->unroll )
            {
                auto graph = &model.graphs[graph_idx];
                
                TRY_DECL(repeats, loop_repeats(component, symbols, *graph, component.loop->count))
                if ( !repeats.is_literal() )
                {
                    return Error(component.position, "unrolled loop must not have dynamic loop count");
                }
                
                std::vector<TensorRef> inputs;
                std::vector<TensorRef> outputs(output_count(component.operation, operators));
                
                for ( auto& [iden, expr] : component.loop->carries )
                {
                    TRY_DECL(input, eval(*expr, symbols, as_tensor(*graph), as_tensor_pack(*graph)))
                    inputs.push_back(input);
                }
                
                for ( auto& [iden, expr] : component.loop->scans )
                {
                    if ( is_tensor_expr(*expr, symbols) )
                    {
                        TRY_DECL(input, eval(*expr, symbols, as_tensor(*graph), as_tensor_pack(*graph)))
                        inputs.push_back(input);
                    }
                }
                
                Dict<Symbol> saved_symbols = symbols;
                
                for ( size_t i = 0; i < repeats.as_int(); ++i )
                {
                    std::vector<TensorRef> locals;
                    
                    if ( component.loop->index )
                    {
                        auto& iden = as_identifier(*component.loop->index).name;
                        symbols.insert_or_assign(iden, Symbol(ValueExpr((int_t)i), Typename::Int));
                    }
                    
                    size_t j = 0;
                    for ( auto& [iden, expr] : component.loop->carries )
                    {
                        auto tensor = i == 0 ? inputs[j++] : outputs[j++];
                        symbols.insert_or_assign(iden.name, Symbol(tensor, tensor.dtype()));
                        
                        if ( i == 0 )
                        {
                            locals.push_back(tensor);
                        }
                    }
                    for ( auto& [iden, expr] : component.loop->scans )
                    {
                        if ( is_tensor_expr(*expr, symbols) )
                        {
                            TRY_DECL(tensor, eval(*expr, symbols, as_tensor(*graph), as_tensor_pack(*graph), i))
                            symbols.insert_or_assign(iden, Symbol(tensor, tensor.dtype()));
                            
                            if ( i == 0 )
                            {
                                locals.push_back(tensor);
                            }
                        }
                        else
                        {
                            TRY_DECL(value, eval(*expr, symbols, i))
                            symbols.insert_or_assign(iden, Symbol(value, value.dtype()));
                        }
                    }
                    
                    TRY_DECL(item_attribs, item_inputs, item_outputs, compose_callable(component.operation, operators, symbols, model, graph_idx, graph_idx, scope, label))
                    
                    if ( i == 0 )
                    {
                        const size_t nlocals = locals.size();
                        add_all(locals, item_inputs);
                        inputs.insert(inputs.end(), locals.begin() + nlocals, locals.end());
                        
                        for ( size_t k = component.loop->carries.size(); k < item_outputs.size(); ++k )
                        {
                            auto& output = *item_outputs[k];
                            outputs[k] = make_tensor_pack(*graph, output.dtype, repeats.as_int(), repeats, output.shape, output.max_shape);
                        }
                    }
                    for ( size_t k = 0; k < component.loop->carries.size(); ++k )
                    {
                        outputs[k] = item_outputs[k];
                    }
                    for ( size_t k = component.loop->carries.size(); k < item_outputs.size(); ++k )
                    {
                        outputs[k].as<TensorPack*>()->items[i] = &*item_outputs[k];
                    }
                }
                
                for ( size_t k = component.loop->carries.size(); k < outputs.size(); ++k )
                {
                    cache_tensor_pack(outputs[k].as<TensorPack*>());
                }
                
                std::swap(symbols, saved_symbols);
                
                rename_results(component.results, outputs, scope);
                TRY_CALL(add_results_to_symbols(component.results, outputs, model.graphs[graph_idx], symbols, scope, component.position))
                
                return std::make_tuple(inputs, outputs);
            }
            else
            {
                TRY_DECL(attribs, inputs, outputs, compose_callable(component.operation, operators, symbols, model, graph_idx, graph_idx, scope, label))
                rename_results(component.results, outputs, scope);
                TRY_CALL(add_results_to_symbols(component.results, outputs, model.graphs[graph_idx], symbols, scope, component.position))
                return std::make_tuple(inputs, outputs);
            }
        }
        
        size_t output_count( const Callable& callable, const Dict<Operator>& operators )
        {
            if ( callable.is<Invocation>() )
            {
                auto& invocation = callable.as<Invocation>();
                auto& op = operators.at(invocation.target);
                return op.outputs.size();
            }
            else
            {
                auto& region = callable.as<Region>();
                return region.yields.size();
            }
        }
        
        ValueExpr subgraph_input_mapping( const std::vector<TensorRef>& all_inputs, const std::vector<TensorRef>& subgraph_inputs )
        {
            std::vector<ValueExpr> values(subgraph_inputs.size());
            for ( size_t i = 0; i < subgraph_inputs.size(); ++i )
            {
                values[i] = (int_t)(std::find(all_inputs.begin(), all_inputs.end(), subgraph_inputs[i]) - all_inputs.begin());
            }
            return ValueExpr::list(std::move(values), Typename::Int);
        }
        
        void add_all( std::vector<TensorRef>& items, const std::vector<TensorRef>& new_items )
        {
            for ( auto& item : new_items )
            {
                if ( std::find(items.begin(), items.end(), item) == items.end() )
                {
                    items.push_back(item);
                }
            }
        }
        
        std::optional<std::string> nested_scope( const std::string& explicit_label, const std::optional<std::string>& scope,
                                                const std::string& auto_label = {} )
        {
            if ( scope )
            {
                auto& label = !explicit_label.empty() ? explicit_label : auto_label;
                if ( label == "~" )
                {
                    return scope;
                }
                if ( !label.empty() )
                {
                    return *scope + label + ".";
                }
            }
            return std::nullopt;
        }
        
        bool shapes_equal( const std::vector<TensorRef>& tensors1, const std::vector<TensorRef>& tensors2 )
        {
            assert(tensors1.size() == tensors2.size());
            for ( size_t i = 0; i < tensors1.size(); ++i )
            {
                if ( !shapes_equal(tensors1[i], tensors2[i]) )
                {
                    return false;
                }
            }
            return true;
        }
        
        bool shapes_equal( const TensorRef& tensor1, const TensorRef& tensor2 )
        {
            assert(tensor1.packed() == tensor2.packed());
            if ( tensor1.packed() )
            {
                if ( tensor1.max_size() != tensor2.max_size() )
                {
                    return false;
                }
                for ( size_t i = 0; i < tensor1.max_size(); ++i )
                {
                    if ( tensor1[i].shape != tensor2[i].shape )
                    {
                        return false;
                    }
                }
                return true;
            }
            else
            {
                return tensor1->shape == tensor2->shape;
            }
        }
        
        Result<std::tuple<Dict<ValueExpr>,std::vector<TensorRef>,std::vector<TensorRef>>>
        compose_callable( const Callable& callable, const Dict<Operator>& operators, const Dict<Symbol>& symbols,
                         Model& model, const size_t ctx_graph_idx, const size_t sub_graph_idx,
                         const std::optional<std::string>& scope, const std::string& auto_label )
        {
            if ( callable.is<Invocation>() )
            {
                auto& invocation = callable.as<Invocation>();
                auto& op = operators.at(invocation.target);
                auto new_scope = op.publish ? nested_scope(invocation.label, scope, auto_label) : std::nullopt;
                if ( op.graph && invocation.label.empty() )
                {
                    new_scope = invocation.target + ".";
                }
                _trace.emplace_back(invocation.target, invocation.position);
                auto result = invoke(invocation, operators, symbols, model, ctx_graph_idx, sub_graph_idx, new_scope);
                _trace.pop_back();
                if ( !result )
                {
                    result.error().trace.emplace_front(invocation.target, invocation.position);
                }
                return result;
            }
            else
            {
                auto& region = callable.as<Region>();
                auto new_scope = nested_scope(region.label, scope);
                TRY_DECL(inputs, outputs, compose_region(region, operators, symbols, model, ctx_graph_idx, sub_graph_idx, new_scope))
                return std::make_tuple(Dict<ValueExpr>{}, std::move(inputs), std::move(outputs));
            }
        }
        
        Result<std::tuple<size_t,std::vector<TensorRef>>>
        compose_subgraph( const Callable& callable, const Dict<Operator>& operators, const Dict<Symbol>& symbols,
                         Model& model, const size_t parent_idx, const std::optional<std::string>& scope, const std::string& auto_label )
        {
            const size_t graph_idx = model.graphs.size();
            
            const std::string label = callable.is<Invocation>() ? callable.as<Invocation>().label : callable.as<Region>().label;
            std::string graph_name = scope && !label.empty() ? *scope + label : next_graph_name();
            
            if ( callable.is<Invocation>() )
            {
                const Invocation& invocation = callable.as<Invocation>();
                const Operator& op = operators.at(invocation.target);
                
                if ( op.graph )
                {
                    if ( label.empty() )
                    {
                        graph_name = invocation.target;
                    }
                    auto it = _subgraphs.find(graph_name);
                    if ( it != _subgraphs.end() )
                    {
                        const SubgraphInfo& bi = it->second;
                        const Graph& graph = model.graphs[bi.index];
                        
                        _trace.emplace_back(invocation.target, invocation.position);
                        auto result = invoke(invocation, operators, symbols, model, parent_idx, graph_idx, scope, true);
                        _trace.pop_back();
                        if ( !result )
                        {
                            result.error().trace.emplace_front(invocation.target, invocation.position);
                            return result.error();
                        }
                        
                        auto& [attribs, inputs, outputs] = *result;
                        
                        if ( bi.attribs != attribs )
                        {
                            return Error(invocation.position, "graph called with different attributes from previous invocation at [%d,%d]; "
                                                              "to call the same graph with different attributes, it must be labelled",
                                         (int)bi.position.line, (int)bi.position.column);
                        }
                        
                        if ( !shapes_equal(inputs, graph.inputs) )
                        {
                            return Error(invocation.position, "graph called with different inputs shapes from previous invocation at [%d,%d]; "
                                                              "to call the same graph with different input shapes, it must be labelled",
                                         (int)bi.position.line, (int)bi.position.column);
                        }
                        return std::make_tuple(bi.index, std::move(inputs));
                    }
                }
            }
            
            new_graph(model, graph_name);
            _contexts.push({});
            
            TRY_DECL(attribs, external_inputs, outputs, compose_callable(callable, operators, symbols, model, parent_idx, graph_idx, scope, auto_label))
            
            auto& graph = model.graphs[graph_idx];
            
            auto inputs = make_tensors_like(graph, external_inputs, std::nullopt, {}, true);
            for ( size_t i = 0; i < external_inputs.size(); ++i )
            {
                replace_tensor(graph, external_inputs[i], inputs[i]);
            }
            
            if ( callable.is<Invocation>() && scope && !label.empty() )
            {
                const Invocation& invocation = callable.as<Invocation>();
                const Operator& op = operators.at(invocation.target);
                
                rename_inputs(op.inputs, inputs, graph_name + ".");
            }
            
            filter_null(external_inputs);
            filter_null(inputs);
            
            graph.inputs = std::move(inputs);
            graph.outputs = std::move(outputs);
            
            _contexts.pop();
            
            if ( callable.is<Invocation>() )
            {
                const Invocation& invocation = callable.as<Invocation>();
                const Operator& op = operators.at(invocation.target);
                
                if ( op.graph )
                {
                    _subgraphs.emplace(graph_name, SubgraphInfo{ graph_idx, std::move(attribs), invocation.position });
                }
            }
            
            return std::make_tuple(graph_idx, std::move(external_inputs));
        }
        
        Result<std::tuple<std::vector<TensorRef>,std::vector<TensorRef>>>
        compose_region( const Region& region, const Dict<Operator>& operators, const Dict<Symbol>& symbols,
                       Model& model, const size_t ctx_graph_idx, const size_t sub_graph_idx, const std::optional<std::string>& scope )
        {
            Dict<Symbol> locals = symbols;
            
            std::vector<TensorRef> inputs;
            std::unordered_set<TensorRef> intermediates;
            for ( auto& component : region.components )
            {
                TRY_DECL(_inputs, _outputs, compose(component, operators, locals, model, sub_graph_idx, scope, false))
                
                auto& tensor_value_exprs = _contexts.top().exprs;
                for ( auto& input : _inputs )
                {
                    auto it = tensor_value_exprs.find(input);
                    if ( it != tensor_value_exprs.end() )
                    {
                        add_accessed_tensors(it->second, inputs);
                    }
                    else if ( !input.is_constant() && !intermediates.count(input) && std::find(inputs.begin(), inputs.end(), input) == inputs.end() )
                    {
                        inputs.push_back(input);
                    }
                }
                for ( auto& output : _outputs )
                {
                    intermediates.insert(output);
                }
            }
            
            std::vector<TensorRef> outputs(region.yields.size());
            
            auto& graph = model.graphs[sub_graph_idx];
            auto& tensor_value_exprs = _contexts.top().exprs;
            
            for ( size_t i = 0; i < region.yields.size(); ++i )
            {
                auto& yield = *region.yields[i];
                TRY_DECL(tensor, eval(yield, locals, as_tensor(graph), as_tensor_pack(graph)))
                
                auto it = tensor_value_exprs.find(tensor);
                if ( it != tensor_value_exprs.end() )
                {
                    add_accessed_tensors(it->second, inputs);
                }
                else if ( !tensor.is_constant() && !intermediates.count(tensor) && std::find(inputs.begin(), inputs.end(), tensor) == inputs.end() )
                {
                    inputs.push_back(tensor);
                }
                
                if ( !intermediates.count(tensor) || std::find(outputs.begin(), outputs.begin() + i, tensor) != outputs.end() )
                {
                    TensorRef output = make_tensors_like(graph, tensor, {}, {});
                    graph.operations.push_back(Operation{ "", {}, {}, { tensor }, { output } });
                    tensor = output;
                }
                
                outputs[i] = tensor;
            }
            
            return std::make_tuple(std::move(inputs), std::move(outputs));
        }
        
        Result<std::tuple<Dict<ValueExpr>,std::vector<TensorRef>,std::vector<TensorRef>>>
        invoke( const Invocation& invocation, const Dict<Operator>& operators, const Dict<Symbol>& symbols,
               Model& model, const size_t context_idx, const size_t graph_idx, const std::optional<std::string>& scope,
               const bool signature_only = false )
        {
            const Operator& op = operators.at(invocation.target);
            auto& context = model.graphs[context_idx];
            auto& graph = model.graphs[graph_idx];
            
            TRY_DECL(types, eval_generic_types(op, invocation.dtypes, invocation.attribs, invocation.args, symbols, invocation.position))
            
            Dict<Symbol> locals;
            for ( auto& type : op.dtypes )
            {
                locals.emplace(type.name, Symbol(ValueExpr(nullptr), types.at(type.name)));
            }
            
            TRY_DECL(inputs, eval_inputs(op.inputs, invocation.args, symbols, locals, context))
            TRY_DECL(attribs, eval_attribs(op.attribs, invocation.attribs, symbols, locals))
            
            for ( auto& param : op.attribs )
            {
                auto it = attribs.find(param.name);
                if ( it != attribs.end() )
                {
                    auto& value = it->second;
                    auto type = resolve_type(param, locals);
                    auto size = eval_dynamic_rank(value);
                    locals.emplace(param.name, Symbol(value, type, value.max_size_or_null(), size));
                }
            }
            
            for ( size_t i = 0; i < inputs.size(); ++i )
            {
                const Param& param = op.inputs[i];
                const TensorRef& input = inputs[i];
                auto type = resolve_type(param, locals);
                locals.emplace(param.name, Symbol(input, type));
            }
            
            std::vector<Assertion> asserts;
            
            std::vector<bool> checked(op.asserts.size(), false);
            TRY_CALL(check_asserts(op.asserts, locals, invocation.position, checked, asserts))
            
            std::vector<bool> declared(op.usings.size(), false);
            for ( size_t i = 0; i < op.usings.size(); ++i )
            {
                auto& usage = op.usings[i];
                if ( !has_undefined_symbols(*usage.expr, locals) )
                {
                    TRY_CALL(check_asserts(op.asserts, locals, invocation.position, checked, asserts))
                    TRY_CALL(eval_using(usage, locals))
                    declared[i] = true;
                }
            }
            
            TRY_DECL(order, Typing::deduction_order(op.attribs, op.inputs, op.usings, op.name))
            
            for ( size_t k = 0; k < order.size(); ++k )
            {
                const size_t i = order[k];
                auto& tensor = inputs[i];
                auto& param = op.inputs[i];
                
                TRY_CALL(deduce_repeats(param, tensor, locals))
                if ( param.shape )
                {
                    TRY_CALL(deduce_shape(param, tensor, 0, tensor.packed() ? tensor.max_size() : 0, tensor.packed() ? tensor.size() : 0,
                                          invocation.position, locals))
                }
            }
            
            TRY_CALL(eval_deferred_inputs(op.inputs, invocation.args, symbols, locals, context, inputs))
            TRY_CALL(eval_deferred_attribs(op.attribs, invocation.attribs, invocation.position, symbols, locals, attribs))
            
            for ( auto& param : op.inputs )
            {
                if ( param.shape && has_undefined_symbols(*param.shape, locals) )
                {
                    return Error(invocation.position, "could not deduce shape of input '%s'", param.name.c_str());
                }
            }
            
            if ( signature_only )
            {
                return std::make_tuple(std::move(attribs), std::move(inputs), std::vector<TensorRef>{});
            }
            
            for ( size_t i = 0; i < inputs.size(); ++i )
            {
                const Param& param = op.inputs[i];
                if ( inputs[i] == nullptr )
                {
                    add_shape_symbols(param.name, param.type.packed, locals);
                }
                else if ( param.shape )
                {
                    TRY_DECL(shape, eval_shape(*param.shape, locals))
                    TRY_DECL(size, param.repeats ? eval_shape_expr(*param.repeats, locals) : ValueExpr(nullptr))
                    add_shape_symbols(param.name, shape, size, locals);
                }
                else
                {
                    auto arg = i < invocation.args.size() ? invocation.args[i] : param.default_value;
                    auto& eval_symbols = arg == param.default_value ? locals : symbols;
                    if ( arg && eval_type(*arg, eval_symbols) != Typename::Type )
                    {
                        TRY_DECL(shape, eval_shape_from_expr(*arg, eval_symbols))
                        TRY_DECL(size, eval_dynamic_rank(*arg, eval_symbols))
                        add_shape_symbols(param.name, shape, size, locals);
                    }
                }
            }
            
            for ( size_t i = 0; i < op.usings.size(); ++i )
            {
                if ( !declared[i] )
                {
                    TRY_CALL(check_asserts(op.asserts, locals, invocation.position, checked, asserts))
                    TRY_CALL(eval_using(op.usings[i], locals))
                    declared[i] = true;
                }
            }
            TRY_CALL(check_asserts(op.asserts, locals, invocation.position, checked, asserts, true))
            
            std::vector<TensorRef> internals;
            for ( auto& param : op.constants )
            {
                auto type = param.type_alias.empty() ? param.type.name : types.at(param.type_alias);
                TRY_DECL(tensor, make_tensors_for_param(graph, param, locals, type, scope, false))
                internals.push_back(tensor);
                locals.emplace(param.name, Symbol(tensor, type));
                add_shape_symbols(param.name, tensor.shape(), tensor.size_or_null(), locals);
            }
            for ( auto& param : op.variables )
            {
                auto type = param.type_alias.empty() ? param.type.name : types.at(param.type_alias);
                TRY_DECL(tensor, make_tensors_for_param(graph, param, locals, type, scope, true))
                internals.push_back(tensor);
                locals.emplace(param.name, Symbol(tensor, type));
                add_shape_symbols(param.name, tensor.shape(), tensor.size_or_null(), locals);
            }
            
            TRY_CALL(add_placeholder_symbols(op.outputs, locals))
            
            bool is_private = op.name.front() == '_';
            bool is_atomic = !is_private && !op.graph && _atomic(invocation.target, types, attribs, inputs);
            bool is_compound = !op.components.empty();
            bool is_primitive = !op.lowerings.empty();
            
            if ( is_compound && !is_atomic )
            {
                if ( op.graph )
                {
                    graph.asserts = std::move(asserts);
                }
                
                const bool propagate_label = can_propagate_label(op, locals);
                for ( auto& component : op.components )
                {
                    TRY_CALL(compose(component, operators, locals, model, graph_idx, scope, propagate_label))
                }
                
                auto outputs = list_tensors(op.outputs, locals);
                TRY_CALL(check_outputs(op.outputs, outputs, locals))
                
                auto& operation = model.graphs[graph_idx].operations.back(); // get graph again based on index as it may be invalidated by compose()
                if ( unqualified_name(operation.name).front() == '_' )       // elevate private primitive
                {
                    operation.name = invocation.target;
                    operation.dtypes = types;
                    operation.attribs = attribs;
                    operation.inputs = inputs;
                    operation.outputs = outputs;
                }

                return std::make_tuple(std::move(attribs), std::move(inputs), std::move(outputs));
            }
            else
            {
                TRY_DECL(outputs, eval_outputs(graph, op.outputs, locals, types, invocation.position, scope))
                
                for ( size_t i = 0; i < outputs.size(); ++i )
                {
                    auto& param = op.outputs[i];
                    auto& tensor = outputs[i];
                    auto type = param.type_alias.empty() ? param.type.name : types.at(param.type_alias);
                    locals.emplace(param.name, Symbol(tensor, type));
                    add_shape_symbols(param.name, tensor.shape(), tensor.size_or_null(), locals);
                }
                
                bool unroll_packs = has_tensor_packs(op) && _unroll(invocation.target, types, attribs, inputs);
                TRY_DECL(contractions, eval_lowerings(op.lowerings, locals, graph, unroll_packs))
                
                auto subexprs = make_subexprs(collect_references(asserts, contractions, outputs, locals));
                
                if ( !(is_primitive && contractions.empty()) )
                {
                    graph.operations.push_back(Operation{ invocation.target, types, attribs, inputs, outputs, internals,
                                                          contractions, asserts, std::move(subexprs) });
                }
                return std::make_tuple(std::move(attribs), std::move(inputs), std::move(outputs));
            }
        }
        
        static bool has_tensor_packs( const Operator& op )
        {
            return has_tensor_packs(op.inputs) || has_tensor_packs(op.outputs) || has_tensor_packs(op.constants) || has_tensor_packs(op.variables);
        }
        
        static bool has_tensor_packs( const std::vector<Param>& params )
        {
            return std::any_of(params.begin(), params.end(), []( const Param& param )
            {
                return param.type.packed;
            });
        }
        
        static std::string unqualified_name( const std::string& qname )
        {
            return qname.substr(qname.find_last_of('.') + 1);
        }
        
        static void add_accessed_tensors( const ValueExpr& expr, std::vector<TensorRef>& tensors )
        {
            preorder_traverse(expr, [&tensors]( const ValueExpr& x )
            {
                if ( x.is_size_access() )
                {
                    tensors.push_back(x.as_size_access().pack);
                }
                else if ( x.is_shape_access() )
                {
                    tensors.push_back(x.as_shape_access().tensor);
                }
            });
        }
        
        static std::vector<ValueExpr::ReferenceExpr*> collect_references( const std::vector<Assertion>& asserts,
                                                                         const std::vector<Contraction>& contractions,
                                                                         const std::vector<TensorRef>& outputs,
                                                                         const Dict<Symbol>& symbols )
        {
            std::vector<ValueExpr::ReferenceExpr*> references;
            for ( auto& contraction : contractions )
            {
                collect_references(contraction.left, symbols, references);
                collect_references(contraction.right, symbols, references);
                for ( auto& [id, expr] : contraction.locals )
                {
                    collect_references(expr, symbols, references);
                }
                for ( auto& [id, expr] : contraction.bounds )
                {
                    collect_references(expr, symbols, references);
                }
            }
            for ( auto& output : outputs )
            {
                for ( auto& expr : output.shape() )
                {
                    collect_references(expr, symbols, references);
                }
            }
            for ( auto& assert : asserts )
            {
                collect_references(assert.condition, symbols, references);
            }
            return references;
        }
        
        static void collect_references( const ValueExpr& expr, const Dict<Symbol>& symbols, std::vector<ValueExpr::ReferenceExpr*>& references )
        {
            recurse(expr, [&]( const ValueExpr& x ){ collect_references(x, symbols, references); }, true);
            if ( expr.is_reference() )
            {
                references.push_back(const_cast<ValueExpr::ReferenceExpr*>(&expr.as_reference()));
            }
        }
        
        OrderedDict<ValueExpr> make_subexprs( const std::vector<ValueExpr::ReferenceExpr*>& references )
        {
            Dict<std::string> remap;
            for ( auto& ref : references )
            {
                auto it = remap.find(ref->name);
                if ( it == remap.end() )
                {
                    it = remap.emplace(ref->name, next_local_name()).first;
                }
                ref->name = it->second;
            }
            
            OrderedDict<ValueExpr> subexprs;
            for ( auto& ref : references )
            {
                auto it = subexprs.find(ref->name);
                if ( it == subexprs.end() )
                {
                    it = subexprs.emplace(ref->name, *ref->target).first;
                }
                ref->target = &it->second;
            }
            return subexprs;
        }
        
        void filter_null( std::vector<TensorRef>& tensors )
        {
            for ( auto it = tensors.begin(); it != tensors.end(); )
            {
                if ( *it == nullptr )
                {
                    it = tensors.erase(it);
                }
                else
                {
                    ++it;
                }
            }
        }
        
        Result<void> add_results_to_symbols( const std::vector<Packable<Typed>>& results, std::vector<TensorRef>& outputs,
                                            Graph& graph, Dict<Symbol>& symbols, const std::optional<std::string>& scope,
                                            const Position& position )
        {
            for ( size_t i = 0; i < results.size(); ++i )
            {
                auto& result = results[i];
                auto& output = outputs[i];
                auto& output_type = output.dtype();
                auto& output_shape = output.shape();
                auto& output_size = output.size_or_null();
                
                assert(output != nullptr);
                
                if ( result.packed() )
                {
                    size_t k = 0;
                    for ( size_t j = 0; j < result.size(); ++j )
                    {
                        auto& item = result[j];
                        
                        auto decl_type = !item.type_alias.empty() ? symbols.at(item.type_alias).type : item.type.name;
                        if ( output_type != Typename::Type && !is_compatible(decl_type, output_type) )
                        {
                            return Error(item.position, "mismatch between declared type '%s' and derived type '%s'",
                                         str(decl_type).c_str(), str(output_type).c_str());
                        }
                        auto& result_type = output_type != Typename::Type ? output_type : decl_type;
                        
                        const size_t n = !item.type.packed ? 1 : output.max_size() > result.size() - 1 ? output.max_size() - result.size() + 1 : 0;
                        if ( k + n <= output.max_size() )
                        {
                            if ( item.type.packed )
                            {
                                if ( item.repeats )
                                {
                                    TRY_DECL(length, eval(*item.repeats, symbols))
                                    if ( (int_t)length != (int_t)n )
                                    {
                                        return Error(item.position, "mismatch between declared and computed pack length (%d vs %d)",
                                                     (int)(int_t)length, (int)n);
                                    }
                                }
                                
                                auto size = output.size() - ValueExpr((int_t)result.size() - 1);
                                auto name = item.name.empty() ? std::string() : scoped_name(scope, item.name);
                                auto pack = make_tensor_pack(graph, result_type, n, size, output.shape(), output.max_shape(), name);
                                for ( size_t i = 0; i < n; ++i )
                                {
                                    pack->items[i] = (Tensor*)&output[k+i];
                                }
                                cache_tensor_pack(pack);
                                
                                if ( !item.name.empty() )
                                {
                                    symbols.insert_or_assign(item.name, Symbol(TensorRef(pack), result_type));
                                }
                            }
                            else
                            {
                                if ( !item.name.empty() )
                                {
                                    symbols.insert_or_assign(item.name, Symbol(TensorRef((Tensor*)&output[k]), result_type));
                                }
                            }
                            
                            if ( item.shape )
                            {
                                auto length = output.packed() ? output.size() - ValueExpr((int_t)result.size() - 1) : 0;
                                TRY_CALL(deduce_repeats(item, output, symbols))
                                TRY_CALL(deduce_shape(item, output, k, n, length, item.position, symbols))
                                TRY_DECL(item_shape, eval_shape(*item.shape, symbols))
                                output.shape() = item_shape;
                                if ( !item.name.empty() )
                                {
                                    add_shape_symbols(item.name, item_shape, output_size, symbols);
                                }
                            }
                            else
                            {
                                if ( !item.name.empty() )
                                {
                                    add_shape_symbols(item.name, output_shape, output_size, symbols);
                                }
                            }
                        }
                        k += n;
                    }
                    
                    if ( k != output.max_size() )
                    {
                        return Error(position, "mismatch between declared and computed result length (%d vs %d)",
                                     (int)k, (int)output.max_size());
                    }
                }
                else
                {
                    auto& item = *result;
                    
                    auto decl_type = !item.type_alias.empty() ? symbols.at(item.type_alias).type : item.type.name;
                    if ( output_type != Typename::Type && !is_compatible(decl_type, output_type))
                    {
                        return Error(item.position, "mismatch between declared type '%s' and derived type '%s'",
                                     str(decl_type).c_str(), str(output_type).c_str());
                    }
                    auto& result_type = output_type != Typename::Type ? output_type : decl_type;
                    
                    if ( item.type.packed && item.repeats )
                    {
                        TRY_DECL(length, eval(*item.repeats, symbols))
                        if ( !canonical_shape_expr_equals(length, output.size()) )
                        {
                            return Error(item.position, "mismatch between declared and computed pack length (%s vs %s)",
                                         str(length).c_str(), str(output.size()).c_str());
                        }
                    }
                    
                    bool is_empty_pack = output.packed() && output.dtype() == Typename::Type;
                    if ( is_empty_pack && decl_type != Typename::Type )
                    {
                        output.as<TensorPack*>()->dtype = decl_type;
                    }
                    
                    if ( !item.name.empty() )
                    {
                        symbols.insert_or_assign(item.name, Symbol(output, result_type));
                    }
                    
                    if ( item.shape )
                    {
                        TRY_CALL(deduce_repeats(item, output, symbols))
                        if ( !is_empty_pack )
                        {
                            TRY_CALL(deduce_shape(item, output, 0, output.packed() ? output.max_size() : 0, 
                                                  output.packed() ? output.size() : 0, item.position, symbols))
                        }
                        TRY_DECL(item_shape, eval_shape(*item.shape, symbols))
                        output.shape() = item_shape;
                        if ( !item.name.empty() )
                        {
                            add_shape_symbols(item.name, item_shape, output_size, symbols);
                        }
                    }
                    else
                    {
                        if ( !item.name.empty() )
                        {
                            add_shape_symbols(item.name, output_shape, output_size, symbols);
                        }
                    }
                }
            }
            return Result<void>();
        }
        
        void add_shape_symbols( const std::string& name, const Shape& shape, const ValueExpr& size, Dict<Symbol>& symbols )
        {
            symbols.insert_or_assign(name + ".shape", Symbol(ValueExpr(shape.data(), shape.size(), Typename::Int), Typename::Int));
            symbols.insert_or_assign(name + ".rank", Symbol(ValueExpr((int_t)shape.size()), Typename::Int));
            if ( size != nullptr )
            {
                symbols.insert_or_assign(name + ".size", Symbol(size, Typename::Int));
            }
        }
        
        void add_shape_symbols( const std::string& name, bool packed, Dict<Symbol>& symbols )
        {
            symbols.insert_or_assign(name + ".shape", Symbol(ValueExpr(nullptr), Typename::Type));
            symbols.insert_or_assign(name + ".rank", Symbol(ValueExpr(nullptr), Typename::Type));
            if ( packed )
            {
                symbols.insert_or_assign(name + ".size", Symbol(ValueExpr(nullptr), Typename::Type));
            }
        }
        
        void remove_shape_symbols( const std::string& name, Dict<Symbol>& symbols )
        {
            symbols.erase(name + ".shape");
            symbols.erase(name + ".rank");
            symbols.erase(name + ".size");
        }
        
        void rename_inputs( const std::vector<Param>& params, std::vector<TensorRef>& inputs,
                           const std::optional<std::string>& scope )
        {
            for ( size_t i = 0; i < params.size(); ++i )
            {
                auto& param = params[i];
                auto& input = inputs[i];
                if ( input != nullptr )
                {
                    if ( input.packed() )
                    {
                        for ( size_t j = 0; j < input.max_size(); ++j )
                        {
                            rename_tensor(input[j], scoped_name(scope, param.name, j));
                        }
                    }
                    else
                    {
                        rename_tensor(*input, scoped_name(scope, param.name));
                    }
                }
            }
        }
        
        void rename_results( const std::vector<Packable<Typed>>& results, std::vector<TensorRef>& outputs,
                            const std::optional<std::string>& scope )
        {
            for ( size_t i = 0; i < results.size(); ++i )
            {
                if ( outputs[i] == nullptr )
                {
                    continue;
                }
                auto& result = results[i];
                auto& output = outputs[i];
                if ( output.packed() )
                {
                    if ( result.packed() )
                    {
                        for ( size_t j = 0, k = 0; j < result.size(); ++j )
                        {
                            auto& item = result[j];
                            if ( item.type.packed )
                            {
                                const size_t count = output.max_size() - result.size() + 1;
                                if ( !item.name.empty() )
                                {
                                    for ( size_t c = 0; c < count; ++c )
                                    {
                                        rename_tensor(output[k+c], scoped_name(scope, item.name, c+1));
                                    }
                                }
                                k += count;
                            }
                            else
                            {
                                if ( !item.name.empty() )
                                {
                                    rename_tensor(output[k], scoped_name(scope, item.name));
                                }
                                k += 1;
                            }
                        }
                    }
                    else
                    {
                        auto& item = *result;
                        
                        if ( !item.name.empty() )
                        {
                            rename_tensor_pack((TensorPack&)output, scoped_name(scope, item.name));
                            for ( size_t j = 0; j < output.max_size(); ++j )
                            {
                                rename_tensor(output[j], scoped_name(scope, item.name, j+1));
                            }
                        }
                    }
                }
                else if ( !result->name.empty() )
                {
                    rename_tensor(*output, scoped_name(scope, result->name));
                }
            }
        }
        
        void rename_tensor( Tensor& tensor, const std::string& name )
        {
            if ( !name.empty() )
            {
                tensor.name = name;
            }
        }
        
        void rename_tensor_pack( TensorPack& pack, const std::string& name )
        {
            if ( !name.empty() )
            {
                pack.name = name;
            }
        }
        
        Result<void> update_branch_output_shapes( std::vector<TensorRef>& outputs, const std::vector<TensorRef>& updates,
                                                 Graph& graph, const Position& position )
        {
            for ( size_t i = 0; i < outputs.size(); ++i )
            {
                auto& output = outputs[i];
                auto& update = updates[i];
                
                if ( output.packed() && update.packed() )
                {
                    for ( size_t j = 0; j < output.max_size() && j < update.max_size(); ++j )
                    {
                        if ( output[j].shape.size() != update[j].shape.size() )
                        {
                            return Error(position, "rank of output %d must be the same for all branches; found %s vs %s",
                                         (int)i, str(output[j].shape).c_str(), str(update[j].shape).c_str());
                        }
                        update_branch_output_shape(output[j].shape, update[j].shape, output[j].max_shape, update[j].max_shape);
                    }
                    if ( update.max_size() > output.max_size() )
                    {
                        auto old_size = output.max_size();
                        auto& items = output.as<TensorPack*>()->items;
                        items.resize(update.max_size());
                        for ( size_t j = old_size; j < update.max_size(); ++j )
                        {
                            items[j] = &*make_tensors_like(graph, TensorRef((Tensor*)&update[j]), std::nullopt, {}, true);
                        }
                    }
                }
                else
                {
                    if ( output->shape.size() != update->shape.size() )
                    {
                        return Error(position, "rank of output %d must be the same for all branches; found %s vs %s",
                                     (int)i, str(output->shape).c_str(), str(update->shape).c_str());
                    }
                    update_branch_output_shape(output->shape, update->shape, output->max_shape, update->max_shape);
                }
            }
            return Result<void>();
        }
        
        void update_branch_output_shape( Shape& shape, const Shape& update, std::vector<int_t>& max_shape, const std::vector<int_t>& max_update )
        {
            for ( size_t i = 0; i < shape.size(); ++i )
            {
                if ( shape[i] != update[i] )
                {
                    dereference(shape[i]);
                }
                if ( max_update[i] > max_shape[i] )
                {
                    max_shape[i] = max_update[i];
                }
            }
        }
        
        Result<void> check_loop_variable_shapes( const std::vector<TensorRef>& inputs, const std::vector<TensorRef>& outputs,
                                                const size_t count, const Position& position )
        {
            for ( size_t i = 0; i < count; ++i )
            {
                auto& input = inputs[i];
                auto& output = outputs[i];
                if ( output.packed() )
                {
                    for ( size_t j = 0; j < output.max_size(); ++j )
                    {
                        if ( input->shape != output[j].shape )
                        {
                            return Error(position, "shape mismatch between input %d and output %d of loop-body (%s vs %s)",
                                         (int)i, (int)i, str(input->shape).c_str(), str(output[j].shape).c_str());
                        }
                    }
                }
                else
                {
                    if ( input->shape != output->shape )
                    {
                        return Error(position, "shape mismatch between input %d and output %d of loop-body (%s vs %s)",
                                     (int)i, (int)i, str(input->shape).c_str(), str(output->shape).c_str());
                    }
                }
            }
            return Result<void>();
        }
        
        void replace_tensor( Graph& graph, const TensorRef& oldRef, const TensorRef& newRef )
        {
            for ( auto& op : graph.operations )
            {
                for ( auto& input : op.inputs )
                {
                    if ( input == oldRef )
                    {
                        input = newRef;
                    }
                }
                for ( auto& output : op.outputs )
                {
                    if ( output == oldRef )
                    {
                        output = newRef;
                    }
                }
                for ( auto& [name, value] : op.attribs )
                {
                    replace_tensor(value, oldRef, newRef);
                }
                for ( auto& assert : op.asserts )
                {
                    replace_tensor(assert.condition, oldRef, newRef);
                    for ( auto& item : assert.args )
                    {
                        replace_tensor(item, oldRef, newRef);
                    }
                }
                for ( auto& contraction : op.contractions )
                {
                    replace_tensor(contraction.left, oldRef, newRef);
                    replace_tensor(contraction.right, oldRef, newRef);
                    for ( auto& [iden, expr] : contraction.locals )
                    {
                        replace_tensor(expr, oldRef, newRef);
                    }
                    for ( auto& [iden, expr] : contraction.bounds )
                    {
                        replace_tensor(expr, oldRef, newRef);
                    }
                }
            }
            for ( auto& tensor : graph.tensors )
            {
                for ( auto& item : tensor->shape )
                {
                    replace_tensor(item, oldRef, newRef);
                }
            }
            for ( auto& pack : graph.packs )
            {
                for ( auto& item : pack->shape )
                {
                    replace_tensor(item, oldRef, newRef);
                }
                replace_tensor(pack->size, oldRef, newRef);
            }
            for ( auto& input : graph.inputs )
            {
                if ( input == oldRef )
                {
                    input = newRef;
                }
            }
            for ( auto& output : graph.outputs )
            {
                if ( output == oldRef )
                {
                    output = newRef;
                }
            }
        }
        
        void replace_tensor( TensorAccess& access, const TensorRef& oldRef, const TensorRef& newRef )
        {
            if ( access.tensor == oldRef )
            {
                access.tensor = newRef;
            }
            for ( auto& index : access.indices )
            {
                replace_tensor(index, oldRef, newRef);
            }
        }
        
        void replace_tensor( ShapeAccess& access, const TensorRef& oldRef, const TensorRef& newRef )
        {
            if ( access.tensor == oldRef )
            {
                access.tensor = newRef;
            }
        }
        
        void replace_tensor( SizeAccess& access, const TensorRef& oldRef, const TensorRef& newRef )
        {
            if ( access.pack == oldRef )
            {
                access.pack = newRef;
            }
        }
        
        void replace_tensor( ValueExpr& expr, const TensorRef& oldRef, const TensorRef& newRef )
        {
            if ( expr.kind() == ValueExpr::SizeAccess )
            {
                replace_tensor(expr.as_size_access(), oldRef, newRef);
            }
            else if ( expr.kind() == ValueExpr::ShapeAccess )
            {
                replace_tensor(expr.as_shape_access(), oldRef, newRef);
            }
            else if ( expr.kind() == ValueExpr::TensorAccess )
            {
                replace_tensor(expr.as_tensor_access(), oldRef, newRef);
            }
            else
            {
                recurse(expr, [&]( ValueExpr& x )
                {
                    replace_tensor(x, oldRef, newRef);
                });
            }
        }
        
        std::vector<TensorRef> list_tensors( const std::vector<Param>& params, const Dict<Symbol>& symbols )
        {
            std::vector<TensorRef> tensors;
            for ( auto& param : params )
            {
                tensors.push_back(symbols.at(param.name).as<TensorRef>());
            }
            return tensors;
        }
        
        Result<std::vector<TensorRef>> eval_inputs( const std::vector<Param>& params, const std::vector<Shared<Expr>>& args,
                                                   const Dict<Symbol>& symbols, const Dict<Symbol>& locals, Graph& graph )
        {
            std::vector<TensorRef> inputs(params.size(), TensorRef(nullptr));
            for ( size_t i = 0; i < args.size(); ++i )
            {
                if ( args[i] )
                {
                    TRY_DECL(tensor, eval(*args[i], symbols, as_tensor(graph), as_tensor_pack(graph)))
                    inputs[i] = tensor;
                }
            }
            for ( size_t i = args.size(); i < params.size(); ++i )
            {
                auto& param = params[i];
                if ( param.default_value && is_const_expr(*param.default_value) )
                {
                    TRY_DECL(value, eval(*param.default_value, locals))
                    auto type = resolve_type(param, locals);
                    inputs[i] = make_constant(graph, value, type);
                }
            }
            return inputs;
        }
        
        Result<void> eval_deferred_inputs( const std::vector<Param>& params, const std::vector<Shared<Expr>>& args,
                                          const Dict<Symbol>& symbols, Dict<Symbol>& locals, Graph& graph, std::vector<TensorRef>& inputs )
        {
            for ( size_t i = args.size(); i < params.size(); ++i )
            {
                auto& param = params[i];
                if ( param.default_value && !is_const_expr(*param.default_value) )
                {
                    TRY_DECL(value, eval(*param.default_value, locals))
                    auto type = resolve_type(param, locals);
                    inputs[i] = make_constant(graph, value, type);
                    locals.insert_or_assign(param.name, Symbol(inputs[i], type));
                }
            }
            return {};
        }
        
        Result<std::vector<TensorRef>> eval_outputs( Graph& graph, const std::vector<Param>& params, const Dict<Symbol>& symbols,
                                                 const Dict<Typename>& dtypes, const Position& position,
                                                 const std::optional<std::string>& scope )
        {
            std::vector<TensorRef> outputs(params.size(), TensorRef(nullptr));
            for ( size_t i = 0; i < outputs.size(); ++i )
            {
                auto& param = params[i];
                auto type = param.type_alias.empty() ? param.type.name : dtypes.at(param.type_alias);
                TRY_DECL(tensor, make_tensors_for_param(graph, param, symbols, type, scope, false))
                outputs[i] = tensor;
            }
            return outputs;
        }
        
        Result<void> check_outputs( const std::vector<Param>& params, const std::vector<TensorRef> outputs,
                                   const Dict<Symbol>& symbols )
        {
            for ( size_t i = 0; i < params.size(); ++i )
            {
                auto& param = params[i];
                auto& output = outputs[i];
                if ( !param.shape || output == nullptr )
                {
                    continue;
                }
                
                if ( param.type.packed )
                {
                    if ( param.repeats )
                    {
                        TRY_DECL(repeats, eval(*param.repeats, symbols))
                        const size_t count = eval_shape_expr_max(repeats);
                        TRY_CALL(check_shape_repeats(*param.shape, symbols, count))
                        
                        if ( !compare_sizes(repeats, output.size(), count, output.max_size()) )
                        {
                            return Error(param.repeats->position, "output pack length (%s) does not match declared output count (%s)",
                                         str(output.size()).c_str(), str(repeats).c_str());
                        }
                    }
                    for ( size_t j = 0; j < output.max_size(); ++j )
                    {
                        auto& composed_shape = output[j].shape;
                        TRY_DECL(declared_shape, eval_shape(*param.shape, symbols, j))
                        if ( !compare_shapes(declared_shape, composed_shape) )
                        {
                            return Error(param.position, "mismatch between composed and declared shapes (%s vs %s) of item %d of output '%s'",
                                         str(composed_shape).c_str(), str(declared_shape).c_str(), (int)j, param.name.c_str());
                        }
                    }
                }
                else
                {
                    auto& composed_shape = output->shape;
                    TRY_DECL(declared_shape, eval_shape(*param.shape, symbols))
                    if ( !compare_shapes(declared_shape, composed_shape) )
                    {
                        return Error(param.position, "mismatch between composed and declared shapes (%s vs %s) of output '%s'",
                                     str(composed_shape).c_str(), str(declared_shape).c_str(), param.name.c_str());
                    }
                }
            }
            return Result<void>();
        }
        
        bool compare_sizes( const ValueExpr& declared_size, const ValueExpr& composed_size,
                           const size_t declared_size_max, const size_t composed_size_max )
        {
            if ( composed_size.is_literal() )
            {
                if ( declared_size.is_placeholder() && composed_size == declared_size.as_placeholder().max_value )
                {
                    return true;
                }
                return composed_size == declared_size;
            }
            else if ( declared_size.is_literal() )
            {
                return composed_size == declared_size;
            }
            else
            {
                return declared_size_max == composed_size_max;
            }
            return true;
        }
        
        bool compare_shapes( const Shape& declared_shape, const Shape& composed_shape )
        {
            if ( declared_shape.size() != composed_shape.size() )
            {
                return false;
            }
            for ( size_t i = 0; i < declared_shape.size(); ++i )
            {
                if ( declared_shape[i] != nullptr && !compare_shapes(declared_shape[i], composed_shape[i]) )
                {
                    return false;
                }
            }
            return true;
        }
        
        bool compare_shapes( const ValueExpr& declared_shape, const ValueExpr& composed_shape )
        {
            if ( composed_shape.is_literal() )
            {
                if ( declared_shape.is_placeholder() && composed_shape == declared_shape.as_placeholder().max_value )
                {
                    return true;
                }
                return composed_shape == declared_shape;
            }
            else if ( declared_shape.is_literal() )
            {
                return composed_shape == declared_shape;
            }
            else
            {
                auto declared_shape_max = eval_shape_expr_max(declared_shape);
                auto composed_shape_max = eval_shape_expr_max(composed_shape);
                return declared_shape_max == composed_shape_max;
            }
        }
        
        Result<void> check_shape_repeats( const Shapedef& shape, const Dict<Symbol>& symbols, const size_t repeats )
        {
            for ( auto item : shape.extents )
            {
                if ( item->kind != Expr::Expand && item->kind != Expr::Range )
                {
                    TRY_DECL(rank, eval_max_rank<true>(*item, symbols))
                    if ( rank && *rank != repeats )
                    {
                        return Error(item->position, "length of spread item (%d) does not match repeat count of parameter (%d)",
                                     (int)*rank, (int)repeats);
                    }
                }
            }
            return Result<void>();
        }
        
        Result<ValueExpr> solve_affine_value( Lexer::Operator op, const ValueExpr& expr_value, const int_t term_value, const Position& position )
        {
            if ( op == Lexer::Operator::Multiply )
            {
                if ( expr_value.is_literal() && expr_value.as_int() % term_value != 0 )
                {
                    return Error(position, "divisibility constraint not met in affine expression (%d not divisible by %d)",
                                 (int)expr_value.as_int(), (int)term_value);
                }
                return expr_value / term_value;
            }
            else if ( op == Lexer::Operator::Plus )
            {
                return expr_value - term_value;
            }
            else if ( op == Lexer::Operator::Minus )
            {
                return expr_value + term_value;
            }
            return Error(position, "invalid operator '%s' in affine expression", Lexer::str(op));
        }
        
        Result<std::pair<std::string,ValueExpr>> eval_affine_id( const Expr& expr, const ValueExpr& expr_value )
        {
            if ( expr_value.is_list() || expr_value.is_uniform() )
            {
                std::vector<ValueExpr> items(expr_value.max_size());
                for ( size_t i = 0; i < expr_value.max_size(); ++i )
                {
                    TRY_DECL(iden, item_value, eval_affine_id(expr, expr_value[i]))
                    items[i] = item_value;
                }
                auto& iden = Typing::find_affine_id(expr);
                return std::make_pair(iden, ValueExpr::list(std::move(items), Typename::Int));
            }
            else
            {
                if ( expr.kind == Expr::Identifier )
                {
                    return std::make_pair(as_identifier(expr).name, expr_value);
                }
                if ( expr.kind == Expr::Binary )
                {
                    auto& binary = as_binary(expr);
                    if ( binary.op == Lexer::Operator::Multiply || binary.op == Lexer::Operator::Plus || binary.op == Lexer::Operator::Minus )
                    {
                        if ( is_const_expr(*binary.left) )
                        {
                            TRY_DECL(left, eval(*binary.left, {}))
                            TRY_DECL(value, solve_affine_value(binary.op, expr_value, left.as_int(), expr.position))
                            return eval_affine_id(*binary.right, value);
                        }
                        else if ( is_const_expr(*binary.right) )
                        {
                            TRY_DECL(right, eval(*binary.right, {}))
                            TRY_DECL(value, solve_affine_value(binary.op, expr_value, right.as_int(), expr.position))
                            return eval_affine_id(*binary.left, value);
                        }
                    }
                }
                return std::make_pair(std::string(), ValueExpr());
            }
        }
        
        Result<Dict<ValueExpr>> eval_attribs( const std::vector<Param>& params, const Dict<Shared<Expr>>& args,
                                             const Dict<Symbol>& symbols, const Dict<Symbol>& locals )
        {
            Dict<ValueExpr> attribs;
            for ( const Param& param : params )
            {
                bool has_constexpr_default = param.default_value && is_const_expr(*param.default_value);
                auto it = args.find(param.name);
                auto expr = it != args.end() ? it->second : has_constexpr_default ? param.default_value : nullptr;
                if ( expr )
                {
                    TRY_DECL(value, eval_optional(*expr, expr == param.default_value ? locals : symbols))
                    if ( !(param.repeats && !value.packed() && value != nullptr) )
                    {
                        attribs.emplace(param.name, value);
                    }
                }
                else if ( param.type.optional )
                {
                    attribs.emplace(param.name, ValueExpr(nullptr));
                }
            }
            return attribs;
        }
        
        Result<void> eval_deferred_attribs( const std::vector<Param>& params, const Dict<Shared<Expr>>& args, const Position& position,
                                           const Dict<Symbol>& symbols, Dict<Symbol>& locals, Dict<ValueExpr>& attribs )
        {
            for ( const Param& param : params )
            {
                if ( !locals.count(param.name) )
                {
                    auto it = args.find(param.name);
                    auto expr = it != args.end() ? it->second : param.default_value;
                    
                    TRY_DECL(value, eval_optional(*expr, expr == param.default_value ? locals : symbols))
                    if ( param.repeats && !value.packed() && value != nullptr )
                    {
                        if ( has_undefined_symbols(*param.repeats, locals) )
                        {
                            return Error(position, "could not deduce rank of attribute '%s'", param.name.c_str());
                        }
                        
                        TRY_DECL(count, eval(*param.repeats, locals))
                        auto max_count = eval_shape_expr_max(count);
                        
                        value = ValueExpr::uniform(value, count, max_count);
                    }
                    else
                    {
                        if ( value == nullptr && !param.type.optional )
                        {
                            return Error(position, "attribute '%s' must be supplied (as its default value evaluates to null)", param.name.c_str());
                        }
                    }
                    
                    auto type = resolve_type(param, locals);
                    auto size = eval_dynamic_rank(value);
                    locals.emplace(param.name, Symbol(value, type, value.max_size_or_null(), size));
                    attribs.emplace(param.name, value);
                }
                if ( param.repeats )
                {
                    TRY_CALL(deduce_attrib_rank(param, position, locals))
                }
            }
            for ( const Param& param : params )
            {
                if ( param.repeats )
                {
                    const Symbol& symbol = locals.at(param.name);
                    if ( symbol.is_null() )
                    {
                        auto& iden = Typing::find_affine_id(*param.repeats);
                        if ( !iden.empty() && !locals.count(iden) )
                        {
                            locals.emplace(iden, Symbol(ValueExpr(nullptr), Typename::Int));
                        }
                    }
                }
            }
            return Result<void>();
        }
        
        Result<void> deduce_attrib_rank( const Param& param, const Position& position, Dict<Symbol>& locals )
        {
            const Symbol& symbol = locals.at(param.name);
            if ( !symbol.is_null() && symbol.packed() )
            {
                if ( !has_undefined_symbols(*param.repeats, locals) )
                {
                    TRY_DECL(count, eval(*param.repeats, locals))
                    if ( symbol.packed() && !canonical_shape_expr_equals(count, symbol.size) )
                    {
                        return Error(position, "argument length %s does not match expected rank %s of attribute '%s'",
                                     str(symbol.size).c_str(), str(count).c_str(), param.name.c_str());
                    }
                }
                else
                {
                    TRY_DECL(iden, value, eval_affine_id(*param.repeats, symbol.size))
                    assert(!iden.empty());
                    auto it = locals.find(iden);
                    if ( value.is_literal() && value.as_int() < 0 && (it == locals.end() || symbol.packed()) )
                    {
                        return Error(position, "rank '%s' deduced to a negative value (%d)", iden.c_str(), (int)value.as_int());
                    }
                    if ( it == locals.end() )
                    {
                        locals.emplace(iden, Symbol(value, Typename::Int));
                    }
                    else if ( symbol.packed() )
                    {
                        auto& size = it->second.as<ValueExpr>();
                        if ( !canonical_shape_expr_equals(size, value) )
                        {
                            return Error(position, "ambiguous deduction of rank '%s' (%s vs %s)",
                                         iden.c_str(), str(size).c_str(), str(value).c_str());
                        }
                    }
                }
            }
            return {};
        }
        
        Result<Dict<Typename>> eval_generic_types( const Operator& op, const std::vector<std::pair<std::string,Typename>>& dtypes,
                                                  const Dict<Shared<Expr>>& attribs, const std::vector<Shared<Expr>>& args,
                                                  const Dict<Symbol>& symbols, const Position& position )
        {
            Dict<Typename> types;
            for ( size_t i = 0; i < dtypes.size(); ++i )
            {
                types[op.dtypes[i].name] = is_abstract(dtypes[i].second) ? symbols.at(dtypes[i].first).type : dtypes[i].second;
            }
            for ( size_t i = dtypes.size(); i < op.dtypes.size(); ++i )
            {
                if ( op.dtypes[i].default_type )
                {
                    types[op.dtypes[i].name] = *op.dtypes[i].default_type;
                }
            }
            
            for ( const Param& param : op.attribs )
            {
                if ( !param.type_alias.empty() )
                {
                    auto it = attribs.find(param.name);
                    if ( it != attribs.end() )
                    {
                        const Typename type = eval_type(*it->second, symbols);
                        if ( type != Typename::Type )
                        {
                            types[param.type_alias] = type;
                        }
                    }
                }
            }
            
            for ( size_t i = 0; i < op.inputs.size(); ++i )
            {
                auto& param = op.inputs[i];
                if ( !param.type_alias.empty() && i < args.size() && args[i] )
                {
                    const Typename type = eval_type(*args[i], symbols);
                    if ( type != Typename::Type )
                    {
                        types[param.type_alias] = type;
                    }
                }
            }
            
            for ( auto& param : op.dtypes )
            {
                auto it = types.find(param.name);
                if ( it == types.end() || it->second == Typename::Type )
                {
                    return Error(position, "could not deduce generic type '%s'", param.name.c_str());
                }
            }
            
            return types;
        }
        
        static bool has_undefined_symbols( const Assert& assert, const Dict<Symbol>& symbols )
        {
            for ( auto& item : assert.prints )
            {
                if ( has_undefined_symbols(*item.second, symbols) )
                {
                    return true;
                }
            }
            if ( assert.message && has_undefined_symbols(*assert.message, symbols) )
            {
                return true;
            }
            return has_undefined_symbols(*assert.expression, symbols);
        }
        
        Result<ValueExpr> flexible_item_rank( const ListExpr& list, const Dict<Symbol>& symbols, const size_t rank )
        {
            size_t items_rank = 0;
            for ( auto item : list.items )
            {
                if ( item->kind == Expr::Expand )
                {
                    auto count = as_expand(*item).count;
                    if ( count )
                    {
                        TRY_DECL(rank, eval(*count, symbols))
                        items_rank += rank.as_int();
                    }
                }
            }
            return ValueExpr((int_t)(rank - items_rank));
        }
        
        Result<void> eval_using( const Using& usage, Dict<Symbol>& symbols )
        {
            static Symbol NullSymbol = Symbol(ValueExpr(nullptr), Typename::Type);
            
            std::optional<size_t> result_rank;
            TRY_DECL(value, eval_optional(*usage.expr, symbols))
            auto type = eval_type(*usage.expr, symbols);
            if ( usage.identifier->kind == Expr::List )
            {
                auto& list = as_list(*usage.identifier);
                
                size_t k = 0;
                bool has_flexible_item = false;
                for ( auto item : list.items )
                {
                    if ( item->kind == Expr::Expand )
                    {
                        auto& expand = as_expand(*item);
                        TRY_DECL(rank, expand.count ? eval(*expand.count, symbols) : flexible_item_rank(list, symbols, value.max_size()))
                        auto ptr = value.as_list().data() + k;
                        has_flexible_item |= !expand.count;
                        if ( expand.item->kind == Expr::Zip )
                        {
                            auto& zip = as_zip(*expand.item);
                            auto item_count = zip.items.size();
                            auto item_rank = (int_t)rank / item_count;
                            for ( size_t i = 0; i < item_count; ++i )
                            {
                                auto symbol = value == nullptr ? NullSymbol : Symbol(ValueExpr(ptr + i, item_rank, item_count, type), type);
                                symbols.emplace(as_identifier(*zip.items[i]).name, symbol);
                            }
                        }
                        else
                        {
                            auto symbol = value == nullptr ? NullSymbol : Symbol(ValueExpr(ptr, rank.as_int(), type), type);
                            symbols.emplace(as_identifier(*expand.item).name, symbol);
                        }
                        k += rank.as_int();
                    }
                    else
                    {
                        auto symbol = value == nullptr ? NullSymbol : Symbol(value[k++], type);
                        symbols.emplace(as_identifier(*item).name, symbol);
                    }
                }
                if ( !has_flexible_item )
                {
                    TRY_MOVE(result_rank, eval_max_rank(list, symbols))
                }
            }
            else
            {
                if ( value.packed() && usage.rank )
                {
                    TRY_DECL(declared_rank, eval(*usage.rank, symbols))
                    result_rank = declared_rank.as_int();
                }
                else
                {
                    TRY_MOVE(result_rank, eval_max_rank<true>(*usage.expr, symbols))
                }
                TRY_DECL(size, eval_dynamic_rank<true>(*usage.expr, symbols))
                auto symbol = value == nullptr ? NullSymbol : Symbol(value, type, result_rank, size);
                symbols.emplace(as_identifier(*usage.identifier).name, symbol);
            }
            if ( value.packed() && result_rank && *result_rank != value.max_size() )
            {
                return Error(usage.position, "expression rank (%d) does not match declared rank (%d)",
                             (int)value.max_size(), (int)*result_rank);
            }
            return Result<void>();
        }
        
        const std::string& get_tensor_access_id( Shared<Expr> expr )
        {
            while ( expr->kind == Expr::Index || expr->kind == Expr::Access )
            {
                expr = expr->kind == Expr::Index ? as_index(*expr).array : as_access(*expr).tensor;
            }
            assert(expr->kind == Expr::Identifier);
            return as_identifier(*expr).name;
        }
        
        Result<bool> has_initializer( const std::vector<Lowering>& lowerings, const size_t idx, const Dict<Symbol>& symbols )
        {
            auto& id = get_tensor_access_id(lowerings[idx].left);
            for ( size_t i = 0; i < idx; ++i )
            {
                auto& lowering = lowerings[i];
                if ( lowering.op == Lexer::Operator::Assign && get_tensor_access_id(lowering.left) == id )
                {
                    TRY_DECL(is_null, eval_null(*lowering.right, symbols))
                    return !is_null;
                }
            }
            return false;
        }
        
        Result<Lowering> generate_initializer( const Lowering& lowering, const Dict<Symbol>& symbols )
        {
            auto type = eval_type(*lowering.right, symbols);
            auto right = implicit_initializer(lowering.op, type, lowering.position);
            if ( !right )
            {
                return Error(lowering.position, "contraction with operator '%s' must have a corresponding explicit initializer",
                             Lexer::str(lowering.op));
            }
            
            Pairs<std::string,Shared<Expr>> bounds;
            for ( auto& bound : lowering.bounds )
            {
                if ( contains_iden(*lowering.left, bound.first) )
                {
                    bounds.push_back(bound);
                }
            }
            return Lowering{ lowering.position, lowering.left, right, Lexer::Operator::Assign, {}, bounds, nullptr, lowering.unroll_index, lowering.unroll_count };
        }
        
        bool contains_iden( const Expr& expr, const std::string& iden )
        {
            return any_of(expr, [&]( const Expr& e )
            {
                return e.kind == Expr::Identifier && as_identifier(e).name == iden;
            });
        }
        
        Result<std::vector<Contraction>> eval_lowerings( const std::vector<Lowering>& lowerings, const Dict<Symbol>& symbols, Graph& graph,
                                                        bool unroll_packs )
        {
            std::vector<Contraction> contractions;
            for ( size_t i = 0; i < lowerings.size(); ++i )
            {
                auto& lowering = lowerings[i];
                if ( lowering.op != Lexer::Operator::Assign )
                {
                    TRY_DECL(has_init, has_initializer(lowerings, i, symbols))
                    if ( !has_init )
                    {
                        TRY_DECL(initializer, generate_initializer(lowering, symbols))
                        TRY_CALL(eval_lowering_unrolled(initializer, symbols, graph, contractions, unroll_packs))
                    }
                }
                TRY_CALL(eval_lowering_unrolled(lowering, symbols, graph, contractions, unroll_packs))
            }
            return contractions;
        }
        
        Result<void> eval_lowering_unrolled( const Lowering& lowering, const Dict<Symbol>& symbols, Graph& graph,
                                            std::vector<Contraction>& contractions, bool unroll_packs )
        {
            if ( !lowering.unroll_count )
            {
                return eval_lowering(lowering, symbols, graph, contractions, unroll_packs);
            }
            
            TRY_DECL(count, eval(*lowering.unroll_count, symbols))
            if ( !count.is_literal() )
            {
                return Error(lowering.unroll_count->position, "unroll count must not depend on dynamic shapes");
            }
            
            auto& _symbols = const_cast<Dict<Symbol>&>(symbols);
            
            for ( size_t i = 0; i < count.as_int(); ++i )
            {
                _symbols.insert_or_assign(lowering.unroll_index, Symbol(ValueExpr((int_t)i), Typename::Int));
                
                TRY_CALL(eval_lowering(lowering, symbols, graph, contractions, unroll_packs))
            }
            
            _symbols.erase(lowering.unroll_index);
            return {};
        }
        
        Result<void> eval_lowering( const Lowering& lowering, const Dict<Symbol>& symbols, Graph& graph, std::vector<Contraction>& contractions,
                                   bool unroll_packs )
        {
            auto& _symbols = const_cast<Dict<Symbol>&>(symbols);
            
            std::vector<std::pair<std::string,ValueExpr>> bounds;
            
            for ( auto& [iden, expr] : lowering.bounds )
            {
                TRY_DECL(rank, eval_static_rank(*expr, symbols))
                if ( rank )
                {
                    for ( size_t i = 0; i < *rank; ++i )
                    {
                        TRY_DECL(value, eval(*expr, symbols, i))
                        bounds.push_back(std::make_pair(iden + "_" + std::to_string(i+1), simplified(std::move(value))));
                    }
                }
                else
                {
                    TRY_DECL(value, eval(*expr, symbols))
                    bounds.push_back(std::make_pair(iden, simplified(std::move(value))));
                }
                
                _symbols.emplace(iden, Symbol(LoopIndex{}, rank));
            }
            
            std::vector<std::pair<std::string,ValueExpr>> locals;
            for ( auto& [iden, expr] : lowering.locals )
            {
                const Typename type = eval_type(*expr, symbols);
                TRY_DECL(rank, eval_max_rank(*expr, symbols))
                if ( rank )
                {
                    for ( size_t i = 0; i < *rank; ++i )
                    {
                        TRY_DECL(value, eval(*expr, symbols, i))
                        locals.emplace_back(iden + "_" + std::to_string(i+1), simplified(std::move(value)));
                    }
                }
                else
                {
                    TRY_DECL(value, eval(*expr, symbols))
                    locals.emplace_back(iden, simplified(std::move(value)));
                }
                _symbols.emplace(iden, Symbol(LoopLocal(), type, rank));
            }
            
            TRY_DECL(is_null, eval_null(*lowering.right, symbols))
            if ( is_null )
            {
                return Result<void>();
            }
            
            TRY_DECL(left, eval(*lowering.left, symbols))
            TRY_DECL(right, eval(*lowering.right, symbols))
            TRY_DECL(cond, lowering.condition ? eval_optional(*lowering.condition, symbols) : ValueExpr(nullptr))
            
            if ( left.packed() && right.packed() && left.size() != right.size() )
            {
                return Error(lowering.position, "pack size mismatch between left-hand-side and right-hand-side (%s vs %s)",
                             str(left.size()).c_str(), str(right.size()).c_str());
            }
            
            for ( auto& [iden, expr] : lowering.locals )
            {
                _symbols.erase(iden);
            }
            for ( auto& [iden, expr] : lowering.bounds )
            {
                _symbols.erase(iden);
            }
            
            if ( _flags & Flags::EliminateTrivialLoops )
            {
                if ( std::any_of(bounds.begin(), bounds.end(), []( const auto& x ){ return x.second == 0; }) )
                {
                    return {};
                }
                for ( auto it = bounds.begin(); it != bounds.end(); )
                {
                    auto& [iden, bound] = *it;
                    if ( bound == 1 )
                    {
                        substitute(left, iden, 0);
                        substitute(right, iden, 0);
                        if ( cond != nullptr )
                        {
                            substitute(cond, iden, 0);
                        }
                        for ( auto& [_, x] : locals )
                        {
                            substitute(x, iden, 0);
                        }
                        for ( auto& [_, x] : bounds )
                        {
                            substitute(x, iden, 0);
                        }
                        
                        it = bounds.erase(it);
                    }
                    else
                    {
                        ++it;
                    }
                }
            }
            
            if ( _flags & Flags::EliminateTrivialLocals )
            {
                for ( auto it = locals.begin(); it != locals.end(); )
                {
                    auto& iden = it->first;
                    auto& expr = it->second;
                    if ( expr.is_identifier() || expr.is_literal() )
                    {
                        substitute(left, iden, expr);
                        substitute(right, iden, expr);
                        if ( cond != nullptr )
                        {
                            substitute(cond, iden, expr);
                        }
                        for ( auto& [_, x] : locals )
                        {
                            substitute(x, iden, expr);
                        }
                        for ( auto& [_, x] : bounds )
                        {
                            substitute(x, iden, expr);
                        }
                        
                        it = locals.erase(it);
                    }
                    else
                    {
                        ++it;
                    }
                }
            }
            
            if ( _flags & Flags::EliminateTrivialBounded )
            {
                eliminate_triavial_bounded(left, bounds);
                eliminate_triavial_bounded(right, bounds);
            }
            
            simplify(left);
            simplify(right);
            simplify(cond);
            
            std::vector<size_t> subscripts;
            for ( size_t i = 0; i < bounds.size(); ++i )
            {
                auto& iden = bounds[i].first;
                if ( contains_iden_in_pack_index(left, iden) || contains_iden_in_pack_index(right, iden) || contains_iden_in_pack_index(cond, iden) ||
                     contains_iden_in_pack_index(locals, iden) || contains_iden_in_pack_index(bounds, iden) )
                {
                    subscripts.push_back(i);
                }
            }
            
            std::vector<size_t> axes;
            for ( size_t i = 0; i < bounds.size(); ++i )
            {
                auto& iden = bounds[i].first;
                if ( !contains_iden(left, iden) )
                {
                    axes.push_back(i);
                }
            }
            
            if ( !(right.is_tensor_access() && right.as_tensor_access().tensor == nullptr) )
            {
                auto contraction = Contraction{ left.as_tensor_access(), right, cond, Lexer::str(lowering.op), locals, bounds, subscripts, axes };
                if ( unroll_packs && !subscripts.empty() && can_unroll_pack_subscripts(contraction) )
                {
                    unroll_pack_subscripts(contraction, contractions);
                }
                else
                {
                    contractions.push_back(std::move(contraction));
                }
            }
            return Result<void>();
        }
        
        size_t find_arg_axis( const TensorAccess& access, const std::vector<std::pair<std::string,ValueExpr>>& bounds )
        {
            size_t axis = bounds.size();
            for ( size_t i = 0; i < bounds.size(); ++i )
            {
                if ( !contains_iden(access, bounds[i].first) )
                {
                    if ( axis != bounds.size() )
                    {
                        return bounds.size();
                    }
                    axis = i;
                }
            }
            return axis;
        }
        
        bool contains_iden( const TensorAccess& access, const std::string& iden )
        {
            for ( auto& index : access.indices )
            {
                if ( contains_iden(index, iden) )
                {
                    return true;
                }
            }
            return false;
        }
        
        bool contains_iden( const ValueExpr& expr, const std::string& iden )
        {
            if ( expr.is_identifier() )
            {
                return expr.as_identifier().name == iden;
            }
            else
            {
                bool contains = false;
                recurse(expr, [&]( const ValueExpr& x )
                {
                    contains |= contains_iden(x, iden);
                });
                return contains;
            }
        }
        
        bool contains_iden_in_pack_index( const ValueExpr& expr, const std::string& iden )
        {
            bool result = false;
            preorder_traverse(expr, [&]( const ValueExpr& x )
            {
                if ( x.is_tensor_access() )
                {
                    auto& access = x.as_tensor_access();
                    if ( !access.item.is_tensor_access() && contains_iden(access.item, iden) )
                    {
                        result = true;
                    }
                }
            });
            return result;
        }
        
        bool contains_iden_in_pack_index( const std::vector<std::pair<std::string,ValueExpr>>& items, const std::string& iden )
        {
            for ( auto& [key,value] : items )
            {
                if ( contains_iden_in_pack_index(value, iden) )
                {
                    return true;
                }
            }
            return false;
        }
        
        Shared<Expr> implicit_initializer( const Lexer::Operator op, const Typename dtype, const Position& position )
        {
            switch ( op )
            {
                case Lexer::Operator::PlusEqual:
                {
                    return dtype == Typename::Real ? (Shared<Expr>)std::make_shared<RealExpr>(position, 0) :
                                                     (Shared<Expr>)std::make_shared<IntExpr>(position, 0);
                }
                case Lexer::Operator::MultiplyEqual:
                {
                    return dtype == Typename::Real ? (Shared<Expr>)std::make_shared<RealExpr>(position, 1) :
                                                     (Shared<Expr>)std::make_shared<IntExpr>(position, 1);
                }
                case Lexer::Operator::GreaterEqual:
                {
                    auto expr = (Shared<Expr>)std::make_shared<RealExpr>(position, -std::numeric_limits<real_t>::infinity());
                    if ( dtype != Typename::Real )
                    {
                        expr = (Shared<Expr>)std::make_shared<CastExpr>(position, dtype, expr);
                    }
                    return expr;
                }
                case Lexer::Operator::LessEqual:
                {
                    auto expr = (Shared<Expr>)std::make_shared<RealExpr>(position, std::numeric_limits<real_t>::infinity());
                    if ( dtype != Typename::Real )
                    {
                        expr = (Shared<Expr>)std::make_shared<CastExpr>(position, dtype, expr);
                    }
                    return expr;
                }
                case Lexer::Operator::AndEqual:
                {
                    return (Shared<Expr>)std::make_shared<BoolExpr>(position, true);
                }
                case Lexer::Operator::OrEqual:
                {
                    return (Shared<Expr>)std::make_shared<BoolExpr>(position, false);
                }
                case Lexer::Operator::MakeEqual:
                {
                    return nullptr;
                }
                default:
                {
                    assert(false);
                    return Shared<Expr>();
                }
            }
        }
        
        static void substitute( ValueExpr& expr, const std::string& iden, const ValueExpr& value )
        {
            if ( expr.is_identifier() && expr.as_identifier().name == iden )
            {
                expr = value;
            }
            else
            {
                recurse(expr, [&]( ValueExpr& x ){ substitute(x, iden, value); });
                Evaluation::simplify(expr);
            }
        }

        static void substitute( Contraction& contraction, const size_t variable, const size_t value )
        {
            auto& iden = contraction.bounds[variable].first;
            
            substitute(contraction.left, iden, (int_t)value);
            substitute(contraction.right, iden, (int_t)value);
            for ( size_t i = variable + 1; i < contraction.bounds.size(); ++i )
            {
                substitute(contraction.bounds[i].second, iden, (int_t)value);
            }
            for ( size_t i = 0; i < contraction.locals.size(); ++i )
            {
                substitute(contraction.locals[i].second, iden, (int_t)value);
            }
            
            contraction.bounds.erase(contraction.bounds.begin() + variable);
            
            erase_value_from_list(contraction.subscripts, variable);
            decrease_values_greater(contraction.subscripts, variable);
            erase_value_from_list(contraction.axes, variable);
            decrease_values_greater(contraction.axes, variable);
        }
        
        static void erase_value_from_list( std::vector<size_t>& items, const size_t value )
        {
            auto it = std::find(items.begin(), items.end(), value);
            if ( it != items.end() )
            {
                items.erase(it);
            }
        }
        
        static void decrease_values_greater( std::vector<size_t>& items, const size_t value )
        {
            for ( auto& item : items )
            {
                if ( item > value )
                {
                    --item;
                }
            }
        }
        
        static void eliminate_triavial_bounded( ValueExpr& expr, const std::vector<std::pair<std::string,ValueExpr>>& bounds )
        {
            preorder_traverse(expr, [&]( ValueExpr& x )
            {
                if ( x.is_tensor_access() )
                {
                    eliminate_trivial_bounded(x.as_tensor_access(), bounds);
                }
            });
        }
        
        static void eliminate_trivial_bounded( ValueExpr::TensorAccessExpr& access, const std::vector<std::pair<std::string,ValueExpr>>& bounds )
        {
            for ( size_t i = 0; i < access.indices.size(); ++i )
            {
                auto& index = access.indices[i];
                if ( index.is_bounded() )
                {
                    auto& bounded = index.as_bounded();
                    if ( bounded.arg.is_identifier() )
                    {
                        auto bound = find_bound(bounded.arg.as_identifier().name, bounds);
                        if ( bound && *bound == access.tensor.shape()[i] )
                        {
                            index = bounded.arg.detach();
                        }
                    }
                }
            }
        }
        
        static const ValueExpr* find_bound( const std::string& name, const std::vector<std::pair<std::string,ValueExpr>>& bounds )
        {
            for ( auto& [iden, expr] : bounds )
            {
                if ( iden == name )
                {
                    return &expr;
                }
            }
            return nullptr;
        }
        
        static bool can_unroll_pack_subscripts( const Contraction& contraction )
        {
            for ( auto& variable : contraction.subscripts )
            {
                auto& range = contraction.bounds[variable].second;
                if ( !range.is_literal() )
                {
                    return false;
                }
            }
            return true;
        }
        
        static void unroll_pack_subscripts( const Contraction& contraction, std::vector<Contraction>& unrolled )
        {
            auto variable = contraction.subscripts.front();
            auto range = (size_t)contraction.bounds[variable].second.as_int();
            
            for ( size_t i = 0; i < range; ++i )
            {
                auto item = contraction;
                substitute(item, variable, i);
                if ( item.subscripts.empty() )
                {
                    unrolled.push_back(std::move(item));
                }
                else
                {
                    unroll_pack_subscripts(item, unrolled);
                }
            }
        }
        
        Result<ValueExpr> eval_default_value( const Expr& expr, const std::vector<std::pair<std::string,Shared<Expr>>>& bounds,
                                             const Shape& shape, const Dict<Symbol>& symbols, std::optional<size_t> idx = std::nullopt )
        {
            if ( bounds.empty() )
            {
                TRY_DECL(value, eval(expr, symbols, idx))
                if ( value.packed() )
                {
                    size_t volume = 1;
                    for ( auto& item : shape )
                    {
                        volume *= item.as_int();
                    }
                    if ( value.max_size() != volume )
                    {
                        return Error(expr.position, "length of default value pack does not match volume of tensor (%d vs %d)",
                                     (int)value.max_size(), (int)volume);
                    }
                }
                return value;
            }
            
            size_t count = 0;
            std::vector<ValueExpr> bound_values(bounds.size());
            for ( size_t i = 0; i < bounds.size(); ++i )
            {
                TRY_DECL(value, eval(*bounds[i].second, symbols))
                count += value.packed() ? value.max_size() : 1;
                bound_values[i] = std::move(value);
            }
            
            std::vector<int_t> limits(count);
            std::vector<int_t*> indices(count);
            for ( size_t i = 0, k = 0; i < bounds.size(); ++i )
            {
                auto& bound = bound_values[i];
                if ( bound.packed() )
                {
                    for ( size_t j = 0; j < bound.max_size(); ++j )
                    {
                        limits[k+j] = bound[j].as_int();
                    }
                }
                else
                {
                    limits[k] = bound.as_int();
                }
                
                Symbol symbol(bound.packed() ? ValueExpr::list((int_t)0, bound.max_size()) : ValueExpr((int_t)0), Typename::Int);
                auto [it,_] = const_cast<Dict<Symbol>&>(symbols).emplace(bounds[i].first, std::move(symbol));
                
                auto& value = it->second.as<ValueExpr>();
                if ( value.is_list() )
                {
                    auto& items = value.as_list();
                    for ( size_t j = 0; j < items.size(); ++j )
                    {
                        indices[k++] = &(int_t&)items[j];
                    }
                }
                else
                {
                    indices[k++] = &(int_t&)value;
                }
            }
            
            bool equals = true;
            for ( size_t i = 0; i < count; ++i )
            {
                if ( shape[i] != limits[i] )
                {
                    equals = false;
                    break;
                }
            }
            if ( !equals )
            {
                return Error(expr.position, "default value shape %s does not match param shape %s",
                             str(limits).c_str(), str(shape).c_str());
            }
            
            const size_t volume = volume_of(limits);
            
            std::vector<ValueExpr> items(volume);
            for ( size_t i = 0; i < volume; ++i )
            {
                TRY_DECL(value, eval(expr, symbols))
                items[i] = value;
                
                for ( size_t j = 0; j < count; ++j )
                {
                    if ( ++(*indices[j]) == limits[j] )
                    {
                        *indices[j] = 0;
                    }
                    else
                    {
                        break;
                    }
                }
            }
            
            for ( auto& [id,_] : bounds )
            {
                const_cast<Dict<Symbol>&>(symbols).erase(id);
            }
            
            return ValueExpr::list(std::move(items), eval_type(expr, symbols));
        }
        
        Result<TensorRef> make_tensors_for_param( Graph& graph, const Param& param, const Dict<Symbol>& symbols, const Typename type,
                                              const std::optional<std::string>& scope, const bool variable )
        {
            if ( param.type.packed )
            {
                TRY_DECL(repeats, eval_shape_expr(*param.repeats, symbols))
                const size_t count = eval_shape_expr_max_checked(repeats, param.repeats->position);
                TRY_CALL(check_shape_repeats(*param.shape, symbols, count))
                
                TRY_DECL(value, param.default_value ? eval(*param.default_value, symbols) : ValueExpr(nullptr))
                
                TRY_DECL(shape, eval_shape(*param.shape, symbols))
                auto max_shape = eval_shape_max_checked(shape, param.shape->position);
                auto name = scoped_name(scope, param.name);
                auto pack = make_tensor_pack(graph, type, count, repeats, shape, max_shape, name);
                
                for ( size_t i = 0; i < count; ++i )
                {
                    TRY_DECL(shape, eval_shape(*param.shape, symbols, i))
                    auto max_shape = eval_shape_max_checked(shape, param.shape->position);
                    auto name = scoped_name(scope, param.name, i+1);
                    pack->items[i] = make_tensor(graph, type, shape, max_shape, name, value.packed() ? value[i] : value, variable);
                }
                cache_tensor_pack(pack);
                return TensorRef(pack);
            }
            else
            {
                TRY_DECL(shape, eval_shape(*param.shape, symbols))
                auto max_shape = eval_shape_max_checked(shape, param.shape->position);
                TRY_DECL(value, param.default_value ? eval_default_value(*param.default_value, param.default_bounds, shape, symbols) : ValueExpr(nullptr))
                auto name = scoped_name(scope, param.name);
                return TensorRef(make_tensor(graph, type, shape, max_shape, name, value, variable));
            }
        }
        
        Tensor* make_tensor( Graph& graph, const Typename dtype, const Shape& shape,
                            const std::vector<int_t>& shape_bound, const std::string& name = {},
                            const ValueExpr& value = ValueExpr(nullptr), const bool variable = false )
        {
            auto _name = !name.empty() ? name : next_tensor_name();
            auto tensor = std::make_unique<Tensor>(Tensor{ _name, dtype, shape, shape_bound, {}, value, variable });
            graph.tensors.push_back(std::move(tensor));
            return graph.tensors.back().get();
        }
        
        TensorPack* make_tensor_pack( Graph& graph, const Typename dtype, const size_t max_size, const ValueExpr& size,
                                     const Shape& shape, const std::vector<int_t>& shape_bound,
                                     const std::string& name = {} )
        {
            auto _name = !name.empty() ? name : next_pack_name();
            auto pack = std::make_unique<TensorPack>(TensorPack{ std::vector<Tensor*>(max_size, nullptr), _name, dtype, shape, shape_bound, size });
            graph.packs.push_back(std::move(pack));
            return graph.packs.back().get();
        }
        
        void cache_tensor_pack( TensorPack* pack )
        {
            _contexts.top().packs.emplace(pack->items, pack);
        }
        
        TensorRef make_tensors_like( Graph& graph, const TensorRef& tensor, const std::optional<std::string>& scope,
                                 const std::string& iden, const bool dereference_shape = false )
        {
            if ( tensor == nullptr )
            {
                return tensor;
            }
            else if ( tensor.packed() )
            {
                auto& size = tensor.size();
                auto pack = make_tensor_pack(graph, tensor.dtype(), tensor.max_size(), size, tensor.shape(), tensor.max_shape(), scoped_name(scope, iden));
                if ( dereference_shape )
                {
                    dereference(pack->shape);
                }
                
                for ( size_t i = 0; i < tensor.max_size(); ++i )
                {
                    auto name = !iden.empty() ? scoped_name(scope, iden, i+1) : iden;
                    pack->items[i] = make_tensor(graph, tensor[i].dtype, tensor[i].shape, tensor[i].max_shape, name, {});
                    if ( dereference_shape )
                    {
                        dereference(pack->items[i]->shape);
                    }
                }
                cache_tensor_pack(pack);
                return pack;
            }
            else
            {
                auto name = !iden.empty() ? scoped_name(scope, iden) : iden;
                auto result = make_tensor(graph, tensor->dtype, tensor->shape, tensor->max_shape, name, {});
                if ( dereference_shape )
                {
                    dereference(result->shape);
                }
                return result;
            }
        }
        
        std::vector<TensorRef> make_tensors_like( Graph& graph, const std::vector<TensorRef>& tensors,
                                                 const std::optional<std::string>& scope, const std::string& iden,
                                                 const bool dereference_shape = false )
        {
            std::vector<TensorRef> duplicates(tensors.size());
            for ( size_t i = 0; i < tensors.size(); ++i )
            {
                duplicates[i] = make_tensors_like(graph, tensors[i], scope, iden, dereference_shape);
            }
            return duplicates;
        }
        
        TensorRef make_constant( Graph& graph, const ValueExpr& value, const Typename type, const Shape& shape = {},
                                const std::vector<int_t>& max_shape = {} )
        {
            if ( value == nullptr )
            {
                return TensorRef(nullptr);
            }
            
            auto& consts = _contexts.top().consts;
            
            const std::string str = std::to_string(value);
            auto it = consts.find(str);
            if ( it == consts.end() )
            {
                if ( value.packed() && shape.empty() )
                {
                    auto pack = make_tensor_pack(graph, type, value.max_size(), (int_t)value.max_size(), shape, max_shape);
                    for ( size_t i = 0; i < value.max_size(); ++i )
                    {
                        pack->items[i] = make_tensor(graph, type, shape, max_shape, {}, value[i]);
                    }
                    cache_tensor_pack(pack);
                    it = consts.emplace(str, TensorRef(pack)).first;
                }
                else
                {
                    auto tensor = make_tensor(graph, type, shape, max_shape, {}, value);
                    it = consts.emplace(str, TensorRef(tensor)).first;
                }
            }
            return it->second;
        }
        
        std::string next_tensor_name()
        {
            return ".T" + std::to_string(_next_tensor_idx++);
        }
        
        std::string next_pack_name()
        {
            return ".P" + std::to_string(_next_pack_idx++);
        }
        
        std::string next_graph_name()
        {
            return ".G" + std::to_string(_next_graph_idx++);
        }
        
        std::string next_local_name()
        {
            return ".L" + std::to_string(_next_local_idx++);
        }
        
        std::string next_placeholder_name()
        {
            return ".S" + std::to_string(_next_placeholder_idx++);
        }
        
        std::string scoped_name( const std::optional<std::string>& scope, const std::string& identifier, const size_t idx = 0 )
        {
            const std::string id = idx ? identifier + ":" + std::to_string(idx) : identifier;
            return scope ? *scope + id : "";
        }
        
        Result<void> check_same_ranks( const TensorRef& tensor, const Position& position, const Typed& param )
        {
            if ( tensor.max_size() )
            {
                for ( size_t i = 1; i < tensor.max_size(); ++i )
                {
                    if ( tensor[i].shape.size() != tensor[0].shape.size() )
                    {
                        return Error(position, "tensor arguments for packed parameter '%s' must have the same rank, found %d vs %d",
                                     param.name.c_str(), (int)tensor[0].shape.size(), (int)tensor[i].shape.size());
                    }
                }
            }
            return Result<void>();
        }
        
        Result<void> deduce_repeats( const Typed& param, const TensorRef& tensor, Dict<Symbol>& symbols )
        {
            if ( param.repeats )
            {
                if ( tensor == nullptr )
                {
                    auto& iden = Typing::find_affine_id(*param.repeats);
                    if ( !iden.empty() )
                    {
                        symbols.emplace(iden, Symbol(ValueExpr(nullptr), Typename::Type));
                    }
                }
                else if ( tensor.packed() )
                {
                    TRY_DECL(iden, count, eval_affine_id(*param.repeats, (int_t)tensor.max_size()))
                    if ( !iden.empty() )
                    {
                        auto pack = tensor.as<TensorPack*>();
                        if ( pack->size.is_literal() )
                        {
                            symbols.emplace(iden, Symbol(ValueExpr(count), Typename::Int));
                        }
                        else
                        {
                            symbols.emplace(iden, Symbol(ValueExpr(SizeAccess{ pack }), Typename::Int));
                        }
                    }
                }
            }
            return {};
        }
        
        Result<void> deduce_shape( const Typed& param, const TensorRef& tensor, const size_t offset, size_t const count,
                                  const ValueExpr& size, const Position& position, Dict<Symbol>& symbols )
        {
            if ( tensor == nullptr )
            {
                if ( param.rank )
                {
                    auto& iden = as_identifier(*param.rank).name;
                    symbols.emplace(iden, Symbol(ValueExpr(nullptr), Typename::Type));
                }
                
                deduce_null_ranks(param, symbols);
                deduce_null_extents(param, symbols);
            }
            else
            {
                if ( tensor.packed() )
                {
                    TRY_CALL(check_same_ranks(tensor, position, param))
                }
                
                if ( param.rank )
                {
                    auto& iden = as_identifier(*param.rank).name;
                    symbols.emplace(iden, Symbol(ValueExpr((int_t)tensor.rank()), Typename::Int));
                }
                
                TRY_CALL(deduce_ranks(param, tensor.rank(), position, symbols))
                TRY_CALL(deduce_extents(param, tensor, offset, count, size, position, symbols))
            }
            return {};
        }
        
        Result<void> deduce_ranks( const Typed& param, const size_t rank, const Position& position, Dict<Symbol>& symbols )
        {
            TRY_DECL(min_rank, deduced_shape_rank(*param.shape, symbols))
            if ( min_rank > rank )
            {
                return Error(position, "expected argument of rank at least %d for parameter '%s', got rank %d",
                             (int)min_rank, param.name.c_str(), (int)rank);
            }
            auto excess = (int_t)(rank - min_rank);
            
            for ( auto item : param.shape->extents )
            {
                if ( item && item->kind == Expr::Expand )
                {
                    auto count = as_expand(*item).count;
                    if ( count )
                    {
                        TRY_DECL(iden, value, eval_affine_id(*count, excess))
                        if ( !iden.empty() )
                        {
                            auto it = symbols.find(iden);
                            if ( it == symbols.end() )
                            {
                                symbols.emplace(iden, Symbol(ValueExpr(value), Typename::Int));
                                excess = 0;
                            }
                            else if ( it->second.as<ValueExpr>() == nullptr )
                            {
                                it->second = Symbol(ValueExpr(value), Typename::Int);
                                excess = 0;
                            }
                        }
                    }
                }
            }
            return Result<void>();
        }
        
        void deduce_null_ranks( const Typed& param, Dict<Symbol>& symbols )
        {
            for ( auto item : param.shape->extents )
            {
                if ( item->kind == Expr::Expand )
                {
                    auto count = as_expand(*item).count;
                    if ( count )
                    {
                        auto& iden = Typing::find_affine_id(*count);
                        if ( !iden.empty() )
                        {
                            auto it = symbols.find(iden);
                            if ( it == symbols.end() )
                            {
                                symbols.emplace(iden, Symbol(ValueExpr(nullptr), Typename::Type));
                            }
                        }
                    }
                }
            }
        }
        
        Result<size_t> deduced_shape_rank( const Shapedef& shape, const Dict<Symbol>& symbols )
        {
            size_t sum = 0;
            for ( auto& item : shape.extents )
            {
                if ( item )
                {
                    TRY_DECL(rank, deduced_shape_item_rank(*item, symbols))
                    sum += rank;
                }
                else
                {
                    sum += 1;
                }
            }
            return sum;
        }
        
        Result<size_t> deduced_shape_item_rank( const Expr& item, const Dict<Symbol>& symbols )
        {
            if ( item.kind == Expr::Expand )
            {
                auto& count = as_expand(item).count;
                if ( count )
                {
                    if ( has_undefined_symbols(*count, symbols) )
                    {
                        return 0;
                    }
                    TRY_DECL(value, eval(*count, symbols))
                    if ( value.is_bool() )
                    {
                        return value.as_bool() ? 1 : 0;
                    }
                    return value.as_int();
                }
                else
                {
                    TRY_DECL(rank, eval_max_rank(item, symbols))
                    return *rank;
                }
            }
            else if ( item.kind == Expr::Range )
            {
                TRY_DECL(rank, eval_max_rank(item, symbols))
                return *rank;
            }
            else
            {
                return 1;
            }
        }
        
        ValueExpr make_size_access_expr( const ValueExpr& value, const TensorRef pack )
        {
            return value.is_literal() ? value : SizeAccess{ pack };
        }
        
        ValueExpr make_shape_access_expr( const ValueExpr& value, const TensorRef tensor, const size_t dim )
        {
            assert(tensor.max_shape()[dim] != -1);
            return value.is_literal() ? value : ShapeAccess{ tensor, (int_t)dim };
        }
        
        ValueExpr make_shape_access_exprs( const ValueExpr& value, const TensorRef tensor, const size_t dim, bool merge_pack = false )
        {
            if ( is_literal(value) )
            {
                return value;
            }
            else if ( value.is_list() && !merge_pack )
            {
                std::vector<ValueExpr> items(value.max_size());
                for ( size_t i = 0; i < value.max_size(); ++i )
                {
                    items[i] = make_shape_access_expr(value[i], tensor, dim + i);
                }
                return ValueExpr::list(std::move(items), Typename::Int);
            }
            else
            {
                return ValueExpr(ShapeAccess{ tensor, (int_t)dim }, Typename::Int, tensor.packed() ? tensor.max_size() : (std::optional<size_t>)std::nullopt);
            }
        }
        
        Result<void> deduce_extents( const Typed& param, const TensorRef tensor, size_t offset, size_t count,
                                    const ValueExpr& size, const Position& position, Dict<Symbol>& symbols )
        {
            size_t k = 0, idx = 0;
            for ( auto item : param.shape->extents )
            {
                bool spread = param.shape->spreads & (1 << idx++);
                if ( !item )
                {
                    k += 1;
                    continue;
                }
                
                TRY_DECL(rank, deduced_shape_item_rank(*item, symbols))
                
                bool bool_count = false;
                
                const bool expand = item->kind == Expr::Expand;
                if ( expand )
                {
                    auto count = as_expand(*item).count;
                    if ( count && eval_type(*count, symbols) == Typename::Bool )
                    {
                        bool_count = true;
                    }
                    item = as_expand(*item).item;
                }
                
                ValueExpr shape_value;
                if ( expand )
                {
                    if ( bool_count )
                    {
                        shape_value = rank ? tensor.shape()[k] : nullptr;
                    }
                    else
                    {
                        shape_value = ValueExpr(tensor.shape().data() + k, rank, Typename::Int);
                        for ( size_t i = 0; i < rank; ++i )
                        {
                            if ( shape_value[i].is_list() )
                            {
                                auto common_value = common_shape_expr(shape_value[i].as_list());
                                if ( common_value == nullptr )
                                {
                                    return Error(position, "ambiguous deduction of shape component %d due to non-uniform pack item shape: %s",
                                                 (int)k+i, std::to_string(shape_value[i]).c_str());
                                }
                                shape_value.as_list()[i] = common_value;
                            }
                        }
                    }
                }
                else if ( spread )
                {
                    std::vector<ValueExpr> items(count);
                    for ( size_t i = 0; i < count; ++i )
                    {
                        items[i] = tensor[i + offset].shape[k];
                    }
                    shape_value = ValueExpr::list(std::move(items), Typename::Int);
                }
                else
                {
                    shape_value = tensor.shape()[k];
                    if ( shape_value.is_list() )
                    {
                        auto common_value = common_shape_expr(shape_value.as_list());
                        if ( common_value == nullptr )
                        {
                            return Error(position, "ambiguous deduction of shape component %d due to non-uniform pack item shape: %s",
                                         (int)k, std::to_string(shape_value).c_str());
                        }
                        shape_value = common_value;
                    }
                }
                
                TRY_DECL(iden, value, eval_affine_id(*item, shape_value))
                
                if ( !iden.empty() )
                {
                    auto it = symbols.find(iden);
                    if ( it == symbols.end() )
                    {
                        if ( spread )
                        {
                            auto shape_expr = make_shape_access_exprs(value, tensor, k, true);
                            auto size_expr = make_size_access_expr(size, tensor);
                            symbols.emplace(iden, Symbol(shape_expr, Typename::Int, count, size_expr));
                        }
                        else if ( value == nullptr )
                        {
                            symbols.emplace(iden, Symbol(value, Typename::Int));
                        }
                        else
                        {
                            auto shape_expr = make_shape_access_exprs(value, tensor, k);
                            symbols.emplace(iden, Symbol(shape_expr, Typename::Int));
                        }
                    }
                    else if ( it->second.is<ValueExpr>() && it->second.as<ValueExpr>() == nullptr && value != nullptr )
                    {
                        auto shape_expr = make_shape_access_exprs(value, tensor, k);
                        it->second = Symbol(shape_expr, Typename::Int);
                    }
                    else if ( it->second.is<ValueExpr>() && is_literal(value) && it->second.as<ValueExpr>() != value )
                    {
                        return Error(position, "ambiguous deduction of shape component '%s' (%s vs %s)",
                                     iden.c_str(), std::to_string(it->second.as<ValueExpr>()).c_str(),
                                     std::to_string(value).c_str());
                    }
                }
                
                k += rank;
            }
            
            auto& shape = tensor.shape();
            
            k = 0;
            for ( auto item : param.shape->extents )
            {
                if ( !item )
                {
                    k += 1;
                    continue;
                }
                
                TRY_DECL(rank, deduced_shape_item_rank(*item, symbols))
                
                const bool expand = item->kind == Expr::Expand;
                if ( expand )
                {
                    item = as_expand(*item).item;
                }
                
                if ( !Typing::is_affine_expr(*item) )   // then it's a literal constant
                {
                    TRY_DECL(value, eval(*item, symbols))
                    
                    if ( expand )
                    {
                        if ( value.is_list() )
                        {
                            if ( !std::equal(value.as_list().begin(), value.as_list().end(), shape.data() + k,
                                             []( const auto& x, const ValueExpr& y ){ return y == (int_t)x; }) )
                            {
                                return Error(position, "expected value '%s' for shape components %d:%d of param '%s', found shape %s",
                                             std::to_string(value).c_str(), (int)k, (int)(k + rank), param.name.c_str(), str(shape).c_str());
                            }
                        }
                        else
                        {
                            auto extent = value.as_int();
                            bool equals = std::all_of(shape.data() + k, shape.data() + k + rank, [&]( const ValueExpr& expr )
                            {
                                return expr == extent;
                            });
                            if ( !equals )
                            {
                                return Error(position, "expected value '%d' for shape components %d:%d of param '%s', found shape %s",
                                             (int)extent, (int)k, (int)(k + rank), param.name.c_str(), str(shape).c_str());
                            }
                        }
                    }
                    else
                    {
                        auto extent = value.as_int();
                        if ( shape[k] != extent )
                        {
                            return Error(position, "expected value '%d' for shape component %d of param '%s', found shape %s",
                                         (int)extent, (int)k, param.name.c_str(), str(shape).c_str());
                        }
                    }
                }
                
                k += rank;
            }
            
            if ( k != shape.size() )
            {
                return Error(position, "deduced rank %d for parameter '%s' does not match actual rank %d",
                             (int)k, param.name.c_str(), (int)shape.size());
            }
            return Result<void>();
        }
        
        void deduce_null_extents( const Typed& param, Dict<Symbol>& symbols )
        {
            for ( auto item : param.shape->extents )
            {
                item = unwrapped(item);
                
                auto& iden = Typing::find_affine_id(*item);
                if ( !iden.empty() )
                {
                    auto it = symbols.find(iden);
                    if ( it == symbols.end() )
                    {
                        symbols.emplace(iden, Symbol(ValueExpr(nullptr), Typename::Type));
                    }
                }
            }
        }
        
        Result<void> check_asserts( const std::vector<Assert>& asserts, const Dict<Symbol>& symbols, const Position& position,
                                   std::vector<bool>& checked, std::vector<Assertion>& dynamic_asserts, bool force = false )
        {
            for ( size_t i = 0; i < asserts.size(); ++i )
            {
                if ( !checked[i] && (force || !has_undefined_symbols(asserts[i], symbols)) )
                {
                    TRY_CALL(check_assert(asserts[i], symbols, position, dynamic_asserts))
                    checked[i] = true;
                }
            }
            return Result<void>();
        }
        
        Result<void> check_assert( const Assert& assert, const Dict<Symbol>& symbols, const Position& position, 
                                  std::vector<Assertion>& dynamic_asserts )
        {
            TRY_DECL(cond, eval_optional(*assert.expression, symbols))
            if ( cond == nullptr )
            {
                return {};
            }
            simplify(cond);
            if ( !cond.is_literal() )
            {
                std::vector<ValueExpr> args;
                TRY_DECL(message, format_assert_message(assert, symbols, &args))
                dynamic_asserts.emplace_back(Assertion{ std::move(cond), std::move(message), std::move(args) });
                return {};
            }
            else if ( !cond )
            {
                TRY_DECL(message, format_assert_message(assert, symbols))
                return Error(position, message.c_str());
            }
            return {};
        }
        
        Result<std::string> format_assert_message( const Assert& assert, const Dict<Symbol>& symbols, std::vector<ValueExpr>* args = nullptr )
        {
            if ( assert.message )
            {
                std::string message;
                if ( assert.message->kind == Expr::Literal )
                {
                    message = as_str(*assert.message).value;
                }
                else
                {
                    auto& format = as_format(*assert.message);
                    size_t offset = 0;
                    for ( auto& sub : format.subs )
                    {
                        message += format.str.substr(offset, sub.first - offset);
                        TRY_DECL(x, eval_optional(*sub.second, symbols))
                        if ( x == nullptr || x.is_literal() || !args )
                        {
                            message += str(x);
                        }
                        else
                        {
                            args->push_back(std::move(x));
                            message += "{}";
                        }
                        offset = sub.first;
                    }
                    message += format.str.substr(offset);
                }
                if ( !assert.prints.empty() )
                {
                    message += "; operator invoked with ";
                    size_t i = 0;
                    for ( auto& item : assert.prints )
                    {
                        if ( i++ )
                        {
                            message += ", ";
                        }
                        message += item.first.empty() ? str(*item.second) : item.first;
                        message += " = ";
                        TRY_DECL(x, eval_optional(*item.second, symbols))
                        if ( x == nullptr || x.is_literal() || !args )
                        {
                            message += str(x);
                        }
                        else
                        {
                            args->push_back(std::move(x));
                            message += "{}";
                        }
                    }
                }
                return message;
            }
            else
            {
                std::string message = "assert failed: '" + str(*assert.expression) + "'";
                
                std::vector<std::string> ids;
                preorder_traverse(*assert.expression, [&]( const Expr& e )
                {
                    if ( e.kind == Expr::Identifier )
                    {
                        ids.push_back(as_identifier(e).name);
                    }
                });
                
                if ( !ids.empty() )
                {
                    message += "; operator invoked with ";
                    for ( auto& iden : ids )
                    {
                        message += iden;
                        message += " = ";
                        auto& x = symbols.at(iden).as<ValueExpr>();
                        if ( x == nullptr || x.is_literal() || !args )
                        {
                            message += str(x);
                        }
                        else
                        {
                            args->push_back(std::move(x));
                            message += "{}";
                        }
                    }
                }
                return message;
            }
        }
        
        Result<void> eval_quantization( Graph& graph, const std::vector<Quantization>& quantization, const Dict<Symbol>& symbols,
                                       const Dict<Operator>& operators )
        {
            Dict<Tensor*> tensors;
            for ( auto& tensor : graph.tensors )
            {
                tensors.emplace(tensor->name, tensor.get());
            }
            
            for ( auto& quant : quantization )
            {
                auto it = tensors.find(quant.tensor);
                if ( it == tensors.end() )
                {
                    report_error(quant.position, "'%s' is not a public tensor identifier", quant.tensor.c_str());
                    continue;
                }
                auto& tensor = *it->second;
                
                auto& op = operators.at(quant.invocation.target);
                auto dtypes = eval_dtypes(op, quant.invocation);
                
                auto& param = op.inputs.front();
                auto param_type = !param.type_alias.empty() ? dtypes.at(param.type_alias) : param.type.name;
                
                if ( !is_compatible(param_type, tensor.dtype) )
                {
                    report_error(quant.position, "tensor type '%s' incompatible with quantization operator's parameter type '%s'",
                                 str(tensor.dtype).c_str(), str(param_type).c_str());
                }
                
                tensor.quant.emplace("op-name", ValueExpr(quant.invocation.target));
                for ( auto& [key, value] : quant.invocation.attribs )
                {
                    TRY_DECL(val, eval_optional(*value, symbols))
                    if ( !is_literal(val) )
                    {
                        report_error(value->position, "quantization attribute must not depend on tensor shapes");
                    }
                    tensor.quant.emplace(key, val);
                }
            }
            return Result<void>();
        }
        
        Dict<Typename> eval_dtypes( const Operator& op, const Invocation& invocation ) const
        {
            Dict<Typename> dtypes;
            for ( size_t i = 0; i < invocation.dtypes.size(); ++i )
            {
                dtypes[op.dtypes[i].name] = invocation.dtypes[i].second;
            }
            return dtypes;
        }
        
        Result<Shape> eval_shape( const Shapedef& shapedef, const Dict<Symbol>& symbols,
                                        const std::optional<size_t> idx = std::nullopt )
        {
            TRY_DECL(rank, shape_rank(shapedef, symbols))
            Shape shape(rank);
            
            size_t k = 0;
            for ( auto item : shapedef.extents )
            {
                if ( !item )
                {
                    shape[k++] = nullptr;
                    continue;
                }
                bool packed = item->kind == Expr::Expand || item->kind == Expr::Range;
                TRY_DECL(count, shape_item_rank(*item, symbols))
                item = unwrapped(item);
                for ( size_t i = 0; i < count; ++i )
                {
                    TRY_DECL(value, item ? eval_shape_expr(*item, symbols, packed ? i : idx) : ValueExpr(nullptr))
                    if ( value.is_literal() )
                    {
                        auto extent = value.as_int();
                        if ( extent < 0 )
                        {
                            return Error(item->position, "extent must be non-negative; found %d", (int)extent);
                        }
                    }
                    shape[k++] = value;
                }
            }
            
            return shape;
        }
        
        Result<ValueExpr> eval_shape_expr( const Expr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx = std::nullopt )
        {
            TRY_CALL(check_shape_expr(expr, symbols))
            TRY_DECL(value, eval(expr, symbols, idx))
            simplify(value);
            return value;
        }
        
        Result<void> check_shape_expr( const Expr& expr, const Dict<Symbol>& symbols )
        {
            if ( expr.kind == Expr::Builtin )
            {
                auto& builtin = as_builtin(expr);
                if ( is_trigonometric_func(builtin.func) || builtin.func == "erf" )
                {
                    TRY_DECL(arg, eval(*builtin.arg, symbols))
                    if ( !is_literal(arg) )
                    {
                        return Error(expr.position, "function '%s' is not allowed with argument that depends on dynamic shapes",
                                     builtin.func.c_str());
                    }
                }
                else if ( builtin.func == "log" || builtin.func == "sqrt" )
                {
                    TRY_DECL(arg, eval(*builtin.arg, symbols))
                    if ( !is_literal(arg) )
                    {
                        auto [min, max] = eval_shape_expr_bounds<real_t>(arg);
                        if ( min < 0 )
                        {
                            report_warning(expr.position, "argument to function '%s' may become negative",
                                           builtin.func.c_str());
                        }
                    }
                }
            }
            else if ( expr.kind == Expr::Binary )
            {
                auto& binary = as_binary(expr);
                if ( binary.op == Lexer::Operator::Power )
                {
                    TRY_DECL(left, eval(*binary.left, symbols))
                    TRY_DECL(right, eval(*binary.right, symbols))
                    
                    if ( left.dtype() == Typename::Real )
                    {
                        auto [left_min, left_max] = eval_shape_expr_bounds<real_t>(left);
                        auto [right_min, right_max] = eval_shape_expr_bounds<real_t>(right);
                        
                        if ( left_min < 0 )
                        {
                            report_warning(expr.position, "base may become negative");
                        }
                        if ( right_min < 0 )
                        {
                            report_warning(expr.position, "power may become negative");
                        }
                    }
                    else
                    {
                        auto [left_min, left_max] = eval_shape_expr_bounds<int_t>(left);
                        auto [right_min, right_max] = eval_shape_expr_bounds<int_t>(right);
                        
                        if ( left_min < 0 )
                        {
                            report_warning(expr.position, "base may become negative");
                        }
                        if ( right_min < 0 )
                        {
                            report_warning(expr.position, "power may become negative");
                        }
                    }
                }
                else if ( binary.op == Lexer::Operator::Divide || binary.op == Lexer::Operator::CeilDivide )
                {
                    TRY_DECL(right, eval(*binary.right, symbols))
                    
                    if ( right.dtype() == Typename::Real )
                    {
                        auto [right_min, right_max] = eval_shape_expr_bounds<real_t>(right);
                        if ( right_min <= 0 && right_max >= 0 )
                        {
                            report_warning(expr.position, "divisor may become zero");
                        }
                    }
                    else
                    {
                        auto [right_min, right_max] = eval_shape_expr_bounds<int_t>(right);
                        if ( right_min <= 0 && right_max >= 0 )
                        {
                            report_warning(expr.position, "divisor may become zero");
                        }
                    }
                }
            }
            return recurse_result(expr, [&]( const Expr& x ){ return check_shape_expr(x, symbols); });
        }
        
        int_t eval_shape_expr_max_checked( const ValueExpr& expr, const Position& position )
        {
            auto [min, max] = eval_shape_expr_bounds<int_t>(expr);
            if ( max < 0 )
            {
                report_error(position, "shape expression is always negative");
            }
            if ( min < 0 )
            {
                report_warning(position, "shape expression may become negative");
            }
            return max;
        }
        
        std::vector<int_t> eval_shape_max_checked( const Shape& shape, const Position& position )
        {
            std::vector<int_t> max_shape(shape.size());
            for ( size_t i = 0; i < shape.size(); ++i )
            {
                auto& extent = shape[i];
                int_t max_extent, min_extent;
                
                if ( extent.packed() )
                {
                    if ( extent.max_size() == 0 )
                    {
                        max_shape[i] = -1;
                        continue;
                    }
                    
                    std::tie(min_extent, max_extent) = eval_shape_expr_bounds(extent, 0);
                    max_shape[i] = max_extent;
                    
                    for ( size_t k = 1; k < extent.max_size(); ++k )
                    {
                        auto [item_min, item_max] = eval_shape_expr_bounds(extent, k);
                        if ( min_extent != item_min || max_extent != item_max )
                        {
                            max_shape[i] = -1;
                            break;
                        }
                    }
                }
                else
                {
                    std::tie(min_extent, max_extent) = eval_shape_expr_bounds(extent);
                    max_shape[i] = max_extent;
                }
                
                if ( max_extent < 0 )
                {
                    report_error(position, "shape expression at dimension %d is always negative", (int)i);
                }
                if ( min_extent < 0 )
                {
                    report_warning(position, "shape expression at dimension %d may become negative", (int)i);
                }
            }
            return max_shape;
        }
        
    private:
        
        static bool is_static_expr( const Expr& expr, const Dict<Symbol>& symbols )
        {
            if ( expr.kind == Expr::Identifier )
            {
                auto it = symbols.find(as_identifier(expr).name);
                return it != symbols.end() && it->second.is<ValueExpr>() && is_literal(it->second.as<ValueExpr>());
            }
            else if ( expr.kind == Expr::Unary )
            {
                if ( as_unary(expr).op == Lexer::Operator::Question )
                {
                    return true;
                }
            }
            return all_recurse(expr, [&]( const Expr& e ){ return is_static_expr(e, symbols); });
        }
        
        static std::string auto_label( const Component& component, const Dict<Symbol>& symbols )
        {
            return Typing::auto_label<Dict<Symbol>,is_static_expr>(component, symbols);
        }
        
        static bool has_single_callable( const Component& component, const Dict<Symbol>& symbols )
        {
            return Typing::has_single_callable<Dict<Symbol>,is_static_expr>(component, symbols);
        }
        
        static bool can_propagate_label( const Operator& op, const Dict<Symbol>& symbols )
        {
            return op.components.size() == 1 && has_single_callable(op.components.front(), symbols) &&
                all_results_are_outputs(op, op.components.front());
        }
        
        static bool all_results_are_outputs( const Operator& op, const Component& component )
        {
            std::set<std::string> output_names;
            for ( auto& param : op.outputs )
            {
                output_names.insert(param.name);
            }
            
            for ( auto result : component.results )
            {
                if ( result.packed() )
                {
                    for ( auto item : result )
                    {
                        if ( !item.name.empty() && !output_names.count(item.name) )
                        {
                            return false;
                        }
                    }
                }
                else
                {
                    if ( !result->name.empty() && !output_names.count(result->name) )
                    {
                        return false;
                    }
                }
            }
            return true;
        }
        
        void dereference( Shape& shape )
        {
            for ( size_t i = 0; i < shape.size(); ++i )
            {
                if ( !shape[i].is_literal() )
                {
                    dereference(shape[i]);
                }
            }
        }
        
        void dereference( ValueExpr& expr )
        {
            auto key = str(canonical_shape_expr(expr));
            auto it = _dereferenced.find(key);
            if ( it == _dereferenced.end() )
            {
                auto max_value = eval_shape_expr_max(expr);
                it = _dereferenced.emplace(key, ValueExpr::placeholder(next_placeholder_name(), max_value)).first;
            }
            expr = it->second;
        }
        
        Shape dereferenced( const Shape& shape )
        {
            Shape result = shape;
            dereference(result);
            return result;
        }
        
        static Typename resolve_type( const Param& param, const Dict<Symbol>& symbols )
        {
            return param.type_alias.empty() ? param.type.name : symbols.at(param.type_alias).type;
        }
        
    private:
        
        template<typename T>
        T volume_of( const std::vector<T>& shape )
        {
            T volume = 1;
            for ( auto extent : shape )
            {
                volume *= extent;
            }
            return volume;
        }
        
        bool is_singular( const Shape& shape )
        {
            return std::all_of(shape.begin(), shape.end(), []( const ValueExpr& x ){ return x == 1; });
        }
        
    private:
        
        void report_error( const Error& error )
        {
            _error(error.position, error.message, error.trace, false);
        }
        
        template<typename... Args>
        void report_error( const Position& position, const char* format, Args&&... args )
        {
            _error(position, Error::format_string(format, std::forward<Args>(args)...), _trace, false);
        }
        
        template<typename... Args>
        void report_warning( const Position& position, const char* format, Args&&... args )
        {
            _error(position, Error::format_string(format, std::forward<Args>(args)...), _trace, true);
        }
        
    private:
        
        struct SubgraphInfo
        {
            size_t index = 0;
            Dict<ValueExpr> attribs;
            Position position;
        };
        
        struct SubgraphContext
        {
            Dict<TensorRef> consts;
            std::unordered_map<Tensors,TensorPack*> packs;
            std::unordered_map<TensorRef,ValueExpr> exprs;
        };
        
    private:
        
        unsigned _flags;
        const ErrorCallback _error;
        const OperationCallback _atomic;
        const OperationCallback _unroll;
        std::stack<SubgraphContext> _contexts;
        Dict<SubgraphInfo> _subgraphs;
        Dict<ValueExpr> _dereferenced;
        StackTrace _trace;
        size_t _next_tensor_idx;
        size_t _next_graph_idx;
        size_t _next_pack_idx;
        size_t _next_local_idx;
        size_t _next_placeholder_idx;
    };
    
}   // namespace sknd


#endif
