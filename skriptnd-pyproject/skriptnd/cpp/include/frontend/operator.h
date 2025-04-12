#ifndef _TS_OPERATOR_H_
#define _TS_OPERATOR_H_

#include "astexpr.h"
#include <vector>
#include <map>
#include <set>


namespace sknd
{

    template<typename K, typename V>
    using Pairs = std::vector<std::pair<K,V>>;

    struct Component;

    
    struct TypeParam
    {
        const Position position;
        const std::string name;
        const Typename base_type;
        const std::optional<Typename> default_type;
    };

    struct Shapedef
    {
        const Position position;
        const std::vector<Shared<Expr>> extents;
        const std::vector<Shared<Expr>> bounds;
        const size_t spreads;
    };

    struct Typed
    {
        const Position position;
        const std::string name;
        const Type type;
        const std::string type_alias;
        const Shared<Expr> rank;
        const Shared<Shapedef> shape;
        const Shared<Expr> repeats;
        const Shared<Expr> repeats_bound;
    };
    
    struct Param : public Typed
    {
        const Shared<Expr> default_value;
        const Pairs<std::string,Shared<Expr>> default_bounds;
    };

    struct Using
    {
        const Position position;
        const Shared<Expr> identifier;
        const Shared<Expr> expr;
        const Shared<Expr> rank;
    };

    struct Assert
    {
        const Position position;
        const Shared<Expr> expression;
        const Shared<Expr> message;
        const Pairs<std::string,Shared<Expr>> prints;
    };
    
    struct Lowering
    {
        const Position position;
        const Shared<Expr> left;
        const Shared<Expr> right;
        const Lexer::Operator op;
        const Pairs<std::string,Shared<Expr>> locals;
        const Pairs<std::string,Shared<Expr>> bounds;
        const Shared<Expr> condition;
        const std::string unroll_index;
        const Shared<Expr> unroll_count;
    };
    
    struct Invocation
    {
        const Position position;
        const std::string label;
        const std::string target;
        const Pairs<std::string,Typename> dtypes;
        const std::map<std::string,Shared<Expr>> attribs;
        const std::vector<Shared<Expr>> args;
    };

    struct Region
    {
        const Position position;
        const std::string label;
        const std::vector<Component> components;
        const std::vector<Shared<Expr>> yields;
    };

    typedef Either<Region,Invocation> Callable;
    
    struct Branch
    {
        const Callable condition;
        const Callable consequent;
    };
    
    struct Loop
    {
        const Pairs<Typed,Shared<Expr>> carries;
        const Pairs<std::string,Shared<Expr>> scans;
        const Shared<Callable> condition;
        const Shared<Expr> count;
        const Shared<IdenfitierExpr> index;
        const bool pretest;
        const bool unroll;
    };
    
    struct Component
    {
        const Position position;
        const std::vector<Packable<Typed>> results;
        const Callable operation;
        const std::vector<Branch> branches;
        const Shared<Loop> loop;
    };
    
    struct Quantization
    {
        const Position position;
        const std::string tensor;
        const Invocation invocation;
    };
    
    struct Operator
    {
        const Position position;
        const bool graph;
        const bool publish;
        const std::string name;
        const std::vector<TypeParam> dtypes;
        const std::vector<Param> attribs;
        const std::vector<Param> inputs;
        const std::vector<Param> outputs;
        const std::vector<Param> constants;
        const std::vector<Param> variables;
        const std::vector<Assert> asserts;
        const std::vector<Using> usings;
        const std::vector<Lowering> lowerings;
        const std::vector<Component> components;
        const std::vector<Component> updates;
        const std::vector<Quantization> quantizations;
    };
    
    inline std::ostream& operator<<( std::ostream& os, const TypeParam& type )
    {
        os << type.name << ": " << str(type.base_type);
        if ( type.default_type )
        {
            os << " = " << str(*type.default_type);
        }
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const Shapedef& shape )
    {
        os << '[';
        for ( size_t i = 0; i < shape.extents.size(); ++i )
        {
            if ( i )
            {
                os << ',';
            }
            os << *shape.extents[i];
        }
        os << ']';
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const Typed& typed )
    {
        os << (typed.name.empty() ? "~" : typed.name);
        
        os << ": ";
        if ( typed.type.optional )
        {
            os << "optional ";
        }
        if ( !typed.type_alias.empty() )
        {
            os << typed.type_alias;
        }
        else
        {
            os << str(typed.type.name);
        }
        
        if ( typed.shape )
        {
			os << *typed.shape;
        }
        if ( typed.type.packed )
        {
            os << "..";
        }
        if ( typed.repeats )
        {
            os << '(' << *typed.repeats << ')';
        }
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const Param& param )
    {
        os << (Typed)param;
        
        if ( param.default_value )
        {
            os << " = " << *param.default_value;
            for ( auto& [id, expr] : param.default_bounds )
            {
                os << ", " << id << " < " << *expr;
            }
        }
        return os;
    }
    
    inline std::ostream& operator<<( std::ostream& os, const Using& usage )
    {
        os << usage.identifier << " = " << *usage.expr;
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const Assert& assert )
    {
        os << *assert.expression;
        if ( assert.message )
        {
            os << ": " << *assert.message;
        }
        for ( auto& print : assert.prints )
        {
            os << ", ";
            if ( !print.first.empty() )
            {
                os << print.first << ": ";
            }
            os << *print.second;
        }
        return os;
    }
    
    inline std::ostream& operator<<( std::ostream& os, const Lowering& lowering )
    {
        if ( lowering.unroll_count )
        {
            os << "unroll..(" << lowering.unroll_index << " -> " << *lowering.unroll_count << ")";
        }
        for ( auto& [iden, expr] : lowering.locals )
        {
            os << iden << " = " << *expr << ", ";
        }
        os << *lowering.left << ' ' << Lexer::str(lowering.op) << ' ' << *lowering.right;
        for ( auto& [iden, expr] : lowering.bounds )
        {
            os << ", " << iden << " < " << *expr;
        }
        if ( lowering.condition )
        {
            os << " | " << *lowering.condition;
        }
        return os;
    }
    
    inline std::ostream& operator<<( std::ostream& os, const Invocation& invocation )
    {
        if ( !invocation.label.empty() )
        {
            os << invocation.label << ':';
        }
        
        os << invocation.target;
        
        if ( invocation.dtypes.size() )
        {
            os << '<';
            for ( size_t i = 0; i < invocation.dtypes.size(); ++i )
            {
                if ( i > 0 )
                {
                    os << ", ";
                }
                os << invocation.dtypes[i].first;
            }
            os << '>';
        }
        
        if ( invocation.attribs.size() )
        {
            os << '{';
            for ( auto it = invocation.attribs.begin(); it != invocation.attribs.end(); ++it )
            {
                if ( it != invocation.attribs.begin() )
                {
                    os << ", ";
                }
                os << it->first << '=' << *it->second;
            }
            os << '}';
        }
        
        os << '(';
        for ( size_t i = 0; i < invocation.args.size(); ++i )
        {
            if ( i )
            {
                os << ", ";
            }
            if ( invocation.args[i] )
            {
                os << *invocation.args[i];
            }
            else
            {
                os << "~";
            }
        }
        os << ')';
        
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const Component& component );

    inline std::ostream& operator<<( std::ostream& os, const Region& region )
    {
        os << "{\n";
        for ( auto& component : region.components )
        {
            os << "\t\t\t" << component << ";\n";
        }
        os << "\t\t\t" << "yield ";
        size_t i = 0;
        for ( auto& yield : region.yields )
        {
            if ( i++ )
            {
                os << ", ";
            }
            os << yield;
        }
        os << ";\n";
        os << "\t\t" << '}';
        return os;
    }
    
    inline std::ostream& operator<<( std::ostream& os, const Component& component )
    {
        for ( size_t i = 0; i < component.results.size(); ++i )
        {
            if ( i )
            {
                os << ", ";
            }
            os << component.results[i];
        }
        
        os << " = ";
        
        if ( component.branches.size() )
        {
            size_t i = 0;
            for ( auto& [condition, consequent] : component.branches )
            {
                os << (i++ == 0 ? "if " : " elif ") << condition << " then " << consequent;
            }
            os << " else " << component.operation;
        }
        else if ( component.loop )
        {
            size_t i = 0;
            for ( auto& [iden, expr] : component.loop->carries )
            {
                os << (i++ ? ", " : "with ") << iden.name;
                if ( iden.shape )
                {
                    os << (!iden.type_alias.empty() ? iden.type_alias : str(iden.type.name)) << *iden.shape;
                }
                os << " = " << *expr;
            }
            if ( !component.loop->carries.empty() && !component.loop->scans.empty() )
            {
                os << ' ';
            }
            size_t j = 0;
            for ( auto& [iden, expr] : component.loop->scans )
            {
                os << (j++ ? ", " : "for ") << iden << " : " << *expr;
            }
            if ( component.loop->condition && component.loop->pretest )
            {
                if ( !component.loop->carries.empty() || !component.loop->scans.empty() )
                {
                    os << ' ';
                }
                os << "while " << *component.loop->condition;
            }
            if ( component.loop && component.loop->unroll )
            {
                os << " unroll ";
            }
            else
            {
                os << " do ";
            }
            if ( component.loop->index || component.loop->count )
            {
                os << ".." << '(';
                if ( component.loop->index )
                {
                    os << *component.loop->index << " -> ";
                }
                if ( component.loop->count )
                {
                    os << *component.loop->count;
                }
                os << ')';
            }
            os << component.operation;
            if ( component.loop->condition && !component.loop->pretest )
            {
                os << " while " << *component.loop->condition;
            }
        }
        else
        {
            os << component.operation;
        }
        
        return os;
    }
    
    inline std::ostream& operator<<( std::ostream& os, const Quantization& quant )
    {
        os << quant.tensor << ": " << quant.invocation.target;
        
        auto& types = quant.invocation.dtypes;
        if ( types.size() )
        {
            os << '<';
            for ( size_t i = 0; i < types.size(); ++i )
            {
                if ( i > 0 )
                {
                    os << ", ";
                }
                os << types[i].first;
            }
            os << '>';
        }
        
        os << '{';
        auto& attribs = quant.invocation.attribs;
        for ( auto it = attribs.begin(); it != attribs.end(); ++it )
        {
            if ( it != attribs.begin() )
            {
                os << ", ";
            }
            os << it->first << '=' << *it->second;
        }
        os << '}';
        
        return os;
    }
    
    template<typename T>
    inline void print_block( std::ostream& os, const std::string& name, const std::vector<T>& items )
    {
        if ( items.size() )
        {
            os << "\t@" << name << " {\n";
            for ( size_t i = 0; i < items.size(); ++i )
            {
                os << "\t\t" << items[i] << ";\n";
            }
            os << "\t}\n";
        }
    }
    
    inline std::ostream& operator<<( std::ostream& os, const Operator& op )
    {
        os << (op.graph ? "graph" : "operator") << ' ' << op.name << " {\n";
        
        print_block(os, "dtype", op.dtypes);
        print_block(os, "attrib", op.attribs);
        print_block(os, "input", op.inputs);
        print_block(os, "output", op.outputs);
        print_block(os, "variable", op.variables);
        print_block(os, "constant", op.constants);
        print_block(os, "assert", op.asserts);
        print_block(os, "using", op.usings);
        print_block(os, "lower", op.lowerings);
        print_block(os, "compose", op.components);
        print_block(os, "update", op.updates);
        print_block(os, "quantize", op.quantizations);
        
        os << "}\n";
        return os;
    }

    
    inline const Position& position( const Callable& callable )
    {
        return callable.is<Invocation>() ? callable.as<Invocation>().position : callable.as<Region>().position;
    }

}   // namespace sknd


#endif
