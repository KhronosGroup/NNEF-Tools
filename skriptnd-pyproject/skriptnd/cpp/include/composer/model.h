#ifndef _TS_MODEL_H_
#define _TS_MODEL_H_

#include <string>
#include <vector>
#include <memory>
#include <forward_list>
#include <iomanip>
#include "types.h"
#include "either.h"
#include "packable.h"
#include "valuexpr.h"
#include "ordereddict.h"
#include "tensorref.h"


namespace ts
{

    typedef std::function<bool( const std::string& name, const std::map<std::string,Typename>& dtypes,
                                const std::map<std::string,ValueExpr>& attribs, const std::vector<TensorRef>& inputs )> OperationCallback;

    inline bool TrueOperationCallback( const std::string&, const std::map<std::string,Typename>&,
                                       const std::map<std::string,ValueExpr>&, const std::vector<TensorRef>& ) { return true; }
    inline bool FalseOperationCallback( const std::string&, const std::map<std::string,Typename>&,
                                       const std::map<std::string,ValueExpr>&, const std::vector<TensorRef>& ) { return false; }
    

    /*
     * Data structure to desribe a tensor assignment via unary/binary/ternary operations over multi-dimensional tensor accesses
     * A contraction may result in reduction or broadcasting accross any combination of dimensions depending on the index variables
     *    shared accross the participating tensor accesses
     */
    struct Contraction
    {
        ValueExpr left;                                                 // resulting tensor
        ValueExpr right;                                                // right hand side of the contraction op
        ValueExpr condition;                                            // condition when the assignment is executed
        std::string assignment;                                         // assignment operator
        std::vector<std::pair<std::string,ValueExpr>> locals;           // expressions defining loop-local variables
        std::vector<std::pair<std::string,ValueExpr>> bounds;           // bounds for tensor index variables
        std::vector<size_t> subscripts;                                 // index variables that are used as pack subscripts
        std::vector<size_t> axes;                                       // reduction axes
    };


    /*
     * Data-structure to represent a single tensor in the graph
     */
    struct Tensor
    {
        std::string name;                                               // name of the tensor in the graph
        Typename dtype;                                                 // data-type of the tensor
        std::vector<ValueExpr> shape;                                   // shape of the tensor, possibly dynamic expressions
        std::vector<int_t> max_shape;                                   // upper bound of tensor shape
        std::map<std::string,ValueExpr> quant;                          // quantization info for the tensor
        ValueExpr value;                                                // value of constant tensors
        bool variable;                                                  // whether the tensor is a variable
    };


    /*
     * Data-structure to represent a (dynamic) pack (list) of tensors
     */
    struct TensorPack
    {
        std::vector<Tensor*> items;                                     // items in the pack
        std::string name;                                               // name of the pack
        Typename dtype;                                                 // dtype of the tensors in the pack
        std::vector<ValueExpr> shape;                                   // (partial) shape of tensors in the pack
        std::vector<int_t> max_shape;                                   // upper bound of tensor shape
        ValueExpr size;                                                 // dynamic size of the pack
    };



    /*
     * Data-structure to represent dynamic asserts
     */
    struct Assertion
    {
        ValueExpr condition;                                            // the condition to evaluate
        std::string message;                                            // the error message in case of failure
        std::vector<ValueExpr> args;                                    // the argmuments that need to be substituted into the formatted message
    };
    
    
    /*
     * Operation data-structure to represent a single operation in the graph
     */
    struct Operation
    {
        std::string name;                                               // (qualified) name of the operation
        std::map<std::string,Typename> dtypes;                          // dictionary of bound generic data types
        std::map<std::string,ValueExpr> attribs;                        // dictionary of attributes (values or shape expressions)
        std::vector<TensorRef> inputs;                                  // list of input tensors, items may be packed
        std::vector<TensorRef> outputs;                                 // list of output tensors, items may be packed
        std::vector<TensorRef> internals;                               // list of internal tensors, items may be packed
        std::vector<Contraction> contractions;                          // list of contractions that define the lowering of the operation
        std::vector<Assertion> asserts;                                 // list of dynamic asserts that need to be checked in run-time
        OrderedDict<ValueExpr> subexprs;                                // dictionary shared sub-expressions
    };
    
    
    /*
     * Graph data-structure that represents a sequence of operations and related tensors
     */
    struct Graph
    {
        std::string name;                                               // name of this graph
        std::vector<Operation> operations;                              // list of operations, in topograpic order
        std::vector<TensorRef> inputs;                                  // list of input tensors, items may be packed
        std::vector<TensorRef> outputs;                                 // list of output tensors, items may be packed
        std::vector<std::unique_ptr<Tensor>> tensors;                   // list of tensors in the graph
        std::vector<std::unique_ptr<TensorPack>> packs;                 // list of tensor packs in the graph
        std::vector<Assertion> asserts;                                 // list of dynamic asserts that need to be checked in run-time
    };
    
    
    /*
     * Model data-structure, list of tensors and operation graphs
     * The main graph is at index 0, other subgraphs are referenced by control-flow ops via indices
     */
    struct Model
    {
        std::string name;                                               // name of the model
        std::vector<Graph> graphs;                                      // list of graphs in the model
    };
    

            
    inline long& _stream_indent( std::ostream& os )
    {
        static int index = std::ios_base::xalloc();
        return os.iword(index);
    }

    inline long& _stream_flags( std::ostream& os )
    {
        static int index = std::ios_base::xalloc();
        return os.iword(index);
    }
            
    inline void _set_flags( std::ostream& os, long flags )
    {
        _stream_flags(os) |= flags;
    }
            
    inline bool _get_flags( std::ostream& os, long flags )
    {
        return (_stream_flags(os) & flags) != 0;
    }
            
    inline void _clear_flags( std::ostream& os, long flags )
    {
        _stream_flags(os) &= ~flags;
    }
            
    inline bool _reset_flags( std::ostream& os, long flags )
    {
        bool value = _get_flags(os, flags);
        _clear_flags(os, flags);
        return value;
    }
    

    inline std::ostream& nobrackets( std::ostream& os )
    {
        _set_flags(os, 0x01);
        return os;
    }

    inline bool fetch_nobrackets( std::ostream& os )
    {
        return _reset_flags(os, 0x01);
    }
            
    struct indent
    {
        int value;
        
        indent( int value ) : value(value) {}
    };
    
    inline std::ostream& operator<<( std::ostream& os, const indent& indent )
    {
        _stream_indent(os) = indent.value;
        return os;
    }
    
    inline int fetch_indent( std::ostream& os )
    {
        int value = (int)_stream_indent(os);
        _stream_indent(os) = 0;
        return value;
    }

    inline std::ostream& operator<<( std::ostream& os, const std::vector<int_t>& shape )
    {
        os << '[';
        for ( size_t i = 0; i < shape.size(); ++i )
        {
            if ( i )
            {
                os << ",";
            }
            os << shape[i];
        }
        os << ']';
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const std::vector<ValueExpr>& shape )
    {
        os << '[';
        for ( size_t i = 0; i < shape.size(); ++i )
        {
            if ( i )
            {
                os << ",";
            }
            os << shape[i];
        }
        os << ']';
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const Tensor& tensor )
    {
        os << tensor.name << ": " << str(tensor.dtype) << tensor.shape;
        if ( tensor.value != nullptr )
        {
            os << " = " << tensor.value;
        }
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const TensorPack& pack )
    {
        if ( pack.name.front() != '.' )
        {
            os << pack.name << ": " << str(pack.dtype) << pack.shape << "..(" << pack.size << ")";
        }
        else
        {
            os << '[';
            for ( size_t i = 0; i < pack.items.size(); ++i )
            {
                if ( i )
                {
                    os << ", ";
                }
                os << *pack.items[i];
            }
            os << ']';
        }
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const TensorRef& ref )
    {
        if ( ref.packed() )
        {
            os << (const TensorPack&)ref;
        }
        else
        {
            os << (const Tensor&)ref;
        }
        return os;
    }
    
    inline std::ostream& operator<<( std::ostream& os, const Assertion& assertion )
    {
        os << assertion.condition << ": " << '"' << assertion.message << '"';
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const Contraction& contraction )
    {
        int indent = fetch_indent(os);
        
        std::string indentation;
        for ( int i = 0; i < indent; ++i )
        {
            indentation += '\t';
        }
        
        for ( auto& [iden, expr] : contraction.locals )
        {
            os << indentation << iden << " = " << expr << ",\n";
        }
        
        os << indentation << contraction.left << ' ' << contraction.assignment << ' ' << nobrackets << contraction.right;
        
        bool first = true;
        for ( auto& [iden,affine] : contraction.bounds )
        {
            os << (first ? "\n" + indentation + "\t" : ", ") << iden << " < " << affine;
            first = false;
        }
        
        if ( contraction.condition != nullptr )
        {
            os << " | " << contraction.condition;
        }
        
        return os;
    }
    
    inline std::ostream& operator<<( std::ostream& os, const Operation& op )
    {
        for ( size_t i = 0; i < op.outputs.size(); ++i )
        {
            if ( i )
            {
                os << ", ";
            }
            os << op.outputs[i];
        }
        
        os << " = ";
        
        if ( op.name.empty() )
        {
            if ( !op.inputs.empty() )
            {
                os << op.inputs.front().name();
            }
            else
            {
                os << op.attribs.at("");
            }
            os << ';';
            return os;
        }
        
        os << op.name;
        
        if ( !op.dtypes.empty() )
        {
            os << '<';
            for ( auto it = op.dtypes.begin(); it != op.dtypes.end(); ++it )
            {
                if ( it != op.dtypes.begin() )
                {
                    os << ',';
                }
                os << str(it->second);
            }
            os << '>';
        }
        
        if ( !op.attribs.empty() )
        {
            os << '{';
            size_t count = 0;
            for ( auto& [name, value] : op.attribs )
            {
                if ( value != nullptr )
                {
                    if ( count++ )
                    {
                        os << ", ";
                    }
                    os << name << '=' << value;
                }
            }
            os << '}';
        }
        
        os << '(';
        for ( size_t i = 0; i < op.inputs.size(); ++i )
        {
            if ( i )
            {
                os << ", ";
            }
            auto& input = op.inputs[i];
            if ( input == nullptr )
            {
                os << "~";
            }
            else if ( input.packed() && input.name().front() == '.' )
            {
                os << '[';
                for ( size_t j = 0; j < input.max_size(); ++j )
                {
                    if ( j )
                    {
                        os << ", ";
                    }
                    os << input[j].name;
                }
                os << ']';
            }
            else
            {
                os << input.name();
            }
        }
        os << ')';
        
        os << ';';
        
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const Graph& graph )
    {
        os << "graph " << graph.name << " {" << std::endl;
        
        os << "\t@input {" << std::endl;
        for ( auto& input : graph.inputs )
        {
            os << "\t\t" << input << ';' << std::endl;
        }
        os << "\t}" << std::endl;
        
        os << "\t@output {" << std::endl;
        for ( auto& output : graph.outputs )
        {
            os << "\t\t" << output << ';' << std::endl;
        }
        os << "\t}" << std::endl;
        
        if ( !graph.asserts.empty() )
        {
            os << "\t@assert {" << std::endl;
            for ( auto& assert : graph.asserts )
            {
                os << "\t\t" << assert << ";" << std::endl;
            }
            os << "\t}" << std::endl;
        }
        
        size_t constants = 0;
        size_t variables = 0;
        for ( auto& tensor : graph.tensors )
        {
            if ( tensor->variable )
            {
                ++variables;
            }
            else if ( tensor->value != nullptr )
            {
                ++constants;
            }
        }
        
        if ( constants )
        {
            os << "\t@constant {" << std::endl;
            for ( auto& tensor : graph.tensors )
            {
                if ( tensor->value != nullptr )
                {
                    os << "\t\t" << *tensor << ';' << std::endl;
                }
            }
            os << "\t}" << std::endl;
        }
        
        if ( variables )
        {
            os << "\t@variable {" << std::endl;
            for ( auto& tensor : graph.tensors )
            {
                if ( tensor->variable )
                {
                    os << "\t\t" << *tensor << ';' << std::endl;
                }
            }
            os << "\t}" << std::endl;
        }
        
        os << "\t@compose {" << std::endl;
        for ( auto& op : graph.operations )
        {
            for ( auto& assert : op.asserts )
            {
                os << "\t\t" << "assert " << assert << ";" << std::endl;
            }
            
            os << "\t\t" << op << std::endl;
            
            for ( auto& [name, expr] : op.subexprs )
            {
                os << "\t\t\t" << name << " = " << expr << std::endl;
            }
            for ( auto& contraction : op.contractions )
            {
                os << indent(3) << contraction << std::endl;
            }
        }
        os << "\t}" << std::endl;
        
        os << "}" << std::endl;
        
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const Model& model )
    {
        for ( auto& graph : model.graphs )
        {
            os << graph << std::endl;
        }
        
        return os;
    }
    
}   // namespace ts


template<>
struct std::hash<ts::TensorRef>
{
    std::size_t operator()( const ts::TensorRef& r ) const noexcept
    {
        if ( r.packed() )
        {
            std::hash<ts::TensorPack*> hasher;
            return hasher(r.as<ts::TensorPack*>());
        }
        else
        {
            std::hash<ts::Tensor*> hasher;
            return hasher(r.as<ts::Tensor*>());
        }
    }
};


#endif
