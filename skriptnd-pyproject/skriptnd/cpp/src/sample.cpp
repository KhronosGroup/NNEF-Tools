#include <fstream>
#include "skriptnd.h"
#include "runtime.h"


int main( int argc, const char * argv[] )
{
    auto error_handler = [&]( const sknd::Position& position, const std::string& message, const sknd::StackTrace& stacktrace, const bool warning )
    {
        std::cout << (warning ? "⚠️ Warning" : "🛑 Error") << " in module '" << position.module << "'";
        if ( position.line )
        {
            std::cout << " [" << position.line << ':' << position.column << "]";
        }
        std::cout << ": " << message << std::endl;
        for ( auto it = stacktrace.rbegin(); it != stacktrace.rend(); ++it )
        {
            auto& [op, pos] = *it;
            std::cout << "\twhile calling operator '" << op << "' in module '" << pos.module;
            std::cout << "' [" << pos.line << ':' << pos.column << "]" << std::endl;
        }
    };
    
    if ( argc < 2 )
    {
        std::cerr << "Input file must be provided" << std::endl;
        return -1;
    }
    
    bool all = false;
    bool verbose = false;
    sknd::OperationCallback atomic = sknd::FalseOperationCallback;
    sknd::OperationCallback unroll = sknd::FalseOperationCallback;
    for ( size_t i = 2; i < argc; ++i )
    {
        const std::string arg = argv[i];
        if ( arg == "--all" )
        {
            all = true;
        }
        else if ( arg == "--verbose" )
        {
            verbose = true;
        }
        else if ( arg == "--atomic" )
        {
            atomic = sknd::TrueOperationCallback;
        }
        else if ( arg == "--unroll" )
        {
            unroll = sknd::TrueOperationCallback;
        }
    }
    
    const std::string fn = argv[1];
    auto beg = fn.find_last_of("\\/") + 1;
    auto end = fn.find_last_of(".");
    const std::string module = fn.substr(beg, end - beg);
    
    std::ifstream is(fn);
    if ( !is )
    {
        std::cerr << "Could not open file: " << fn << std::endl;
        return -1;
    }
    
    auto graph_names = sknd::enum_graph_names(is);
    for ( auto& graph_name : graph_names )
    {
        is.close();
        is.open(fn);
        
        auto model = sknd::read_model(is, module.c_str(), graph_name, "skriptnd/stdlib/", "", error_handler, atomic, unroll);
        if ( model )
        {
            std::cout << "✅ Succesfully parsed graph '" + graph_name + "'" << std::endl;
            if ( verbose )
            {
                std::cout << *model << std::endl;
            }
        }
        else
        {
            std::cout << "❌ Parse failed for graph '" + graph_name + "'" << std::endl;
        }
        if ( !all )
        {
            break;
        }
    }
    
    return 0;
}
